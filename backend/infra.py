"""
infra.py — Infrastructure Upgrades
=====================================
Consolidates two infrastructure improvements into one module:

  1. LLM Router       — circuit breaker + exponential backoff + fallback model support
                        Drop-in replacement for the raw groq_llm callable in main.py
  2. Persistent Queue — Postgres-backed job queue that survives server restarts
                        Drop-in replacement for job_system.py's threading-based dispatch

Backward compatibility:
  - groq_llm callable in main.py is replaced with LLMRouter.call — same signature
  - create_job / update_job / get_job_status / dispatch_job in job_system.py
    continue to work unchanged; this module adds claim-based recovery on top

New DB columns required (run once):
  ALTER TABLE agent_jobs ADD COLUMN IF NOT EXISTS claimed_by       VARCHAR(128);
  ALTER TABLE agent_jobs ADD COLUMN IF NOT EXISTS claim_expires_at TIMESTAMPTZ;
  ALTER TABLE agent_jobs ADD COLUMN IF NOT EXISTS retry_count      INTEGER DEFAULT 0;
  ALTER TABLE agent_jobs ADD COLUMN IF NOT EXISTS max_retries      INTEGER DEFAULT 3;
"""

import os
import time
import uuid
import socket
import threading
import traceback
import json
from dataclasses import dataclass, field
from typing import Optional, Callable
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import SessionLocal


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — LLM ROUTER
# ═══════════════════════════════════════════════════════════════

@dataclass
class LLMProvider:
    """
    Configuration for a single LLM provider.

    client_fn signature: (system_prompt: str, user_prompt: str) -> str
    priority: lower number = tried first
    """
    name:      str
    client_fn: Callable[[str, str], str]
    priority:  int = 0


class RateLimitError(Exception):
    """Raised when an LLM provider returns a rate limit response."""
    pass


class LLMRouter:
    """
    Multi-provider LLM router with:
      - Priority ordering (lower priority number = preferred provider)
      - Per-provider circuit breaker (opens on repeated failures, auto-resets)
      - Exponential backoff retry within a provider before failing over
      - Transparent fallback to next provider in the list

    Usage:
        router = LLMRouter([
            LLMProvider("groq-primary",   groq_llm_fn,    priority=0),
            LLMProvider("groq-fallback",  groq_llm_fn_2,  priority=1),
        ])
        # Then pass router.call as the groq_llm argument everywhere
        response = router.call(system_prompt, user_prompt)
    """

    # Circuit breaker config
    CIRCUIT_OPEN_DURATION  = 60    # seconds before a tripped circuit auto-resets
    CIRCUIT_FAILURE_THRESH = 3     # consecutive failures before circuit opens

    def __init__(self, providers: list[LLMProvider]):
        self._providers      = sorted(providers, key=lambda p: p.priority)
        self._circuit_open   = {}   # provider_name → open_until (timestamp)
        self._failure_counts = {}   # provider_name → consecutive failure count
        self._lock           = threading.Lock()
        self._request_log: list[dict] = []   # recent request history (last 100)

    def _is_circuit_open(self, name: str) -> bool:
        open_until = self._circuit_open.get(name, 0)
        if time.time() < open_until:
            return True
        # Auto-reset: if circuit was open but time has passed, reset failure count
        if name in self._circuit_open and time.time() >= open_until:
            with self._lock:
                self._circuit_open.pop(name, None)
                self._failure_counts[name] = 0
        return False

    def _record_failure(self, name: str):
        with self._lock:
            count = self._failure_counts.get(name, 0) + 1
            self._failure_counts[name] = count
            if count >= self.CIRCUIT_FAILURE_THRESH:
                open_until = time.time() + self.CIRCUIT_OPEN_DURATION
                self._circuit_open[name] = open_until
                print(f"[LLMRouter] ⚡ Circuit OPEN for '{name}' — cooling off for {self.CIRCUIT_OPEN_DURATION}s")

    def _record_success(self, name: str):
        with self._lock:
            self._failure_counts[name] = 0
            self._circuit_open.pop(name, None)

    def _log_request(self, provider: str, success: bool, latency_ms: float, error: str = ""):
        entry = {
            "ts":          time.time(),
            "provider":    provider,
            "success":     success,
            "latency_ms":  round(latency_ms, 1),
            "error":       error
        }
        self._request_log.append(entry)
        if len(self._request_log) > 100:
            self._request_log = self._request_log[-100:]

    def call(
        self,
        system_prompt: str,
        user_prompt:   str,
        max_retries:   int = 3,
        base_delay:    float = 1.0
    ) -> str:
        """
        Call the LLM with automatic failover and retry.

        Args:
            system_prompt: System/instruction prompt
            user_prompt:   User message
            max_retries:   Max attempts per provider before failing over
            base_delay:    Base delay for exponential backoff (seconds)

        Returns:
            LLM response string

        Raises:
            RuntimeError if all providers fail or are circuit-broken
        """
        last_error = ""

        for provider in self._providers:
            if self._is_circuit_open(provider.name):
                print(f"[LLMRouter] Skipping '{provider.name}' — circuit open")
                continue

            for attempt in range(1, max_retries + 1):
                t_start = time.time()
                try:
                    response = provider.client_fn(system_prompt, user_prompt)
                    latency  = (time.time() - t_start) * 1000
                    self._record_success(provider.name)
                    self._log_request(provider.name, True, latency)
                    if attempt > 1 or provider != self._providers[0]:
                        print(f"[LLMRouter] ✅ Response from '{provider.name}' (attempt {attempt})")
                    return response

                except RateLimitError as e:
                    latency = (time.time() - t_start) * 1000
                    delay   = base_delay * (2 ** (attempt - 1))
                    last_error = str(e)
                    self._log_request(provider.name, False, latency, f"RateLimit: {e}")
                    print(f"[LLMRouter] Rate limit on '{provider.name}' attempt {attempt} — waiting {delay:.1f}s")
                    if attempt < max_retries:
                        time.sleep(delay)

                except Exception as e:
                    latency = (time.time() - t_start) * 1000
                    last_error = str(e)
                    self._log_request(provider.name, False, latency, str(e))
                    self._record_failure(provider.name)
                    print(f"[LLMRouter] ❌ '{provider.name}' attempt {attempt} failed: {e}")
                    if attempt < max_retries:
                        time.sleep(base_delay * attempt)
                    break   # Move to next provider after non-rate-limit errors

        raise RuntimeError(
            f"All LLM providers failed or are circuit-broken. Last error: {last_error}"
        )

    def status(self) -> dict:
        """Return current router health status."""
        provider_status = {}
        for p in self._providers:
            open_until = self._circuit_open.get(p.name, 0)
            provider_status[p.name] = {
                "circuit_open":    time.time() < open_until,
                "open_until":      open_until if time.time() < open_until else None,
                "failure_count":   self._failure_counts.get(p.name, 0),
                "priority":        p.priority
            }
        recent = self._request_log[-20:]
        success_rate = (
            sum(1 for r in recent if r["success"]) / len(recent)
            if recent else 1.0
        )
        return {
            "providers":     provider_status,
            "success_rate_recent_20": round(success_rate, 3),
            "total_requests_logged":  len(self._request_log)
        }


def build_groq_router(groq_client, model_primary: str = "llama-3.1-8b-instant") -> LLMRouter:
    """
    Factory function to create a Groq-backed LLMRouter.
    Wraps the groq client into the LLMProvider interface.

    Provides a primary and fallback model using the same Groq client.
    If Groq itself goes down, both will fail — add a second provider for true redundancy.

    Usage in main.py:
        from infra import build_groq_router, LLMRouter
        router = build_groq_router(groq_client)
        groq_llm = router.call   # drop-in replacement
    """
    def _make_fn(model_name: str):
        def _call(system_prompt: str, user_prompt: str) -> str:
            try:
                response = groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt}
                    ],
                    temperature=0.2,
                )
                return response.choices[0].message.content
            except Exception as e:
                err_str = str(e).lower()
                if "rate_limit" in err_str or "429" in err_str:
                    raise RateLimitError(str(e))
                raise
        return _call

    providers = [
        LLMProvider(
            name=f"groq-{model_primary}",
            client_fn=_make_fn(model_primary),
            priority=0
        ),
    ]

    # Add fallback model if a different one is available
    fallback_model = "llama-3.1-70b-versatile"
    if fallback_model != model_primary:
        providers.append(LLMProvider(
            name=f"groq-{fallback_model}",
            client_fn=_make_fn(fallback_model),
            priority=1
        ))

    return LLMRouter(providers)


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — PERSISTENT JOB QUEUE
# ═══════════════════════════════════════════════════════════════

# Worker identity — unique per process instance
WORKER_ID = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"

# How long a worker can hold a job claim before it's considered abandoned
CLAIM_TIMEOUT_SECONDS = 300   # 5 minutes


def claim_pending_job(db: Session) -> Optional[dict]:
    """
    Atomically claim one pending job for processing.
    Uses SELECT FOR UPDATE SKIP LOCKED to prevent double-claiming.
    Returns the job dict if claimed, None if no pending jobs.

    Automatically reclaims abandoned jobs (claimed_by is set but claim expired).
    """
    now_ts = time.time()

    # First: reclaim abandoned jobs (worker died mid-execution)
    db.execute(
        text("""
            UPDATE agent_jobs
            SET status = 'pending',
                claimed_by = NULL,
                claim_expires_at = NULL
            WHERE status = 'running'
            AND claim_expires_at IS NOT NULL
            AND claim_expires_at < now()
            AND retry_count < max_retries
        """)
    )
    db.commit()

    # Claim one pending job
    row = db.execute(
        text("""
            UPDATE agent_jobs
            SET status = 'running',
                claimed_by = :worker_id,
                claim_expires_at = now() + INTERVAL ':timeout seconds',
                retry_count = retry_count + 1
            WHERE id = (
                SELECT id FROM agent_jobs
                WHERE status = 'pending'
                AND (retry_count < max_retries OR max_retries IS NULL)
                ORDER BY created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING id, project_id, query, retry_count
        """.replace(":timeout seconds", f"{CLAIM_TIMEOUT_SECONDS} seconds")),
        {"worker_id": WORKER_ID}
    ).fetchone()
    db.commit()

    if not row:
        return None

    return {
        "job_id":      row[0],
        "project_id":  row[1],
        "query":       row[2],
        "retry_count": row[3]
    }


def extend_claim(job_id: int, db: Session):
    """
    Extend the claim expiry for a long-running job.
    Call this periodically from long tasks to prevent abandonment reclaim.
    """
    db.execute(
        text("""
            UPDATE agent_jobs
            SET claim_expires_at = now() + INTERVAL ':timeout seconds'
            WHERE id = :jid AND claimed_by = :worker_id
        """.replace(":timeout seconds", f"{CLAIM_TIMEOUT_SECONDS} seconds")),
        {"jid": job_id, "worker_id": WORKER_ID}
    )
    db.commit()


def complete_job(job_id: int, result: dict, db: Session):
    """Mark a job as completed with its result."""
    db.execute(
        text("""
            UPDATE agent_jobs
            SET status = 'completed',
                result = CAST(:result AS jsonb),
                claimed_by = NULL,
                claim_expires_at = NULL,
                updated_at = now()
            WHERE id = :jid
        """),
        {"result": json.dumps(result), "jid": job_id}
    )
    db.commit()


def fail_job(job_id: int, error: str, db: Session):
    """
    Mark a job as failed. If retry_count < max_retries, reset to pending
    so it can be reclaimed and retried.
    """
    row = db.execute(
        text("SELECT retry_count, max_retries FROM agent_jobs WHERE id=:jid"),
        {"jid": job_id}
    ).fetchone()

    if row and row[0] < (row[1] or 3):
        # Retry: reset to pending
        db.execute(
            text("""
                UPDATE agent_jobs
                SET status = 'pending',
                    error = :error,
                    claimed_by = NULL,
                    claim_expires_at = NULL,
                    updated_at = now()
                WHERE id = :jid
            """),
            {"error": error[:500], "jid": job_id}
        )
        print(f"[PersistentQueue] Job #{job_id} reset to pending for retry (attempt {row[0]}/{row[1]})")
    else:
        # Exhausted retries: mark permanently failed
        db.execute(
            text("""
                UPDATE agent_jobs
                SET status = 'failed',
                    error = :error,
                    claimed_by = NULL,
                    claim_expires_at = NULL,
                    updated_at = now()
                WHERE id = :jid
            """),
            {"error": error[:500], "jid": job_id}
        )
        print(f"[PersistentQueue] Job #{job_id} permanently failed")

    db.commit()


def dispatch_persistent_job(
    job_id:  int,
    task_fn: Callable,
    **kwargs
) -> threading.Thread:
    """
    Drop-in replacement for job_system.dispatch_job().
    Uses persistent queue semantics: task failures trigger retry logic
    rather than simply marking the job failed.

    Same interface as the original dispatch_job — backward compatible.
    """
    def _run():
        db = SessionLocal()
        try:
            result = task_fn(job_id=job_id, **kwargs)
            complete_job(job_id, result, db)
        except Exception as e:
            tb = traceback.format_exc()
            fail_job(job_id, f"{e}\n{tb[:400]}", db)
        finally:
            db.close()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread


def start_persistent_worker(task_registry: dict[str, Callable], poll_interval: int = 5):
    """
    Start a background polling worker that continuously claims and executes
    jobs from the persistent queue.

    task_registry maps job_type strings to callable task functions.
    For the current system, there's only one task type ("agent"), but this
    architecture supports future multi-type queues.

    This worker is an alternative to dispatch_persistent_job() — use one or the other.
    For the current system, dispatch_persistent_job() is simpler and recommended.

    Args:
        task_registry:  dict of {"job_type": task_fn}
        poll_interval:  seconds between queue polls when idle
    """
    def _worker_loop():
        print(f"[PersistentQueue] Worker started ({WORKER_ID})")
        while True:
            db = SessionLocal()
            try:
                job = claim_pending_job(db)
                if not job:
                    db.close()
                    time.sleep(poll_interval)
                    continue

                job_id     = job["job_id"]
                project_id = job["project_id"]
                query      = job["query"]
                print(f"[PersistentQueue] Claimed job #{job_id} (project {project_id}, attempt {job['retry_count']})")

                # Default task function: run_agent
                task_fn = task_registry.get("agent")
                if not task_fn:
                    fail_job(job_id, "No task function registered for type 'agent'", db)
                    db.close()
                    continue

                try:
                    result = task_fn(job_id=job_id, query=query, project_id=project_id)
                    complete_job(job_id, result, db)
                    print(f"[PersistentQueue] Job #{job_id} completed")
                except Exception as e:
                    tb = traceback.format_exc()
                    fail_job(job_id, f"{e}\n{tb[:400]}", db)

            except Exception as outer_e:
                print(f"[PersistentQueue] Worker loop error: {outer_e}")
            finally:
                try:
                    db.close()
                except Exception:
                    pass

    thread = threading.Thread(target=_worker_loop, daemon=True, name="persistent-queue-worker")
    thread.start()
    print(f"[PersistentQueue] Background worker thread started")
    return thread


# ═══════════════════════════════════════════════════════════════
# INTEGRATION GUIDE — main.py changes
# ═══════════════════════════════════════════════════════════════
#
# ── 1. Replace groq_llm with router ──────────────────────────
#
# BEFORE (in main.py):
#   groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#   def groq_llm(system_prompt: str, user_prompt: str):
#       response = groq_client.chat.completions.create(...)
#       return response.choices[0].message.content
#
# AFTER:
#   from infra import build_groq_router
#   groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#   _router     = build_groq_router(groq_client, model_primary="llama-3.1-8b-instant")
#   groq_llm    = _router.call   # same (system_prompt, user_prompt) -> str interface
#
#
# ── 2. Add router status endpoint ────────────────────────────
#
# @app.get("/llm-status/")
# def llm_status(key_project_id: int = Depends(require_api_key)):
#     return _router.status()
#
#
# ── 3. Replace dispatch_job with persistent version ──────────
#
# BEFORE (in /modify-code-async/):
#   from job_system import create_job, dispatch_job, get_job_status
#   dispatch_job(job_id, _task, ...)
#
# AFTER:
#   from job_system import create_job, get_job_status
#   from infra import dispatch_persistent_job
#   dispatch_persistent_job(job_id, _task, ...)
#
#   (create_job and get_job_status from job_system are unchanged)
#
#
# ── 4. DB migrations (run once) ──────────────────────────────
#
# ALTER TABLE agent_jobs ADD COLUMN IF NOT EXISTS claimed_by       VARCHAR(128);
# ALTER TABLE agent_jobs ADD COLUMN IF NOT EXISTS claim_expires_at TIMESTAMPTZ;
# ALTER TABLE agent_jobs ADD COLUMN IF NOT EXISTS retry_count      INTEGER DEFAULT 0;
# ALTER TABLE agent_jobs ADD COLUMN IF NOT EXISTS max_retries      INTEGER DEFAULT 3;