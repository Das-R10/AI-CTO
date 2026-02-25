"""
job_system.py — Background Job Queue
=====================================
Runs agent tasks in a background thread.
Stores status in Postgres agent_jobs table.
No external infrastructure required.
"""

import threading
import traceback
from typing import Optional, Callable
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import SessionLocal


# ─────────────────────────────────────────────
# JOB MANAGEMENT
# ─────────────────────────────────────────────

def create_job(project_id: int, query: str, db: Session) -> int:
    """Create a pending job record, return job_id."""
    row = db.execute(
        text("""
            INSERT INTO agent_jobs (project_id, query, status)
            VALUES (:pid, :query, 'pending')
            RETURNING id
        """),
        {"pid": project_id, "query": query}
    ).fetchone()
    db.commit()
    return row[0]


def update_job(job_id: int, status: str, result: dict = None, error: str = None):
    """Update job status — uses its own DB session (runs in background thread)."""
    db = SessionLocal()
    try:
        db.execute(
            text("""
                UPDATE agent_jobs
                SET status=:status,
                    result=CAST(:result AS jsonb),
                    error=:error,
                    updated_at=now()
                WHERE id=:id
            """),
            {
                "status": status,
                "result": __import__("json").dumps(result) if result else None,
                "error":  error,
                "id":     job_id
            }
        )
        db.commit()
    finally:
        db.close()


def get_job_status(job_id: int, db: Session) -> Optional[dict]:
    row = db.execute(
        text("SELECT id, project_id, query, status, result, error, created_at, updated_at FROM agent_jobs WHERE id=:id"),
        {"id": job_id}
    ).fetchone()
    if not row:
        return None
    return {
        "job_id":     row[0],
        "project_id": row[1],
        "query":      row[2],
        "status":     row[3],
        "result":     row[4],
        "error":      row[5],
        "created_at": str(row[6]),
        "updated_at": str(row[7])
    }


# ─────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────

def dispatch_job(
    job_id: int,
    task_fn: Callable,
    **kwargs
):
    """
    Run task_fn(**kwargs) in a background thread.
    Updates job status to running → completed/failed automatically.
    task_fn must accept job_id as a kwarg for progress updates.
    """
    def _run():
        update_job(job_id, "running")
        try:
            result = task_fn(job_id=job_id, **kwargs)
            update_job(job_id, "completed", result=result)
        except Exception as e:
            tb = traceback.format_exc()
            update_job(job_id, "failed", error=f"{e}\n{tb[:500]}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread