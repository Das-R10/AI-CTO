"""
autonomous_ops.py — Autonomous Operation Layer
===============================================
Consolidates all proactive/scheduled intelligence into one module:

  1. Timeline Engine      — code evolution history, change velocity, function history
  2. Performance Monitor  — benchmark capture + regression detection
  3. Behavioral Diff      — pre/post function output comparison
  4. Drift Detector       — architectural decay detection against snapshots
  5. Autonomous Scheduler — APScheduler-backed cron jobs (daily audit, weekly refactor,
                            6-hour drift check, daily memory decay)

This module is entirely additive — it does not modify any existing file.
All existing endpoints and functions remain unchanged.

Integration: call setup_scheduler(DATABASE_URL) from main.py startup event.
See INTEGRATION GUIDE at the bottom.

New DB tables required (run once):
  ALTER TABLE project_memory ADD COLUMN IF NOT EXISTS confidence     FLOAT   DEFAULT 1.0;
  ALTER TABLE project_memory ADD COLUMN IF NOT EXISTS superseded_by  INTEGER REFERENCES project_memory(id);
  ALTER TABLE project_memory ADD COLUMN IF NOT EXISTS decay_rate     FLOAT   DEFAULT 0.01;

  CREATE TABLE IF NOT EXISTS performance_baselines (
      id            SERIAL PRIMARY KEY,
      project_id    INTEGER REFERENCES projects(id),
      file_id       INTEGER REFERENCES files(id),
      function_name VARCHAR(255),
      mean_ms       FLOAT,
      p95_ms        FLOAT,
      input_hash    VARCHAR(32),
      recorded_at   TIMESTAMPTZ DEFAULT now()
  );

  CREATE TABLE IF NOT EXISTS architectural_snapshots (
      id          SERIAL PRIMARY KEY,
      project_id  INTEGER REFERENCES projects(id),
      snapshot    JSONB,
      created_at  TIMESTAMPTZ DEFAULT now()
  );

  CREATE TABLE IF NOT EXISTS drift_reports (
      id          SERIAL PRIMARY KEY,
      project_id  INTEGER REFERENCES projects(id),
      report      JSONB,
      violations  INTEGER,
      created_at  TIMESTAMPTZ DEFAULT now()
  );

  CREATE TABLE IF NOT EXISTS scheduled_job_log (
      id          SERIAL PRIMARY KEY,
      job_name    VARCHAR(128),
      project_id  INTEGER,
      status      VARCHAR(32),
      result      JSONB,
      started_at  TIMESTAMPTZ,
      finished_at TIMESTAMPTZ
  );
"""

import ast
import re
import os
import sys
import json
import time
import hashlib
import tempfile
import subprocess
import statistics
from dataclasses import dataclass, field
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import SessionLocal


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — TIMELINE ENGINE
# ═══════════════════════════════════════════════════════════════

def get_file_change_velocity(file_id: int, db: Session, days: int = 30) -> float:
    """
    Returns the average number of modifications per day for a file
    over the last N days. Uses file_versions table as the source of truth.
    """
    row = db.execute(
        text("""
            SELECT COUNT(*) FROM file_versions
            WHERE file_id = :fid
            AND created_at >= now() - INTERVAL ':days days'
        """.replace(":days days", f"{days} days")),
        {"fid": file_id}
    ).fetchone()
    count = row[0] if row else 0
    return round(count / days, 3)


def get_function_history(function_name: str, project_id: int, db: Session) -> list[dict]:
    """
    Returns a chronological log of every agent decision that touched this function.
    Pulls from project_memory where memory_type='decision' and content references the function.
    """
    rows = db.execute(
        text("""
            SELECT id, content, created_at
            FROM project_memory
            WHERE project_id = :pid
            AND memory_type = 'decision'
            AND content ILIKE :fn_pattern
            ORDER BY created_at ASC
        """),
        {"pid": project_id, "fn_pattern": f"%{function_name}%"}
    ).fetchall()

    return [
        {"memory_id": r[0], "content": r[1], "at": str(r[2])}
        for r in rows
    ]


def get_code_evolution_summary(project_id: int, db: Session) -> dict:
    """
    Returns a project-level code evolution summary:
      - Most volatile files (highest change velocity)
      - Most stable files (lowest change velocity)
      - Total modifications in last 30 days
      - Quality trend (current vs 30 days ago snapshot)
    """
    from models import File as DBFile

    files = db.query(DBFile).filter(DBFile.project_id == project_id).all()
    velocity_map = {}

    for f in files:
        velocity_map[f.filename] = get_file_change_velocity(f.id, db, days=30)

    sorted_files = sorted(velocity_map.items(), key=lambda x: x[1], reverse=True)
    total_mods   = db.execute(
        text("""
            SELECT COUNT(*) FROM file_versions fv
            JOIN files f ON f.id = fv.file_id
            WHERE f.project_id = :pid
            AND fv.created_at >= now() - INTERVAL '30 days'
        """),
        {"pid": project_id}
    ).scalar() or 0

    return {
        "project_id":      project_id,
        "total_mods_30d":  total_mods,
        "most_volatile":   [{"file": f, "mods_per_day": v} for f, v in sorted_files[:5]],
        "most_stable":     [{"file": f, "mods_per_day": v} for f, v in sorted_files[-5:] if v == 0],
    }


def decay_old_memories(project_id: int, db: Session, half_life_days: int = 90):
    """
    Reduce the confidence score of memories older than half_life_days.
    Memories at exactly half_life_days old will have confidence ~0.5x original.
    Uses exponential decay: confidence *= e^(-decay_rate * days_old)
    Skips memories with superseded_by set (already invalidated).
    """
    import math

    rows = db.execute(
        text("""
            SELECT id, confidence, created_at
            FROM project_memory
            WHERE project_id = :pid
            AND superseded_by IS NULL
            AND created_at < now() - INTERVAL ':hl days'
        """.replace(":hl days", f"{half_life_days} days")),
        {"pid": project_id}
    ).fetchall()

    decay_rate = math.log(2) / half_life_days   # so that at half_life, confidence halves

    for row in rows:
        mem_id, current_conf, created_at = row
        # Calculate days old via Python (avoids SQL dialect issues)
        from datetime import datetime, timezone
        if hasattr(created_at, "replace"):
            age_days = (datetime.now(timezone.utc) - created_at.replace(tzinfo=timezone.utc)).days
        else:
            age_days = half_life_days

        new_conf = max(0.05, current_conf * math.exp(-decay_rate * age_days))
        db.execute(
            text("UPDATE project_memory SET confidence = :c WHERE id = :id"),
            {"c": round(new_conf, 4), "id": mem_id}
        )

    db.commit()
    print(f"[Timeline] Decayed {len(rows)} old memories for project {project_id}")


def mark_superseded_memories(project_id: int, new_memory_id: int, topic_keywords: list[str], db: Session):
    """
    When a new memory is stored about a topic, mark older memories on the same
    topic as superseded. Uses keyword matching against content.
    Call this after store_memory() for 'decision' and 'architecture' types.
    """
    if not topic_keywords:
        return

    like_clauses = " OR ".join(f"content ILIKE '%{kw}%'" for kw in topic_keywords[:5])
    rows = db.execute(
        text(f"""
            SELECT id FROM project_memory
            WHERE project_id = :pid
            AND id != :new_id
            AND superseded_by IS NULL
            AND ({like_clauses})
            ORDER BY created_at ASC
        """),
        {"pid": project_id, "new_id": new_memory_id}
    ).fetchall()

    for row in rows:
        db.execute(
            text("UPDATE project_memory SET superseded_by = :new_id WHERE id = :old_id"),
            {"new_id": new_memory_id, "old_id": row[0]}
        )
    db.commit()
    if rows:
        print(f"[Timeline] Marked {len(rows)} memories as superseded by memory #{new_memory_id}")


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — PERFORMANCE MONITOR
# ═══════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    function_name: str
    mean_ms:       float
    p95_ms:        float
    iterations:    int
    input_hash:    str


def _generate_benchmark_harness(source_code: str, function_name: str, groq_llm) -> str:
    """
    Ask the LLM to generate a minimal timing harness for the target function.
    Returns Python code that prints timing results as JSON.
    """
    prompt = f"""
Generate a Python timing harness for the function '{function_name}'.

Source code:
{source_code[:3000]}

Rules:
- Import the function directly (the source is in the same module scope — use exec to load it)
- Call the function with 3-5 representative inputs
- Run it 20 times and collect timing in milliseconds
- Print ONLY a JSON object: {{"mean_ms": X, "p95_ms": Y, "iterations": 20}}
- No markdown, no explanation
- Use time.perf_counter() for timing
- Handle any exceptions gracefully (print {{"mean_ms": -1, "p95_ms": -1, "iterations": 0}} on error)
"""
    raw = groq_llm(
        "You are a Python performance engineer. Return only a timing harness script.",
        prompt
    )
    return re.sub(r"```(?:python)?\n?", "", raw).replace("```", "").strip()


def capture_benchmark(
    source_code:   str,
    function_name: str,
    project_id:    int,
    file_id:       int,
    groq_llm,
    db:            Session,
    timeout:       int = 15
) -> Optional[BenchmarkResult]:
    """
    Generate a benchmark harness, execute it, parse results,
    and store the baseline in performance_baselines.
    Returns BenchmarkResult or None if benchmarking fails.
    """
    harness_code = _generate_benchmark_harness(source_code, function_name, groq_llm)

    # Inject the source code into the harness via exec
    full_script = f"""
import sys, json, time

_source = {repr(source_code)}
_ns = {{}}
try:
    exec(compile(_source, '<source>', 'exec'), _ns)
except Exception as e:
    print(json.dumps({{"mean_ms": -1, "p95_ms": -1, "iterations": 0, "error": str(e)}}))
    sys.exit(0)

# Inject into module namespace for the harness
globals().update(_ns)

{harness_code}
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full_script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout.strip()
        if not output:
            return None
        data     = json.loads(output)
        mean_ms  = float(data.get("mean_ms", -1))
        p95_ms   = float(data.get("p95_ms", -1))
        iters    = int(data.get("iterations", 0))

        if mean_ms < 0 or iters == 0:
            print(f"[Perf] Benchmark for '{function_name}' produced no valid results")
            return None

        # Store baseline
        input_hash = hashlib.md5(harness_code.encode()).hexdigest()[:8]
        db.execute(
            text("""
                INSERT INTO performance_baselines
                    (project_id, file_id, function_name, mean_ms, p95_ms, input_hash)
                VALUES (:pid, :fid, :fn, :mean, :p95, :ih)
            """),
            {"pid": project_id, "fid": file_id, "fn": function_name,
             "mean": mean_ms, "p95": p95_ms, "ih": input_hash}
        )
        db.commit()
        print(f"[Perf] Baseline for '{function_name}': mean={mean_ms:.2f}ms p95={p95_ms:.2f}ms")

        return BenchmarkResult(
            function_name=function_name,
            mean_ms=mean_ms, p95_ms=p95_ms,
            iterations=iters, input_hash=input_hash
        )

    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        print(f"[Perf] Benchmark failed for '{function_name}': {e}")
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def detect_regression(
    function_name: str,
    project_id:    int,
    new_mean_ms:   float,
    db:            Session,
    threshold:     float = 0.25   # 25% slower = regression
) -> Optional[str]:
    """
    Compare new benchmark against stored baseline.
    Returns a warning string if regression detected, None if clean.
    """
    row = db.execute(
        text("""
            SELECT mean_ms FROM performance_baselines
            WHERE project_id = :pid AND function_name = :fn
            ORDER BY recorded_at DESC LIMIT 1
        """),
        {"pid": project_id, "fn": function_name}
    ).fetchone()

    if not row or row[0] <= 0:
        return None   # No baseline to compare against

    baseline_ms = row[0]
    ratio       = (new_mean_ms - baseline_ms) / baseline_ms

    if ratio > threshold:
        msg = (f"Performance regression in '{function_name}': "
               f"{baseline_ms:.1f}ms → {new_mean_ms:.1f}ms "
               f"({ratio*100:.0f}% slower)")
        print(f"[Perf] ⚠️  {msg}")
        return msg

    print(f"[Perf] ✅ No regression in '{function_name}': {baseline_ms:.1f}ms → {new_mean_ms:.1f}ms")
    return None


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — BEHAVIORAL DIFF ENGINE
# ═══════════════════════════════════════════════════════════════

@dataclass
class BehavioralSnapshot:
    function_name: str
    test_cases:    list[dict]   # [{input_repr, output_repr}]


def capture_behavioral_snapshot(
    source_code:   str,
    function_name: str,
    groq_llm,
    timeout:       int = 10
) -> Optional[BehavioralSnapshot]:
    """
    Generate and run a small input/output probe for a function.
    Captures {input, output} pairs before a modification.
    These are compared after modification to detect unexpected behavioral changes.
    """
    probe_prompt = f"""
Write a Python script that calls the function '{function_name}' with 4-6 representative inputs
and prints the results as a JSON array.

Source code:
{source_code[:2000]}

Rules:
- Inject the source using exec() with a namespace
- Call the function with diverse inputs (normal, edge, boundary)
- Print ONLY: [{{"input": "repr_of_input", "output": "repr_of_output"}}, ...]
- Handle exceptions: use "ERROR: <msg>" as the output string
- No markdown
"""
    raw = groq_llm(
        "You are a Python test engineer. Write input/output probes. Return only valid Python.",
        probe_prompt
    )
    probe_code = re.sub(r"```(?:python)?\n?", "", raw).replace("```", "").strip()

    full_script = f"""
import json, sys
_source = {repr(source_code)}
_ns = {{}}
try:
    exec(compile(_source, '<source>', 'exec'), _ns)
    globals().update(_ns)
except Exception as e:
    print(json.dumps([]))
    sys.exit(0)

{probe_code}
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full_script)
        tmp_path = f.name

    try:
        result  = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout.strip()
        if not output:
            return None
        cases = json.loads(output)
        if not isinstance(cases, list):
            return None
        print(f"[BehavioralDiff] Captured {len(cases)} I/O case(s) for '{function_name}'")
        return BehavioralSnapshot(function_name=function_name, test_cases=cases)
    except Exception as e:
        print(f"[BehavioralDiff] Snapshot failed for '{function_name}': {e}")
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def compare_behavioral_snapshots(before: BehavioralSnapshot, after: BehavioralSnapshot) -> dict:
    """
    Compare pre- and post-modification I/O snapshots.
    Returns {changed: bool, diffs: list, unexpected_changes: int}
    """
    if not before or not after:
        return {"changed": False, "diffs": [], "unexpected_changes": 0}

    diffs = []
    before_map = {c.get("input"): c.get("output") for c in before.test_cases}
    after_map  = {c.get("input"): c.get("output") for c in after.test_cases}

    for inp, out_before in before_map.items():
        out_after = after_map.get(inp)
        if out_after is None:
            diffs.append({"input": inp, "before": out_before, "after": "MISSING"})
        elif str(out_before) != str(out_after):
            diffs.append({"input": inp, "before": out_before, "after": out_after})

    return {
        "changed":            len(diffs) > 0,
        "diffs":              diffs,
        "unexpected_changes": len(diffs)
    }


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — DRIFT DETECTOR
# ═══════════════════════════════════════════════════════════════

@dataclass
class DriftReport:
    project_id:        int
    new_violations:    list[str]
    coupling_delta:    float     # positive = more coupled (worse)
    complexity_delta:  float     # positive = more complex (worse)
    circular_deps:     list[tuple]
    volatile_modules:  list[str]
    risk_level:        str       # "low" | "medium" | "high"


def _build_current_snapshot(project_id: int, db: Session) -> dict:
    """
    Capture the current architectural state as a JSON-serializable dict.
    Used for both snapshot storage and drift comparison.
    """
    from architectural_awareness import build_dependency_graph
    from code_quality import analyze_project_quality

    dep_graph = build_dependency_graph(project_id, db)
    quality   = analyze_project_quality(project_id, db)

    # Calculate average coupling (imported_by count) and complexity score
    coupling_scores = [len(data.get("imported_by", [])) for data in dep_graph.values()]
    avg_coupling    = statistics.mean(coupling_scores) if coupling_scores else 0.0

    quality_score = quality.get("overall_score", 100)

    # Detect circular dependencies
    circular = []
    for filename, data in dep_graph.items():
        for importer in data.get("imported_by", []):
            importer_data = dep_graph.get(importer, {})
            if filename in importer_data.get("imports_from", []):
                pair = tuple(sorted([filename, importer]))
                if pair not in circular:
                    circular.append(pair)

    return {
        "dep_graph":      dep_graph,
        "avg_coupling":   avg_coupling,
        "quality_score":  quality_score,
        "circular_deps":  [list(c) for c in circular],
    }


def take_architectural_snapshot(project_id: int, db: Session) -> int:
    """
    Store the current architectural state as a baseline snapshot.
    Returns the snapshot ID.
    Call this after successful agent runs and on a schedule.
    """
    snapshot = _build_current_snapshot(project_id, db)
    row = db.execute(
        text("""
            INSERT INTO architectural_snapshots (project_id, snapshot)
            VALUES (:pid, CAST(:snap AS jsonb))
            RETURNING id
        """),
        {"pid": project_id, "snap": json.dumps(snapshot)}
    ).fetchone()
    db.commit()
    print(f"[Drift] Snapshot #{row[0]} taken for project {project_id}")
    return row[0]


def detect_drift(project_id: int, db: Session) -> DriftReport:
    """
    Compare the current architectural state against the most recent snapshot.
    Returns a DriftReport with violations, coupling delta, and risk level.
    """
    # Load last snapshot
    row = db.execute(
        text("""
            SELECT snapshot FROM architectural_snapshots
            WHERE project_id = :pid
            ORDER BY created_at DESC LIMIT 1
        """),
        {"pid": project_id}
    ).fetchone()

    current = _build_current_snapshot(project_id, db)

    if not row:
        # No baseline — store one and return clean report
        take_architectural_snapshot(project_id, db)
        return DriftReport(
            project_id=project_id,
            new_violations=[], coupling_delta=0.0, complexity_delta=0.0,
            circular_deps=[], volatile_modules=[], risk_level="low"
        )

    baseline = row[0]

    # Coupling delta
    coupling_delta = current["avg_coupling"] - baseline.get("avg_coupling", 0.0)

    # Quality delta (negative = code got worse)
    quality_delta = current["quality_score"] - baseline.get("quality_score", 100)
    complexity_delta = -quality_delta   # invert: positive = more complex

    # New circular dependencies
    old_circulars = {tuple(c) for c in baseline.get("circular_deps", [])}
    new_circulars = {tuple(c) for c in current["circular_deps"]}
    new_circles   = [list(c) for c in (new_circulars - old_circulars)]

    # Volatile modules (files that didn't exist in baseline)
    old_files     = set(baseline.get("dep_graph", {}).keys())
    current_files = set(current["dep_graph"].keys())
    new_files     = list(current_files - old_files)

    # Compose violations
    violations = []
    if coupling_delta > 1.5:
        violations.append(f"Average coupling increased by {coupling_delta:.1f} (files have more dependents)")
    if complexity_delta > 10:
        violations.append(f"Code quality score dropped by {complexity_delta:.0f} points")
    if new_circles:
        violations.append(f"New circular dependencies: {new_circles}")

    # Risk level
    if len(violations) >= 3 or len(new_circles) >= 2:
        risk = "high"
    elif violations:
        risk = "medium"
    else:
        risk = "low"

    report = DriftReport(
        project_id=project_id,
        new_violations=violations,
        coupling_delta=round(coupling_delta, 3),
        complexity_delta=round(complexity_delta, 3),
        circular_deps=new_circles,
        volatile_modules=new_files,
        risk_level=risk
    )

    # Store report
    db.execute(
        text("""
            INSERT INTO drift_reports (project_id, report, violations)
            VALUES (:pid, CAST(:rep AS jsonb), :v)
        """),
        {"pid": project_id, "rep": json.dumps({
            "new_violations":   violations,
            "coupling_delta":   report.coupling_delta,
            "complexity_delta": report.complexity_delta,
            "circular_deps":    new_circles,
            "volatile_modules": new_files,
            "risk_level":       risk
        }), "v": len(violations)}
    )
    db.commit()

    print(f"[Drift] Project {project_id}: risk={risk} | violations={len(violations)} | coupling_delta={coupling_delta:.2f}")
    return report


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — AUTONOMOUS SCHEDULER
# ═══════════════════════════════════════════════════════════════

def _log_scheduled_job(job_name: str, project_id: Optional[int], status: str,
                       result: dict, started_at: float, db: Session):
    from datetime import datetime, timezone
    db.execute(
        text("""
            INSERT INTO scheduled_job_log (job_name, project_id, status, result, started_at, finished_at)
            VALUES (:jn, :pid, :s, CAST(:r AS jsonb), :sa, :fa)
        """),
        {
            "jn":  job_name,
            "pid": project_id,
            "s":   status,
            "r":   json.dumps(result),
            "sa":  datetime.fromtimestamp(started_at, tz=timezone.utc),
            "fa":  datetime.now(timezone.utc)
        }
    )
    db.commit()


def _get_all_project_ids(db: Session) -> list[int]:
    rows = db.execute(text("SELECT id FROM projects ORDER BY id")).fetchall()
    return [r[0] for r in rows]


# ── Scheduled Tasks ─────────────────────────────────────────────

def daily_quality_audit(project_id: Optional[int] = None):
    """
    Run quality analysis on all projects (or a specific one).
    Log results. Alert if score drops > 5 points vs yesterday's snapshot.
    """
    from code_quality import analyze_project_quality
    t0 = time.time()
    db = SessionLocal()
    try:
        pids = [project_id] if project_id else _get_all_project_ids(db)
        for pid in pids:
            try:
                report = analyze_project_quality(pid, db)
                score  = report.get("overall_score", 0)
                print(f"[Scheduler] Quality audit — project {pid}: {score}/100")
                _log_scheduled_job("daily_quality_audit", pid, "completed",
                                   {"score": score, "grade": report.get("overall_grade")}, t0, db)
            except Exception as e:
                print(f"[Scheduler] Quality audit failed for project {pid}: {e}")
                _log_scheduled_job("daily_quality_audit", pid, "failed", {"error": str(e)}, t0, db)
    finally:
        db.close()


def drift_check_all_projects(project_id: Optional[int] = None):
    """
    Run architectural drift detection on all projects (or a specific one).
    """
    t0 = time.time()
    db = SessionLocal()
    try:
        pids = [project_id] if project_id else _get_all_project_ids(db)
        for pid in pids:
            try:
                report = detect_drift(pid, db)
                _log_scheduled_job("drift_check", pid, "completed", {
                    "risk_level":    report.risk_level,
                    "violations":    report.new_violations,
                    "coupling_delta": report.coupling_delta
                }, t0, db)
            except Exception as e:
                print(f"[Scheduler] Drift check failed for project {pid}: {e}")
                _log_scheduled_job("drift_check", pid, "failed", {"error": str(e)}, t0, db)
    finally:
        db.close()


def weekly_refactor_pass(project_id: Optional[int] = None):
    """
    Identify functions with complexity issues and run targeted refactors.
    Uses the full orchestrator (with approval gate) so high-risk refactors
    are queued for human review rather than auto-applied.
    """
    from code_quality import analyze_project_quality
    from multi_agent import orchestrate

    t0 = time.time()
    db = SessionLocal()
    try:
        pids = [project_id] if project_id else _get_all_project_ids(db)
        for pid in pids:
            try:
                # Import model + groq from the running app context
                # These are module-level singletons in main.py
                from main import model, groq_llm

                report   = analyze_project_quality(pid, db)
                refactored = []

                for filename, file_report in report.get("reports", {}).items():
                    complexity_issues = [
                        i for i in file_report.get("issues", [])
                        if i["category"] == "complexity" and i["severity"] == "warning"
                    ]
                    if not complexity_issues:
                        continue

                    instruction = (
                        f"Refactor {filename} to reduce complexity: "
                        + "; ".join(i["message"] for i in complexity_issues[:3])
                    )
                    print(f"[Scheduler] Refactor pass: {filename} — {len(complexity_issues)} issue(s)")

                    result = orchestrate(
                        query=instruction, project_id=pid,
                        db=db, model=model, groq_llm=groq_llm,
                        skip_approval_gate=False   # always gate autonomous refactors
                    )
                    refactored.append({
                        "file":   filename,
                        "status": result.status,
                        "approval_id": result.approval_id
                    })

                _log_scheduled_job("weekly_refactor", pid, "completed",
                                   {"refactored": refactored}, t0, db)
            except Exception as e:
                print(f"[Scheduler] Refactor pass failed for project {pid}: {e}")
                _log_scheduled_job("weekly_refactor", pid, "failed", {"error": str(e)}, t0, db)
    finally:
        db.close()


def decay_all_project_memories():
    """Apply temporal decay to memories across all projects."""
    t0 = time.time()
    db = SessionLocal()
    try:
        pids = _get_all_project_ids(db)
        for pid in pids:
            try:
                decay_old_memories(pid, db, half_life_days=90)
            except Exception as e:
                print(f"[Scheduler] Memory decay failed for project {pid}: {e}")
        _log_scheduled_job("memory_decay", None, "completed", {"projects": len(pids)}, t0, db)
    finally:
        db.close()


def setup_scheduler(database_url: str):
    """
    Initialize and start the APScheduler background scheduler.
    Uses SQLAlchemy job store so scheduled jobs survive server restarts.

    Call this from main.py:
        from autonomous_ops import setup_scheduler
        @app.on_event("startup")
        async def startup():
            setup_scheduler(DATABASE_URL)

    Install APScheduler if not already present:
        pip install apscheduler
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
        from apscheduler.executors.pool import ThreadPoolExecutor
    except ImportError:
        print("[Scheduler] ⚠️  APScheduler not installed — scheduler disabled.")
        print("[Scheduler]    Install with: pip install apscheduler")
        return None

    jobstores = {
        "default": SQLAlchemyJobStore(url=database_url, tablename="apscheduler_jobs")
    }
    executors = {
        "default": ThreadPoolExecutor(max_workers=4)
    }

    scheduler = BackgroundScheduler(
        jobstores=jobstores,
        executors=executors,
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 3600}
    )

    # Daily quality audit at 2am
    scheduler.add_job(
        daily_quality_audit, trigger="cron", hour=2, minute=0,
        id="daily_quality_audit", replace_existing=True
    )

    # Weekly refactor pass every Sunday at 3am
    scheduler.add_job(
        weekly_refactor_pass, trigger="cron", day_of_week="sun", hour=3, minute=0,
        id="weekly_refactor", replace_existing=True
    )

    # Drift check every 6 hours
    scheduler.add_job(
        drift_check_all_projects, trigger="interval", hours=6,
        id="drift_check", replace_existing=True
    )

    # Memory decay daily at 1am
    scheduler.add_job(
        decay_all_project_memories, trigger="cron", hour=1, minute=0,
        id="memory_decay", replace_existing=True
    )

    scheduler.start()
    print("[Scheduler] ✅ Autonomous scheduler started (4 jobs registered)")
    return scheduler


# ═══════════════════════════════════════════════════════════════
# INTEGRATION GUIDE — main.py additions
# ═══════════════════════════════════════════════════════════════
#
# ── 1. Startup event ──────────────────────────────────────────
#
# from autonomous_ops import setup_scheduler
# from database import DATABASE_URL
#
# @app.on_event("startup")
# async def startup():
#     setup_scheduler(DATABASE_URL)
#
#
# ── 2. New endpoints ──────────────────────────────────────────
#
# from autonomous_ops import (
#     get_code_evolution_summary, detect_drift,
#     take_architectural_snapshot, get_function_history
# )
#
# @app.get("/evolution/")
# def code_evolution(project_id: int, db: Session = Depends(get_db),
#                    key_project_id: int = Depends(require_api_key)):
#     verify_project_access(project_id, key_project_id)
#     return get_code_evolution_summary(project_id, db)
#
# @app.get("/drift/")
# def drift_report(project_id: int, db: Session = Depends(get_db),
#                  key_project_id: int = Depends(require_api_key)):
#     verify_project_access(project_id, key_project_id)
#     report = detect_drift(project_id, db)
#     return {
#         "risk_level":      report.risk_level,
#         "violations":      report.new_violations,
#         "coupling_delta":  report.coupling_delta,
#         "circular_deps":   report.circular_deps,
#         "volatile_modules": report.volatile_modules
#     }
#
# @app.post("/snapshot/")
# def take_snapshot(project_id: int, db: Session = Depends(get_db),
#                   key_project_id: int = Depends(require_api_key)):
#     verify_project_access(project_id, key_project_id)
#     snap_id = take_architectural_snapshot(project_id, db)
#     return {"snapshot_id": snap_id}
#
# @app.get("/function-history/")
# def function_history(project_id: int, function_name: str,
#                      db: Session = Depends(get_db),
#                      key_project_id: int = Depends(require_api_key)):
#     verify_project_access(project_id, key_project_id)
#     return {"function": function_name, "history": get_function_history(function_name, project_id, db)}
#
#
# ── 3. DB migrations (run once) ────────────────────────────────
#







