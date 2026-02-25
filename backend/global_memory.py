"""
global_memory.py — Cross-Project Pattern Learning
==================================================
Observes error/failure patterns across ALL projects.
After 3+ occurrences of the same pattern, promotes it
to global_memory as an actionable lesson.
Injects high-confidence lessons into planner prompts.
"""

import hashlib
import re
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text

PROMOTION_THRESHOLD = 3    # occurrences before a pattern is promoted
HIGH_CONFIDENCE_THRESHOLD = 0.7


def _fingerprint(error_text: str) -> str:
    """Normalize and hash an error for deduplication."""
    normalized = re.sub(r"line \d+", "line N", error_text.lower())
    normalized = re.sub(r"'[^']*'", "'X'", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()[:300]
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _to_lesson(error_text: str, memory_type: str) -> str:
    """Convert raw error text into a generalized lesson string."""
    # Simple heuristics — the LLM call happens at observation time
    if "signatur" in error_text.lower():
        return "Changing function signatures frequently breaks callers — preserve signatures unless explicitly requested."
    if "import" in error_text.lower():
        return "Import errors often follow DB or heavy-dependency file modifications — verify imports after refactoring."
    if "test" in error_text.lower() and "fail" in error_text.lower():
        return "Refactors frequently break existing tests — always re-run impacted tests after modifications."
    if "timeout" in error_text.lower():
        return "Generated code sometimes introduces infinite loops — add termination conditions to all loops."
    return f"Recurring {memory_type}: {error_text[:120]}"


def observe_error(
    error_text: str,
    memory_type: str,
    model,
    db: Session
):
    """
    Record one occurrence of an error pattern.
    If it has occurred PROMOTION_THRESHOLD times, promote to global_memory.
    """
    fp = _fingerprint(error_text)

    existing = db.execute(
        text("SELECT id, count, content FROM error_observations WHERE fingerprint=:fp"),
        {"fp": fp}
    ).fetchone()

    if existing:
        new_count = existing[1] + 1
        db.execute(
            text("UPDATE error_observations SET count=:c, last_seen=now() WHERE id=:id"),
            {"c": new_count, "id": existing[0]}
        )
        db.commit()

        if new_count == PROMOTION_THRESHOLD:
            _promote_to_global(existing[2], memory_type, new_count, model, db)
    else:
        db.execute(
            text("INSERT INTO error_observations (fingerprint, content, count) VALUES (:fp,:c,1)"),
            {"fp": fp, "c": error_text[:400]}
        )
        db.commit()


def _promote_to_global(content: str, memory_type: str, count: int, model, db: Session):
    """Promote a frequently-observed pattern to the global_memory table."""
    lesson    = _to_lesson(content, memory_type)
    confidence = min(1.0, count / 10.0)

    # Check if already promoted (by content similarity shortcut)
    existing = db.execute(
        text("SELECT id FROM global_memory WHERE content=:c"),
        {"c": lesson}
    ).fetchone()

    if existing:
        db.execute(
            text("UPDATE global_memory SET occurrence_count=occurrence_count+1, last_seen_at=now(), confidence=:conf WHERE id=:id"),
            {"conf": confidence, "id": existing[0]}
        )
    else:
        embedding = model.encode(lesson).tolist()
        db.execute(
            text("""
                INSERT INTO global_memory (memory_type, content, occurrence_count, confidence, embedding)
                VALUES (:mt, :c, :oc, :conf, CAST(:emb AS vector))
            """),
            {"mt": memory_type, "c": lesson, "oc": count, "conf": confidence, "emb": embedding}
        )

    db.commit()
    print(f"[GlobalMemory] ✅ Promoted pattern to global memory: {lesson[:80]}")


def retrieve_global_lessons(query: str, model, db: Session, top_k: int = 3) -> list[str]:
    """
    Retrieve high-confidence global lessons relevant to the current query.
    Returns list of lesson strings for prompt injection.
    """
    embedding = model.encode(query).tolist()

    rows = db.execute(
        text("""
            SELECT content, confidence FROM global_memory
            WHERE confidence >= :threshold
            ORDER BY embedding <-> CAST(:emb AS vector)
            LIMIT :k
        """),
        {"threshold": HIGH_CONFIDENCE_THRESHOLD, "emb": embedding, "k": top_k}
    ).fetchall()

    return [row[0] for row in rows]


def format_global_lessons_for_prompt(lessons: list[str]) -> str:
    if not lessons:
        return ""
    lines = ["=== GLOBAL LESSONS FROM PAST FAILURES ==="]
    for lesson in lessons:
        lines.append(f"  ⚠️  {lesson}")
    return "\n".join(lines) + "\n"