"""
memory_engine.py — Long-Term Project Memory Engine
====================================================
Provides persistent, vector-searchable memory for each project.

Memory types:
  - "architecture"  : structural constraints (e.g. "follow Clean Architecture")
  - "decision"      : past agent actions and outcomes
  - "constraint"    : explicit rules set by the user
  - "style"         : coding style preferences

Integration:
  from memory_engine import store_memory, retrieve_relevant_memory
"""

import re
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MAX_CONTENT_LENGTH = 1000   # characters per memory entry
TOP_K_DEFAULT      = 5


# ─────────────────────────────────────────────
# STORE MEMORY
# ─────────────────────────────────────────────

def store_memory(
    project_id: int,
    memory_type: str,
    content: str,
    model,
    db: Session
) -> Optional[int]:
    """
    Store a memory entry for a project.

    Args:
        project_id:   Project to associate memory with
        memory_type:  One of "architecture", "decision", "constraint", "style"
        content:      Human-readable memory text (truncated to 1000 chars)
        model:        SentenceTransformer embedding model
        db:           SQLAlchemy session

    Returns:
        Inserted memory row ID, or None on failure
    """
    VALID_TYPES = {"architecture", "decision", "constraint", "style"}
    if memory_type not in VALID_TYPES:
        print(f"[Memory] ⚠️  Invalid memory_type '{memory_type}' — skipping")
        return None

    # Truncate to avoid bloating the DB and the prompt
    content = content.strip()[:MAX_CONTENT_LENGTH]
    if not content:
        print(f"[Memory] ⚠️  Empty content — skipping")
        return None

    try:
        embedding = model.encode(content).tolist()

        result = db.execute(
            text("""
                INSERT INTO project_memory (project_id, memory_type, content, embedding)
                VALUES (:project_id, :memory_type, :content, CAST(:embedding AS vector))
                RETURNING id
            """),
            {
                "project_id":   project_id,
                "memory_type":  memory_type,
                "content":      content,
                "embedding":    embedding
            }
        )
        db.commit()
        row_id = result.fetchone()[0]
        print(f"[Memory] ✅ Stored {memory_type} memory (id={row_id}) for project {project_id}")
        return row_id

    except Exception as e:
        print(f"[Memory] ❌ Failed to store memory: {e}")
        db.rollback()
        return None


# ─────────────────────────────────────────────
# RETRIEVE RELEVANT MEMORY
# ─────────────────────────────────────────────

def retrieve_relevant_memory(
    project_id: int,
    query: str,
    model,
    db: Session,
    top_k: int = TOP_K_DEFAULT
) -> list[dict]:
    """
    Retrieve the most relevant memory entries for a query using vector similarity.

    Args:
        project_id:  Project to search within
        query:       The current user instruction or context string
        model:       SentenceTransformer embedding model
        db:          SQLAlchemy session
        top_k:       Maximum number of memories to return (default 5)

    Returns:
        List of dicts: [{id, memory_type, content, created_at}, ...]
        Empty list on failure or if no memories exist.
    """
    try:
        embedding = model.encode(query).tolist()

        rows = db.execute(
            text("""
                SELECT id, memory_type, content, created_at
                FROM project_memory
                WHERE project_id = :project_id
                ORDER BY embedding <-> CAST(:embedding AS vector)
                LIMIT :top_k
            """),
            {
                "project_id": project_id,
                "embedding":  embedding,
                "top_k":      top_k
            }
        ).fetchall()

        memories = [
            {
                "id":          row[0],
                "memory_type": row[1],
                "content":     row[2],
                "created_at":  str(row[3])
            }
            for row in rows
        ]

        if memories:
            print(f"[Memory] Retrieved {len(memories)} memory entries for project {project_id}")
        else:
            print(f"[Memory] No memories found for project {project_id}")

        return memories

    except Exception as e:
        print(f"[Memory] ❌ Memory retrieval failed (non-fatal): {e}")
        return []


# ─────────────────────────────────────────────
# FORMAT MEMORY FOR PROMPT INJECTION
# ─────────────────────────────────────────────

def format_memories_for_prompt(memories: list[dict]) -> str:
    """
    Format retrieved memories into a structured string for LLM prompt injection.
    Returns empty string if no memories.
    """
    if not memories:
        return ""

    lines = []
    for m in memories:
        tag = m["memory_type"].upper()
        lines.append(f"[{tag}] {m['content']}")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# BUILD DECISION MEMORY CONTENT
# ─────────────────────────────────────────────

def build_decision_memory(
    instruction: str,
    plan_goal: str,
    steps_executed: int,
    results: list[dict]
) -> str:
    """
    Build a structured decision memory string from a completed agent run.
    Kept under MAX_CONTENT_LENGTH.
    """
    files_modified = []
    correction_notes = []

    for r in results:
        output = r.get("output") or ""
        if output:
            # Extract filename from output like "Modified 'foo.py' successfully"
            match = re.search(r"'([^']+\.[^']+)'", output)
            if match:
                files_modified.append(match.group(1))
            # Extract correction attempt info
            attempt_match = re.search(r"fixed in (\d+) attempt", output)
            if attempt_match:
                correction_notes.append(f"{attempt_match.group(1)} correction attempt(s) needed")

    files_str   = ", ".join(set(files_modified)) if files_modified else "none"
    correct_str = "; ".join(correction_notes) if correction_notes else "no corrections needed"

    content = (
        f"INSTRUCTION: {instruction[:200]} | "
        f"GOAL: {plan_goal[:150]} | "
        f"STEPS: {steps_executed} | "
        f"FILES: {files_str} | "
        f"CORRECTIONS: {correct_str}"
    )

    return content[:MAX_CONTENT_LENGTH]