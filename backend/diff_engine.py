"""
diff_engine.py — Change-Diff Visualization Layer (Phase 2)
===========================================================
Provides full transparency into what AI CTO modifies before any commit touches
the repository. This module is a pure observation layer — it never modifies files,
never alters execution results, and never blocks the pipeline.

Responsibilities:
  1. Capture "before" snapshots of files that will be modified (pre-execution).
  2. After execution, compare before vs. after and generate unified diffs.
  3. Store diffs in the change_diffs table (project_id, branch_name, filename).
  4. Score each diff for risk level: low | medium | high.
  5. Provide retrieval helpers for the /diff/ endpoint and orchestrator enrichment.

Risk Scoring Logic:
  - HIGH   : public interface changes (def/class signatures), >100 lines changed,
             deletions > additions (destructive), or import changes
  - MEDIUM : 20–100 lines changed, or any function/method body modified
  - LOW    : <20 lines, only additions, comments or whitespace only

Integration:
  from diff_engine import snapshot_files_before, generate_and_store_diffs, get_diffs_for_branch

All functions are non-raising — failures are logged and return safe empty defaults
so the execution pipeline is never blocked by a diff failure.
"""

import difflib
import re
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

DIFF_PREVIEW_LINES  = 200   # lines shown in orchestrator response preview
HIGH_RISK_THRESHOLD = 100   # net line changes to qualify as HIGH risk
MED_RISK_THRESHOLD  = 20    # net line changes to qualify as MEDIUM risk

# Patterns that indicate a public interface change (HIGH risk)
_PUBLIC_INTERFACE_RE = re.compile(
    r"^[+-]\s*(def |class |async def )",
    re.MULTILINE
)
_IMPORT_CHANGE_RE = re.compile(
    r"^[+-]\s*(import |from .+ import )",
    re.MULTILINE
)


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — PRE-EXECUTION SNAPSHOT
# ═══════════════════════════════════════════════════════════════

def snapshot_files_before(
    filenames:  list[str],
    project_id: int,
    db:         Session
) -> dict[str, str]:
    """
    Capture the current content of each file from the DB *before* execution runs.

    This must be called BEFORE run_agent_from_dag() so we have a true baseline
    to diff against. Files that don't exist yet (new files) are stored as empty string.

    Args:
        filenames:  List of filenames the DAG plan intends to modify.
        project_id: Project scope.
        db:         SQLAlchemy session.

    Returns:
        { "filename.py": "original content ...", ... }
        Missing files map to "" (empty string = new file).
    """
    snapshots: dict[str, str] = {}
    for filename in filenames:
        try:
            row = db.execute(
                text("""
                    SELECT content FROM files
                    WHERE project_id = :pid AND filename = :fn
                    ORDER BY created_at DESC LIMIT 1
                """),
                {"pid": project_id, "fn": filename}
            ).fetchone()
            snapshots[filename] = row[0] if row else ""
        except Exception as e:
            print(f"[DiffEngine] ⚠️  Could not snapshot '{filename}': {e}")
            snapshots[filename] = ""

    print(f"[DiffEngine] 📸 Captured {len(snapshots)} pre-execution snapshot(s)")
    return snapshots


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — DIFF GENERATION
# ═══════════════════════════════════════════════════════════════

def _generate_unified_diff(
    filename:        str,
    before_content:  str,
    after_content:   str
) -> str:
    """
    Generate a unified diff string between before and after content.
    Returns empty string if content is identical.
    """
    before_lines = before_content.splitlines(keepends=True)
    after_lines  = after_content.splitlines(keepends=True)

    diff_lines = list(difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm=""
    ))

    return "\n".join(diff_lines)


def score_diff_risk(diff_text: str, filename: str) -> str:
    """
    Analyse a unified diff and return a risk level: "low" | "medium" | "high".

    HIGH conditions (any one is sufficient):
      - Public interface changed (def/class/async def added or removed)
      - Import statements changed
      - Net line changes > HIGH_RISK_THRESHOLD
      - Deletions significantly exceed additions (destructive ratio > 2:1)

    MEDIUM conditions:
      - Net line changes between MED_RISK_THRESHOLD and HIGH_RISK_THRESHOLD
      - Any function/method body modified (heuristic: lines with indented logic)

    LOW: everything else.
    """
    if not diff_text:
        return "low"

    added_lines   = [l for l in diff_text.splitlines() if l.startswith("+") and not l.startswith("+++")]
    removed_lines = [l for l in diff_text.splitlines() if l.startswith("-") and not l.startswith("---")]
    net_changes   = len(added_lines) + len(removed_lines)

    # HIGH: public interface or imports touched
    if _PUBLIC_INTERFACE_RE.search(diff_text):
        return "high"
    if _IMPORT_CHANGE_RE.search(diff_text):
        return "high"

    # HIGH: large volume of changes
    if net_changes > HIGH_RISK_THRESHOLD:
        return "high"

    # HIGH: destructive — far more deletions than additions
    if removed_lines and len(removed_lines) > len(added_lines) * 2 and len(removed_lines) > 10:
        return "high"

    # MEDIUM: moderate volume
    if net_changes > MED_RISK_THRESHOLD:
        return "medium"

    # MEDIUM: any function body change (indented modified lines)
    body_change = any(
        re.match(r"^[+-]\s{4,}", line) for line in diff_text.splitlines()
    )
    if body_change:
        return "medium"

    return "low"


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — STORE DIFFS
# ═══════════════════════════════════════════════════════════════

def generate_and_store_diffs(
    snapshots:    dict[str, str],
    project_id:   int,
    branch_name:  str,
    db:           Session
) -> list[dict]:
    """
    For each file in snapshots, load the current (post-execution) content from DB,
    generate the unified diff, score its risk, and persist to change_diffs table.

    This is called AFTER run_agent_from_dag() completes successfully.
    It is purely additive — it never modifies files or execution state.

    Args:
        snapshots:   { filename: original_content } from snapshot_files_before().
        project_id:  Project scope.
        branch_name: The ai-change-* branch name (may be None for non-Git projects).
        db:          SQLAlchemy session.

    Returns:
        List of diff summary dicts:
        [{ filename, lines_added, lines_removed, risk_level, has_changes }, ...]
    """
    summaries = []
    branch    = branch_name or "no-branch"

    for filename, before_content in snapshots.items():
        try:
            # Load post-execution content
            row = db.execute(
                text("""
                    SELECT content FROM files
                    WHERE project_id = :pid AND filename = :fn
                    ORDER BY created_at DESC LIMIT 1
                """),
                {"pid": project_id, "fn": filename}
            ).fetchone()

            after_content = row[0] if row else before_content

            # Generate diff
            diff_text  = _generate_unified_diff(filename, before_content, after_content)
            risk_level = score_diff_risk(diff_text, filename)

            # Count lines
            diff_lines    = diff_text.splitlines()
            lines_added   = sum(1 for l in diff_lines if l.startswith("+") and not l.startswith("+++"))
            lines_removed = sum(1 for l in diff_lines if l.startswith("-") and not l.startswith("---"))
            has_changes   = bool(diff_text.strip())

            # Persist to DB
            if has_changes:
                db.execute(
                    text("""
                        INSERT INTO change_diffs
                            (project_id, branch_name, filename, diff_text, risk_level,
                             lines_added, lines_removed, created_at)
                        VALUES
                            (:pid, :branch, :fn, :diff, :risk,
                             :added, :removed, NOW())
                        ON CONFLICT (project_id, branch_name, filename)
                        DO UPDATE SET
                            diff_text     = EXCLUDED.diff_text,
                            risk_level    = EXCLUDED.risk_level,
                            lines_added   = EXCLUDED.lines_added,
                            lines_removed = EXCLUDED.lines_removed,
                            created_at    = NOW()
                    """),
                    {
                        "pid":     project_id,
                        "branch":  branch,
                        "fn":      filename,
                        "diff":    diff_text,
                        "risk":    risk_level,
                        "added":   lines_added,
                        "removed": lines_removed,
                    }
                )
                print(f"[DiffEngine] 💾 Stored diff for '{filename}' "
                      f"(+{lines_added}/-{lines_removed}, risk={risk_level})")
            else:
                print(f"[DiffEngine] ℹ️  No changes detected in '{filename}' — skipping")

            summaries.append({
                "filename":      filename,
                "lines_added":   lines_added,
                "lines_removed": lines_removed,
                "risk_level":    risk_level,
                "has_changes":   has_changes
            })

        except Exception as e:
            print(f"[DiffEngine] ⚠️  Failed to process diff for '{filename}': {e}")
            summaries.append({
                "filename":      filename,
                "lines_added":   0,
                "lines_removed": 0,
                "risk_level":    "unknown",
                "has_changes":   False,
                "error":         str(e)
            })

    try:
        db.commit()
    except Exception as e:
        print(f"[DiffEngine] ⚠️  DB commit failed: {e}")
        db.rollback()

    return summaries


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — RETRIEVAL
# ═══════════════════════════════════════════════════════════════

def get_diffs_for_branch(
    project_id:   int,
    branch_name:  str,
    db:           Session,
    preview_only: bool = False
) -> list[dict]:
    """
    Retrieve all stored diffs for a given project + branch.

    Args:
        project_id:   Project scope.
        branch_name:  The ai-change-* branch name.
        db:           SQLAlchemy session.
        preview_only: If True, truncate diff_text to DIFF_PREVIEW_LINES lines.

    Returns:
        List of dicts: [{ filename, diff_text, lines_added, lines_removed,
                          risk_level, created_at }, ...]
        Empty list if none found.
    """
    try:
        rows = db.execute(
            text("""
                SELECT filename, diff_text, lines_added, lines_removed,
                       risk_level, created_at
                FROM change_diffs
                WHERE project_id = :pid AND branch_name = :branch
                ORDER BY risk_level DESC, filename ASC
            """),
            {"pid": project_id, "branch": branch_name}
        ).fetchall()
    except Exception as e:
        print(f"[DiffEngine] ⚠️  Failed to retrieve diffs: {e}")
        return []

    result = []
    for row in rows:
        diff_text = row[1] or ""
        if preview_only:
            lines     = diff_text.splitlines()
            diff_text = "\n".join(lines[:DIFF_PREVIEW_LINES])
            if len(lines) > DIFF_PREVIEW_LINES:
                diff_text += f"\n... [{len(lines) - DIFF_PREVIEW_LINES} more lines truncated]"

        result.append({
            "filename":      row[0],
            "diff_text":     diff_text,
            "lines_added":   row[2],
            "lines_removed": row[3],
            "risk_level":    row[4],
            "created_at":    str(row[5])
        })

    return result


def compute_overall_risk(diff_summaries: list[dict]) -> str:
    """
    Aggregate individual file risk levels into a single overall risk label.
    Any HIGH file → overall HIGH. Any MEDIUM (no HIGH) → MEDIUM. All LOW → LOW.
    """
    if not diff_summaries:
        return "low"
    levels = {d.get("risk_level", "low") for d in diff_summaries}
    if "high" in levels:
        return "high"
    if "medium" in levels:
        return "medium"
    return "low"


def build_diff_preview(diff_summaries: list[dict]) -> list[dict]:
    """
    Build a compact diff preview list suitable for the orchestrator response.
    Each entry includes file metadata and a truncated diff snippet (first 200 lines).
    Entries with no changes are excluded.
    """
    return [
        {
            "filename":      s["filename"],
            "lines_added":   s["lines_added"],
            "lines_removed": s["lines_removed"],
            "risk_level":    s["risk_level"],
        }
        for s in diff_summaries
        if s.get("has_changes", False)
    ]