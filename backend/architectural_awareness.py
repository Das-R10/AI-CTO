"""
architectural_awareness.py — Cross-File Impact Analysis Engine
===============================================================
Drop next to agent_executor.py.

What this adds:
  BEFORE: Agent modifies a function with no idea if other files use it.
  NOW:    Before any modification, the engine:
            1. Finds all files in the project that reference the target function/class
            2. Scores the impact (how many callers, which files)
            3. Injects the impact report into the LLM prompt so it modifies carefully
            4. After modification, checks all impacted files for breakage
            5. Auto-patches broken callers if the signature changed

Integration:
  In agent_executor.py, call analyze_impact() before the LLM prompt is built
  inside tool_modify_function(), and call check_callers_after_change() after
  the file is saved. See INTEGRATION GUIDE at the bottom.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class CallerInfo:
    filename: str
    file_id: int
    call_sites: list[int]      # line numbers where the function is called
    import_style: str          # "direct", "from_import", "attribute"


@dataclass
class ImpactReport:
    target_function: str
    target_file: str
    callers: list[CallerInfo]
    total_call_sites: int
    risk_level: str            # "none", "low", "medium", "high"
    summary: str               # human-readable for LLM prompt injection


@dataclass
class CallerCheckResult:
    filename: str
    file_id: int
    is_broken: bool
    issues: list[str]
    suggested_fix: Optional[str] = None


# ─────────────────────────────────────────────
# 1. FIND ALL CALLERS OF A FUNCTION/CLASS
# ─────────────────────────────────────────────

def _find_callers_in_code(
    code: str,
    target_name: str,
    source_filename: str
) -> list[int]:
    """
    Parse `code` and return line numbers where `target_name` is called.
    Handles: direct calls, attribute calls (obj.method), and name references.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    call_lines = []
    for node in ast.walk(tree):
        # Direct call: target_name(...)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == target_name:
                call_lines.append(node.lineno)
            # Attribute call: something.target_name(...)
            elif isinstance(node.func, ast.Attribute) and node.func.attr == target_name:
                call_lines.append(node.lineno)

        # Name reference (not a call — e.g. passed as argument)
        elif isinstance(node, ast.Name) and node.id == target_name:
            if not isinstance(getattr(node, '_parent', None), ast.Call):
                call_lines.append(node.lineno)

    return sorted(set(call_lines))


def _detect_import_style(code: str, source_filename: str) -> str:
    """
    Check how the source file is imported in this code.
    Returns: "direct", "from_import", "attribute", or "none"
    """
    module_name = source_filename.replace(".py", "").replace("/", ".").replace("\\", ".")
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "none"

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and (module_name in (node.module or "")):
                return "from_import"
        if isinstance(node, ast.Import):
            for alias in node.names:
                if module_name in alias.name:
                    return "direct"
    return "none"


def analyze_impact(
    target_function: Optional[str],
    target_file: str,
    project_id: int,
    db: Session
) -> ImpactReport:
    """
    Main entry point. Given a function/class name and the file it lives in,
    find all other files in the project that reference it.

    Returns an ImpactReport with callers, risk level, and a prompt-ready summary.
    """
    from models import File as DBFile

    if not target_function:
        return ImpactReport(
            target_function="(whole file)",
            target_file=target_file,
            callers=[],
            total_call_sites=0,
            risk_level="low",
            summary="No specific function targeted — whole-file modification. No cross-file impact analysis needed."
        )

    # Get all OTHER files in the project
    all_files = db.query(DBFile).filter(
        DBFile.project_id == project_id,
        DBFile.filename != target_file
    ).all()

    callers: list[CallerInfo] = []
    total_sites = 0

    for f in all_files:
        sites = _find_callers_in_code(f.content, target_function, target_file)
        if sites:
            import_style = _detect_import_style(f.content, target_file)
            callers.append(CallerInfo(
                filename=f.filename,
                file_id=f.id,
                call_sites=sites,
                import_style=import_style
            ))
            total_sites += len(sites)

    # Also check the DB-stored dependency graph for broader coverage
    db_callers = db.execute(
        text("""
            SELECT f.filename, f.id
            FROM file_dependencies fd
            JOIN files f ON f.id = fd.source_file_id
            WHERE fd.target = :target
            AND f.project_id = :project_id
            AND f.filename != :source_file
        """),
        {"target": target_function, "project_id": project_id, "source_file": target_file}
    ).fetchall()

    # Add any callers found in DB but not already in callers list
    existing_filenames = {c.filename for c in callers}
    for row in db_callers:
        if row[0] not in existing_filenames:
            callers.append(CallerInfo(
                filename=row[0],
                file_id=row[1],
                call_sites=[],   # DB doesn't store line numbers
                import_style="unknown"
            ))

    # Determine risk level
    if total_sites == 0 and not db_callers:
        risk = "none"
    elif total_sites <= 2:
        risk = "low"
    elif total_sites <= 6:
        risk = "medium"
    else:
        risk = "high"

    # Build human-readable summary for LLM prompt injection
    if not callers:
        summary = (
            f"Cross-file impact: NONE. '{target_function}' is not called by any other file in this project. "
            f"Safe to modify freely."
        )
    else:
        caller_lines = []
        for c in callers:
            if c.call_sites:
                caller_lines.append(f"  • {c.filename} — calls on lines {c.call_sites}")
            else:
                caller_lines.append(f"  • {c.filename} — references detected (exact lines unknown)")

        summary = (
            f"⚠️ CROSS-FILE IMPACT DETECTED — Risk level: {risk.upper()}\n"
            f"'{target_function}' is used by {len(callers)} other file(s) "
            f"({total_sites} call site(s) total):\n"
            + "\n".join(caller_lines) + "\n\n"
            f"INSTRUCTIONS: You MUST preserve the existing function signature "
            f"(name, parameter names, parameter count, return type) unless the user "
            f"explicitly asked to change it. Changing the signature will break the callers listed above."
        )

    print(f"[Impact] '{target_function}' in '{target_file}': {risk} risk, {len(callers)} callers, {total_sites} sites")

    return ImpactReport(
        target_function=target_function,
        target_file=target_file,
        callers=callers,
        total_call_sites=total_sites,
        risk_level=risk,
        summary=summary
    )


# ─────────────────────────────────────────────
# 2. CHECK CALLERS AFTER MODIFICATION
# ─────────────────────────────────────────────

def _extract_signature(code: str, function_name: str) -> Optional[dict]:
    """
    Extract the signature of a function from code.
    Returns: {name, args, has_return_annotation} or None
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return {
                "name": node.name,
                "args": [arg.arg for arg in node.args.args],
                "arg_count": len(node.args.args),
                "has_defaults": len(node.args.defaults) > 0,
                "has_return": node.returns is not None,
            }
    return None


def check_callers_after_change(
    impact: ImpactReport,
    new_code: str,
    project_id: int,
    db: Session,
    groq_llm
) -> list[CallerCheckResult]:
    """
    After a function is modified, check if any caller files are now broken.
    For each caller:
      1. Re-parse caller to find call sites
      2. Compare expected args with new signature
      3. If mismatch → generate a fix suggestion

    Returns list of CallerCheckResult — one per impacted file.
    """
    if not impact.callers or impact.risk_level == "none":
        return []

    print(f"\n[Impact] Checking {len(impact.callers)} caller(s) after modifying '{impact.target_function}'...")

    new_sig = _extract_signature(new_code, impact.target_function)
    results = []

    for caller in impact.callers:
        issues = []
        suggested_fix = None

        # Load current caller content from DB
        row = db.execute(
            text("SELECT content FROM files WHERE id = :fid"),
            {"fid": caller.file_id}
        ).fetchone()

        if not row:
            continue

        caller_content = row[0]

        # Check if the function is still callable with the new signature
        if new_sig:
            try:
                tree = ast.parse(caller_content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        fn = node.func
                        fn_name = (
                            fn.id if isinstance(fn, ast.Name)
                            else fn.attr if isinstance(fn, ast.Attribute)
                            else None
                        )
                        if fn_name == impact.target_function:
                            n_args_passed = len(node.args) + len(node.keywords)
                            expected = new_sig["arg_count"]

                            # Skip self in method context
                            if "self" in new_sig["args"]:
                                expected -= 1

                            if not new_sig["has_defaults"] and n_args_passed != expected:
                                issues.append(
                                    f"Line {node.lineno}: passes {n_args_passed} arg(s) "
                                    f"but '{impact.target_function}' now expects {expected}"
                                )
            except SyntaxError:
                issues.append("Caller file has syntax errors — cannot verify compatibility")

        is_broken = len(issues) > 0

        if is_broken:
            print(f"[Impact] ⚠️  '{caller.filename}' may be broken: {issues}")

            # Ask LLM to suggest a fix
            fix_prompt = (
                f"The function '{impact.target_function}' was modified and now has this signature:\n"
                f"{new_sig}\n\n"
                f"This caller file references it and may be broken:\n"
                f"{caller_content[:2000]}\n\n"
                f"Issues detected:\n" + "\n".join(issues) + "\n\n"
                f"Return ONLY the corrected caller file. No markdown. Preserve all other logic."
            )
            suggested_fix = groq_llm(
                "You are an expert Python engineer fixing cross-file compatibility issues.",
                fix_prompt
            )
            suggested_fix = re.sub(r"```.*?\n", "", suggested_fix).replace("```", "").strip()
        else:
            print(f"[Impact] ✅ '{caller.filename}' looks compatible with new signature")

        results.append(CallerCheckResult(
            filename=caller.filename,
            file_id=caller.file_id,
            is_broken=is_broken,
            issues=issues,
            suggested_fix=suggested_fix
        ))

    return results


def apply_caller_fixes(
    check_results: list[CallerCheckResult],
    db: Session,
    model,
    groq_llm
):
    """
    For any broken callers, save the LLM-suggested fix to the DB and re-embed.
    Saves a version snapshot before overwriting.
    """
    from models import File as DBFile, CodeChunk

    fixed_files = []

    for result in check_results:
        if not result.is_broken or not result.suggested_fix:
            continue

        file = db.query(DBFile).filter(DBFile.id == result.file_id).first()
        if not file:
            continue

        # Validate the fix first
        try:
            ast.parse(result.suggested_fix)
        except SyntaxError as e:
            print(f"[Impact] ❌ Fix for '{result.filename}' has syntax error: {e} — skipping")
            continue

        # Save version before patching
        db.execute(
            text("INSERT INTO file_versions (file_id, content) VALUES (:fid, :c)"),
            {"fid": file.id, "c": file.content}
        )
        db.commit()

        # Apply fix
        file.content = result.suggested_fix
        db.commit()

        # Re-embed
        db.execute(text("DELETE FROM code_chunks WHERE file_id = :fid"), {"fid": file.id})
        db.commit()
        chunks = re.split(r"\n(?=def |class )", result.suggested_fix)
        for chunk in chunks:
            if chunk.strip():
                embedding = model.encode(chunk).tolist()
                db.add(CodeChunk(file_id=file.id, content=chunk, embedding=embedding))
        db.commit()

        print(f"[Impact] ✅ Auto-patched '{result.filename}'")
        fixed_files.append(result.filename)

    return fixed_files


# ─────────────────────────────────────────────
# 3. DEPENDENCY GRAPH ENDPOINT HELPER
# ─────────────────────────────────────────────

def build_dependency_graph(project_id: int, db: Session) -> dict:
    """
    Build a full cross-file dependency graph for a project.
    Returns a dict suitable for the /dependency-graph/ endpoint.

    Format:
    {
      "sample.py": {
        "exports": ["add", "multiply"],
        "imported_by": ["main.py", "test_utils.py"],
        "imports_from": ["math", "os"]
      },
      ...
    }
    """
    from models import File as DBFile

    files = db.query(DBFile).filter(DBFile.project_id == project_id).all()
    graph = {}

    for f in files:
        # What does this file export (define)?
        exports = []
        try:
            tree = ast.parse(f.content)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    exports.append(node.name)
        except SyntaxError:
            pass

        # What does this file import?
        imports_from = db.execute(
            text("""
                SELECT DISTINCT target FROM file_dependencies
                WHERE source_file_id = :fid AND dependency_type = 'import'
            """),
            {"fid": f.id}
        ).fetchall()

        # What files import from this file?
        imported_by = db.execute(
            text("""
                SELECT DISTINCT f2.filename
                FROM file_dependencies fd
                JOIN files f2 ON f2.id = fd.source_file_id
                WHERE fd.target IN (
                    SELECT name FROM architectural_elements WHERE file_id = :fid
                )
                AND f2.project_id = :project_id
                AND f2.id != :fid
            """),
            {"fid": f.id, "project_id": project_id}
        ).fetchall()

        graph[f.filename] = {
            "file_id":      f.id,
            "exports":      exports,
            "imports_from": [r[0] for r in imports_from],
            "imported_by":  [r[0] for r in imported_by],
        }

    return graph


# ─────────────────────────────────────────────
# INTEGRATION GUIDE
# ─────────────────────────────────────────────
#
# ── In agent_executor.py, update tool_modify_function() ──
#
# from architectural_awareness import analyze_impact, check_callers_after_change, apply_caller_fixes
#
# STEP 1: Before building the LLM prompt, run impact analysis:
#
#   impact = analyze_impact(step.target_function, file.filename, project_id, db)
#
#   if impact.risk_level != "none":
#       # Inject impact warning into the prompt
#       prompt = (
#           f"=== ARCHITECTURAL CONTEXT ===\n{impact.summary}\n\n"
#           f"Modify ONLY this function as instructed.\n"
#           f"Return ONLY the updated function definition. No markdown.\n\n"
#           f"Function:\n{function_source}\n\n"
#           f"Instruction:\n{step.instruction}"
#       )
#
# STEP 2: After saving the modified file, check callers:
#
#   caller_results = check_callers_after_change(impact, correction.final_code, project_id, db, groq_llm)
#   fixed = apply_caller_fixes(caller_results, db, model, groq_llm)
#
#   broken = [r for r in caller_results if r.is_broken]
#   impact_note = ""
#   if fixed:
#       impact_note = f" | auto-patched callers: {fixed}"
#   elif broken:
#       impact_note = f" | ⚠️ broken callers detected: {[r.filename for r in broken]}"
#
#   return StepResult(
#       ...,
#       output=f"Modified '{file.filename}' successfully{attempts_note}{impact_note}"
#   )
#
# ── Add /dependency-graph/ endpoint to main.py ──
#
# from architectural_awareness import build_dependency_graph
#
# @app.get("/dependency-graph/")
# def dependency_graph(
#     project_id: int,
#     db: Session = Depends(get_db),
#     key_project_id: int = Depends(require_api_key)
# ):
#     verify_project_access(project_id, key_project_id)
#     return build_dependency_graph(project_id, db)