"""
agent_executor.py — Agentic Plan -> Execute Engine
====================================================
Changes from previous version:
  - correction_validator: validate_correction() wired into all three tool functions
    after self_correction loop. Adds Layer 2 (postcondition) + Layer 3 (critic).
  - test_spec_engine: lock_spec_tests() called BEFORE fix generation in
    tool_modify_function. validate_fix_against_spec() called AFTER.
    This breaks the circular LLM bias where tests are written to confirm the fix.
  - Original code snapshot captured before any LLM call for critic comparison.
  - Validation feedback injected back into correction loop on failure.
  - MAX_VALIDATION_ATTEMPTS controls how many re-correction cycles are allowed.
"""

import ast
import re
import os
import json
from typing import Optional
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session
from sqlalchemy import text
from self_correction import self_correct_code
from architectural_awareness import analyze_impact, check_callers_after_change, apply_caller_fixes
from memory_engine import (
    store_memory,
    retrieve_relevant_memory,
    format_memories_for_prompt,
    build_decision_memory
)
from execution_hardening import (
    SandboxController,
    SandboxConfig,
    get_sandbox_controller,
    static_security_scan,
)
from correction_validator import (
    validate_correction,
    should_skip_postcondition,
    format_validation_feedback
)
from test_spec_engine import (
    lock_spec_tests,
    validate_fix_against_spec,
    format_spec_failure_feedback
)

# ── Global safe_mode flag ────────────────────────────────────────────────────
_GLOBAL_SAFE_MODE: bool = os.environ.get("AICTO_SAFE_MODE", "0").strip() == "1"

MAX_FILE_CHARS = 6000

# Maximum validation+re-correction attempts before accepting best available code
MAX_VALIDATION_ATTEMPTS = 3


# ─────────────────────────────────────────────
# 1. PYDANTIC SCHEMAS
# ─────────────────────────────────────────────

class PlanStep(BaseModel):
    step_number: int
    action: str
    target_file: Optional[str] = None
    target_function: Optional[str] = None
    instruction: str


class ExecutionPlan(BaseModel):
    goal: str
    steps: list[PlanStep]


class StepResult(BaseModel):
    step_number: int
    action: str
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────
# 2. PLAN GENERATOR
# ─────────────────────────────────────────────

def build_plan_prompt(file_list: list[str], memory_section: str = "") -> str:
    files_section = (
        "\n".join(f"  - {f}" for f in file_list)
        if file_list else "  (no files uploaded yet)"
    )

    memory_block = ""
    if memory_section:
        memory_block = f"""
=== PROJECT MEMORY ===
{memory_section}

You MUST respect project memory and architectural constraints.
Avoid repeating past failures.
Avoid changing public signatures unless explicitly requested.
======================

"""

    return f"""
You are an AI CTO planning engine.

The project contains these files:
{files_section}
{memory_block}
Given a user instruction and codebase context, return a JSON execution plan.

Rules:
- Return ONLY valid JSON, no markdown, no explanation.
- Each step must have a single, atomic action.
- Valid actions: "modify_function", "create_file", "explain", "refactor_file"
- target_file MUST be one of the filenames listed above (exact spelling).
- If no specific file is mentioned, pick the most relevant one from the list.
- Keep steps minimal — do not add unnecessary steps.

JSON format:
{{
  "goal": "short description of overall goal",
  "steps": [
    {{
      "step_number": 1,
      "action": "modify_function",
      "target_file": "sample.py",
      "target_function": "add",
      "instruction": "Add input validation to reject negative values"
    }}
  ]
}}
"""


def generate_structured_plan(
    query: str,
    context: str,
    groq_llm,
    file_list: list[str],
    max_retries: int = 3,
    project_id: Optional[int] = None,
    model=None,
    db: Optional[Session] = None
) -> ExecutionPlan:
    memory_section = ""
    if project_id is not None and model is not None and db is not None:
        try:
            memories = retrieve_relevant_memory(project_id, query, model, db, top_k=5)
            memory_section = format_memories_for_prompt(memories)
        except Exception as e:
            print(f"[Planner] Memory retrieval failed (non-fatal): {e}")

    system_prompt = build_plan_prompt(file_list, memory_section)
    user_prompt   = f"Codebase context:\n{context}\n\nUser instruction:\n{query}"
    last_error    = None

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            user_prompt += (
                f"\n\nAttempt {attempt} — previous failed: {last_error}"
                f"\nReturn ONLY valid JSON. target_file must be one of: {file_list}"
            )

        raw = groq_llm(system_prompt, user_prompt).strip()
        raw = re.sub(r"```(?:json)?\n?", "", raw).replace("```", "").strip()

        try:
            data = json.loads(raw)
            plan = ExecutionPlan(**data)
            for step in plan.steps:
                if step.target_file:
                    step.target_file = _resolve_filename(step.target_file, file_list)
            print(f"[Planner] Validated on attempt {attempt}: {len(plan.steps)} step(s)")
            for s in plan.steps:
                print(f"          Step {s.step_number}: {s.action} -> '{s.target_file}' / fn='{s.target_function}'")
            return plan
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = str(e)
            print(f"[Planner] Attempt {attempt} failed: {last_error}")

    raise ValueError(f"Plan generation failed after {max_retries} attempts. Last: {last_error}")


def _resolve_filename(llm_name: str, file_list: list[str]) -> str:
    if not llm_name or not file_list:
        return llm_name
    if llm_name in file_list:
        return llm_name
    llm_lower = llm_name.lower()
    for f in file_list:
        if f.lower() == llm_lower:
            return f
    for f in file_list:
        if llm_lower in f.lower() or f.lower() in llm_lower:
            return f
    return llm_name


# ─────────────────────────────────────────────
# 3. FILE RESOLUTION HELPERS
# ─────────────────────────────────────────────

def _get_file_exact(project_id: int, filename: str, db: Session):
    from models import File as DBFile
    return db.query(DBFile).filter(
        DBFile.project_id == project_id, DBFile.filename == filename
    ).first()


def _get_file_fuzzy(project_id: int, filename: str, db: Session):
    from models import File as DBFile
    all_files = db.query(DBFile).filter(DBFile.project_id == project_id).all()
    low = filename.lower()
    for f in all_files:
        if f.filename.lower() == low:
            return f
    for f in all_files:
        if low in f.filename.lower() or f.filename.lower() in low:
            return f
    return None


def _get_latest_file(project_id: int, db: Session):
    from models import File as DBFile
    return (
        db.query(DBFile)
        .filter(DBFile.project_id == project_id)
        .order_by(DBFile.id.desc())
        .first()
    )


def _get_best_file(project_id: int, target_file: Optional[str], db: Session):
    if target_file:
        f = _get_file_exact(project_id, target_file, db)
        if f:
            print(f"[FileResolver] Exact: '{target_file}'")
            return f
        f = _get_file_fuzzy(project_id, target_file, db)
        if f:
            print(f"[FileResolver] Fuzzy: '{target_file}' -> '{f.filename}'")
            return f
        print(f"[FileResolver] No match for '{target_file}', using latest")
    f = _get_latest_file(project_id, db)
    if f:
        print(f"[FileResolver] Fallback to latest: '{f.filename}'")
    return f


def _get_project_filenames(project_id: int, db: Session) -> list[str]:
    from models import File as DBFile
    rows = db.query(DBFile.filename).filter(DBFile.project_id == project_id).all()
    return [r[0] for r in rows]


# ─────────────────────────────────────────────
# 4. SHARED UTILITIES
# ─────────────────────────────────────────────

def _save_version(file, db: Session):
    db.execute(
        text("INSERT INTO file_versions (file_id, content) VALUES (:fid, :c)"),
        {"fid": file.id, "c": file.content}
    )
    db.commit()


def _reembed_file(file, model, db: Session):
    from models import CodeChunk
    db.execute(text("DELETE FROM code_chunks WHERE file_id = :fid"), {"fid": file.id})
    db.commit()
    chunks = re.split(r"\n(?=def |class )", file.content)
    for chunk in chunks:
        if chunk.strip():
            embedding = model.encode(chunk).tolist()
            db.add(CodeChunk(file_id=file.id, content=chunk, embedding=embedding))
    db.commit()


def _resolve_function(file_content: str, target_function: Optional[str]):
    if not target_function:
        return None, None
    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        return None, None
    low = target_function.lower()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == target_function:
            return node, ast.get_source_segment(file_content, node)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.lower() == low:
            print(f"[FnResolver] Case-insensitive: '{target_function}' -> '{node.name}'")
            return node, ast.get_source_segment(file_content, node)
    print(f"[FnResolver] '{target_function}' not found — will modify whole file")
    return None, None


def _determine_run_mode(code: str) -> bool:
    SKIP_IF_IMPORTS = {
        "sqlalchemy", "fastapi", "sentence_transformers",
        "groq", "pgvector", "torch", "tensorflow", "sklearn",
        "pandas", "numpy", "cv2", "PIL", "flask", "django"
    }
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in SKIP_IF_IMPORTS:
                        print(f"[SelfCorrect] Heavy import '{alias.name}' detected -> syntax-only mode")
                        return False
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in SKIP_IF_IMPORTS:
                    print(f"[SelfCorrect] Heavy import '{node.module}' detected -> syntax-only mode")
                    return False
    except SyntaxError:
        pass
    return True


def _truncate_file_for_prompt(content: str, label: str = "file") -> str:
    if len(content) <= MAX_FILE_CHARS:
        return content
    truncated = content[:MAX_FILE_CHARS]
    last_newline = truncated.rfind("\n")
    if last_newline > MAX_FILE_CHARS // 2:
        truncated = truncated[:last_newline]
    print(
        f"[Agent] {label} too large ({len(content)} chars) -> "
        f"truncated to {len(truncated)} chars for LLM prompt"
    )
    return truncated + "\n\n# ... [remaining file truncated — preserve all code below this point] ..."


# ─────────────────────────────────────────────
# 5. TOOL IMPLEMENTATIONS
# ─────────────────────────────────────────────

def tool_modify_function(
    step: PlanStep,
    project_id: int,
    db: Session,
    model,
    groq_llm,
    enable_tests: bool = False
) -> StepResult:
    file = _get_best_file(project_id, step.target_file, db)
    if not file:
        return StepResult(
            step_number=step.step_number, action=step.action,
            success=False, error="No files found in this project"
        )

    _save_version(file, db)

    target_node, function_source = _resolve_function(file.content, step.target_function)

    # ── Capture original code snapshot BEFORE any LLM call ─────
    # This is the baseline the critic compares against.
    # Captured here so even re-correction attempts compare against
    # the true original, not a partially-modified intermediate.
    original_code_snapshot = function_source if function_source else file.content

    # ── STEP A: Generate spec tests BEFORE the fix ─────────────
    # Tests are generated from instruction + original code only.
    # The fix does not exist yet — by design. This prevents the
    # circular bias where the model writes tests to confirm its own output.
    spec_suite = None
    _run_enabled = _determine_run_mode(file.content)

    if _run_enabled and step.target_function:
        try:
            spec_suite = lock_spec_tests(
                original_code=original_code_snapshot,
                function_name=step.target_function,
                instruction=step.instruction,
                groq_llm=groq_llm
            )
            if spec_suite.is_trustworthy:
                print(
                    f"[Agent] Spec suite locked: "
                    f"{len(spec_suite.locked_tests)} trustworthy test(s), "
                    f"kill_rate={spec_suite.mutation_kill_rate:.2f}"
                )
            else:
                print(
                    f"[Agent] Spec suite weak "
                    f"(kill_rate={spec_suite.mutation_kill_rate:.2f}) — soft check only"
                )
        except Exception as e:
            print(f"[Agent] Spec test generation failed (non-fatal): {e}")
            spec_suite = None

    # ── Architectural Impact Analysis ──────────────────────────
    impact = analyze_impact(step.target_function, file.filename, project_id, db)
    arch_ctx = (
        f"=== ARCHITECTURAL CONTEXT ===\n{impact.summary}\n\n"
        if impact.risk_level != "none" else ""
    )

    # ── STEP B: Generate the fix ───────────────────────────────
    if function_source:
        prompt = (
            f"{arch_ctx}"
            f"Modify ONLY this function as instructed.\n"
            f"Return ONLY the updated function definition. No markdown.\n\n"
            f"Function:\n{function_source}\n\n"
            f"Instruction:\n{step.instruction}"
        )
    else:
        safe_content = _truncate_file_for_prompt(file.content, label=file.filename)
        prompt = (
            f"{arch_ctx}"
            f"Modify this file as instructed. Return ONLY valid Python code. No markdown.\n\n"
            f"File content:\n{safe_content}\n\n"
            f"Instruction:\n{step.instruction}"
        )

    raw       = groq_llm("You are an expert Python engineer. Return valid Python only.", prompt)
    generated = re.sub(r"```.*?\n", "", raw).replace("```", "").strip()

    # Splice back into the full file if we targeted a specific function
    if function_source and target_node:
        start     = target_node.lineno - 1
        end       = target_node.end_lineno
        lines     = file.content.splitlines()
        indent    = lines[start][:len(lines[start]) - len(lines[start].lstrip())]
        gen_lines = generated.splitlines()
        min_ind   = min((len(l) - len(l.lstrip()) for l in gen_lines if l.strip()), default=0)
        indented  = [indent + l[min_ind:] if l.strip() else l for l in gen_lines]
        updated   = "\n".join(lines[:start] + indented + lines[end:])
    else:
        updated = generated

    # ── STEP C: Self-Correction Loop (sandbox, syntax + execution) ─
    _controller = get_sandbox_controller(
        safe_mode=_GLOBAL_SAFE_MODE or (not _run_enabled),
    )
    correction = _controller.run_corrected(
        code=updated,
        original_instruction=step.instruction,
        groq_llm=groq_llm,
    )

    if not correction.success:
        return StepResult(
            step_number=step.step_number, action=step.action,
            success=False,
            error=f"Self-correction failed after {correction.attempts} attempts: {correction.error}"
        )

    # ── STEP D: Behavioral Validation (Layers 2 + 3) ──────────
    # NEW exit condition — replaces bare "no exception" check.
    # Layer 2: postcondition contract (deterministic execution + behavioral contract)
    # Layer 3: semantic critic (separate LLM call in adversarial role)
    # On failure, feedback is injected back into a re-correction loop.
    final_code      = correction.final_code
    validation_note = ""

    skip_post = (
        not _run_enabled
        or should_skip_postcondition(final_code, step.target_function)
    )

    for val_attempt in range(1, MAX_VALIDATION_ATTEMPTS + 1):
        validation = validate_correction(
            original_code=original_code_snapshot,
            new_code=final_code,
            function_name=step.target_function or "",
            instruction=step.instruction,
            groq_llm=groq_llm,
            skip_postcondition=skip_post,
            skip_critic=False
        )

        if validation.passed:
            print(f"[Agent] Validation passed (attempt {val_attempt})")
            break

        if val_attempt < MAX_VALIDATION_ATTEMPTS:
            feedback = format_validation_feedback(validation)
            print(
                f"[Agent] Validation failed at layer '{validation.layer_failed}' — "
                f"re-correcting (attempt {val_attempt + 1})"
            )

            re_prompt = (
                f"Your previous fix was rejected by the validation system. Here is why:\n\n"
                f"{feedback}\n\n"
                f"Original instruction: {step.instruction}\n\n"
                f"Previous (rejected) code:\n{final_code}\n\n"
                f"Rewrite the function to fix ALL issues above. "
                f"Return ONLY valid Python. No markdown."
            )
            raw2  = groq_llm(
                "You are an expert Python engineer. Fix the issues described. "
                "Return valid Python only.",
                re_prompt
            )
            fixed = re.sub(r"```.*?\n", "", raw2).replace("```", "").strip()

            re_correction = _controller.run_corrected(
                code=fixed,
                original_instruction=step.instruction,
                groq_llm=groq_llm,
            )
            if re_correction.success:
                final_code = re_correction.final_code
            else:
                print(
                    f"[Agent] Re-correction syntax failed — "
                    f"keeping previous code for next validation attempt"
                )
        else:
            # All validation attempts exhausted — proceed with best code, log warning
            if validation.critic:
                validation_note = (
                    f" | validation warning ({validation.layer_failed}): "
                    f"{validation.critic.verdict_reason}"
                )
            elif validation.postcondition:
                failures_preview = "; ".join(validation.postcondition.failures[:2])
                validation_note = (
                    f" | validation warning (postcondition): {failures_preview}"
                )
            print(
                f"[Agent] Validation did not fully pass after "
                f"{MAX_VALIDATION_ATTEMPTS} attempt(s) — proceeding with best code"
            )

    # ── STEP E: Spec Test Validation ───────────────────────────
    # Run the locked spec tests (generated BEFORE the fix) against the fix.
    # This is the structural solution to the circular LLM bias.
    spec_note = ""
    if spec_suite and spec_suite.is_trustworthy and spec_suite.locked_tests:
        spec_result = validate_fix_against_spec(
            fix_code=final_code,
            function_name=step.target_function,
            spec_suite=spec_suite
        )
        if spec_result.passed:
            spec_note = f" | {len(spec_suite.locked_tests)} spec test(s) passed"
        else:
            spec_note = (
                f" | {spec_result.tests_failed}/{spec_result.tests_run} "
                f"spec test(s) failed"
            )
            print(
                f"[Agent] Spec test failures:\n"
                f"{format_spec_failure_feedback(spec_result, spec_suite)}"
            )

    # ── STEP F: Save and re-embed ──────────────────────────────
    file.content = final_code
    db.commit()
    _reembed_file(file, model, db)

    # ── Cross-file caller compatibility check ──────────────────
    caller_results = check_callers_after_change(impact, final_code, project_id, db, groq_llm)
    fixed_files    = apply_caller_fixes(caller_results, db, model, groq_llm)
    broken_files   = [
        r.filename for r in caller_results
        if r.is_broken and r.filename not in fixed_files
    ]

    attempts_note = (
        f" (corrected in {correction.attempts} attempt(s))"
        if correction.attempts > 1 else ""
    )
    impact_note = ""
    if fixed_files:
        impact_note = f" | auto-patched callers: {fixed_files}"
    elif broken_files:
        impact_note = f" | callers may need review: {broken_files}"
    elif impact.callers:
        impact_note = f" | {len(impact.callers)} caller file(s) verified compatible"

    if enable_tests:
        from test_engine import (
            generate_and_validate_tests, store_test,
            find_impacted_tests, rerun_impacted_tests
        )
        test_filename = f"tests/test_{file.filename}"
        test_code, test_result = generate_and_validate_tests(
            source_code=final_code,
            filename=file.filename,
            instruction=step.instruction,
            groq_llm=groq_llm
        )
        store_test(project_id, file.id, test_filename, test_code, test_result, job_id=None, db=db)

        test_impact  = find_impacted_tests(step.target_function or "", project_id, db)
        test_summary = rerun_impacted_tests(
            test_impact, final_code, file.filename,
            project_id, job_id=None, db=db, groq_llm=groq_llm
        )
        if test_summary["needs_review"]:
            attempts_note += f" | tests need review: {test_summary['needs_review']}"

    return StepResult(
        step_number=step.step_number, action=step.action,
        success=True,
        output=(
            f"Modified '{file.filename}' successfully"
            f"{attempts_note}{validation_note}{spec_note}{impact_note}"
        )
    )


def tool_create_file(
    step: PlanStep,
    project_id: int,
    db: Session,
    model,
    groq_llm,
    enable_tests: bool = False
) -> StepResult:
    from models import File as DBFile, PageIndex

    prompt  = (
        f"Create a new Python file called '{step.target_file}'.\n"
        f"Return ONLY valid Python code, no markdown.\n\n"
        f"Instruction:\n{step.instruction}"
    )
    raw     = groq_llm("You are an expert Python engineer.", prompt)
    content = re.sub(r"```.*?\n", "", raw).replace("```", "").strip()

    _run_enabled = _determine_run_mode(content)
    _controller  = get_sandbox_controller(
        safe_mode=_GLOBAL_SAFE_MODE or (not _run_enabled),
    )
    correction = _controller.run_corrected(
        code=content,
        original_instruction=step.instruction,
        groq_llm=groq_llm,
    )

    if not correction.success:
        return StepResult(
            step_number=step.step_number, action=step.action,
            success=False,
            error=f"Generated file could not be corrected: {correction.error}"
        )

    # Semantic critic on new file — original_code is empty (nothing existed before)
    final_code      = correction.final_code
    validation_note = ""

    validation = validate_correction(
        original_code="",
        new_code=final_code,
        function_name="",
        instruction=step.instruction,
        groq_llm=groq_llm,
        skip_postcondition=True,   # new file: no baseline function to probe
        skip_critic=False
    )
    if not validation.passed and validation.critic:
        validation_note = f" | critic note: {validation.critic.verdict_reason}"

    db_file = DBFile(
        project_id=project_id,
        filename=step.target_file or "generated.py",
        content=final_code
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)

    from main import generate_file_summary, extract_python_structure, extract_dependencies
    summary   = generate_file_summary(final_code)
    embedding = model.encode(summary).tolist()
    db.add(PageIndex(
        project_id=project_id,
        file_id=db_file.id,
        filename=db_file.filename,
        summary=summary,
        embedding=embedding
    ))
    db.commit()

    extract_python_structure(db_file.id, final_code, db)
    extract_dependencies(db_file.id, final_code, db)
    _reembed_file(db_file, model, db)

    attempts_note = (
        f" (corrected in {correction.attempts} attempt(s))"
        if correction.attempts > 1 else ""
    )

    if enable_tests:
        from test_engine import generate_and_validate_tests, store_test
        test_filename = f"tests/test_{db_file.filename}"
        test_code, test_result = generate_and_validate_tests(
            source_code=final_code,
            filename=db_file.filename,
            instruction=step.instruction,
            groq_llm=groq_llm
        )
        store_test(project_id, db_file.id, test_filename, test_code, test_result, job_id=None, db=db)
        if not test_result.passed:
            attempts_note += f" | generated tests failing — review {test_filename}"

    return StepResult(
        step_number=step.step_number, action=step.action,
        success=True,
        output=f"Created '{db_file.filename}'{attempts_note}{validation_note}"
    )


def tool_explain(
    step: PlanStep,
    project_id: int,
    db: Session,
    model,
    groq_llm,
    enable_tests: bool = False
) -> StepResult:
    file    = _get_best_file(project_id, step.target_file, db)
    context = file.content[:3000] if file else "No file found."
    answer  = groq_llm(
        "You are an expert AI CTO. Answer clearly and concisely.",
        f"Context:\n{context}\n\nQuestion:\n{step.instruction}"
    )
    return StepResult(
        step_number=step.step_number, action=step.action,
        success=True, output=answer
    )


def tool_refactor_file(
    step: PlanStep,
    project_id: int,
    db: Session,
    model,
    groq_llm,
    enable_tests: bool = False
) -> StepResult:
    file = _get_best_file(project_id, step.target_file, db)
    if not file:
        return StepResult(
            step_number=step.step_number, action=step.action,
            success=False, error="No files found in this project"
        )

    _save_version(file, db)

    # Capture original snapshot for critic comparison
    original_code_snapshot = file.content

    safe_content = _truncate_file_for_prompt(file.content, label=file.filename)

    prompt = (
        f"Refactor the following Python file.\n"
        f"- Preserve all existing function signatures and class names\n"
        f"- Return ONLY the complete refactored file, no markdown\n\n"
        f"Instruction: {step.instruction}\n\n"
        f"File:\n{safe_content}"
    )
    raw        = groq_llm("You are a senior Python refactoring expert.", prompt)
    refactored = re.sub(r"```.*?\n", "", raw).replace("```", "").strip()

    _run_enabled = _determine_run_mode(refactored)
    _controller  = get_sandbox_controller(
        safe_mode=_GLOBAL_SAFE_MODE or (not _run_enabled),
    )
    correction = _controller.run_corrected(
        code=refactored,
        original_instruction=step.instruction,
        groq_llm=groq_llm,
    )

    if not correction.success:
        return StepResult(
            step_number=step.step_number, action=step.action,
            success=False,
            error=f"Refactor could not be corrected: {correction.error}"
        )

    final_code      = correction.final_code
    validation_note = ""

    # Semantic critic for refactors — postcondition skipped for whole-file changes
    validation = validate_correction(
        original_code=original_code_snapshot,
        new_code=final_code,
        function_name="",
        instruction=step.instruction,
        groq_llm=groq_llm,
        skip_postcondition=True,   # whole-file refactor: no single function contract
        skip_critic=False
    )
    if not validation.passed and validation.critic:
        # Critic failure on refactors is a warning, not a hard block
        # (refactors change structure, which the critic may conservatively flag)
        validation_note = f" | critic note: {validation.critic.verdict_reason}"
        print(f"[Agent] Refactor critic: {validation.critic.verdict_reason}")

    file.content = final_code
    db.commit()
    _reembed_file(file, model, db)

    attempts_note = (
        f" (corrected in {correction.attempts} attempt(s))"
        if correction.attempts > 1 else ""
    )

    if enable_tests:
        from test_engine import (
            generate_and_validate_tests, store_test,
            find_impacted_tests, rerun_impacted_tests
        )
        test_filename = f"tests/test_{file.filename}"
        test_code, test_result = generate_and_validate_tests(
            source_code=final_code,
            filename=file.filename,
            instruction=step.instruction,
            groq_llm=groq_llm
        )
        store_test(project_id, file.id, test_filename, test_code, test_result, job_id=None, db=db)

        test_impact  = find_impacted_tests("", project_id, db)
        test_summary = rerun_impacted_tests(
            test_impact, final_code, file.filename,
            project_id, job_id=None, db=db, groq_llm=groq_llm
        )
        if test_summary["needs_review"]:
            attempts_note += f" | tests need review: {test_summary['needs_review']}"

    return StepResult(
        step_number=step.step_number, action=step.action,
        success=True,
        output=f"Refactored '{file.filename}'{attempts_note}{validation_note}"
    )


# ─────────────────────────────────────────────
# 6. ACTION DISPATCHER
# ─────────────────────────────────────────────

TOOL_MAP = {
    "modify_function": tool_modify_function,
    "modify_file":     tool_modify_function,
    "edit_file":       tool_modify_function,
    "create_file":     tool_create_file,
    "new_file":        tool_create_file,
    "explain":         tool_explain,
    "describe":        tool_explain,
    "refactor_file":   tool_refactor_file,
    "refactor":        tool_refactor_file,
}


# ─────────────────────────────────────────────
# 7. MAIN ENTRY POINT
# ─────────────────────────────────────────────

def run_agent(
    query: str,
    project_id: int,
    db: Session,
    model,
    groq_llm,
    context: str = "",
    enable_tests: bool = False
) -> dict:
    print(f"\n[Agent] Query: {query!r}")

    file_list = _get_project_filenames(project_id, db)
    print(f"[Agent] Project files: {file_list}")

    if not file_list:
        return {"error": "No files uploaded to this project yet."}

    try:
        plan = generate_structured_plan(
            query, context, groq_llm, file_list,
            project_id=project_id, model=model, db=db
        )
    except ValueError as e:
        return {"error": f"Planning failed: {e}"}

    print(f"[Agent] Goal: {plan.goal} | {len(plan.steps)} step(s)")

    results: list[dict] = []

    for step in plan.steps:
        print(
            f"\n[Agent] Step {step.step_number}: {step.action} -> "
            f"'{step.target_file}' / fn='{step.target_function}'"
        )

        step.action = step.action.lower().replace(" ", "_")
        tool_fn = TOOL_MAP.get(step.action)
        if not tool_fn:
            result = StepResult(
                step_number=step.step_number, action=step.action,
                success=False, error=f"Unknown action '{step.action}'"
            )
        else:
            try:
                result = tool_fn(
                    step, project_id, db, model, groq_llm,
                    enable_tests=enable_tests
                )
            except Exception as e:
                result = StepResult(
                    step_number=step.step_number, action=step.action,
                    success=False, error=f"Unexpected error: {e}"
                )

        results.append(result.model_dump())

        if not result.success:
            print(f"[Agent] Step {step.step_number} failed: {result.error}")
            _attempt_rollback(step, project_id, db, model)
            return {
                "goal":             plan.goal,
                "status":           "failed",
                "failed_at_step":   step.step_number,
                "error":            result.error,
                "completed_steps":  results
            }

    print(f"\n[Agent] All {len(plan.steps)} step(s) completed.")

    try:
        memory_content = build_decision_memory(
            instruction=query,
            plan_goal=plan.goal,
            steps_executed=len(plan.steps),
            results=results
        )
        store_memory(
            project_id=project_id,
            memory_type="decision",
            content=memory_content,
            model=model,
            db=db
        )
    except Exception as e:
        print(f"[Agent] Memory storage failed (non-fatal): {e}")

    return {
        "goal":            plan.goal,
        "status":          "success",
        "steps_executed":  len(plan.steps),
        "results":         results
    }


def _attempt_rollback(step: PlanStep, project_id: int, db: Session, model):
    try:
        file = _get_best_file(project_id, step.target_file, db)
        if not file:
            return
        row = db.execute(
            text(
                "SELECT content FROM file_versions "
                "WHERE file_id = :fid ORDER BY created_at DESC LIMIT 1"
            ),
            {"fid": file.id}
        ).fetchone()
        if row:
            file.content = row[0]
            db.commit()
            _reembed_file(file, model, db)
            print(f"[Rollback] Restored '{file.filename}'")
        else:
            print(f"[Rollback] No saved version for '{file.filename}'")
    except Exception as e:
        print(f"[Rollback] Failed: {e}")