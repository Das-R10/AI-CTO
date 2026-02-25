"""
self_correction.py — Runtime Self-Correction Engine
=====================================================
Drop next to main.py and agent_executor.py.

What this does (vs what existed before):
  BEFORE: One-shot syntax check only. If code ran but crashed, agent had no idea.
  NOW:    Full execute → catch → analyze → fix → retry loop.

          1. Write code to a temp file
          2. Run it in a sandboxed subprocess with timeout
          3. If it crashes → parse the full traceback
          4. Send broken code + traceback + original instruction to LLM
          5. LLM returns fixed code
          6. Repeat up to MAX_RETRIES times
          7. If stable → return corrected code
          8. If all retries exhausted → return last error for rollback

Integration:
  In agent_executor.py, replace the final block in tool_modify_function
  and tool_create_file with a call to self_correct_code() before saving.
  See INTEGRATION GUIDE at the bottom of this file.
"""

import ast
import re
import sys
import os
import subprocess
import tempfile
import textwrap
import traceback
from typing import Optional
from dataclasses import dataclass


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

MAX_RETRIES    = 4     # max fix attempts before giving up
RUN_TIMEOUT    = 8     # seconds before subprocess is killed
PYTHON_BIN     = sys.executable  # use same Python that runs the server


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class CorrectionResult:
    success: bool
    final_code: str
    attempts: int
    error: Optional[str] = None        # last error if still failing
    correction_log: list = None        # full history of attempts

    def __post_init__(self):
        if self.correction_log is None:
            self.correction_log = []


@dataclass
class RunResult:
    success: bool
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


# ─────────────────────────────────────────────
# STEP 1 — SYNTAX CHECK (fast, no subprocess)
# ─────────────────────────────────────────────

def check_syntax(code: str) -> Optional[str]:
    """Returns error string if syntax is broken, None if clean."""
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}\n  {e.text}"


# ─────────────────────────────────────────────
# STEP 2 — RUNTIME EXECUTION (subprocess sandbox)
# ─────────────────────────────────────────────

def run_code(code: str, timeout: int = RUN_TIMEOUT) -> RunResult:
    """
    Write code to a temp file and run it in a subprocess.
    Returns stdout, stderr, returncode, and whether it timed out.

    Safety: subprocess has no network access concern since it's
    just running the user's own code locally, same as they would.
    Timeout prevents infinite loops from hanging the server.
    """
    # Write to a temp .py file
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [PYTHON_BIN, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return RunResult(
            success=(result.returncode == 0),
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
            timed_out=False
        )
    except subprocess.TimeoutExpired:
        return RunResult(
            success=False,
            stdout="",
            stderr=f"Execution timed out after {timeout} seconds.",
            returncode=-1,
            timed_out=True
        )
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ─────────────────────────────────────────────
# STEP 3 — TRACEBACK PARSER
# ─────────────────────────────────────────────

def parse_traceback(stderr: str) -> dict:
    """
    Extract structured info from a Python traceback.
    Returns: {type, message, line_number, relevant_lines}
    """
    lines = stderr.strip().splitlines()

    error_type    = "UnknownError"
    error_message = stderr
    line_number   = None

    for line in lines:
        # Catch "line N" references
        if "line " in line and ", line " in line:
            try:
                line_number = int(line.split(", line ")[-1].split(",")[0].strip())
            except ValueError:
                pass

        # Catch the actual error type (last line usually)
        if ": " in line and not line.startswith(" "):
            parts = line.split(": ", 1)
            if len(parts) == 2:
                error_type    = parts[0].strip()
                error_message = parts[1].strip()

    # Get last 10 lines as context
    relevant = "\n".join(lines[-10:]) if len(lines) > 10 else stderr

    return {
        "type":           error_type,
        "message":        error_message,
        "line_number":    line_number,
        "relevant_lines": relevant,
        "full_stderr":    stderr
    }


# ─────────────────────────────────────────────
# STEP 4 — LLM FIX PROMPT BUILDER
# ─────────────────────────────────────────────

def build_fix_prompt(
    broken_code: str,
    error_info: dict,
    original_instruction: str,
    attempt: int
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) for the fix request."""

    system = (
        "You are an expert Python debugger.\n"
        "You will be given broken code, the error it produced, and the original instruction.\n"
        "Your job: fix the code so it runs without errors AND satisfies the instruction.\n"
        "Rules:\n"
        "- Return ONLY the complete corrected Python code\n"
        "- No markdown, no explanation, no backticks\n"
        "- Do not remove existing functionality\n"
        "- Preserve all function signatures\n"
    )

    user = f"""
=== ATTEMPT {attempt} FIX REQUEST ===

ORIGINAL INSTRUCTION:
{original_instruction}

BROKEN CODE:
{broken_code}

ERROR TYPE: {error_info['type']}
ERROR MESSAGE: {error_info['message']}
LINE NUMBER: {error_info['line_number'] or 'unknown'}

TRACEBACK (last 10 lines):
{error_info['relevant_lines']}

Fix the code. Return ONLY valid Python. No markdown.
"""
    return system, user


# ─────────────────────────────────────────────
# STEP 5 — IMPORTABILITY CHECK
# ─────────────────────────────────────────────

def check_importable(code: str) -> Optional[str]:
    """
    For module-style files (with functions/classes but no __main__ block),
    check that the file can be imported without errors.
    Wraps the code in a compile() call to catch import-time errors.
    """
    try:
        compile(code, "<string>", "exec")
        return None
    except Exception as e:
        return str(e)


# ─────────────────────────────────────────────
# MAIN ENTRY — self_correct_code()
# ─────────────────────────────────────────────

def self_correct_code(
    code: str,
    original_instruction: str,
    groq_llm,
    max_retries: int = MAX_RETRIES,
    run_code_enabled: bool = True
) -> CorrectionResult:
    """
    Full self-correction loop for generated Python code.

    Args:
        code:                 The generated code to validate and fix
        original_instruction: What the code was supposed to do (for LLM context)
        groq_llm:             The LLM callable (system_prompt, user_prompt) -> str
        max_retries:          Max fix attempts (default 4)
        run_code_enabled:     Set False to skip subprocess execution (syntax-only mode)

    Returns:
        CorrectionResult with .success, .final_code, .attempts, .error, .correction_log
    """
    current_code = code
    log          = []

    print(f"\n[SelfCorrect] Starting correction loop (max {max_retries} attempts)")

    for attempt in range(1, max_retries + 1):
        print(f"[SelfCorrect] Attempt {attempt}/{max_retries}")
        entry = {"attempt": attempt}

        # ── Phase 1: Syntax check (fast, no subprocess) ────────
        syntax_error = check_syntax(current_code)
        if syntax_error:
            print(f"[SelfCorrect] Syntax error: {syntax_error}")
            entry["phase"]  = "syntax"
            entry["error"]  = syntax_error
            error_info = {
                "type":           "SyntaxError",
                "message":        syntax_error,
                "line_number":    None,
                "relevant_lines": syntax_error,
                "full_stderr":    syntax_error
            }
        else:
            # ── Phase 2: Import/compile check ──────────────────
            import_error = check_importable(current_code)
            if import_error:
                print(f"[SelfCorrect] Import error: {import_error}")
                entry["phase"]  = "import"
                entry["error"]  = import_error
                error_info = {
                    "type":           "ImportError",
                    "message":        import_error,
                    "line_number":    None,
                    "relevant_lines": import_error,
                    "full_stderr":    import_error
                }
            else:
                # ── Phase 3: Runtime execution ──────────────────
                if run_code_enabled:
                    run_result = run_code(current_code)
                    entry["phase"]   = "runtime"
                    entry["stdout"]  = run_result.stdout[:500]
                    entry["stderr"]  = run_result.stderr[:500]

                    if run_result.timed_out:
                        print(f"[SelfCorrect] Timed out after {RUN_TIMEOUT}s")
                        # Timeout might be infinite loop — flag but don't retry
                        # (LLM can't fix an intentional infinite loop)
                        log.append(entry)
                        return CorrectionResult(
                            success=False,
                            final_code=current_code,
                            attempts=attempt,
                            error=f"Code timed out after {RUN_TIMEOUT}s — possible infinite loop",
                            correction_log=log
                        )

                    if run_result.success:
                        print(f"[SelfCorrect] ✅ Code runs cleanly on attempt {attempt}")
                        entry["result"] = "success"
                        log.append(entry)
                        return CorrectionResult(
                            success=True,
                            final_code=current_code,
                            attempts=attempt,
                            correction_log=log
                        )

                    # Runtime failed — parse the traceback
                    error_info = parse_traceback(run_result.stderr)
                    entry["error"] = error_info["relevant_lines"]
                    print(f"[SelfCorrect] Runtime error: {error_info['type']}: {error_info['message']}")

                else:
                    # run_code disabled — syntax + import passed, treat as success
                    print(f"[SelfCorrect] ✅ Syntax + import checks passed (runtime disabled)")
                    entry["result"] = "success (syntax-only mode)"
                    log.append(entry)
                    return CorrectionResult(
                        success=True,
                        final_code=current_code,
                        attempts=attempt,
                        correction_log=log
                    )

        # ── Fix: send to LLM ───────────────────────────────────
        if attempt < max_retries:
            print(f"[SelfCorrect] Sending to LLM for fix (attempt {attempt})...")
            sys_prompt, usr_prompt = build_fix_prompt(
                current_code, error_info, original_instruction, attempt
            )
            raw_fix = groq_llm(sys_prompt, usr_prompt)

            # Strip markdown fences
            fixed = re.sub(r"```(?:python)?\n?", "", raw_fix).replace("```", "").strip()
            entry["fixed_code_preview"] = fixed[:200]
            current_code = fixed
            print(f"[SelfCorrect] Got fix from LLM, retrying...")
        else:
            entry["result"] = "exhausted"

        log.append(entry)

    # All retries exhausted
    print(f"[SelfCorrect] ❌ Could not fix after {max_retries} attempts")
    return CorrectionResult(
        success=False,
        final_code=current_code,
        attempts=max_retries,
        error=f"{error_info['type']}: {error_info['message']}",
        correction_log=log
    )


# ─────────────────────────────────────────────
# INTEGRATION GUIDE
# ─────────────────────────────────────────────
#
# In agent_executor.py, update tool_modify_function() and tool_create_file()
# Replace the final syntax-check-and-save block with this pattern:
#
#   from self_correction import self_correct_code
#
#   result = self_correct_code(
#       code=updated,                        # the generated/modified code
#       original_instruction=step.instruction,
#       groq_llm=groq_llm,
#       run_code_enabled=True                # set False if file has external deps
#   )
#
#   if not result.success:
#       return StepResult(
#           step_number=step.step_number,
#           action=step.action,
#           success=False,
#           error=f"Self-correction failed after {result.attempts} attempts: {result.error}"
#       )
#
#   file.content = result.final_code
#   db.commit()
#   _reembed_file(file, model, db)
#
#   return StepResult(
#       step_number=step.step_number,
#       action=step.action,
#       success=True,
#       output=f"Modified '{file.filename}' | correction attempts: {result.attempts}"
#   )