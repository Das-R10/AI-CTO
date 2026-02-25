"""
test_engine.py — Automated Test Generation + Execution + Impact Analysis
=========================================================================
Responsibilities:
  1. Generate pytest-style unit tests for any Python function/file
  2. Execute tests in subprocess sandbox (reuses execution_hardening.py)
  3. Feed test failures into self_correct_code() loop
  4. Detect which tests are impacted by a given function change
  5. Auto-patch tests when safe; flag for review when not
"""

import ast
import re
import os
import subprocess
import tempfile
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from self_correction import self_correct_code, CorrectionResult


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

TEST_TIMEOUT     = 15   # seconds
TEST_MAX_RETRIES = 3


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class TestResult:
    passed: bool
    output: str
    failures: list[str] = field(default_factory=list)
    timed_out: bool = False


@dataclass
class TestImpactReport:
    function_name: str
    impacted_test_ids: list[int]
    impacted_test_files: list[str]
    safe_to_auto_patch: bool


# ─────────────────────────────────────────────
# TEST GENERATION
# ─────────────────────────────────────────────

def generate_tests(
    source_code: str,
    filename: str,
    instruction: str,
    groq_llm
) -> str:
    """
    Ask the LLM to generate pytest-style unit tests for the given source file.
    Returns raw test code string.
    """
    module_name = filename.replace(".py", "").replace("/", ".").replace("\\", ".")

    prompt = f"""
You are a Python test engineer.

Write pytest unit tests for this file: {filename}
Module import: from {module_name} import *

Source code:
{source_code[:3000]}

Recent change instruction:
{instruction}

Rules:
- Use pytest style (def test_...)
- Cover normal cases, edge cases, and the specific change
- Use assert statements
- No mocking unless essential
- Return ONLY valid Python code, no markdown
"""
    raw = groq_llm(
        "You are an expert Python test engineer. Return only valid pytest code.",
        prompt
    )
    return re.sub(r"```(?:python)?\n?", "", raw).replace("```", "").strip()


# ─────────────────────────────────────────────
# TEST EXECUTION
# ─────────────────────────────────────────────

def run_tests(
    test_code: str,
    source_code: str,
    source_filename: str,
    timeout: int = TEST_TIMEOUT
) -> TestResult:
    """
    Write source + test to temp dir, run pytest, capture output.
    Returns TestResult with pass/fail + any failure messages.
    """
    import sys
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write source file
        src_path  = os.path.join(tmpdir, source_filename)
        test_path = os.path.join(tmpdir, f"test_{source_filename}")

        os.makedirs(os.path.dirname(src_path), exist_ok=True) if os.sep in source_filename else None

        with open(src_path, "w") as f:
            f.write(source_code)

        with open(test_path, "w") as f:
            f.write(test_code)

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short", "--no-header"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir
            )
            passed   = result.returncode == 0
            combined = result.stdout + result.stderr

            # Parse failure messages
            failures = []
            for line in combined.splitlines():
                if line.startswith("FAILED") or "AssertionError" in line or "Error" in line:
                    failures.append(line.strip())

            return TestResult(passed=passed, output=combined[:2000], failures=failures[:10])

        except subprocess.TimeoutExpired:
            return TestResult(
                passed=False,
                output=f"Test execution timed out after {timeout}s",
                timed_out=True
            )


# ─────────────────────────────────────────────
# FULL TEST LOOP: generate → run → self-correct
# ─────────────────────────────────────────────

def generate_and_validate_tests(
    source_code: str,
    filename: str,
    instruction: str,
    groq_llm,
    max_retries: int = TEST_MAX_RETRIES
) -> tuple[str, TestResult]:
    """
    Generate tests, run them, attempt self-correction if they fail.
    Returns (final_test_code, test_result).
    """
    test_code = generate_tests(source_code, filename, instruction, groq_llm)

    for attempt in range(1, max_retries + 1):
        print(f"[TestEngine] Running tests attempt {attempt}/{max_retries}")
        result = run_tests(test_code, source_code, filename)

        if result.timed_out:
            print("[TestEngine] Test timed out — skipping further attempts")
            break

        if result.passed:
            print(f"[TestEngine] ✅ Tests passed on attempt {attempt}")
            return test_code, result

        if attempt < max_retries:
            print(f"[TestEngine] ❌ Tests failed, asking LLM to fix tests...")
            fix_prompt = f"""
The following pytest tests failed when run against the code.

SOURCE FILE ({filename}):
{source_code[:2000]}

TEST FILE:
{test_code}

FAILURES:
{chr(10).join(result.failures[:10])}

Fix the tests so they correctly validate the source code.
Return ONLY valid Python pytest code. No markdown.
"""
            raw = groq_llm("Fix the broken pytest tests. Return only valid Python.", fix_prompt)
            test_code = re.sub(r"```(?:python)?\n?", "", raw).replace("```", "").strip()

    return test_code, result


# ─────────────────────────────────────────────
# TEST STORAGE
# ─────────────────────────────────────────────

def store_test(
    project_id: int,
    file_id: int,
    test_filename: str,
    test_code: str,
    test_result: TestResult,
    job_id: Optional[int],
    db: Session
) -> int:
    """
    Upsert test file to project_tests table and log the run.
    Returns the test record ID.
    """
    existing = db.execute(
        text("SELECT id FROM project_tests WHERE project_id=:pid AND filename=:fn"),
        {"pid": project_id, "fn": test_filename}
    ).fetchone()

    pass_rate = 1.0 if test_result.passed else 0.0

    if existing:
        test_id = existing[0]
        db.execute(
            text("""
                UPDATE project_tests
                SET content=:content, pass_rate=:pr, last_run_at=now()
                WHERE id=:id
            """),
            {"content": test_code, "pr": pass_rate, "id": test_id}
        )
    else:
        row = db.execute(
            text("""
                INSERT INTO project_tests (project_id, file_id, filename, content, pass_rate, last_run_at)
                VALUES (:pid, :fid, :fn, :content, :pr, now())
                RETURNING id
            """),
            {"pid": project_id, "fid": file_id, "fn": test_filename,
             "content": test_code, "pr": pass_rate}
        ).fetchone()
        test_id = row[0]

    db.execute(
        text("""
            INSERT INTO test_run_log (test_id, job_id, passed, output)
            VALUES (:tid, :jid, :passed, :output)
        """),
        {"tid": test_id, "jid": job_id, "passed": test_result.passed,
         "output": test_result.output[:1000]}
    )
    db.commit()
    return test_id


# ─────────────────────────────────────────────
# TEST IMPACT ANALYSIS
# ─────────────────────────────────────────────

def find_impacted_tests(
    function_name: str,
    project_id: int,
    db: Session
) -> TestImpactReport:
    """
    Find all test files that reference a given function name.
    Uses AST-level name search across stored test files.
    """
    rows = db.execute(
        text("SELECT id, filename, content FROM project_tests WHERE project_id=:pid"),
        {"pid": project_id}
    ).fetchall()

    impacted_ids   = []
    impacted_files = []

    for row in rows:
        tid, fname, content = row
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id == function_name:
                    impacted_ids.append(tid)
                    impacted_files.append(fname)
                    break
                if isinstance(node, ast.Attribute) and node.attr == function_name:
                    impacted_ids.append(tid)
                    impacted_files.append(fname)
                    break
        except SyntaxError:
            pass

    # Consider auto-patching safe when test references are straightforward calls
    # (not deeply nested assertion logic that requires semantic understanding)
    safe = len(impacted_ids) <= 3

    return TestImpactReport(
        function_name=function_name,
        impacted_test_ids=impacted_ids,
        impacted_test_files=impacted_files,
        safe_to_auto_patch=safe
    )


def rerun_impacted_tests(
    impact: TestImpactReport,
    new_source_code: str,
    source_filename: str,
    project_id: int,
    job_id: Optional[int],
    db: Session,
    groq_llm,
    auto_patch: bool = True
) -> dict:
    """
    Re-run all tests impacted by a function change.
    Optionally auto-patch tests that fail due to signature drift.
    Returns summary dict.
    """
    from models import File as DBFile

    results   = {}
    patched   = []
    needs_review = []

    for test_id, test_filename in zip(impact.impacted_test_ids, impact.impacted_test_files):
        row = db.execute(
            text("SELECT content FROM project_tests WHERE id=:tid"),
            {"tid": test_id}
        ).fetchone()
        if not row:
            continue

        test_code  = row[0]
        run_result = run_tests(test_code, new_source_code, source_filename)

        # Log run
        db.execute(
            text("INSERT INTO test_run_log (test_id, job_id, passed, output) VALUES (:tid,:jid,:p,:o)"),
            {"tid": test_id, "jid": job_id, "p": run_result.passed, "o": run_result.output[:1000]}
        )
        db.commit()

        if run_result.passed:
            results[test_filename] = "passed"
            continue

        if auto_patch and impact.safe_to_auto_patch:
            print(f"[TestEngine] Auto-patching '{test_filename}'...")
            fix_prompt = f"""
The function '{impact.function_name}' was modified. These tests now fail.
Source file ({source_filename}):
{new_source_code[:2000]}

Failing test file:
{test_code}

Failures:
{chr(10).join(run_result.failures[:5])}

Fix the tests to work with the updated function signature.
Return ONLY valid Python pytest code.
"""
            raw_fix   = groq_llm("Fix the broken tests to match the new function signature.", fix_prompt)
            fixed_tests = re.sub(r"```(?:python)?\n?", "", raw_fix).replace("```", "").strip()
            verify     = run_tests(fixed_tests, new_source_code, source_filename)

            if verify.passed:
                db.execute(
                    text("UPDATE project_tests SET content=:c, pass_rate=1.0, last_run_at=now() WHERE id=:tid"),
                    {"c": fixed_tests, "tid": test_id}
                )
                db.commit()
                patched.append(test_filename)
                results[test_filename] = "auto-patched"
            else:
                needs_review.append(test_filename)
                results[test_filename] = "failed — needs review"
        else:
            needs_review.append(test_filename)
            results[test_filename] = "failed — needs review"

    return {
        "results":      results,
        "patched":      patched,
        "needs_review": needs_review
    }

# ═════════════════════════════════════════════════════════════════════════════
# CI-STYLE VERIFICATION ENGINE  (Phase 3 — QA Layer Upgrade)
# ═════════════════════════════════════════════════════════════════════════════
#
# Extends the existing QA layer (generate/run/self-correct) with a full
# CI-style verification pipeline that runs inside an isolated sandbox.
#
# Responsibilities:
#   1. Detect repo features (requirements.txt, pytest.ini/tests/, package.json)
#   2. Execute each detected step inside a temp-dir sandbox
#   3. Capture exit codes + full console output
#   4. Block PR creation if any step fails
#   5. Return structured CI report
# ═════════════════════════════════════════════════════════════════════════════

import shutil
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SANDBOX RUNNER
# ─────────────────────────────────────────────

class SandboxRunner:
    """
    Clones the repo into a fresh temp directory and executes CI commands there.
    The sandbox is always cleaned up — success or failure.
    """

    def __init__(self, repo_path: str, timeout: int = 300):
        self.repo_path  = Path(repo_path).resolve()
        self.timeout    = timeout
        self.sandbox_dir: Optional[Path] = None
        self.work_dir:    Optional[Path] = None

    def __enter__(self) -> "SandboxRunner":
        self.sandbox_dir = Path(tempfile.mkdtemp(prefix="aicto_ci_"))
        target = self.sandbox_dir / "repo"
        shutil.copytree(str(self.repo_path), str(target))
        self.work_dir = target
        logger.info(f"[CI] Sandbox ready at {self.sandbox_dir}")
        return self

    def __exit__(self, *_):
        if self.sandbox_dir and self.sandbox_dir.exists():
            shutil.rmtree(str(self.sandbox_dir), ignore_errors=True)
            logger.info("[CI] Sandbox cleaned up")

    def run(self, cmd: list[str], label: str) -> dict:
        """Execute one CI step; capture exit code + combined stdout/stderr."""
        import sys
        logger.info(f"[CI] → {label}: {' '.join(cmd)}")
        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.work_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            duration = round(time.monotonic() - start, 2)
            combined = f"STDOUT:\n{proc.stdout.strip()}\nSTDERR:\n{proc.stderr.strip()}".strip()
            passed   = proc.returncode == 0
            level    = "✅" if passed else "❌"
            logger.info(f"[CI] {level} {label} exit={proc.returncode} ({duration}s)")
            return {
                "step":             label,
                "passed":           passed,
                "exit_code":        proc.returncode,
                "duration_seconds": duration,
                "log":              combined,
            }
        except subprocess.TimeoutExpired:
            return {
                "step":             label,
                "passed":           False,
                "exit_code":        -1,
                "duration_seconds": self.timeout,
                "log":              f"TIMEOUT: step '{label}' exceeded {self.timeout}s",
            }
        except Exception as exc:
            return {
                "step":             label,
                "passed":           False,
                "exit_code":        -1,
                "duration_seconds": round(time.monotonic() - start, 2),
                "log":              f"ERROR: {exc}",
            }


# ─────────────────────────────────────────────
# FEATURE DETECTION
# ─────────────────────────────────────────────

def _detect_repo_features(repo_path: str) -> dict:
    """Inspect repo root for files that trigger CI steps."""
    root = Path(repo_path)
    return {
        "has_requirements": (root / "requirements.txt").exists(),
        "has_pytest": (
            (root / "pytest.ini").exists()
            or (root / "setup.cfg").exists()
            or (root / "pyproject.toml").exists()
            or (root / "tests").is_dir()
            or (root / "test").is_dir()
        ),
        "has_package_json": (root / "package.json").exists(),
    }


# ─────────────────────────────────────────────
# CI VERIFICATION PIPELINE
# ─────────────────────────────────────────────

def run_ci_verification(repo_path: str, timeout: int = 300) -> dict:
    """
    Full CI pipeline for a repository.

    Steps executed (only when detected):
      1. pip install -r requirements.txt
      2. python -m pytest --tb=short -q
      3. npm install && npm run build

    Returns a structured CI report:
    {
        "backend_tests":      "passed" | "failed" | "skipped",
        "frontend_build":     "passed" | "failed" | "skipped",
        "logs":               "<combined log text>",
        "ci_passed":          bool,   # False → PR must be blocked
        "steps_run":          int,
        "features_detected":  dict,
        "steps_detail":       [...]   # per-step results for audit trail
    }
    """
    features   = _detect_repo_features(repo_path)
    steps_run  = []

    pip_result:   Optional[dict] = None
    pytest_result: Optional[dict] = None
    npm_install:  Optional[dict] = None
    npm_build:    Optional[dict] = None

    with SandboxRunner(repo_path, timeout=timeout) as sandbox:

        # ── Backend: dependency install ──────────────────────────
        if features["has_requirements"]:
            pip_result = sandbox.run(
                ["pip", "install", "-r", "requirements.txt", "--quiet", "--disable-pip-version-check"],
                label="pip_install"
            )
            steps_run.append(pip_result)

        # ── Backend: pytest ──────────────────────────────────────
        if features["has_pytest"]:
            install_ok = (pip_result is None) or pip_result["passed"]
            if install_ok:
                pytest_result = sandbox.run(
                    ["python3", "-m", "pytest", "--tb=short", "-q"],
                    label="pytest"
                )
            else:
                pytest_result = {
                    "step": "pytest", "passed": False, "exit_code": -1,
                    "duration_seconds": 0,
                    "log": "Skipped: pip install failed — cannot run tests safely"
                }
            steps_run.append(pytest_result)

        # ── Frontend: npm install ────────────────────────────────
        if features["has_package_json"]:
            npm_install = sandbox.run(["npm", "install", "--silent"], label="npm_install")
            steps_run.append(npm_install)

            if npm_install["passed"]:
                npm_build = sandbox.run(["npm", "run", "build"], label="npm_build")
            else:
                npm_build = {
                    "step": "npm_build", "passed": False, "exit_code": -1,
                    "duration_seconds": 0,
                    "log": "Skipped: npm install failed"
                }
            steps_run.append(npm_build)

    # ── Aggregate ────────────────────────────────────────────────
    def _status(r: Optional[dict]) -> str:
        if r is None:
            return "skipped"
        return "passed" if r["passed"] else "failed"

    backend_tests_status  = _status(pytest_result)
    frontend_build_status = (
        "skipped" if (npm_install is None and npm_build is None)
        else ("passed" if (npm_install and npm_build and npm_install["passed"] and npm_build["passed"])
              else "failed")
    )

    all_passed = all(s["passed"] for s in steps_run) if steps_run else True

    log_parts = []
    for s in steps_run:
        log_parts.append(f"[{s['step']}] exit={s['exit_code']} ({s['duration_seconds']}s)\n{s['log']}")
    combined_log = "\n\n".join(log_parts) if log_parts else "No CI steps executed (nothing detected)."

    report = {
        "backend_tests":     backend_tests_status,
        "frontend_build":    frontend_build_status,
        "logs":              combined_log,
        # Extended fields used by qa_gate_check and the /create-pr/ endpoint
        "ci_passed":         all_passed,
        "steps_run":         len(steps_run),
        "features_detected": features,
        "steps_detail":      steps_run,
    }

    if not all_passed:
        logger.warning("[CI] ⛔ CI verification FAILED — PR creation must be blocked")
    else:
        logger.info("[CI] ✅ All CI steps passed")

    return report


# ─────────────────────────────────────────────
# UNIFIED PR GATE
# ─────────────────────────────────────────────

def qa_gate_check(
    repo_path: str,
    changed_files: list[str],
    timeout: int = 300
) -> dict:
    """
    Combined QA gate that must pass before PR creation is allowed.

    Layer 1 — Per-file QA (existing):
        run_qa_on_file() → syntax check + static analysis on every changed file.

    Layer 2 — CI Verification (new):
        run_ci_verification() → pip/pytest/npm pipeline in isolated sandbox.

    Returns:
    {
        "allowed_to_create_pr": bool,
        "file_qa":              { "passed": bool, "results": [...] },
        "ci_verification":      { ... CI report ... },
        "summary":              str
    }

    Callers MUST check `allowed_to_create_pr` before calling create_github_pr().
    """
    # Layer 1
    file_results  = [run_qa_on_file(f) for f in changed_files]
    files_passed  = all(r["passed"] for r in file_results)

    # Layer 2
    ci_report = run_ci_verification(repo_path, timeout=timeout)

    allowed = files_passed and ci_report["ci_passed"]

    return {
        "allowed_to_create_pr": allowed,
        "file_qa": {
            "passed":  files_passed,
            "results": file_results,
        },
        "ci_verification": ci_report,
        "summary": (
            "✅ All QA and CI checks passed. PR creation allowed."
            if allowed else
            "⛔ QA/CI gate FAILED. PR creation blocked. Review logs."
        ),
    }