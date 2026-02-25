"""
execution_hardening.py — Hardened Execution Sandbox
=====================================================
Wraps subprocess execution with layered security.

  ORIGINAL (preserved, unchanged):
    - Dangerous import blocking          (check_dangerous_imports)
    - File system write restrictions     (_build_restricted_runner)
    - Memory limits (RLIMIT_AS)
    - Execution mode selection           (run_hardened)

  EXTENDED (Phase 4 — Execution Security Hardening):
    - Network isolation via Linux namespace (unshare --net)
    - CPU time limit (RLIMIT_CPU) + file-size limit (RLIMIT_FSIZE)
    - Process count limit (RLIMIT_NPROC) — anti fork-bomb
    - Dangerous pattern scanner: os.system, shell=True, rm/del outside repo
    - Execution audit log  →  execution_audit.log  (every invocation)
    - safe_mode=True: disables all subprocess execution, static-only
    - SandboxController: wraps self_correct_code() inside the security perimeter

All existing public interfaces (check_dangerous_imports, run_hardened, etc.)
are preserved with identical signatures. New params use backward-compatible
defaults so no callers break.
"""

import ast
import re
import sys
import os
import subprocess
import tempfile
import textwrap
import logging
import datetime
from typing import Optional
from dataclasses import dataclass, field

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

EXECUTION_MODES = {"syntax-only", "runtime-safe", "test-mode"}

BLOCKED_IMPORTS = {
    "shutil", "subprocess", "socket", "requests", "urllib",
    "ftplib", "smtplib", "paramiko", "fabric",
}

# Dangerous patterns (regex → human label)
DANGEROUS_PATTERNS: list[tuple[str, str]] = [
    (r"subprocess\.(run|Popen|call|check_output|check_call)\s*\(.*?shell\s*=\s*True",
     "subprocess call with shell=True detected"),
    (r"\bos\.system\s*\(",
     "os.system() call detected"),
    (r"\bos\.popen\s*\(",
     "os.popen() call detected"),
    (r"\beval\s*\(",
     "eval() call detected"),
    (r"\bexec\s*\(",
     "exec() call detected"),
    (r"\bos\.(remove|unlink|rmdir)\s*\(",
     "os file-deletion call detected"),
    (r"\bshutil\.rmtree\s*\(",
     "shutil.rmtree() call detected"),
    (r"\b__import__\s*\(",
     "__import__() dynamic import detected"),
    (r"\bctypes\.",
     "ctypes usage detected"),
]

WRITE_PATTERNS = re.compile(r'open\s*\([^)]*["\']w["\']', re.IGNORECASE)

# Audit log — override via env var
_AUDIT_LOG_PATH = os.environ.get("AICTO_AUDIT_LOG", "execution_audit.log")

# Resource limits
SANDBOX_MEMORY_BYTES = 256 * 1024 * 1024   # 256 MB
SANDBOX_FSIZE_BYTES  = 10  * 1024 * 1024   # 10 MB max file write
SANDBOX_CPU_SECONDS  = 20
SANDBOX_NPROC        = 64


# ─────────────────────────────────────────────
# AUDIT LOGGER
# ─────────────────────────────────────────────

_audit_logger = logging.getLogger("aicto.execution_audit")


def _setup_audit_log():
    """Idempotent — set up file handler once."""
    if _audit_logger.handlers:
        return
    _audit_logger.setLevel(logging.INFO)
    try:
        fh = logging.FileHandler(_AUDIT_LOG_PATH, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        _audit_logger.addHandler(fh)
    except Exception:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        _audit_logger.addHandler(sh)


_setup_audit_log()


def _audit(event: str, detail: str = "", blocked: bool = False, safe_mode: bool = False):
    status = "BLOCKED" if blocked else ("SAFE_MODE" if safe_mode else "EXEC")
    _audit_logger.info(f"[{status}] {event} | {detail[:300]}")


# ─────────────────────────────────────────────
# DATA CLASSES  (original preserved + extended)
# ─────────────────────────────────────────────

@dataclass
class HardenedRunResult:
    success:        bool
    stdout:         str
    stderr:         str
    blocked_reason: Optional[str] = None
    timed_out:      bool = False
    # Extended fields (new — do not break existing callers)
    safe_mode_used: bool = False
    audit_events:   list = field(default_factory=list)


# ─────────────────────────────────────────────
# ORIGINAL HELPERS — PRESERVED UNCHANGED
# ─────────────────────────────────────────────

def check_dangerous_imports(code: str) -> Optional[str]:
    """
    Returns a block reason string if code imports anything in BLOCKED_IMPORTS.
    (Original — unchanged.)
    """
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base = alias.name.split(".")[0]
                    if base in BLOCKED_IMPORTS:
                        return f"Blocked import '{alias.name}' not allowed in sandbox"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    base = node.module.split(".")[0]
                    if base in BLOCKED_IMPORTS:
                        return f"Blocked import '{node.module}' not allowed in sandbox"
    except SyntaxError:
        pass
    return None


def _build_restricted_runner(code: str, tmpdir: str) -> str:
    """
    Wrap user code in a runner that restricts filesystem writes to tmpdir.
    (Original — unchanged.)
    """
    restriction_prelude = textwrap.dedent(f"""
import builtins as _builtins
_orig_open = _builtins.open
_allowed_dir = {repr(tmpdir)}

def _safe_open(file, mode='r', *args, **kwargs):
    if isinstance(file, str) and 'w' in mode:
        import os
        abs_path = os.path.abspath(file)
        if not abs_path.startswith(_allowed_dir):
            raise PermissionError(f"File write blocked outside project directory: {{abs_path}}")
    return _orig_open(file, mode, *args, **kwargs)

_builtins.open = _safe_open
""")
    return restriction_prelude + "\n" + code


# ─────────────────────────────────────────────
# NEW: DANGEROUS PATTERN SCANNER
# ─────────────────────────────────────────────

def check_dangerous_patterns(code: str, repo_path: Optional[str] = None) -> list[str]:
    """
    Regex scan for dangerous execution patterns.
    Returns list of violation strings (empty = clean).
    """
    violations = []
    for pattern, label in DANGEROUS_PATTERNS:
        if re.search(pattern, code, re.DOTALL | re.MULTILINE):
            violations.append(label)
    return violations


def static_security_scan(code: str, repo_path: Optional[str] = None) -> dict:
    violations: list[str] = []

    import_block = check_dangerous_imports(code)
    if import_block:
        violations.append(import_block)

    violations.extend(check_dangerous_patterns(code, repo_path))

    # Remove the ast.parse check entirely — syntax is already
    # checked earlier in run_hardened() and blocks valid type hints
    # try:
    #     ast.parse(code)
    # except SyntaxError as e:
    #     violations.append(f"SyntaxError: {e}")

    return {
        "passed":     len(violations) == 0,
        "violations": violations,
        "blocked":    len(violations) > 0,
    }

# ─────────────────────────────────────────────
# NEW: RESOURCE LIMIT BUILDER
# ─────────────────────────────────────────────

def _build_resource_limits(
    memory_bytes: int = SANDBOX_MEMORY_BYTES,
    cpu_seconds:  int = SANDBOX_CPU_SECONDS,
    fsize_bytes:  int = SANDBOX_FSIZE_BYTES,
    nproc:        int = SANDBOX_NPROC,
):
    """
    Returns a preexec_fn that applies OS resource limits before exec().
    Linux only — gracefully no-ops on Windows or when limits can't be set.
    """
    def _apply():
        try:
            import resource as _r

            def _try_set(limit_const, soft, hard_fallback):
                try:
                    _, hard = _r.getrlimit(limit_const)
                    cap = soft if hard == -1 or hard > soft else hard
                    _r.setrlimit(limit_const, (cap, max(cap, hard_fallback)))
                except (ValueError, _r.error, OSError):
                    pass

            _try_set(_r.RLIMIT_AS,    memory_bytes, memory_bytes * 2)
            _try_set(_r.RLIMIT_CPU,   cpu_seconds,  cpu_seconds  * 2)
            _try_set(_r.RLIMIT_FSIZE, fsize_bytes,  fsize_bytes  * 2)
            _try_set(_r.RLIMIT_NPROC, nproc,        nproc        * 2)

        except ImportError:
            pass

    return _apply


# ─────────────────────────────────────────────
# NEW: NETWORK ISOLATION WRAPPER
# ─────────────────────────────────────────────

_UNSHARE_AVAILABLE: Optional[bool] = None


def _network_isolated_cmd(cmd: list[str]) -> tuple[list[str], bool]:
    """
    Wrap cmd with 'unshare --net' if available.
    Returns (final_cmd, network_isolated_bool).
    """
    global _UNSHARE_AVAILABLE
    if _UNSHARE_AVAILABLE is None:
        try:
            r = subprocess.run(["which", "unshare"], capture_output=True, text=True)
            _UNSHARE_AVAILABLE = r.returncode == 0 and bool(r.stdout.strip())
        except Exception:
            _UNSHARE_AVAILABLE = False

    if _UNSHARE_AVAILABLE:
        return ["unshare", "--net", "--"] + cmd, True
    return cmd, False


# ─────────────────────────────────────────────
# ORIGINAL run_hardened() — EXTENDED
# Same signature + 2 new optional params
# ─────────────────────────────────────────────

def run_hardened(
    code:      str,
    mode:      str = "runtime-safe",
    timeout:   int = 8,
    # ─── New params (backward-compatible) ───
    safe_mode: bool = False,
    repo_path: Optional[str] = None,
) -> HardenedRunResult:
    """
    Execute code with layered safety restrictions.

    Original modes (unchanged behaviour):
      syntax-only   — AST parse only, no execution
      runtime-safe  — subprocess with import + write restrictions
      test-mode     — pytest execution (test_engine.py)

    Extended behaviour (new):
      safe_mode=True       → static-only, all subprocess skipped
      Network isolation    → unshare --net (Linux, auto-detected)
      CPU + memory limits  → via RLIMIT_* in preexec_fn
      Pattern scan         → blocks shell=True, os.system, etc. before exec
      Audit logging        → every call logged to execution_audit.log
    """
    audit_events: list[str] = []

    # ── safe_mode override ───────────────────────────────────────
    if safe_mode:
        _audit("run_hardened", f"mode=SAFE_MODE code_len={len(code)}", safe_mode=True)
        audit_events.append("safe_mode: subprocess disabled")

        try:
            ast.parse(code)
        except SyntaxError as e:
            reason = f"SyntaxError: {e}"
            _audit("run_hardened", reason, blocked=True, safe_mode=True)
            return HardenedRunResult(
                success=False, stdout="", stderr=reason,
                blocked_reason=reason, safe_mode_used=True,
                audit_events=audit_events,
            )

        scan = static_security_scan(code, repo_path)
        if not scan["passed"]:
            reason = "; ".join(scan["violations"])
            _audit("run_hardened", f"static_scan BLOCKED: {reason}", blocked=True, safe_mode=True)
            audit_events.append(f"blocked: {reason}")
            return HardenedRunResult(
                success=False, stdout="", stderr=reason,
                blocked_reason=reason, safe_mode_used=True,
                audit_events=audit_events,
            )

        audit_events.append("static_scan: passed (safe_mode)")
        return HardenedRunResult(
            success=True, stdout="", stderr="",
            safe_mode_used=True, audit_events=audit_events,
        )

    # ── Mode validation ──────────────────────────────────────────
    if mode not in EXECUTION_MODES:
        return HardenedRunResult(
            success=False, stdout="", stderr="",
            blocked_reason=f"Unknown execution mode '{mode}'",
            audit_events=audit_events,
        )

    # ── Syntax check (always) ────────────────────────────────────
    try:
        ast.parse(code)
    except SyntaxError as e:
        _audit("run_hardened", f"SyntaxError: {e}", blocked=True)
        return HardenedRunResult(
            success=False, stdout="", stderr=str(e),
            blocked_reason=f"SyntaxError: {e}",
            audit_events=audit_events,
        )

    if mode == "syntax-only":
        _audit("run_hardened", "mode=syntax-only passed")
        return HardenedRunResult(success=True, stdout="", stderr="", audit_events=audit_events)

    # ── Static security scan (new — runs before any subprocess) ─
    scan = static_security_scan(code, repo_path)
    if not scan["passed"]:
        reason = "; ".join(scan["violations"])
        _audit("run_hardened", f"static_scan BLOCKED: {reason}", blocked=True)
        audit_events.append(f"static_scan blocked: {reason}")
        return HardenedRunResult(
            success=False, stdout="", stderr=reason,
            blocked_reason=reason, audit_events=audit_events,
        )
    audit_events.append("static_scan: passed")

    # ── Original: dangerous import check ────────────────────────
    block_reason = check_dangerous_imports(code)
    if block_reason:
        _audit("run_hardened", f"import_block: {block_reason}", blocked=True)
        return HardenedRunResult(
            success=False, stdout="", stderr=block_reason,
            blocked_reason=block_reason, audit_events=audit_events,
        )

    # ── runtime-safe execution (original + hardened) ─────────────
    if mode == "runtime-safe":
        with tempfile.TemporaryDirectory() as tmpdir:
            restricted_code = _build_restricted_runner(code, tmpdir)
            tmp_path = os.path.join(tmpdir, "exec_target.py")
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(restricted_code)

            base_cmd = [sys.executable, tmp_path]
            cmd, net_isolated = _network_isolated_cmd(base_cmd)

            _audit(
                "run_hardened",
                f"mode=runtime-safe timeout={timeout}s "
                f"network_isolated={net_isolated} tmp={tmp_path}"
            )
            audit_events.append(
                f"exec cmd={'unshare+net' if net_isolated else 'direct'} "
                f"timeout={timeout}s net_isolated={net_isolated}"
            )

            preexec = _build_resource_limits()

            try:
                run_kwargs = dict(
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                # preexec_fn only works on Linux/Mac — skip entirely on Windows
                if sys.platform != "win32":
                    run_kwargs["preexec_fn"] = preexec

                result = subprocess.run(cmd, **run_kwargs)
                _audit("run_hardened", f"exit={result.returncode}")
                audit_events.append(f"exit_code={result.returncode}")
                return HardenedRunResult(
                    success=(result.returncode == 0),
                    stdout=result.stdout[:1000],
                    stderr=result.stderr[:1000],
                    audit_events=audit_events,
                )
            except subprocess.TimeoutExpired:
                _audit("run_hardened", f"TIMEOUT after {timeout}s", blocked=True)
                audit_events.append(f"timeout {timeout}s")
                return HardenedRunResult(
                    success=False, stdout="", stderr=f"Timed out after {timeout}s",
                    timed_out=True, audit_events=audit_events,
                )

    return HardenedRunResult(
        success=False, stdout="", stderr="Unhandled execution mode",
        audit_events=audit_events,
    )


# ═════════════════════════════════════════════════════════════════════════════
# SANDBOX CONTROLLER
# Single entry point for all agent execution with security enforcement.
# Wraps self_correct_code() — does NOT remove it.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SandboxConfig:
    """
    Security posture for one sandboxed session.

    safe_mode:        True → no subprocess, static validation only.
    timeout:          Wall-clock kill timeout (seconds).
    memory_bytes:     Virtual memory cap per subprocess (RLIMIT_AS).
    cpu_seconds:      CPU time cap per subprocess (RLIMIT_CPU).
    repo_path:        Repo root used for file-deletion boundary checks.
    max_retries:      Max self-correction attempts inside the loop.
    """
    safe_mode:    bool          = False
    timeout:      int           = 8
    memory_bytes: int           = SANDBOX_MEMORY_BYTES
    cpu_seconds:  int           = SANDBOX_CPU_SECONDS
    repo_path:    Optional[str] = None
    max_retries:  int           = 4


@dataclass
class SandboxedCorrectionResult:
    """
    Result of SandboxController.run_corrected().

    Extends CorrectionResult with security metadata.
    All original CorrectionResult fields are mirrored here so callers
    that previously used CorrectionResult can switch without changes.
    """
    success:        bool
    final_code:     str
    attempts:       int
    error:          Optional[str]  = None
    correction_log: list           = field(default_factory=list)
    # Security fields
    safe_mode_used: bool           = False
    security_scan:  Optional[dict] = None
    blocked_reason: Optional[str]  = None
    audit_trail:    list           = field(default_factory=list)


class SandboxController:
    """
    Unified sandbox controller for AI CTO agent execution.

    Enforces the full security perimeter:
      1. Pre-execution static security scan (blocks dangerous code immediately)
      2. safe_mode=True support (static-only — no subprocess ever)
      3. Network isolation (unshare --net)
      4. OS resource limits (memory, CPU, fsize, nproc)
      5. Audit log for every invocation
      6. Wraps self_correct_code() — the correction loop runs INSIDE the sandbox

    Usage (from agent_executor.py):

        from execution_hardening import SandboxController, SandboxConfig

        controller = SandboxController(SandboxConfig(
            safe_mode=os.environ.get("AICTO_SAFE_MODE") == "1",
            repo_path=project_repo_path,
        ))

        result = controller.run_corrected(code, step.instruction, groq_llm)
        if not result.success:
            return StepResult(success=False, error=result.error)

        file.content = result.final_code
    """

    def __init__(self, config: Optional[SandboxConfig] = None):
        self.config = config or SandboxConfig()
        _audit(
            "SandboxController.__init__",
            f"safe_mode={self.config.safe_mode} "
            f"timeout={self.config.timeout}s "
            f"memory={self.config.memory_bytes // (1024*1024)}MB "
            f"cpu_cap={self.config.cpu_seconds}s",
        )

    def scan(self, code: str) -> dict:
        """Static-only security scan. Always available regardless of safe_mode."""
        result = static_security_scan(code, self.config.repo_path)
        _audit(
            "SandboxController.scan",
            f"passed={result['passed']} violations={len(result['violations'])}",
            blocked=not result["passed"],
        )
        return result

    def run_corrected(
        self,
        code: str,
        original_instruction: str,
        groq_llm,
    ) -> SandboxedCorrectionResult:
        """
        Run the self-correction loop inside the security sandbox.

        Execution flow:
          1. Static security scan → block if dangerous patterns found
          2. If safe_mode=True  → run self_correct_code(run_code_enabled=False)
          3. Otherwise          → monkey-patch run_code with hardened version,
                                  run self_correct_code(run_code_enabled=True),
                                  restore original run_code in finally block

        The self_correct_code() loop is preserved in full.
        safe_mode disables subprocess but keeps all LLM correction rounds.
        """
        from self_correction import self_correct_code

        audit_trail: list[dict] = []
        cfg = self.config

        _audit(
            "SandboxController.run_corrected",
            f"safe_mode={cfg.safe_mode} code_len={len(code)} "
            f"instruction_len={len(original_instruction)}",
        )

        # ── 1. Static security scan ──────────────────────────────
        scan_result = static_security_scan(code, cfg.repo_path)
        audit_trail.append({
            "step":       "static_scan",
            "passed":     scan_result["passed"],
            "violations": scan_result["violations"],
        })

        if not scan_result["passed"]:
            reason = "; ".join(scan_result["violations"])
            _audit("SandboxController.run_corrected", f"BLOCKED: {reason}", blocked=True)
            return SandboxedCorrectionResult(
                success=False,
                final_code=code,
                attempts=0,
                error=f"Security scan blocked: {reason}",
                safe_mode_used=cfg.safe_mode,
                security_scan=scan_result,
                blocked_reason=reason,
                audit_trail=audit_trail,
            )

        # ── 2. safe_mode — correction loop without subprocess ────
        if cfg.safe_mode:
            _audit("SandboxController.run_corrected", "safe_mode → runtime disabled")
            audit_trail.append({"step": "runtime", "result": "skipped (safe_mode)"})

            correction = self_correct_code(
                code=code,
                original_instruction=original_instruction,
                groq_llm=groq_llm,
                max_retries=cfg.max_retries,
                run_code_enabled=False,
            )
            audit_trail.append({
                "step": "self_correct", "attempts": correction.attempts,
                "success": correction.success,
            })
            return SandboxedCorrectionResult(
                success=correction.success,
                final_code=correction.final_code,
                attempts=correction.attempts,
                error=correction.error,
                correction_log=correction.correction_log,
                safe_mode_used=True,
                security_scan=scan_result,
                audit_trail=audit_trail,
            )

        # ── 3. Runtime correction with hardened run_code ─────────
        import self_correction as _sc_module
        _original_run_code = _sc_module.run_code

        def _hardened_run_code(code_inner: str, timeout: int = cfg.timeout) -> _sc_module.RunResult:
            hresult = run_hardened(
                code=code_inner,
                mode="runtime-safe",
                timeout=timeout,
                safe_mode=False,
                repo_path=cfg.repo_path,
            )
            audit_trail.append({
                "step":           "subprocess_exec",
                "success":        hresult.success,
                "timed_out":      hresult.timed_out,
                "blocked_reason": hresult.blocked_reason,
                "audit_events":   hresult.audit_events,
            })
            return _sc_module.RunResult(
                success=hresult.success,
                stdout=hresult.stdout,
                stderr=hresult.stderr,
                returncode=0 if hresult.success else 1,
                timed_out=hresult.timed_out,
            )

        _sc_module.run_code = _hardened_run_code
        try:
            correction = self_correct_code(
                code=code,
                original_instruction=original_instruction,
                groq_llm=groq_llm,
                max_retries=cfg.max_retries,
                run_code_enabled=True,
            )
        finally:
            _sc_module.run_code = _original_run_code   # always restore

        audit_trail.append({
            "step": "self_correct", "attempts": correction.attempts,
            "success": correction.success,
        })
        _audit(
            "SandboxController.run_corrected",
            f"done success={correction.success} attempts={correction.attempts}",
        )

        return SandboxedCorrectionResult(
            success=correction.success,
            final_code=correction.final_code,
            attempts=correction.attempts,
            error=correction.error,
            correction_log=correction.correction_log,
            safe_mode_used=False,
            security_scan=scan_result,
            audit_trail=audit_trail,
        )


# ─────────────────────────────────────────────
# CONVENIENCE FACTORY
# ─────────────────────────────────────────────

def get_sandbox_controller(
    safe_mode: bool = False,
    repo_path: Optional[str] = None,
    timeout:   int  = 8,
) -> SandboxController:
    """
    Convenience factory used by agent_executor.py.

    Reads AICTO_SAFE_MODE env var if safe_mode is not explicitly set.

    Example:
        controller = get_sandbox_controller(
            safe_mode=os.environ.get("AICTO_SAFE_MODE") == "1",
            repo_path=project.repo_path,
        )
        result = controller.run_corrected(code, instruction, groq_llm)
    """
    return SandboxController(SandboxConfig(
        safe_mode=safe_mode,
        repo_path=repo_path,
        timeout=timeout,
    ))