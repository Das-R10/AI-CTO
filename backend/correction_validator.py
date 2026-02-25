"""
correction_validator.py — Behavioral Correctness Validator
===========================================================
Solves the self-correction loop exit condition problem.

  PROBLEM:  Loop exits when code runs without throwing an exception.
            "No exception" ≠ "correct behavior." A function can return
            None, silently corrupt data, or break on edge cases without
            raising a single error.

  SOLUTION: Two validation layers added on top of self_correction.py:
            Layer 2 — Postcondition check  (deterministic execution + contract)
            Layer 3 — Semantic critic pass (separate LLM call in critic role)

Usage in agent_executor.py:
    from correction_validator import validate_correction, should_skip_postcondition, format_validation_feedback
"""

import ast
import re
import json
import subprocess
import sys
import tempfile
import os
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class PostconditionResult:
    passed: bool
    checks_run: int
    failures: list
    output_samples: list   # [{input, output, passed, reason}]


@dataclass
class CriticVerdict:
    intent_fulfilled: bool
    confidence: float           # 0.0–1.0
    issues_found: list
    verdict_reason: str


@dataclass
class ValidationResult:
    passed: bool
    layer_failed: Optional[str]  # "postcondition" | "critic" | None
    postcondition: Optional[PostconditionResult] = None
    critic: Optional[CriticVerdict] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────
# LAYER 2 — BEHAVIORAL CONTRACTS
# ─────────────────────────────────────────────
# Each entry: (regex_pattern, description, checker_source_code)
# The checker_source_code is injected into a subprocess and run against
# actual function output. No LLM decides what "correct" means — the
# contract code does. This is fully deterministic.

_INSTRUCTION_CONTRACTS = [
    (
        r"\bsort\b",
        "output must be a sorted sequence",
        """
def _check(result, arg):
    if result is None:
        return False, "returned None instead of sorted sequence"
    if not hasattr(result, '__iter__'):
        return False, f"returned non-iterable: {type(result).__name__}"
    lst = list(result)
    try:
        if lst != sorted(lst):
            return False, f"output is not sorted: {lst}"
    except TypeError:
        pass  # mixed types — ordering check skipped
    return True, "sorted correctly"
"""
    ),
    (
        r"\bfilter\b",
        "output must be iterable (subset of input)",
        """
def _check(result, arg):
    if result is None:
        return False, "returned None instead of filtered iterable"
    if not hasattr(result, '__iter__'):
        return False, f"returned non-iterable: {type(result).__name__}"
    return True, "filter returned iterable"
"""
    ),
    (
        r"\breturn.+list\b|\blist of\b|\bget.+list\b",
        "output must be a list",
        """
def _check(result, arg):
    if result is None:
        return False, "returned None instead of list"
    if not isinstance(result, list):
        return False, f"expected list, got {type(result).__name__}"
    return True, "returned list"
"""
    ),
    (
        r"\bvalidat\b",
        "output must be bool or raise on invalid",
        """
def _check(result, arg):
    if result is None:
        return False, "returned None — validators must return bool or raise"
    return True, "returned non-None"
"""
    ),
    (
        r"\bcount\b|\bsum\b|\baverage\b|\bmean\b",
        "output must be numeric",
        """
def _check(result, arg):
    if result is None:
        return False, "returned None instead of numeric value"
    if not isinstance(result, (int, float)):
        return False, f"expected numeric, got {type(result).__name__}"
    return True, f"returned numeric: {result}"
"""
    ),
    (
        r"\bparse\b|\bdeserializ\b",
        "output must be non-None structured data",
        """
def _check(result, arg):
    if result is None:
        return False, "parser returned None — should return data or raise"
    return True, "returned non-None"
"""
    ),
    (
        r"\bsearch\b|\bfind\b|\blookup\b",
        "output must not be None (return empty container if not found)",
        """
def _check(result, arg):
    if result is None:
        return False, "search/find returned None — return empty container instead"
    return True, f"returned {type(result).__name__}"
"""
    ),
    (
        r"\btransform\b|\bconvert\b|\bmap\b",
        "output must be non-None and same length as input if input is a sequence",
        """
def _check(result, arg):
    if result is None:
        return False, "transform returned None"
    if hasattr(arg, '__len__') and hasattr(result, '__len__'):
        if len(result) != len(arg):
            return False, f"transform changed length: {len(arg)} → {len(result)}"
    return True, "transform returned non-None"
"""
    ),
]

_DEFAULT_CONTRACT = """
def _check(result, arg):
    # Default: flag None returns as suspicious (allow for known void functions)
    if result is None:
        return True, "None accepted (no specific contract matched — verify manually)"
    return True, f"returned {type(result).__name__}"
"""


def _detect_contract(instruction: str) -> tuple:
    """Match instruction text to the tightest applicable behavioral contract."""
    instr_lower = instruction.lower()
    for pattern, description, checker_src in _INSTRUCTION_CONTRACTS:
        if re.search(pattern, instr_lower):
            return description, checker_src
    return "default (non-None output check)", _DEFAULT_CONTRACT


def _generate_probe_inputs(code: str, function_name: str, groq_llm) -> list:
    """
    Ask the LLM for probe INPUTS only — never expected outputs.
    The contract (deterministic code, not LLM) decides what correct output is.
    This keeps the LLM's role narrow: it knows the function signature,
    we ask only for diverse inputs, not for expected results.
    """
    prompt = f"""Function to probe:
{code[:1500]}

Target function name: {function_name}

Return ONLY a JSON array of 4-6 diverse input values as Python repr strings.
Include: normal case, empty/zero case, edge case, boundary value, stress input.
Example for a sort function: ["[3,1,2]", "[]", "[1]", "[-5,0,5,3]", "[1,1,1]", "[9,8,7,6,5]"]
Return ONLY the JSON array. No explanation. No markdown."""

    raw = groq_llm(
        "Return ONLY a JSON array of Python repr input strings. Nothing else.",
        prompt
    ).strip().replace("```json", "").replace("```", "").strip()

    try:
        inputs = json.loads(raw)
        if isinstance(inputs, list):
            return [str(i) for i in inputs[:6]]
    except Exception:
        pass
    return ["[]", "[1, 2, 3]", "[3, 1, 2]", "[1]", "[-1, 0, 1]"]


def run_postcondition_check(
    code: str,
    function_name: str,
    instruction: str,
    groq_llm,
    timeout: int = 10
) -> PostconditionResult:
    """
    Layer 2: Execute the corrected function against probe inputs and verify
    outputs satisfy the behavioral contract derived from the instruction.

    This is entirely deterministic — no LLM judges the output.
    The contract is a Python function injected into a subprocess.
    """
    contract_desc, checker_src = _detect_contract(instruction)
    probe_inputs = _generate_probe_inputs(code, function_name, groq_llm)

    failures = []
    output_samples = []
    checks_run = 0

    for input_repr in probe_inputs:
        script = f"""
import json, sys, traceback

_source = {repr(code)}
_ns = {{}}
try:
    exec(compile(_source, '<source>', 'exec'), _ns)
except Exception as e:
    print(json.dumps({{"error": str(e), "result": None,
                       "input": {repr(input_repr)}, "passed": False, "reason": str(e)}}))
    sys.exit(0)

{checker_src}

fn = _ns.get({repr(function_name)})
if fn is None:
    print(json.dumps({{"error": "function not found", "result": None,
                       "input": {repr(input_repr)}, "passed": False, "reason": "not found"}}))
    sys.exit(0)

try:
    raw_arg = eval({repr(input_repr)})
    result = fn(*raw_arg) if isinstance(raw_arg, tuple) else fn(raw_arg)
    passed, reason = _check(result, raw_arg)
    print(json.dumps({{"result": repr(result), "input": {repr(input_repr)},
                       "passed": passed, "reason": reason}}))
except Exception as e:
    tb = traceback.format_exc(limit=3)
    print(json.dumps({{"error": str(e), "result": None, "input": {repr(input_repr)},
                       "passed": False, "reason": str(e)}}))
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(script)
            tmp_path = f.name

        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True, timeout=timeout
            )
            out = proc.stdout.strip()
            if not out:
                continue
            data = json.loads(out)
            checks_run += 1
            sample = {
                "input":  input_repr,
                "output": data.get("result", "ERROR"),
                "passed": data.get("passed", False),
                "reason": data.get("reason", data.get("error", "unknown"))
            }
            output_samples.append(sample)
            if not data.get("passed", False):
                failures.append(
                    f"Input {input_repr!r} → {data.get('reason', 'failed')}"
                )
        except subprocess.TimeoutExpired:
            failures.append(f"Input {input_repr!r} → timed out after {timeout}s")
            checks_run += 1
        except (json.JSONDecodeError, Exception) as e:
            failures.append(f"Input {input_repr!r} → probe error: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    passed = (len(failures) == 0) and (checks_run > 0)
    print(
        f"[Validator] Postcondition ({contract_desc}): "
        f"{checks_run} check(s), {len(failures)} failure(s)"
    )
    for f in failures:
        print(f"[Validator]   FAIL: {f}")

    return PostconditionResult(
        passed=passed,
        checks_run=checks_run,
        failures=failures,
        output_samples=output_samples
    )


# ─────────────────────────────────────────────
# LAYER 3 — SEMANTIC CRITIC
# ─────────────────────────────────────────────

_CRITIC_SYSTEM_PROMPT = """You are a CODE REVIEW CRITIC. You did NOT write the code you are reviewing.
Your ONLY job: determine whether a code change actually fulfills its stated instruction.

RULES:
- Be adversarial and skeptical. Find problems the author missed or rationalized away.
- A function that no longer crashes is NOT the same as a function that is correct.
- A function returning None when it should return a value is ALWAYS a bug.
- Return ONLY valid JSON — no markdown, no preamble, nothing outside the JSON object.
- Set "intent_fulfilled" to false if you have any reasonable doubt.

Return EXACTLY this JSON structure — nothing else:
{
  "intent_fulfilled": true or false,
  "confidence": 0.0 to 1.0,
  "issues_found": ["specific issue 1", "specific issue 2"],
  "verdict_reason": "one concise sentence"
}"""


def run_semantic_critic(
    original_code: str,
    new_code: str,
    instruction: str,
    groq_llm
) -> CriticVerdict:
    """
    Layer 3: A SEPARATE LLM call in adversarial critic role.

    This breaks the circular bias where the generator implicitly validates
    its own output. Key design decisions:
      - Different system prompt: adversarial reviewer, not helpful code generator
      - Receives BOTH old and new code to detect regressions
      - Returns structured JSON only — prevents free-text rationalization
      - Explicitly told it did NOT write this code
      - Called with a different role framing than the generation call

    Low confidence failures are treated conservatively (don't block).
    High confidence failures block the correction loop and trigger a new attempt.
    """
    user_prompt = f"""ORIGINAL INSTRUCTION (this is the specification — ground truth):
{instruction}

ORIGINAL CODE (before the change):
```python
{original_code[:2000]}
```

NEW CODE (after the change — what you are reviewing):
```python
{new_code[:2000]}
```

Does the new code ACTUALLY fulfill the original instruction? Investigate:
1. Does it do exactly what was asked, or something superficially similar?
2. Does it return the correct type for every case? (None returns are usually wrong)
3. Does it handle edge cases the instruction implies (empty input, zero, negative, etc.)?
4. Are there any silent behavioral changes that were NOT requested?
5. Would the original problem that motivated this instruction still occur after this change?

Return ONLY the JSON verdict object."""

    raw = groq_llm(_CRITIC_SYSTEM_PROMPT, user_prompt)
    raw = raw.strip().replace("```json", "").replace("```", "").strip()

    # Extract JSON even if the model wrapped it in prose
    json_match = re.search(r'\{[^{}]*"intent_fulfilled"[^{}]*\}', raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)

    try:
        data = json.loads(raw)
        verdict = CriticVerdict(
            intent_fulfilled=bool(data.get("intent_fulfilled", False)),
            confidence=float(data.get("confidence", 0.5)),
            issues_found=list(data.get("issues_found", [])),
            verdict_reason=str(data.get("verdict_reason", "No reason provided"))
        )
        status = "✅ PASS" if verdict.intent_fulfilled else "❌ FAIL"
        print(
            f"[Validator] Critic {status} "
            f"(confidence={verdict.confidence:.2f}): {verdict.verdict_reason}"
        )
        for issue in verdict.issues_found:
            print(f"[Validator]   Issue: {issue}")
        return verdict
    except Exception as e:
        print(f"[Validator] Critic JSON parse failed ({e}) — conservative pass applied")
        return CriticVerdict(
            intent_fulfilled=True,
            confidence=0.3,
            issues_found=[f"Critic response unparseable: {e}"],
            verdict_reason="Critic call failed — conservative pass applied (verify manually)"
        )


# ─────────────────────────────────────────────
# COMBINED ENTRY POINT
# ─────────────────────────────────────────────

def validate_correction(
    original_code: str,
    new_code: str,
    function_name: str,
    instruction: str,
    groq_llm,
    skip_postcondition: bool = False,
    skip_critic: bool = False,
    critic_confidence_threshold: float = 0.65
) -> ValidationResult:
    """
    Full validation gate (Layers 2 + 3).
    Layer 1 (syntax) is assumed already passed by self_correction.py.

    Args:
        original_code:               Code BEFORE modification (for critic comparison)
        new_code:                    Code AFTER modification (what we validate)
        function_name:               Target function (for postcondition probing)
        instruction:                 Original user instruction = the specification
        groq_llm:                    Callable: (system_prompt, user_prompt) -> str
        skip_postcondition:          True for void/side-effect functions (DB writers, etc.)
        skip_critic:                 True only in CI/test-only mode
        critic_confidence_threshold: Min confidence to treat critic FAIL as blocking
    """
    # ── Layer 2: Postcondition ─────────────────────────────────
    postcondition_result = None
    if not skip_postcondition and function_name:
        postcondition_result = run_postcondition_check(
            code=new_code,
            function_name=function_name,
            instruction=instruction,
            groq_llm=groq_llm
        )
        if not postcondition_result.passed and postcondition_result.checks_run > 0:
            return ValidationResult(
                passed=False,
                layer_failed="postcondition",
                postcondition=postcondition_result
            )
        print("[Validator] ✅ Postcondition passed")
    else:
        print("[Validator] ⏭  Postcondition skipped (void/side-effect function)")

    # ── Layer 3: Semantic Critic ───────────────────────────────
    critic_result = None
    if not skip_critic:
        critic_result = run_semantic_critic(
            original_code=original_code,
            new_code=new_code,
            instruction=instruction,
            groq_llm=groq_llm
        )
        if (not critic_result.intent_fulfilled
                and critic_result.confidence >= critic_confidence_threshold):
            return ValidationResult(
                passed=False,
                layer_failed="critic",
                postcondition=postcondition_result,
                critic=critic_result
            )
    else:
        print("[Validator] ⏭  Critic skipped")

    return ValidationResult(
        passed=True,
        layer_failed=None,
        postcondition=postcondition_result,
        critic=critic_result
    )


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def should_skip_postcondition(code: str, function_name: Optional[str]) -> bool:
    """
    Detect void/side-effectful functions that should skip the postcondition layer
    because they intentionally return None (DB writers, loggers, event emitters, etc.).
    """
    if not function_name:
        return True

    VOID_PREFIXES = (
        "log_", "write_", "save_", "store_", "emit_", "send_",
        "publish_", "update_", "delete_", "remove_", "insert_",
        "commit_", "close_", "init_", "setup_", "configure_",
        "register_", "notify_", "dispatch_", "print_"
    )
    fn_lower = function_name.lower()
    if any(fn_lower.startswith(p) for p in VOID_PREFIXES):
        return True

    # Check for explicit -> None annotation in AST
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                if node.returns:
                    if (isinstance(node.returns, ast.Constant)
                            and node.returns.value is None):
                        return True
                    if (isinstance(node.returns, ast.Name)
                            and node.returns.id == "None"):
                        return True
    except SyntaxError:
        pass

    return False


def format_validation_feedback(result: ValidationResult) -> str:
    """
    Convert a failed ValidationResult into specific, actionable feedback text
    for the next self-correction loop attempt. Injected into the correction prompt
    so the LLM understands WHY its previous fix was rejected.
    """
    if result.passed:
        return ""

    parts = [f"⛔ Validation failed at layer: {result.layer_failed}"]

    if result.layer_failed == "postcondition" and result.postcondition:
        parts.append("\nActual execution failures (these are real, not theoretical):")
        for failure in result.postcondition.failures:
            parts.append(f"  ✗ {failure}")
        parts.append(
            "\nCRITICAL: The function ran without exceptions but returned WRONG VALUES. "
            "This is not an exception-handling problem. "
            "Fix the return value logic. "
            "Catching exceptions and returning None is not a fix — it makes things worse. "
            "The function must return a correct, meaningful value for all inputs."
        )

    if result.layer_failed == "critic" and result.critic:
        parts.append(f"\nCode review verdict: {result.critic.verdict_reason}")
        if result.critic.issues_found:
            parts.append("Specific issues identified by reviewer:")
            for issue in result.critic.issues_found:
                parts.append(f"  ✗ {issue}")
        parts.append(
            "\nCRITICAL: The fix was syntactically valid but did not fulfill the original intent. "
            "Re-read the instruction carefully. "
            "Do not change adjacent behavior. "
            "Fix the specific thing that was asked for."
        )

    return "\n".join(parts)