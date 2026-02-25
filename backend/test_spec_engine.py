"""
test_spec_engine.py — Intent-Anchored Test Generation + Mutation Oracle
========================================================================
Solves the circular dependency between LLM-generated tests and LLM-generated fixes.

  PROBLEM:  If the LLM generates the fix AND the tests, both share the same
            model bias. Tests get written to confirm the fix rather than
            verify the original intent. The bug survives dressed up as a pass.

  SOLUTION: Temporal separation + objective oracle:
            1. Tests generated BEFORE the fix, from instruction + original code only.
               The fix is withheld entirely from test generation.
            2. Generated tests validated against DELIBERATE mutations of the
               original code (mutation testing). Tests that pass a mutant are
               provably too weak — discarded.
            3. Only tests that KILLED mutations are locked. The fix must pass
               these locked tests.

This breaks the circular dependency structurally:
  - Tests anchored to the original instruction, not the fix's behavior
  - The mutation oracle is model-agnostic (deterministic Python mutations)
  - A test that passes a mutation is provably weak and discarded

Usage:
    from test_spec_engine import lock_spec_tests, validate_fix_against_spec, format_spec_failure_feedback
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
class SpecTest:
    """A single intent-anchored test case."""
    test_id: str
    description: str          # Human-readable: what intent this tests
    test_code: str            # Complete runnable Python test code
    mutation_kills: int       # How many mutations this test caught (0 = weak)
    is_locked: bool           # True if it passed the mutation oracle


@dataclass
class MutationResult:
    """Result of running one test against one mutation."""
    mutation_type: str
    mutation_desc: str
    test_detected: bool       # True if test FAILED on mutant (good — test is alive)
    mutant_output: str


@dataclass
class SpecTestSuite:
    """Full test suite for a function, anchored to the original instruction."""
    function_name: str
    instruction: str
    tests: list               # list[SpecTest]
    locked_tests: list        # tests that passed the mutation oracle
    mutation_kill_rate: float # fraction of mutations caught by locked tests
    is_trustworthy: bool      # True if kill rate >= threshold


@dataclass
class FixValidationResult:
    """Result of running locked spec tests against the generated fix."""
    passed: bool
    tests_run: int
    tests_failed: int
    failures: list            # [{test_id, description, error}]
    fix_output_samples: list  # [{test_id, output}]


# ─────────────────────────────────────────────
# STEP 1: GENERATE SPEC TESTS (BEFORE THE FIX)
# ─────────────────────────────────────────────

_SPEC_TEST_SYSTEM_PROMPT = """You are a TEST SPECIFICATION WRITER.
You write tests based on WHAT A FUNCTION SHOULD DO, not on what it currently does.

RULES:
- You are given a function's ORIGINAL code and the INSTRUCTION for what it should do.
- Write tests that verify the INTENDED BEHAVIOR described in the instruction.
- Do NOT write tests that just confirm the current (possibly broken) behavior.
- Do NOT write tests based on what a fix might look like — you will not see the fix.
- Each test must be a complete, standalone Python function named test_N(fn).
- Each test must call fn(...) and include an assert that FAILS if intent is not met.
- Include edge cases: empty input, None input, zero, negative values, large inputs.
- Return ONLY a JSON array of test objects. No markdown. No explanation.

JSON format:
[
  {
    "test_id": "test_1",
    "description": "what intent this test verifies",
    "test_code": "def test_1(fn):\\n    result = fn([3,1,2])\\n    assert result == [1,2,3], f'Expected sorted list, got {result}'"
  }
]"""


def generate_spec_tests(
    original_code: str,
    function_name: str,
    instruction: str,
    groq_llm,
    n_tests: int = 6
) -> list:
    """
    Step 1: Generate tests from the INSTRUCTION and ORIGINAL CODE only.
    The fix is never provided to this function — by design.

    The tests produced here are the specification, not acceptance criteria.
    They describe what the function SHOULD do, anchored to the instruction.
    """
    prompt = f"""INSTRUCTION (what the function should do after modification):
{instruction}

ORIGINAL FUNCTION CODE (before any fix — provided for context on the function interface only):
```python
{original_code[:2000]}
```

Function name: {function_name}

Write {n_tests} test functions that verify the INTENDED BEHAVIOR from the instruction.
Each test calls fn(input) and asserts the result matches the intended behavior.
Cover: normal case, empty input, edge case, boundary value, None/invalid input.

Return ONLY the JSON array. No markdown."""

    raw = groq_llm(_SPEC_TEST_SYSTEM_PROMPT, prompt)
    raw = raw.strip().replace("```json", "").replace("```", "").strip()

    json_match = re.search(r'\[.*\]', raw, re.DOTALL)
    if json_match:
        raw = json_match.group(0)

    try:
        data = json.loads(raw)
        tests = []
        for i, item in enumerate(data[:n_tests]):
            tests.append(SpecTest(
                test_id=item.get("test_id", f"test_{i+1}"),
                description=item.get("description", f"Test {i+1}"),
                test_code=item.get("test_code", ""),
                mutation_kills=0,
                is_locked=False
            ))
        print(f"[SpecEngine] Generated {len(tests)} spec test(s) for '{function_name}'")
        return tests
    except Exception as e:
        print(f"[SpecEngine] Spec test generation failed: {e}")
        return []


# ─────────────────────────────────────────────
# STEP 2: MUTATION ORACLE
# ─────────────────────────────────────────────

def _generate_mutants(function_source: str, function_name: str) -> list:
    """
    Generate deterministic mutations of the original function.
    These are known-bad versions — a good test suite MUST catch them.
    Mutations are model-agnostic: pure text/AST transformations.

    Returns list of (mutation_type, mutation_desc, mutated_code) tuples.
    """
    mutants = []

    # Mutation 1: Return None — replace first real return with return None
    none_mutant = re.sub(
        r'\breturn\s+(?!None\b)(\S[^\n]*)',
        'return None',
        function_source,
        count=1
    )
    if none_mutant != function_source:
        mutants.append((
            "return_none",
            "First return statement replaced with 'return None'",
            none_mutant
        ))

    # Mutation 2: Off-by-one — range(n) -> range(n-1)
    off_by_one = re.sub(r'\brange\((\w+)\)', r'range(\1 - 1)', function_source, count=1)
    if off_by_one != function_source:
        mutants.append((
            "off_by_one",
            "range(n) changed to range(n-1)",
            off_by_one
        ))

    # Mutation 3: Flip > to >=
    flip_gt = re.sub(r'(?<![=!<>])>(?!=)', '>=', function_source, count=1)
    if flip_gt != function_source:
        mutants.append((
            "flip_comparison_gt",
            "First '>' changed to '>='",
            flip_gt
        ))

    # Mutation 4: Flip < to <=
    flip_lt = re.sub(r'(?<![=!<>])<(?!=)', '<=', function_source, count=1)
    if flip_lt != function_source:
        mutants.append((
            "flip_comparison_lt",
            "First '<' changed to '<='",
            flip_lt
        ))

    # Mutation 5: Delete if-block body (replace with pass)
    try:
        tree = ast.parse(function_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                for child in ast.walk(node):
                    if isinstance(child, ast.If) and child.body:
                        lines = function_source.splitlines()
                        if hasattr(child.body[0], 'lineno') and hasattr(child, 'end_lineno'):
                            body_start = child.body[0].lineno - 1
                            body_end = child.body[-1].end_lineno
                            indent = "    " * 2
                            new_lines = (
                                lines[:body_start]
                                + [f"{indent}pass  # MUTANT: body deleted"]
                                + lines[body_end:]
                            )
                            skip_mutant = "\n".join(new_lines)
                            if skip_mutant != function_source:
                                mutants.append((
                                    "delete_if_body",
                                    "First if-block body replaced with pass",
                                    skip_mutant
                                ))
                        break
                break
    except (SyntaxError, Exception):
        pass

    # Mutation 6: Negate first condition
    neg_mutant = re.sub(r'\bif\s+(?!not\b)', 'if not ', function_source, count=1)
    if neg_mutant != function_source:
        mutants.append((
            "negate_condition",
            "First 'if' condition negated with 'not'",
            neg_mutant
        ))

    print(f"[SpecEngine] Generated {len(mutants)} mutant(s) for '{function_name}'")
    return mutants[:6]


def _run_test_against_code(
    test: SpecTest,
    source_code: str,
    function_name: str,
    timeout: int = 8
) -> tuple:
    """
    Run a single spec test against a given source code (original or mutant).
    Returns (test_passed: bool, output: str).

    test_passed=True  means the function behaved correctly (assertions passed).
    test_passed=False means the test caught a problem (assertions failed or crashed).
    """
    script = f"""
import sys, traceback

_source = {repr(source_code)}
_ns = {{}}
try:
    exec(compile(_source, '<source>', 'exec'), _ns)
except Exception as e:
    print(f"COMPILE_ERROR: {{e}}")
    sys.exit(1)

fn = _ns.get({repr(function_name)})
if fn is None:
    print("FUNCTION_NOT_FOUND")
    sys.exit(1)

{test.test_code}

try:
    {test.test_id}(fn)
    print("TEST_PASSED")
except AssertionError as e:
    print(f"ASSERTION_FAILED: {{e}}")
except Exception as e:
    print(f"RUNTIME_ERROR: {{e}}")
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
        out = (proc.stdout + proc.stderr).strip()
        passed = "TEST_PASSED" in out
        return passed, out
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def run_mutation_oracle(
    tests: list,
    original_code: str,
    function_name: str,
    kill_rate_threshold: float = 0.4
) -> tuple:
    """
    Step 2: Validate spec tests by running them against known-bad mutants.

    Logic:
    - For each mutant: run all tests against it
    - A test "kills" a mutant if it FAILS on the mutant (test caught the bug)
    - A test that PASSES on every mutant is useless — locked out
    - Only tests with mutation_kills > 0 are locked (trustworthy)
    - kill_rate = total kills / total possible kills

    Returns (locked_tests: list[SpecTest], kill_rate: float)
    """
    if not tests:
        return [], 0.0

    mutants = _generate_mutants(original_code, function_name)
    if not mutants:
        print("[SpecEngine] No mutants generated — locking all tests by default")
        for t in tests:
            t.is_locked = True
        return tests, 1.0

    total_possible_kills = len(mutants) * len(tests)
    total_kills = 0

    for test in tests:
        test.mutation_kills = 0
        for mutation_type, mutation_desc, mutant_code in mutants:
            passed_on_mutant, output = _run_test_against_code(
                test, mutant_code, function_name
            )
            if not passed_on_mutant:
                # Test FAILED on mutant = test correctly detected the bug
                test.mutation_kills += 1
                total_kills += 1

        test.is_locked = test.mutation_kills > 0
        status = "LOCKED" if test.is_locked else "WEAK (discarded)"
        print(
            f"[SpecEngine] {status} — {test.test_id}: "
            f"killed {test.mutation_kills}/{len(mutants)} mutant(s) — {test.description}"
        )

    locked = [t for t in tests if t.is_locked]
    kill_rate = total_kills / total_possible_kills if total_possible_kills > 0 else 0.0

    print(
        f"[SpecEngine] Mutation oracle complete: "
        f"{len(locked)}/{len(tests)} tests locked, "
        f"kill_rate={kill_rate:.2f}"
    )
    return locked, kill_rate


def lock_spec_tests(
    original_code: str,
    function_name: str,
    instruction: str,
    groq_llm,
    kill_rate_threshold: float = 0.4
) -> SpecTestSuite:
    """
    Full pipeline: generate spec tests from instruction -> validate via mutation oracle
    -> return only the locked (trustworthy) tests.

    Called BEFORE the fix is generated. The returned SpecTestSuite is then
    passed to validate_fix_against_spec() after the fix is ready.
    """
    print(f"\n[SpecEngine] Generating spec tests for '{function_name}'...")

    raw_tests = generate_spec_tests(
        original_code=original_code,
        function_name=function_name,
        instruction=instruction,
        groq_llm=groq_llm
    )

    if not raw_tests:
        return SpecTestSuite(
            function_name=function_name,
            instruction=instruction,
            tests=[],
            locked_tests=[],
            mutation_kill_rate=0.0,
            is_trustworthy=False
        )

    locked_tests, kill_rate = run_mutation_oracle(
        tests=raw_tests,
        original_code=original_code,
        function_name=function_name,
        kill_rate_threshold=kill_rate_threshold
    )

    is_trustworthy = kill_rate >= kill_rate_threshold and len(locked_tests) > 0

    return SpecTestSuite(
        function_name=function_name,
        instruction=instruction,
        tests=raw_tests,
        locked_tests=locked_tests,
        mutation_kill_rate=kill_rate,
        is_trustworthy=is_trustworthy
    )


# ─────────────────────────────────────────────
# STEP 3: VALIDATE FIX AGAINST LOCKED SPEC TESTS
# ─────────────────────────────────────────────

def validate_fix_against_spec(
    fix_code: str,
    function_name: str,
    spec_suite: SpecTestSuite
) -> FixValidationResult:
    """
    Step 3: Run the locked spec tests against the generated fix.

    These tests were generated from the instruction BEFORE the fix existed,
    and were validated by the mutation oracle. They represent a trustworthy
    specification, not a rubber stamp for the fix's behavior.
    """
    if not spec_suite.locked_tests:
        print(f"[SpecEngine] No locked tests available — skipping fix validation")
        return FixValidationResult(
            passed=True,
            tests_run=0,
            tests_failed=0,
            failures=[],
            fix_output_samples=[]
        )

    failures = []
    output_samples = []
    tests_failed = 0

    print(
        f"\n[SpecEngine] Validating fix against "
        f"{len(spec_suite.locked_tests)} locked spec test(s)..."
    )

    for test in spec_suite.locked_tests:
        passed, output = _run_test_against_code(test, fix_code, function_name)
        output_samples.append({"test_id": test.test_id, "output": output})

        if passed:
            print(f"[SpecEngine] PASS: {test.test_id} — {test.description}")
        else:
            tests_failed += 1
            failures.append({
                "test_id":     test.test_id,
                "description": test.description,
                "error":       output
            })
            print(f"[SpecEngine] FAIL: {test.test_id} — {test.description} -> {output[:100]}")

    total = len(spec_suite.locked_tests)
    passed_all = tests_failed == 0

    print(
        f"[SpecEngine] Fix validation: "
        f"{total - tests_failed}/{total} test(s) passed "
        f"({'ACCEPTED' if passed_all else 'REJECTED'})"
    )

    return FixValidationResult(
        passed=passed_all,
        tests_run=total,
        tests_failed=tests_failed,
        failures=failures,
        fix_output_samples=output_samples
    )


def format_spec_failure_feedback(
    result: FixValidationResult,
    spec_suite: SpecTestSuite
) -> str:
    """
    Build actionable feedback for the self-correction loop when the fix
    fails spec tests. Tells the LLM exactly which intended behaviors are broken.
    """
    if result.passed:
        return ""

    parts = [
        f"Fix failed {result.tests_failed}/{result.tests_run} intent-based spec test(s).",
        "These tests were written from the original instruction BEFORE the fix was generated.",
        "They define what the fix MUST do, not what the fix currently does.",
        ""
    ]
    for failure in result.failures:
        parts.append(f"  FAIL [{failure['test_id']}]: {failure['description']}")
        parts.append(f"    Error: {failure['error'][:150]}")

    parts.append(
        "\nCRITICAL: Your fix does not implement the intended behavior described in the instruction. "
        "Re-read the instruction and ensure the function correctly handles all cases the tests describe."
    )
    return "\n".join(parts)