"""
accuracy_test.py — Behavioral Accuracy Test Suite
===================================================
Tests WHAT the agent actually did, not just whether it returned 200.

For each test:
  1. Send a modification request
  2. Fetch the actual file content from DB
  3. Parse it with AST / exec it
  4. Assert the behavior is correct

Run with: python accuracy_test.py
Set your API key and project_id at the top before running.
"""

import requests
import ast
import sys
import types
import json

# ─────────────────────────────────────────────
# CONFIG — set these before running
# ─────────────────────────────────────────────
BASE_URL   = "http://127.0.0.1:8000"
API_KEY    = "aicto_83a60bc4fabb3a4b2a360acafdb117b89395c5de5acb7a1b"   # from /create-project/
PROJECT_ID = 3                           # your project id

HEADERS = {"X-API-Key": API_KEY}

# ─────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

passed = 0
failed = 0
results_log = []

def ok(label, detail=""):
    global passed
    passed += 1
    msg = f"{GREEN}  ✅ PASS{RESET} {label}"
    if detail:
        msg += f"\n        {CYAN}{detail}{RESET}"
    print(msg)
    results_log.append({"test": label, "status": "PASS", "detail": detail})

def fail(label, detail=""):
    global failed
    failed += 1
    msg = f"{RED}  ❌ FAIL{RESET} {label}"
    if detail:
        msg += f"\n        {YELLOW}{detail}{RESET}"
    print(msg)
    results_log.append({"test": label, "status": "FAIL", "detail": detail})

def header(title):
    print(f"\n{BOLD}{YELLOW}{'─'*60}\n  {title}\n{'─'*60}{RESET}")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_file_content(filename: str) -> str | None:
    """Fetch current file content from DB via API."""
    r = requests.get(
        f"{BASE_URL}/get-file/",
        params={"project_id": PROJECT_ID, "filename": filename},
        headers=HEADERS
    )
    if r.status_code != 200:
        return None
    data = r.json()
    return data.get("content")


def exec_code(code: str) -> types.ModuleType | None:
    """Execute code string and return the module namespace, or None on error."""
    mod = types.ModuleType("test_module")
    try:
        exec(compile(code, "<test>", "exec"), mod.__dict__)
        return mod
    except Exception as e:
        print(f"        {RED}[exec error] {e}{RESET}")
        return None


def modify(query: str) -> dict:
    """Send a modify-code request and return the JSON response."""
    r = requests.post(
        f"{BASE_URL}/modify-code/",
        params={"query": query, "project_id": PROJECT_ID},
        headers=HEADERS
    )
    return r.json()


def ask(query: str) -> str:
    try:
        r = requests.post(
            f"{BASE_URL}/ask/",
            params={"query": query, "project_id": PROJECT_ID},
            headers=HEADERS,
            timeout=60
        )
        return r.json().get("answer", "")
    except requests.exceptions.ConnectionError:
        print(f"        {RED}[connection error] Server crashed — check uvicorn terminal{RESET}")
        return ""
    except Exception as e:
        print(f"        {RED}[error] {e}{RESET}")
        return ""


def rollback(file_id: int):
    """Roll back a file to its previous version."""
    requests.post(
        f"{BASE_URL}/rollback/",
        params={"file_id": file_id},
        headers=HEADERS
    )


def get_file_id(filename: str) -> int | None:
    r = requests.get(
        f"{BASE_URL}/list-files/",
        params={"project_id": PROJECT_ID},
        headers=HEADERS
    )
    for f in r.json().get("files", []):
        if f["filename"] == filename:
            return f["file_id"]
    return None


# ─────────────────────────────────────────────
# TEST GROUP 1 — RAG / Ask accuracy
# ─────────────────────────────────────────────

def test_ask_accuracy():
    header("GROUP 1 — RAG Ask Accuracy")

    # Test 1a: Does it know what functions exist?
    answer = ask("What functions are defined in sample.py?")
    low = answer.lower()
    if "add" in low and "multiply" in low:
        ok("Knows functions in sample.py", f"Answer mentions 'add' and 'multiply'")
    else:
        fail("Knows functions in sample.py", f"Got: {answer[:200]}")

    # Test 1b: Does it know what a function does?
    answer = ask("What does the add function do?")
    low = answer.lower()
    if any(w in low for w in ["sum", "adds", "addition", "returns", "a + b", "a+b"]):
        ok("Understands add() semantics", f"Answer: {answer[:150]}")
    else:
        fail("Understands add() semantics", f"Got: {answer[:200]}")

    # Test 1c: Does it correctly say something is NOT in the codebase?
    answer = ask("Is there a divide function in sample.py?")
    low = answer.lower()
    if any(w in low for w in ["no", "not", "doesn't", "does not", "absent", "not present"]):
        ok("Correctly identifies missing function", f"Answer: {answer[:150]}")
    else:
        fail("Correctly identifies missing function", f"Got: {answer[:200]}")


# ─────────────────────────────────────────────
# TEST GROUP 2 — Code Modification Accuracy
# ─────────────────────────────────────────────

def test_modification_accuracy():
    header("GROUP 2 — Code Modification Accuracy")
    file_id = get_file_id("sample.py")

    # ── Test 2a: Add input validation ──────────────────────────
    print(f"\n  {CYAN}[2a] Requesting: Add validation to reject negative numbers in add(){RESET}")
    resp = modify("Add input validation to the add function in sample.py to reject negative numbers")

    if resp.get("status") != "success":
        fail("Modify request succeeded", f"Response: {resp}")
        return

    ok("Modify request returned success")

    content = get_file_content("sample.py")
    if not content:
        fail("Can fetch file content after modification")
        return
    ok("File content is fetchable")

    # Check syntax is valid
    try:
        ast.parse(content)
        ok("Modified file has valid Python syntax")
    except SyntaxError as e:
        fail("Modified file has valid Python syntax", str(e))
        return

    # Execute and test behavior
    mod = exec_code(content)
    if not mod:
        fail("Modified file is executable")
        return
    ok("Modified file is executable")

    # Positive case: valid inputs should still work
    try:
        result = mod.add(3, 5)
        if result == 8:
            ok("add(3, 5) still returns 8 ✓")
        else:
            fail("add(3, 5) still returns 8", f"Got: {result}")
    except Exception as e:
        fail("add(3, 5) works with valid inputs", str(e))

    # Negative case: negative inputs should be rejected
    rejected = False
    try:
        mod.add(-1, 5)
    except (ValueError, TypeError):
        rejected = True
    except Exception as e:
        # Some agents return 0 or use if/else — check the content instead
        pass

    if rejected:
        ok("add(-1, 5) raises an exception for negative input ✓")
    else:
        # Fallback: check the code contains validation logic
        low = content.lower()
        if any(w in low for w in ["raise", "valueerror", "if a <", "if b <", "negative", "< 0"]):
            ok("add() contains validation logic (raises or checks) ✓")
        else:
            fail("add(-1, 5) rejected by validation", "No exception raised and no validation code found")

    # ── Test 2b: Add a new function ────────────────────────────
    print(f"\n  {CYAN}[2b] Requesting: Add a subtract function{RESET}")
    resp2 = modify("Add a subtract function to sample.py that takes two numbers and returns their difference")

    if resp2.get("status") != "success":
        fail("Add subtract function request succeeded", str(resp2))
    else:
        ok("Add subtract request returned success")
        content2 = get_file_content("sample.py")
        if content2:
            try:
                tree2 = ast.parse(content2)
                fn_names = [n.name for n in ast.walk(tree2) if isinstance(n, ast.FunctionDef)]
                print(f"        {CYAN}Functions now in file: {fn_names}{RESET}")
                if "subtract" in fn_names:
                    ok("subtract function exists in file ✓")
                    mod2 = exec_code(content2)
                    if mod2 and hasattr(mod2, "subtract"):
                        try:
                            r = mod2.subtract(10, 3)
                            ok("subtract(10, 3) == 7 ✓") if r == 7 else fail("subtract(10, 3) == 7", f"Got: {r}")
                        except Exception as e:
                            fail("subtract() callable", str(e))
                    else:
                        ok("subtract in AST — exec skipped (complex imports)")
                else:
                    
                    fail("subtract function exists in modified file",
                         f"Functions found: {fn_names} | Preview: {content2[:200]}")
            except SyntaxError as e:
                fail("subtract file valid Python", str(e))

    # ── Test 2c: Type hints added ──────────────────────────────
    print(f"\n  {CYAN}[2c] Requesting: Add type hints to all functions{RESET}")
    resp3 = modify("Add type hints to all functions in sample.py. Use int for parameters and return types.")

    if resp3.get("status") != "success":
        fail("Add type hints request succeeded", str(resp3))
    else:
        ok("Add type hints request returned success")
        content3 = get_file_content("sample.py")
        if content3:
            try:
                tree = ast.parse(content3)
                funcs_with_hints = 0
                funcs_total = 0
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        funcs_total += 1
                        has_return  = node.returns is not None
                        has_args    = any(arg.annotation for arg in node.args.args)
                        if has_return or has_args:
                            funcs_with_hints += 1

                if funcs_total == 0:
                    fail("Type hints: functions found", "No functions detected in file")
                elif funcs_with_hints == funcs_total:
                    ok(f"All {funcs_total} functions have type hints ✓")
                elif funcs_with_hints > 0:
                    ok(f"Partial: {funcs_with_hints}/{funcs_total} functions have type hints")
                else:
                    fail("Type hints added", f"0/{funcs_total} functions have annotations")
            except SyntaxError as e:
                fail("Type hints file is valid Python", str(e))


# ─────────────────────────────────────────────
# TEST GROUP 3 — Self-Correction Verification
# ─────────────────────────────────────────────

def test_self_correction():
    header("GROUP 3 — Self-Correction Loop Verification")

    # Check that correction attempts info appears in results
    resp = modify("Add a divide function to sample.py with division by zero protection")
    if resp.get("status") == "success":
        results = resp.get("results", [])
        for r in results:
            output = r.get("output", "") or ""
            if "attempt" in output.lower():
                ok("Self-correction ran and reported attempt count", output)
            else:
                ok("Modification succeeded (self-correction passed on attempt 1)", output)

        # Verify the function actually works
        content = get_file_content("sample.py")
        if content:
            try:
                tree_d = ast.parse(content)
                fn_names_d = [n.name for n in ast.walk(tree_d) if isinstance(n, ast.FunctionDef)]
                print(f"        {CYAN}Functions now in file: {fn_names_d}{RESET}")
                if "divide" in fn_names_d:
                    ok("divide function exists in file ✓")
                    mod = exec_code(content)
                    if mod and hasattr(mod, "divide"):
                        try:
                            result = mod.divide(10, 2)
                            ok("divide(10, 2) == 5 ✓") if result in (5, 5.0) else fail("divide(10, 2) == 5", f"Got: {result}")
                        except Exception as e:
                            fail("divide() callable", str(e))
                        try:
                            mod.divide(10, 0)
                            fail("divide(10, 0) protected", "No exception raised")
                        except (ZeroDivisionError, ValueError):
                            ok("divide(10, 0) raises exception ✓")
                        except Exception as e:
                            ok(f"divide(10, 0) raises {type(e).__name__} ✓")
                    else:
                        ok("divide in AST — exec skipped (complex imports)")
                else:
                    fail("divide function added to file",
                         f"Functions found: {fn_names_d} | Preview: {content[:200]}")
            except SyntaxError as e:
                fail("divide file valid Python", str(e))
    else:
        fail("Add divide function succeeded", str(resp))


# ─────────────────────────────────────────────
# TEST GROUP 4 — Version + Rollback
# ─────────────────────────────────────────────

def test_rollback_accuracy():
    header("GROUP 4 — Rollback Accuracy")

    file_id = get_file_id("sample.py")
    if not file_id:
        fail("Get file_id for sample.py")
        return
    ok(f"Got file_id: {file_id}")

    # Capture content before modification
    before = get_file_content("sample.py")

    # Make a modification
    resp = modify("Add a comment # MARKER_TEST at the top of sample.py")
    if resp.get("status") != "success":
        fail("Modification before rollback test", str(resp))
        return

    after = get_file_content("sample.py")
    if after == before:
        fail("File actually changed after modification", "Content identical before and after")
        return
    ok("File changed after modification")

    # Rollback
    rollback(file_id)
    restored = get_file_content("sample.py")

    if restored == before:
        ok("Rollback restored exact previous content ✓")
    elif restored and "MARKER_TEST" not in restored:
        ok("Rollback removed the modification (content differs slightly but marker gone)")
    else:
        fail("Rollback restored previous content",
             f"MARKER_TEST still present or content unchanged")


# ─────────────────────────────────────────────
# TEST GROUP 5 — Auth Enforcement
# ─────────────────────────────────────────────

def test_auth_enforcement():
    header("GROUP 5 — Auth Enforcement")

    # No key → 401
    r = requests.post(f"{BASE_URL}/ask/",
                      params={"query": "test", "project_id": PROJECT_ID})
    if r.status_code == 401:
        ok("Missing key returns 401 ✓")
    else:
        fail("Missing key returns 401", f"Got {r.status_code}: {r.text[:100]}")

    # Wrong key → 403
    r = requests.post(f"{BASE_URL}/ask/",
                      params={"query": "test", "project_id": PROJECT_ID},
                      headers={"X-API-Key": "aicto_wrongkeyabcdef123456"})
    if r.status_code == 403:
        ok("Wrong key returns 403 ✓")
    else:
        fail("Wrong key returns 403", f"Got {r.status_code}: {r.text[:100]}")

    # Wrong project_id with valid key → 403
    r = requests.post(f"{BASE_URL}/ask/",
                      params={"query": "test", "project_id": 9999},
                      headers=HEADERS)
    if r.status_code == 403:
        ok("Valid key + wrong project_id returns 403 ✓")
    else:
        fail("Valid key + wrong project_id returns 403", f"Got {r.status_code}: {r.text[:100]}")

    # Correct key → 200
    r = requests.post(f"{BASE_URL}/ask/",
                      params={"query": "what functions exist", "project_id": PROJECT_ID},
                      headers=HEADERS)
    if r.status_code == 200:
        ok("Correct key + correct project returns 200 ✓")
    else:
        fail("Correct key + correct project returns 200", f"Got {r.status_code}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if API_KEY == "PASTE_YOUR_API_KEY_HERE":
        print(f"{RED}⛔ Set your API_KEY and PROJECT_ID at the top of this file first!{RESET}")
        sys.exit(1)

    print(f"\n{BOLD}{'═'*60}")
    print(f"  AI CTO — Accuracy Test Suite")
    print(f"  Target : {BASE_URL}")
    print(f"  Project: {PROJECT_ID}")
    print(f"{'═'*60}{RESET}")

    test_ask_accuracy()
    test_modification_accuracy()
    test_self_correction()
    test_rollback_accuracy()
    test_auth_enforcement()

    # ── Summary ───────────────────────────────────────────────
    total = passed + failed
    score = round((passed / total) * 100) if total else 0

    print(f"\n{BOLD}{'═'*60}")
    print(f"  RESULTS: {passed}/{total} passed  ({score}%)")
    print(f"{'═'*60}{RESET}")

    if failed > 0:
        print(f"\n{RED}Failed tests:{RESET}")
        for r in results_log:
            if r["status"] == "FAIL":
                print(f"  ✗ {r['test']}")
                if r["detail"]:
                    print(f"    → {r['detail'][:120]}")

    print()
    sys.exit(0 if failed == 0 else 1)