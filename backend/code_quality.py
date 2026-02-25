"""
code_quality.py — Code Quality Analyzer
=========================================
Drop next to main.py.

Analyzes Python files and produces a structured quality report covering:
  - Duplicate functions (same name or same body)
  - Unused imports
  - Missing docstrings
  - Naming convention violations (snake_case)
  - Long functions (complexity proxy)
  - Overall quality score 0-100

Integration: see INTEGRATION GUIDE at the bottom.
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class QualityIssue:
    category:  str          # "duplicate", "unused_import", "naming", "docstring", "complexity"
    severity:  str          # "error", "warning", "info"
    line:      Optional[int]
    message:   str
    fix_hint:  str = ""


@dataclass
class QualityReport:
    filename:       str
    score:          int               # 0-100
    grade:          str               # A / B / C / D / F
    issues:         list[QualityIssue] = field(default_factory=list)
    functions:      list[str]          = field(default_factory=list)
    classes:        list[str]          = field(default_factory=list)
    imports:        list[str]          = field(default_factory=list)
    summary:        str = ""

    def to_dict(self) -> dict:
        return {
            "filename":  self.filename,
            "score":     self.score,
            "grade":     self.grade,
            "summary":   self.summary,
            "functions": self.functions,
            "classes":   self.classes,
            "issues": [
                {
                    "category": i.category,
                    "severity": i.severity,
                    "line":     i.line,
                    "message":  i.message,
                    "fix_hint": i.fix_hint
                }
                for i in self.issues
            ]
        }


# ─────────────────────────────────────────────
# INDIVIDUAL CHECKS
# ─────────────────────────────────────────────

def _check_duplicates(tree: ast.Module, issues: list[QualityIssue]):
    """Detect functions/methods with identical names (shadowing) or identical bodies."""
    seen_names:  dict[str, int] = {}   # name → first line
    seen_bodies: dict[str, str] = {}   # body_hash → name

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name

            # Duplicate name
            if name in seen_names and name not in ("__init__", "__str__", "__repr__"):
                issues.append(QualityIssue(
                    category="duplicate",
                    severity="error",
                    line=node.lineno,
                    message=f"Function '{name}' defined again at line {node.lineno} (first at line {seen_names[name]})",
                    fix_hint=f"Remove or rename the duplicate '{name}' function"
                ))
            else:
                seen_names[name] = node.lineno

            # Duplicate body (same logic, different name)
            try:
                body_src = ast.dump(ast.Module(body=node.body, type_ignores=[]))
                body_key = re.sub(r'\s+', ' ', body_src)
                if body_key in seen_bodies and len(node.body) > 1:
                    issues.append(QualityIssue(
                        category="duplicate",
                        severity="warning",
                        line=node.lineno,
                        message=f"'{name}' has identical logic to '{seen_bodies[body_key]}' — possible duplicate",
                        fix_hint=f"Consider merging '{name}' and '{seen_bodies[body_key]}' into one function"
                    ))
                else:
                    seen_bodies[body_key] = name
            except Exception:
                pass


def _check_unused_imports(tree: ast.Module, code: str, issues: list[QualityIssue]) -> list[str]:
    """Find imports that are never referenced in the file body."""
    imported_names: dict[str, int] = {}  # name → line

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".")[0]
                imported_names[local_name] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    local_name = alias.asname or alias.name
                    imported_names[local_name] = node.lineno

    # Check which names are actually used
    used_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used_names.add(node.id)
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                used_names.add(node.value.id)

    all_imports = list(imported_names.keys())
    for name, line in imported_names.items():
        if name not in used_names:
            issues.append(QualityIssue(
                category="unused_import",
                severity="warning",
                line=line,
                message=f"Import '{name}' is never used",
                fix_hint=f"Remove 'import {name}' at line {line}"
            ))

    return all_imports


def _check_naming(tree: ast.Module, issues: list[QualityIssue]):
    """Check snake_case for functions/variables, PascalCase for classes."""
    snake_re  = re.compile(r'^[a-z_][a-z0-9_]*$')
    pascal_re = re.compile(r'^[A-Z][a-zA-Z0-9]*$')

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip dunder methods
            if not (node.name.startswith("__") and node.name.endswith("__")):
                if not snake_re.match(node.name):
                    issues.append(QualityIssue(
                        category="naming",
                        severity="warning",
                        line=node.lineno,
                        message=f"Function '{node.name}' should use snake_case",
                        fix_hint=f"Rename to '{_to_snake(node.name)}'"
                    ))

        elif isinstance(node, ast.ClassDef):
            if not pascal_re.match(node.name):
                issues.append(QualityIssue(
                    category="naming",
                    severity="warning",
                    line=node.lineno,
                    message=f"Class '{node.name}' should use PascalCase",
                    fix_hint=f"Rename to '{_to_pascal(node.name)}'"
                ))


def _to_snake(name: str) -> str:
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    s = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', s)
    return s.lower()


def _to_pascal(name: str) -> str:
    return ''.join(word.capitalize() for word in name.split('_'))


def _check_docstrings(tree: ast.Module, issues: list[QualityIssue]):
    """Flag functions and classes without docstrings."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Skip short/private/dunder functions
            is_dunder  = node.name.startswith("__") and node.name.endswith("__")
            is_private = node.name.startswith("_") and not is_dunder
            body_len   = len(node.body)

            if is_dunder or (is_private and body_len <= 3):
                continue

            has_docstring = (
                body_len > 0
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            )

            if not has_docstring and body_len > 2:
                kind = "Class" if isinstance(node, ast.ClassDef) else "Function"
                issues.append(QualityIssue(
                    category="docstring",
                    severity="info",
                    line=node.lineno,
                    message=f"{kind} '{node.name}' has no docstring",
                    fix_hint=f'Add a docstring: """{node.name} — describe what it does."""'
                ))


def _check_complexity(tree: ast.Module, issues: list[QualityIssue]):
    """Flag functions that are too long or deeply nested."""
    MAX_LINES  = 50
    MAX_NESTING = 4

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn_len = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0

            if fn_len > MAX_LINES:
                issues.append(QualityIssue(
                    category="complexity",
                    severity="warning",
                    line=node.lineno,
                    message=f"Function '{node.name}' is {fn_len} lines long (max recommended: {MAX_LINES})",
                    fix_hint=f"Break '{node.name}' into smaller helper functions"
                ))

            # Check nesting depth
            max_depth = _max_nesting_depth(node)
            if max_depth > MAX_NESTING:
                issues.append(QualityIssue(
                    category="complexity",
                    severity="warning",
                    line=node.lineno,
                    message=f"Function '{node.name}' has nesting depth {max_depth} (max recommended: {MAX_NESTING})",
                    fix_hint=f"Reduce nesting in '{node.name}' using early returns or helper functions"
                ))


def _max_nesting_depth(node, current=0) -> int:
    """Recursively calculate max nesting depth inside a function."""
    max_depth = current
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
            max_depth = max(max_depth, _max_nesting_depth(child, current + 1))
        else:
            max_depth = max(max_depth, _max_nesting_depth(child, current))
    return max_depth


# ─────────────────────────────────────────────
# SCORE CALCULATOR
# ─────────────────────────────────────────────

def _calculate_score(issues: list[QualityIssue], total_functions: int) -> tuple[int, str]:
    """
    Start at 100, deduct per issue.
    Errors: -15, Warnings: -5, Info: -1 (min score 0)
    """
    score = 100
    for issue in issues:
        if issue.severity == "error":
            score -= 15
        elif issue.severity == "warning":
            score -= 5
        elif issue.severity == "info":
            score -= 1

    score = max(0, score)

    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"

    return score, grade


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def analyze_quality(code: str, filename: str) -> QualityReport:
    """
    Run all quality checks on a Python file.
    Returns a QualityReport with score, grade, and detailed issues.
    """
    issues: list[QualityIssue] = []
    functions = []
    classes   = []

    # Parse
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return QualityReport(
            filename=filename,
            score=0,
            grade="F",
            issues=[QualityIssue(
                category="syntax",
                severity="error",
                line=e.lineno,
                message=f"Syntax error: {e.msg}",
                fix_hint="Fix the syntax error before quality analysis"
            )],
            summary=f"File has syntax errors — quality analysis skipped."
        )

    # Collect structure info
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)

    # Run all checks
    _check_duplicates(tree, issues)
    all_imports = _check_unused_imports(tree, code, issues)
    _check_naming(tree, issues)
    _check_docstrings(tree, issues)
    _check_complexity(tree, issues)

    # Score
    score, grade = _calculate_score(issues, len(functions))

    # Build summary
    errors   = sum(1 for i in issues if i.severity == "error")
    warnings = sum(1 for i in issues if i.severity == "warning")
    infos    = sum(1 for i in issues if i.severity == "info")

    if score >= 90:
        verdict = "Excellent — clean, well-structured code"
    elif score >= 80:
        verdict = "Good — minor improvements possible"
    elif score >= 70:
        verdict = "Fair — several issues to address"
    elif score >= 60:
        verdict = "Poor — significant quality issues"
    else:
        verdict = "Critical — major refactoring recommended"

    summary = (
        f"{verdict}. Score: {score}/100 (Grade {grade}). "
        f"{len(functions)} function(s), {len(classes)} class(es). "
        f"{errors} error(s), {warnings} warning(s), {infos} suggestion(s)."
    )

    print(f"[Quality] '{filename}': {score}/100 ({grade}) — {errors}E {warnings}W {infos}I")

    return QualityReport(
        filename=filename,
        score=score,
        grade=grade,
        issues=issues,
        functions=functions,
        classes=classes,
        imports=all_imports,
        summary=summary
    )


def analyze_project_quality(project_id: int, db) -> dict:
    """
    Run quality analysis on every file in a project.
    Returns per-file reports and an overall project score.
    """
    from models import File as DBFile
    from sqlalchemy.orm import Session

    files = db.query(DBFile).filter(DBFile.project_id == project_id).all()
    if not files:
        return {"error": "No files in project"}

    reports     = {}
    total_score = 0

    for f in files:
        if not f.filename.endswith(".py"):
            continue
        report = analyze_quality(f.content, f.filename)
        reports[f.filename] = report.to_dict()
        total_score += report.score

    avg_score = round(total_score / len(reports)) if reports else 0

    if avg_score >= 90:
        project_grade = "A"
    elif avg_score >= 80:
        project_grade = "B"
    elif avg_score >= 70:
        project_grade = "C"
    elif avg_score >= 60:
        project_grade = "D"
    else:
        project_grade = "F"

    return {
        "project_id":    project_id,
        "overall_score": avg_score,
        "overall_grade": project_grade,
        "files_analyzed": len(reports),
        "reports":       reports
    }


# ─────────────────────────────────────────────
# INTEGRATION GUIDE
# ─────────────────────────────────────────────
#
# ── STEP 1: Add to main.py imports ──
#
# from code_quality import analyze_quality, analyze_project_quality
#
#
# ── STEP 2: Add quality report to /modify-code/ response ──
#
# In modify_code(), after run_agent() returns:
#
#   result = run_agent(...)
#   if result.get("status") == "success":
#       # Run quality check on the modified file
#       file = db.query(DBFile).filter(
#           DBFile.project_id == project_id,
#           DBFile.filename == result["results"][0].get("output","").split("'")[1]
#       ).first() if result.get("results") else None
#       # Simpler: just analyze via the endpoint below
#
#
# ── STEP 3: Add quality endpoints ──
#
# from code_quality import analyze_quality, analyze_project_quality
#
# @app.get("/quality/")
# def get_quality(
#     project_id: int,
#     filename: str,
#     db: Session = Depends(get_db),
#     key_project_id: int = Depends(require_api_key)
# ):
#     verify_project_access(project_id, key_project_id)
#     file = db.query(DBFile).filter(
#         DBFile.project_id == project_id,
#         DBFile.filename == filename
#     ).first()
#     if not file:
#         return {"error": "File not found"}
#     return analyze_quality(file.content, file.filename).to_dict()
#
#
# @app.get("/quality/project/")
# def get_project_quality(
#     project_id: int,
#     db: Session = Depends(get_db),
#     key_project_id: int = Depends(require_api_key)
# ):
#     verify_project_access(project_id, key_project_id)
#     return analyze_project_quality(project_id, db)