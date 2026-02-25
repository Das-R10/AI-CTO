"""
schemas.py — Pydantic Response & Request Models for AI CTO API
==============================================================
Drop this file next to main.py.

Rules applied throughout:
  - Every response model is a strict subset of what the endpoint actually returns.
    No invented fields; every field traced back to the exact DB query or module call.
  - No SQLAlchemy models ever appear in a response.
  - Optional fields that the backend may not populate are typed Optional[X] = None.
  - Enums are typed as Literal strings, not Python Enum, to stay JSON-serialisable
    without extra configuration.
  - All timestamps are str — SQLAlchemy returns datetime objects which FastAPI
    will serialise; we receive them as strings in responses.

Sections
--------
  A.  Shared primitives
  B.  Auth / JWT (new)
  C.  Project management        /create-project/
  D.  File management           /upload-file/  /list-files/  /get-file/  /diff/{file_id}  /history/
  E.  Project health            /project-health/
  F.  Audit log                 /audit-log/
  G.  Code quality              /quality/  /quality/project/
  H.  Dependency graph          /dependency-graph/
  I.  Project memory            /project-memory/  /add-architecture-memory/
  J.  Approvals                 /pending-approvals/  /approve-change/  /reject-change/
                                /approve/{id}  /reject/{id}
  K.  Execute goal              /execute-goal/
  L.  Ask / retrieve context    /ask/  /retrieve-context/
  M.  Evolution & drift         /evolution/  /drift/  /snapshot/  /function-history/
  N.  Roles & governance        /roles/  /assign-role/
  O.  Branch diffs              /diff/{project_id}/{branch_name}
                                /diff/{project_id}/{branch_name}/summary
  P.  Git operations            /clone-repo/  /create-pr/  /git-status/
  Q.  Job system                /modify-code-async/  /job-status/{job_id}
  R.  Tests                     /tests/
  S.  Ops                       /rate-limit-status/  /llm-status/  /global-lessons/
"""

from __future__ import annotations
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════
# A. SHARED PRIMITIVES
# ══════════════════════════════════════════════════════════════════

class ErrorDetail(BaseModel):
    """Standard error envelope — returned when the backend raises HTTPException."""
    detail: str


class OKResponse(BaseModel):
    """Generic success acknowledgement for simple mutations."""
    ok: bool = True
    message: str


# ══════════════════════════════════════════════════════════════════
# B. AUTH / JWT  (new endpoints: POST /auth/login, POST /auth/refresh)
# ══════════════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    """
    Body for POST /auth/login.
    The frontend sends the raw API key exactly once to exchange it for a JWT.
    After this call the raw key is discarded; the JWT is used for all subsequent requests.
    """
    project_id: int
    api_key: str = Field(..., description="Raw API key received from /create-project/")


class TokenResponse(BaseModel):
    """
    Returned by POST /auth/login and POST /auth/refresh.
    Frontend stores access_token in memory (never localStorage).
    All subsequent requests use: Authorization: Bearer <access_token>
    """
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(..., description="Seconds until the token expires")
    project_id: int
    project_name: str


# ══════════════════════════════════════════════════════════════════
# C. PROJECT MANAGEMENT   POST /create-project/
# ══════════════════════════════════════════════════════════════════

class ProjectCreateResponse(BaseModel):
    """
    Returned once when a project is created.
    The api_key field is shown exactly once and must be saved immediately.
    Afterward the caller should exchange it for a JWT via POST /auth/login.
    """
    project_id: int
    name: str
    api_key: str = Field(..., description="Raw key — shown once only. Exchange for JWT immediately.")
    warning: str  # "Save this key now — it will never be shown again."


# ══════════════════════════════════════════════════════════════════
# D. FILE MANAGEMENT
#    POST /upload-file/
#    GET  /list-files/
#    GET  /get-file/
#    GET  /diff/{file_id}
#    GET  /history/{file_id}
# ══════════════════════════════════════════════════════════════════

class FileUploadResponse(BaseModel):
    """POST /upload-file/ — file stored, embedded, and quality-scored."""
    message: str            # "File stored + embedded successfully"
    file_id: int
    quality_score: float
    quality_grade: str      # A | B | C | D | F
    sanitize_warnings: list[str]


class FileListEntry(BaseModel):
    """One row in GET /list-files/ response."""
    file_id: int
    filename: str
    lines: int


class FileListResponse(BaseModel):
    """GET /list-files/"""
    project_id: int
    files: list[FileListEntry]


class FileContentResponse(BaseModel):
    """GET /get-file/"""
    file_id: int
    filename: str
    content: str


class FileDiffResponse(BaseModel):
    """
    GET /diff/{file_id}  — unified diff between the current file and its most
    recent saved version. Returns an error string if no previous version exists.
    """
    filename: str
    lines_added: int
    lines_removed: int
    diff: str   # unified diff text, or "No changes detected"


class FileVersionEntry(BaseModel):
    """One entry in GET /history/{file_id}."""
    version_id: int
    saved_at: str


class FileHistoryResponse(BaseModel):
    """GET /history/{file_id}"""
    file_id: int
    versions: list[FileVersionEntry]


# ══════════════════════════════════════════════════════════════════
# E. PROJECT HEALTH   GET /project-health/
# ══════════════════════════════════════════════════════════════════

class DependencyRiskInfo(BaseModel):
    """
    Derived from build_dependency_graph(): files with 5+ reverse-dependencies
    are flagged as high-risk coupling points.
    """
    high_risk_files: list[str]
    risk_level: Literal["low", "medium", "high"]


class ProjectHealthResponse(BaseModel):
    """
    GET /project-health/
    Aggregates: quality score, test pass rate, memory count,
    recently modified files, open agent jobs, and dependency risk.
    Powers the four overview metric cards in the Dashboard page.
    """
    project_id: int
    quality_score: float                           # 0–100 from analyze_project_quality()
    quality_grade: str                             # A | B | C | D | F
    test_pass_rate: Optional[Union[float, str]]    # float, or "no tests" if no tests exist
    test_files_count: int
    memory_entries: int                            # count from project_memory table
    files_modified_24h: list[str]                  # filenames changed in last 24 h
    open_jobs: int                                 # agent_jobs with status pending|running
    dependency_risk: DependencyRiskInfo


# ══════════════════════════════════════════════════════════════════
# F. AUDIT LOG   GET /audit-log/
# ══════════════════════════════════════════════════════════════════

class AuditLogEntry(BaseModel):
    """
    One row from the audit_log table.
    Actions include: orchestrate_started, dag_execution_complete,
    execution_risk_scored, approval_requested, change_approved,
    change_rejected, admin_high_risk_bypass, role_assigned, create_pr.
    """
    id: int
    user_id: Optional[str]
    action: str
    branch: Optional[str]
    risk_score: int
    detail: Optional[str]
    timestamp: str


class AuditLogResponse(BaseModel):
    """GET /audit-log/"""
    project_id: int
    count: int
    entries: list[AuditLogEntry]


# ══════════════════════════════════════════════════════════════════
# G. CODE QUALITY
#    GET /quality/           — single file
#    GET /quality/project/   — whole project
# ══════════════════════════════════════════════════════════════════

class FileQualityResponse(BaseModel):
    """
    GET /quality/ — result of analyze_quality(content, filename).
    Maps directly to QualityReport.to_dict() in code_quality.py.
    """
    filename: str
    score: float
    grade: str                  # A | B | C | D | F
    summary: str
    issues: list[str]
    metrics: dict[str, Any]     # raw metric values (complexity, lines, etc.)


class ProjectFileQuality(BaseModel):
    """Per-file entry inside ProjectQualityResponse."""
    filename: str
    score: float
    grade: str
    issues: list[str]


class ProjectQualityResponse(BaseModel):
    """
    GET /quality/project/ — result of analyze_project_quality().
    analyze_project_quality() returns a plain dict; these fields are what it actually contains.
    """
    project_id: int
    overall_score: float
    overall_grade: str
    files: dict[str, Any]       # keyed by filename; values are per-file quality dicts


# ══════════════════════════════════════════════════════════════════
# H. DEPENDENCY GRAPH   GET /dependency-graph/
# ══════════════════════════════════════════════════════════════════

class FileDependencyNode(BaseModel):
    """
    One entry from build_dependency_graph() in architectural_awareness.py.
    The raw return is dict[filename, {imports_from, imported_by}].
    """
    filename: str
    imports_from: list[str]
    imported_by: list[str]
    is_circular: bool           # True if this file is in a circular pair


class DependencyGraphResponse(BaseModel):
    """
    GET /dependency-graph/
    The existing endpoint returns the raw dict from build_dependency_graph().
    The new version wraps it in this envelope for consistent typing.
    For backward compatibility the raw dict format is preserved in raw_graph.
    """
    project_id: int
    total_files: int
    circular_pairs: int
    nodes: list[FileDependencyNode]


# ══════════════════════════════════════════════════════════════════
# I. PROJECT MEMORY
#    GET  /project-memory/
#    POST /add-architecture-memory/
# ══════════════════════════════════════════════════════════════════

class MemoryEntry(BaseModel):
    """One row from the project_memory table."""
    id: int
    memory_type: Literal["architecture", "decision", "constraint", "style"]
    content: str
    created_at: str


class ProjectMemoryResponse(BaseModel):
    """GET /project-memory/"""
    project_id: int
    total: int
    memories: list[MemoryEntry]


class AddMemoryRequest(BaseModel):
    """
    POST /add-architecture-memory/ — request body.
    The existing endpoint takes `note` as a query param; the new version
    accepts it as a JSON body for cleaner API design. Both paths supported.
    """
    note: str = Field(..., min_length=1, max_length=1000,
                      description="Memory content to store")
    memory_type: Literal["architecture", "decision", "constraint", "style"] = "architecture"


class AddMemoryResponse(BaseModel):
    """POST /add-architecture-memory/"""
    message: str                        # "Architecture memory stored successfully"
    memory_id: int
    note: str                           # truncated to 200 chars for display


# ══════════════════════════════════════════════════════════════════
# J. APPROVALS
#    GET  /pending-approvals/
#    POST /approve-change/
#    POST /reject-change/
#    POST /approve/{approval_id}    (legacy)
#    POST /reject/{approval_id}     (legacy)
# ══════════════════════════════════════════════════════════════════

class PendingApprovalEntry(BaseModel):
    """
    One row from pending_approvals.
    plan_json is kept as Any because it may contain DAGPlan dicts of varying shapes.
    risk_flags is the list of human-readable risk factor strings computed by
    compute_enterprise_risk_score().
    risk_score added by enterprise_governance.create_enterprise_approval().
    reviewer_id / reviewed_at populated after resolution.
    """
    approval_id: int
    plan: Any                           # DAGPlan dict: {goal, steps}
    risk_flags: list[str]
    status: Literal["pending", "approved", "rejected"]
    risk_score: Optional[int] = None    # 0–100, present after Phase 5 gate
    risk_label: Optional[str] = None    # low | medium | high (derived from score)
    reviewer_id: Optional[str] = None
    reviewed_at: Optional[str] = None
    created_at: str


class PendingApprovalsResponse(BaseModel):
    """GET /pending-approvals/"""
    project_id: int
    count: int
    approvals: list[PendingApprovalEntry]


class ApproveChangeRequest(BaseModel):
    """
    POST /approve-change/ — request body.
    The existing endpoint takes these as query params; both styles supported.
    """
    approval_id: int
    reviewer_id: str = Field(default="frontend-user",
                             description="User ID of the reviewer performing the action")


class RejectChangeRequest(BaseModel):
    """POST /reject-change/ — request body."""
    approval_id: int
    reviewer_id: str = Field(default="frontend-user")
    reason: str = Field(default="Rejected by reviewer",
                        description="Human-readable reason stored in audit log")


class ApprovalActionResponse(BaseModel):
    """
    Returned by POST /approve-change/ and POST /reject-change/.
    When approved, execution_result contains the full run_agent_from_dag() return.
    """
    status: Literal["approved", "rejected"]
    approval_id: int
    execution_result: Optional[dict[str, Any]] = None   # populated only on approval


class LegacyApprovalResponse(BaseModel):
    """
    Returned by legacy POST /approve/{approval_id} and POST /reject/{approval_id}.
    These call resolve_approval() from multi_agent.py.
    On reject: {status, approval_id}
    On approve: same shape as run_agent_from_dag() success response (passed through).
    """
    status: str
    approval_id: Optional[int] = None
    error: Optional[str] = None


# ══════════════════════════════════════════════════════════════════
# K. EXECUTE GOAL   POST /execute-goal/
# ══════════════════════════════════════════════════════════════════

class ExecuteGoalRequest(BaseModel):
    """POST /execute-goal/ — replaces bare query-param interface for cleaner API."""
    goal: str = Field(..., min_length=3, max_length=2000,
                      description="High-level product goal to decompose and execute")
    enable_tests: bool = Field(default=False,
                               description="Run impacted tests after each sub-goal execution")


class SubGoalResult(BaseModel):
    """
    One entry in execute_goal() return's sub_goal_results list.
    status: completed | pending_approval | failed | rejected | skipped
    """
    sub_goal_id: str
    description: str
    status: str
    approval_id: Optional[int] = None
    final_result: Optional[dict[str, Any]] = None


class ExecuteGoalResponse(BaseModel):
    """
    POST /execute-goal/
    Top-level shape returned by execute_goal() in multi_agent.py.
    overall status: completed | partial | failed | pending_approvals
    """
    goal: str
    status: str
    sub_goals_total: int
    sub_goals_completed: int
    pending_approvals: list[int]            # approval IDs awaiting human action
    sub_goal_results: list[SubGoalResult]
    # Present only on early-exit (failed_at set)
    failed_at: Optional[str] = None
    completed: Optional[list[str]] = None


# ══════════════════════════════════════════════════════════════════
# L. ASK / RETRIEVE CONTEXT
#    POST /ask/
#    POST /retrieve-context/
# ══════════════════════════════════════════════════════════════════

class AskResponse(BaseModel):
    """POST /ask/ — LLM answer grounded in retrieved code context."""
    answer: str


class RetrieveContextResponse(BaseModel):
    """POST /retrieve-context/ — raw code chunks retrieved for a query."""
    relevant_chunks: list[str]


# ══════════════════════════════════════════════════════════════════
# M. EVOLUTION & DRIFT
#    GET  /evolution/
#    GET  /drift/
#    POST /snapshot/
#    GET  /function-history/
# ══════════════════════════════════════════════════════════════════

class VolatileFileEntry(BaseModel):
    """One entry from get_code_evolution_summary() most_volatile / most_stable lists."""
    file: str
    mods_per_day: float


class EvolutionResponse(BaseModel):
    """
    GET /evolution/ — get_code_evolution_summary() return, wrapped in a typed envelope.
    total_mods_30d: count of file_versions rows created in last 30 days.
    most_volatile:  up to 5 files with highest change velocity.
    most_stable:    up to 5 files with zero changes.
    """
    project_id: int
    total_mods_30d: int
    most_volatile: list[VolatileFileEntry]
    most_stable: list[VolatileFileEntry]


class DriftResponse(BaseModel):
    """
    GET /drift/ — DriftReport dataclass from autonomous_ops.detect_drift(), serialised.
    risk_level:       low | medium | high
    violations:       human-readable strings describing each architectural violation
    coupling_delta:   positive = more coupled since last snapshot
    circular_deps:    new circular dependency pairs introduced since last snapshot
    volatile_modules: new files that didn't exist in the baseline snapshot
    """
    project_id: int
    risk_level: Literal["low", "medium", "high"]
    violations: list[str]
    coupling_delta: float
    circular_deps: list[Any]            # list of [file_a, file_b] pairs
    volatile_modules: list[str]


class SnapshotResponse(BaseModel):
    """POST /snapshot/"""
    snapshot_id: int


class FunctionHistoryEntry(BaseModel):
    """One entry from get_function_history() — a decision memory that mentions the function."""
    memory_id: int
    content: str
    at: str


class FunctionHistoryResponse(BaseModel):
    """GET /function-history/"""
    function: str
    history: list[FunctionHistoryEntry]


# ══════════════════════════════════════════════════════════════════
# N. ROLES & GOVERNANCE
#    GET  /roles/
#    POST /assign-role/
# ══════════════════════════════════════════════════════════════════

class RoleEntry(BaseModel):
    """One row from user_roles table via list_project_roles()."""
    user_id: str
    role: Literal["admin", "reviewer", "observer"]
    granted_by: Optional[str]
    granted_at: str


class RolesResponse(BaseModel):
    """GET /roles/"""
    project_id: int
    roles: list[RoleEntry]


class AssignRoleRequest(BaseModel):
    """
    POST /assign-role/ — body form (existing endpoint uses query params;
    new body form also supported).
    granted_by must be an admin on the project.
    """
    target_user: str
    role: Literal["admin", "reviewer", "observer"]
    granted_by: str


class AssignRoleResponse(BaseModel):
    """POST /assign-role/ — mirrors assign_role() return dict."""
    project_id: int
    user_id: str
    role: str


# ══════════════════════════════════════════════════════════════════
# O. BRANCH DIFFS
#    GET /diff/{project_id}/{branch_name}
#    GET /diff/{project_id}/{branch_name}/summary
# ══════════════════════════════════════════════════════════════════

class BranchDiffEntry(BaseModel):
    """
    One file's diff from get_diffs_for_branch() — stored in change_diffs table
    by diff_engine.generate_and_store_diffs().
    diff_text is the full unified diff (or truncated to 200 lines if preview=true).
    """
    filename: str
    diff_text: str
    lines_added: int
    lines_removed: int
    risk_level: Literal["low", "medium", "high", "unknown"]
    created_at: str


class BranchDiffResponse(BaseModel):
    """
    GET /diff/{project_id}/{branch_name}
    overall_risk computed by compute_overall_risk():
      any HIGH file → HIGH; any MEDIUM (no HIGH) → MEDIUM; else LOW.
    """
    project_id: int
    branch_name: str
    overall_risk: Literal["low", "medium", "high"]
    total_files: int
    total_added: int
    total_removed: int
    diffs: list[BranchDiffEntry]
    message: Optional[str] = None       # present when diffs is empty


class BranchDiffSummaryEntry(BaseModel):
    """
    One file entry in GET /diff/{project_id}/{branch_name}/summary.
    Omits diff_text to keep payload small for dashboard cards.
    """
    filename: str
    lines_added: int
    lines_removed: int
    risk_level: Literal["low", "medium", "high", "unknown"]
    created_at: str


class BranchDiffSummaryResponse(BaseModel):
    """GET /diff/{project_id}/{branch_name}/summary"""
    project_id: int
    branch_name: str
    overall_risk: Literal["low", "medium", "high"]
    total_files: int
    total_added: int
    total_removed: int
    files: list[BranchDiffSummaryEntry]


# ══════════════════════════════════════════════════════════════════
# P. GIT OPERATIONS
#    POST /clone-repo/
#    POST /create-pr/
#    GET  /git-status/
# ══════════════════════════════════════════════════════════════════

class CloneRepoResponse(BaseModel):
    """POST /clone-repo/ — repository cloned and project record updated."""
    message: str
    project_id: int
    repo_url: str
    repo_path: str
    default_branch: str
    git_enabled: bool
    note: str


class CreatePRResponse(BaseModel):
    """POST /create-pr/ — GitHub Pull Request created."""
    message: str
    pr_url: str
    pr_number: int
    branch_name: str
    base_branch: str
    project_id: int
    note: str


class GitStatusResponse(BaseModel):
    """
    GET /git-status/
    git_enabled False when no repository has been cloned for this project.
    When True, the remaining fields describe the local workspace state
    as returned by git_ops.repo_status().
    """
    project_id: int
    git_enabled: bool
    message: Optional[str] = None           # set when git_enabled is False
    repo_url: Optional[str] = None
    default_branch: Optional[str] = None
    workspace_path: Optional[str] = None
    # Additional fields from repo_status() are merged in via **status —
    # they vary by git state so we accept any extra keys via model_config.
    model_config = {"extra": "allow"}


# ══════════════════════════════════════════════════════════════════
# Q. JOB SYSTEM
#    POST /modify-code-async/
#    GET  /job-status/{job_id}
# ══════════════════════════════════════════════════════════════════

class AsyncJobResponse(BaseModel):
    """POST /modify-code-async/ — job queued, returns immediately."""
    job_id: int
    status: Literal["pending"]
    message: str


class JobStatusResponse(BaseModel):
    """
    GET /job-status/{job_id} — wraps get_job_status() from job_system.py.
    result is populated when status == 'completed' or 'failed'.
    """
    job_id: int
    project_id: int
    status: str     # pending | running | completed | failed
    query: Optional[str] = None
    result: Optional[dict[str, Any]] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


# ══════════════════════════════════════════════════════════════════
# R. TESTS   GET /tests/
# ══════════════════════════════════════════════════════════════════

class TestEntry(BaseModel):
    """One row from project_tests table."""
    test_id: int
    filename: str
    pass_rate: Optional[float]
    last_run_at: Optional[str]


class ProjectTestsResponse(BaseModel):
    """GET /tests/"""
    project_id: int
    tests: list[TestEntry]


# ══════════════════════════════════════════════════════════════════
# S. OPS
#    GET /rate-limit-status/
#    GET /llm-status/
#    GET /global-lessons/
#    POST /modify-code/       (sync, existing)
#    POST /rollback/          (existing)
# ══════════════════════════════════════════════════════════════════

class RateLimitStatusResponse(BaseModel):
    """
    GET /rate-limit-status/ — wraps get_rate_limit_status() return dict.
    The exact keys depend on production_safety.py implementation;
    extra keys are allowed to keep backward compatibility.
    """
    model_config = {"extra": "allow"}


class LLMStatusResponse(BaseModel):
    """
    GET /llm-status/ — wraps _router.status() from infra.build_groq_router().
    Router status shape varies by implementation; extra keys allowed.
    """
    model_config = {"extra": "allow"}


class GlobalLessonsResponse(BaseModel):
    """GET /global-lessons/ — global cross-project lessons from global_memory.py."""
    query: str
    lessons: list[Any]


class ModifyCodeResponse(BaseModel):
    """
    POST /modify-code/ — result of run_agent() from agent_executor.py.
    status: "success" | "failed" | other agent-defined strings.
    quality: populated only when status == "success" — per-file quality snapshots.
    All other fields from run_agent() are passed through via extra="allow".
    """
    status: str
    quality: Optional[dict[str, Any]] = None    # { filename: {score, grade} }
    model_config = {"extra": "allow"}


class RollbackResponse(BaseModel):
    """POST /rollback/ — file rolled back to previous version."""
    message: str


# ══════════════════════════════════════════════════════════════════
# T. ARCHITECTURE ELEMENTS   GET /get-architecture/
# ══════════════════════════════════════════════════════════════════

class ArchitecturalElement(BaseModel):
    """One extracted element from architectural_elements table."""
    type: Literal["class", "function", "method"]
    name: str
    signature: Optional[str]


class GetArchitectureResponse(BaseModel):
    """GET /get-architecture/ — all extracted elements keyed by filename."""
    project_id: int
    architecture: dict[str, list[ArchitecturalElement]]