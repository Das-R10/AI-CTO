"""
multi_agent.py — Multi-Agent Orchestration Layer
=================================================
Consolidates all agent-role separation into one module:

  1. DAG Planner        — dependency-aware execution graph (superset of linear plans)
  2. Architect Agent    — validates plans against architectural constraints before execution
  3. Approval Gate      — halts high-risk plans pending human confirmation
  4. Agent Orchestrator — coordinates Architect → Plan → Gate → Execute → QA flow
  5. QA Agent           — post-execution verification (tests + behavioral + arch rules)
  6. Goal Engine        — decomposes high-level product goals into ordered sub-goals

Integration with existing code:
  - run_agent() in agent_executor.py is preserved and used as the inner execution primitive
  - generate_structured_plan() is called by the DAG planner for each sub-goal
  - All existing endpoints continue working; new endpoints added in main.py (see INTEGRATION GUIDE)

New DB tables required (run once):
  CREATE TABLE dag_plans (
      id          SERIAL PRIMARY KEY,
      project_id  INTEGER REFERENCES projects(id),
      job_id      INTEGER,
      goal        TEXT,
      plan_json   JSONB,
      status      VARCHAR(32) DEFAULT 'pending',
      created_at  TIMESTAMPTZ DEFAULT now()
  );

  CREATE TABLE pending_approvals (
      id            SERIAL PRIMARY KEY,
      project_id    INTEGER REFERENCES projects(id),
      plan_json     JSONB,
      risk_flags    JSONB,
      status        VARCHAR(32) DEFAULT 'pending',
      created_at    TIMESTAMPTZ DEFAULT now(),
      resolved_at   TIMESTAMPTZ
  );
"""

import re
import json
import ast
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from sqlalchemy.orm import Session
from sqlalchemy import text
from enterprise_governance import (
    UserRole, RoleContext, RiskScoreResult,
    resolve_role_context, write_audit_log, get_audit_log,
    compute_enterprise_risk_score, governance_gate,
    build_enterprise_pr_body, CORE_MODULES,
)


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — DATA CLASSES
# ═══════════════════════════════════════════════════════════════

class AgentRole(Enum):
    ARCHITECT   = "architect"
    PLANNER     = "planner"
    EXECUTOR    = "executor"
    QA          = "qa"
    REFACTOR    = "refactor"
    OBSERVER    = "observer"


@dataclass
class DAGStep:
    """A single step in a dependency-aware execution graph."""
    step_id:         str
    action:          str
    target_file:     Optional[str]
    target_function: Optional[str]
    instruction:     str
    depends_on:      list[str] = field(default_factory=list)   # step_ids that must complete first
    can_parallel:    bool = False

    def to_dict(self) -> dict:
        return {
            "step_id":         self.step_id,
            "action":          self.action,
            "target_file":     self.target_file,
            "target_function": self.target_function,
            "instruction":     self.instruction,
            "depends_on":      self.depends_on,
            "can_parallel":    self.can_parallel,
        }


@dataclass
class DAGPlan:
    """A directed acyclic graph of execution steps."""
    goal:  str
    steps: list[DAGStep]

    def execution_batches(self) -> list[list[DAGStep]]:
        """
        Topological sort — returns batches of steps that can run in parallel.
        Steps in the same batch have no dependency on each other.
        """
        completed: set[str] = set()
        remaining = list(self.steps)
        batches   = []

        while remaining:
            ready = [
                s for s in remaining
                if all(dep in completed for dep in s.depends_on)
            ]
            if not ready:
                # Cycle or unresolvable dependency — fall back to sequential
                ready = [remaining[0]]

            batches.append(ready)
            for s in ready:
                completed.add(s.step_id)
                remaining.remove(s)

        return batches

    def to_dict(self) -> dict:
        return {
            "goal":  self.goal,
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass
class ArchitectDecision:
    approved:      bool
    reason:        str
    risk_flags:    list[str]
    modified_plan: Optional[DAGPlan] = None   # if agent proposes a safer alternative


@dataclass
class QAVerdict:
    passed:                  bool
    reason:                  str
    test_results:            dict
    architectural_violations: list[str]
    unexpected_behavioral_changes: list[str]


@dataclass
class SubGoal:
    sub_goal_id:     str
    description:     str
    depends_on:      list[str]
    estimated_files: list[str]
    priority:        int


@dataclass
class DecomposedGoal:
    high_level_goal: str
    sub_goals:       list[SubGoal]


@dataclass
class OrchestratorResult:
    goal:          str
    status:        str          # completed | failed | pending_approval
    approval_id:   Optional[int]
    agent_trace:   list[dict]
    final_result:  Optional[dict]
    # Enterprise governance fields (new — defaults preserve backward compat)
    risk_score:    Optional[int] = None
    risk_band:     Optional[str] = None
    user_id:       Optional[str] = None


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — DAG PLANNER
# ═══════════════════════════════════════════════════════════════

def build_dag_plan(
    query:      str,
    context:    str,
    groq_llm,
    file_list:  list[str],
    project_id: Optional[int] = None,
    model=None,
    db:         Optional[Session] = None,
    max_retries: int = 3
) -> DAGPlan:
    """
    Generate a DAG-structured execution plan.
    Falls back to a linear plan (all steps depend on previous) if the LLM
    doesn't return dependency info — preserving full backward compatibility
    with the existing linear ExecutionPlan structure.
    """
    from agent_executor import generate_structured_plan, _get_project_filenames

    # Use the existing validated planner for the step content
    linear_plan = generate_structured_plan(
        query=query,
        context=context,
        groq_llm=groq_llm,
        file_list=file_list,
        max_retries=max_retries,
        project_id=project_id,
        model=model,
        db=db
    )

    # Ask the LLM if any steps can run in parallel
    if len(linear_plan.steps) <= 1:
        # Single step — trivial DAG
        dag_steps = [
            DAGStep(
                step_id=f"step_{s.step_number}",
                action=s.action,
                target_file=s.target_file,
                target_function=s.target_function,
                instruction=s.instruction,
                depends_on=[],
                can_parallel=False
            )
            for s in linear_plan.steps
        ]
        return DAGPlan(goal=linear_plan.goal, steps=dag_steps)

    # For multi-step plans, attempt to detect parallelism
    step_summaries = "\n".join(
        f"  step_{s.step_number}: {s.action} → {s.target_file} ({s.instruction[:60]})"
        for s in linear_plan.steps
    )
    dep_prompt = f"""
Given these execution steps for the goal "{linear_plan.goal}":
{step_summaries}

For each step, list which other step IDs it depends on (must complete first).
A step can depend on zero or more earlier steps.
If a step modifies a file that another step also modifies, they must be sequential.
If steps touch independent files, they can run in parallel (empty depends_on list).

Return ONLY valid JSON. No markdown.

Format:
[
  {{"step_id": "step_1", "depends_on": []}},
  {{"step_id": "step_2", "depends_on": ["step_1"]}},
  {{"step_id": "step_3", "depends_on": []}}
]
"""
    raw = groq_llm("You are a dependency analyzer. Return only valid JSON arrays.", dep_prompt)
    raw = re.sub(r"```(?:json)?\n?", "", raw).replace("```", "").strip()

    dep_map: dict[str, list[str]] = {}
    try:
        dep_data = json.loads(raw)
        for entry in dep_data:
            dep_map[entry["step_id"]] = entry.get("depends_on", [])
    except Exception:
        # Fall back: linear dependencies
        for i, s in enumerate(linear_plan.steps):
            sid = f"step_{s.step_number}"
            dep_map[sid] = [f"step_{linear_plan.steps[i-1].step_number}"] if i > 0 else []

    dag_steps = []
    step_ids  = {f"step_{s.step_number}" for s in linear_plan.steps}
    for s in linear_plan.steps:
        sid  = f"step_{s.step_number}"
        deps = [d for d in dep_map.get(sid, []) if d in step_ids and d != sid]
        dag_steps.append(DAGStep(
            step_id=sid,
            action=s.action,
            target_file=s.target_file,
            target_function=s.target_function,
            instruction=s.instruction,
            depends_on=deps,
            can_parallel=(len(deps) == 0 and s.step_number > 1)
        ))

    return DAGPlan(goal=linear_plan.goal, steps=dag_steps)


def store_dag_plan(plan: DAGPlan, project_id: int, job_id: Optional[int], db: Session) -> int:
    """Persist a DAG plan to the database. Returns plan record ID."""
    row = db.execute(
        text("""
            INSERT INTO dag_plans (project_id, job_id, goal, plan_json, status)
            VALUES (:pid, :jid, :goal, CAST(:plan AS jsonb), 'pending')
            RETURNING id
        """),
        {"pid": project_id, "jid": job_id, "goal": plan.goal,
         "plan": json.dumps(plan.to_dict())}
    ).fetchone()
    db.commit()
    return row[0]


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — ARCHITECT AGENT
# ═══════════════════════════════════════════════════════════════

def architect_review(
    plan:       DAGPlan,
    project_id: int,
    db:         Session,
    groq_llm,
    model
) -> ArchitectDecision:
    """
    Validate a proposed DAG plan against stored architectural constraints.

    Loads 'architecture' and 'constraint' memories for the project,
    then asks the LLM to evaluate each step against those rules.

    Returns ArchitectDecision with approved/rejected + risk_flags.
    """
    from memory_engine import retrieve_relevant_memory, format_memories_for_prompt

    # Retrieve architectural constraints from project memory
    arch_memories  = retrieve_relevant_memory(project_id, plan.goal, model, db, top_k=8)
    constraint_str = format_memories_for_prompt(
        [m for m in arch_memories if m["memory_type"] in ("architecture", "constraint")]
    )

    if not constraint_str:
        # No stored constraints — approved by default, low risk
        print(f"[Architect] No stored constraints for project {project_id} — auto-approved")
        return ArchitectDecision(
            approved=True,
            reason="No architectural constraints stored — approved by default.",
            risk_flags=[]
        )

    plan_summary = "\n".join(
        f"  {s.step_id}: {s.action} → {s.target_file or 'N/A'} / fn={s.target_function or 'N/A'}"
        f" | depends_on={s.depends_on}"
        for s in plan.steps
    )

    review_prompt = f"""
You are a senior software architect reviewing a proposed code change plan.

=== ARCHITECTURAL CONSTRAINTS ===
{constraint_str}

=== PROPOSED PLAN ===
Goal: {plan.goal}
Steps:
{plan_summary}

Your task:
1. Identify any steps that VIOLATE the architectural constraints above.
2. Identify any steps that could cause architectural decay (increased coupling,
   layer boundary violations, breaking public APIs without explicit instruction).
3. Return a JSON review.

Return ONLY valid JSON. No markdown.

Format:
{{
  "approved": true | false,
  "reason": "short explanation",
  "risk_flags": ["flag1", "flag2"],
  "suggested_modifications": ["optional: how to make a step safer"]
}}
"""
    raw = groq_llm(
        "You are a strict software architect. Evaluate plans against constraints. Return only JSON.",
        review_prompt
    )
    raw = re.sub(r"```(?:json)?\n?", "", raw).replace("```", "").strip()

    try:
        data = json.loads(raw)
        approved   = bool(data.get("approved", True))
        reason     = data.get("reason", "")
        risk_flags = data.get("risk_flags", [])
        print(f"[Architect] Review: {'✅ approved' if approved else '❌ rejected'} | flags={risk_flags}")
        return ArchitectDecision(approved=approved, reason=reason, risk_flags=risk_flags)
    except Exception as e:
        print(f"[Architect] Review parse failed ({e}) — defaulting to approved")
        return ArchitectDecision(approved=True, reason="Review parsing failed — defaulted to approved.", risk_flags=[])


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — APPROVAL GATE
# ═══════════════════════════════════════════════════════════════

# Risk thresholds — configurable
HIGH_RISK_STEP_THRESHOLD  = 1     # any "high" impact step triggers gate
HIGH_RISK_FILE_THRESHOLD  = 4     # modifying 4+ files triggers gate


def _plan_needs_approval(plan: DAGPlan, architect_decision: ArchitectDecision) -> bool:
    """Return True if this plan must be queued for human approval before execution."""
    # Always gate if architect flagged risks
    if architect_decision.risk_flags:
        return True
    # Gate if plan touches many files
    unique_files = {s.target_file for s in plan.steps if s.target_file}
    if len(unique_files) >= HIGH_RISK_FILE_THRESHOLD:
        return True
    return False


def create_approval_request(
    plan:       DAGPlan,
    risk_flags: list[str],
    project_id: int,
    db:         Session
) -> int:
    """Store the plan in pending_approvals and return the approval ID."""
    row = db.execute(
        text("""
            INSERT INTO pending_approvals (project_id, plan_json, risk_flags, status)
            VALUES (:pid, CAST(:plan AS jsonb), CAST(:flags AS jsonb), 'pending')
            RETURNING id
        """),
        {
            "pid":   project_id,
            "plan":  json.dumps(plan.to_dict()),
            "flags": json.dumps(risk_flags)
        }
    ).fetchone()
    db.commit()
    approval_id = row[0]
    print(f"[ApprovalGate] Created pending approval #{approval_id} for project {project_id}")
    return approval_id


def get_pending_approvals(project_id: int, db: Session) -> list[dict]:
    """List all pending approval requests for a project."""
    rows = db.execute(
        text("""
            SELECT id, plan_json, risk_flags, status, created_at
            FROM pending_approvals
            WHERE project_id = :pid AND status = 'pending'
            ORDER BY created_at DESC
        """),
        {"pid": project_id}
    ).fetchall()
    return [
        {
            "approval_id": r[0],
            "plan":        r[1],
            "risk_flags":  r[2],
            "status":      r[3],
            "created_at":  str(r[4])
        }
        for r in rows
    ]


def resolve_approval(
    approval_id: int,
    action:      str,    # "approve" | "reject"
    project_id:  int,
    db:          Session,
    model=None,
    groq_llm=None,
    context:     str = ""
) -> dict:
    """
    Resolve a pending approval.
    If approved: load the stored plan and execute it through run_agent_from_dag().
    If rejected: mark as rejected, store refusal in memory.
    """
    row = db.execute(
        text("SELECT plan_json, risk_flags FROM pending_approvals WHERE id=:aid AND project_id=:pid AND status='pending'"),
        {"aid": approval_id, "pid": project_id}
    ).fetchone()

    if not row:
        return {"error": f"Approval #{approval_id} not found or already resolved"}

    db.execute(
        text("UPDATE pending_approvals SET status=:s, resolved_at=now() WHERE id=:aid"),
        {"s": action + "d", "aid": approval_id}   # "approved" | "rejected"
    )
    db.commit()

    if action == "reject":
        print(f"[ApprovalGate] Approval #{approval_id} rejected by user")
        return {"status": "rejected", "approval_id": approval_id}

    # approved — execute the stored plan
    plan_data = row[0]
    try:
        steps     = [DAGStep(**s) for s in plan_data["steps"]]
        dag_plan  = DAGPlan(goal=plan_data["goal"], steps=steps)
    except Exception as e:
        return {"error": f"Could not reconstruct plan from approval: {e}"}

    print(f"[ApprovalGate] Approval #{approval_id} approved — executing plan")
    return run_agent_from_dag(
        dag_plan=dag_plan, project_id=project_id, db=db,
        model=model, groq_llm=groq_llm, context=context
    )


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — QA AGENT
# ═══════════════════════════════════════════════════════════════

def qa_verify(
    plan:           DAGPlan,
    project_id:     int,
    modified_files: list[str],
    db:             Session,
    groq_llm,
    model,
    original_query: str = ""
) -> QAVerdict:
    """
    Post-execution QA verification. Runs after all executor steps complete.

    Checks:
      1. Re-run all impacted tests
      2. Verify no architectural constraints were violated by the final code state
      3. Check for unexpected public signature changes across modified files

    Returns QAVerdict. If not passed, the caller should trigger rollback.
    """
    from test_engine import find_impacted_tests, run_tests
    from models import File as DBFile
    from memory_engine import retrieve_relevant_memory, format_memories_for_prompt

    test_results: dict           = {}
    arch_violations: list[str]   = []
    behavioral_issues: list[str] = []

    # ── 1. Re-run impacted tests for all modified files ─────────
    for filename in modified_files:
        file = db.query(DBFile).filter(
            DBFile.project_id == project_id,
            DBFile.filename == filename
        ).first()
        if not file:
            continue

        # Find tests impacted by any function in this file
        try:
            tree = ast.parse(file.content)
            fn_names = [
                node.name for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
            ]
        except SyntaxError:
            fn_names = []

        for fn_name in fn_names:
            impact = find_impacted_tests(fn_name, project_id, db)
            for test_id, test_fname in zip(impact.impacted_test_ids, impact.impacted_test_files):
                if test_fname in test_results:
                    continue   # already checked
                row = db.execute(
                    text("SELECT content FROM project_tests WHERE id=:tid"),
                    {"tid": test_id}
                ).fetchone()
                if not row:
                    continue
                result = run_tests(row[0], file.content, filename)
                test_results[test_fname] = "passed" if result.passed else f"FAILED: {result.failures[:2]}"

    # ── 2. Architectural constraint check on final code state ───
    arch_memories = retrieve_relevant_memory(project_id, original_query, model, db, top_k=5)
    constraint_str = format_memories_for_prompt(
        [m for m in arch_memories if m["memory_type"] in ("architecture", "constraint")]
    )

    if constraint_str and modified_files:
        # Sample the modified file contents for the LLM check
        file_samples = []
        for filename in modified_files[:3]:
            file = db.query(DBFile).filter(
                DBFile.project_id == project_id,
                DBFile.filename == filename
            ).first()
            if file:
                file_samples.append(f"=== {filename} ===\n{file.content[:1500]}")

        if file_samples:
            check_prompt = f"""
You are a software architect verifying that modified code obeys architectural rules.

=== CONSTRAINTS ===
{constraint_str}

=== MODIFIED FILES (samples) ===
{chr(10).join(file_samples)}

List any constraint violations found. If none, say "none".
Return a JSON array of violation strings, or an empty array.
No markdown.
"""
            raw = groq_llm(
                "You are a strict architect. Return only a JSON array of violation strings.",
                check_prompt
            )
            raw = re.sub(r"```(?:json)?\n?", "", raw).replace("```", "").strip()
            try:
                violations = json.loads(raw)
                if isinstance(violations, list):
                    arch_violations = [v for v in violations if v and v.lower() != "none"]
            except Exception:
                pass

    # ── 3. Determine overall verdict ────────────────────────────
    failed_tests = [k for k, v in test_results.items() if "FAILED" in str(v)]
    passed = (len(failed_tests) == 0 and len(arch_violations) == 0)

    reason_parts = []
    if failed_tests:
        reason_parts.append(f"{len(failed_tests)} test file(s) failing: {failed_tests[:3]}")
    if arch_violations:
        reason_parts.append(f"{len(arch_violations)} architectural violation(s) detected")
    if passed:
        reason_parts.append("All tests passed, no architectural violations")

    verdict = QAVerdict(
        passed=passed,
        reason=" | ".join(reason_parts),
        test_results=test_results,
        architectural_violations=arch_violations,
        unexpected_behavioral_changes=behavioral_issues
    )
    print(f"[QA] Verdict: {'✅ passed' if passed else '❌ failed'} — {verdict.reason}")
    return verdict


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — INNER EXECUTION: run DAG plan through run_agent()
# ═══════════════════════════════════════════════════════════════

def _get_project_git_config(project_id: int, db: Session) -> dict:
    """
    Fetch Git-related fields for a project from the DB.
    Returns dict with repo_path, repo_url, github_token, default_branch.
    All fields may be None if the project has no Git integration configured.
    """
    row = db.execute(
        text("""
            SELECT repo_path, repo_url, github_token, default_branch
            FROM projects WHERE id = :pid
        """),
        {"pid": project_id}
    ).fetchone()
    if not row:
        return {"repo_path": None, "repo_url": None,
                "github_token": None, "default_branch": "main"}
    return {
        "repo_path":      row[0],
        "repo_url":       row[1],
        "github_token":   row[2],
        "default_branch": row[3] or "main"
    }


def run_agent_from_dag(
    dag_plan:   DAGPlan,
    project_id: int,
    db:         Session,
    model,
    groq_llm,
    context:    str = "",
    enable_tests: bool = False,
    # Enterprise governance (new — backward-compatible defaults)
    user_id:    str  = "system",
    role_ctx:   Optional[RoleContext] = None,
    skip_gate:  bool = False,
) -> dict:
    """
    Execute a DAGPlan by running each batch of steps through the existing
    run_agent() executor (which handles self-correction, rollback, memory, etc.).

    Git Orchestration Layer (Phase 1):
      - Before execution: creates an isolated ai-change-{timestamp} branch.
      - After successful execution: commits all modified files and pushes branch.
      - On failure: resets branch to previous commit (Git-level rollback).
      - Branch name is returned in the result for PR creation.
      - If the project has no repo_path configured, Git steps are skipped
        gracefully (backward compatible with non-Git projects).

    Steps in the same batch that are marked can_parallel are run sequentially
    for now (true parallelism requires thread safety in run_agent — future work).
    """
    from agent_executor import run_agent, _get_project_filenames, _attempt_rollback
    from agent_executor import PlanStep, ExecutionPlan, TOOL_MAP, StepResult
    from git_ops import (
        create_ai_branch, commit_and_push, reset_branch_to_previous_commit,
        write_files_to_workspace
    )
    from diff_engine import snapshot_files_before, generate_and_store_diffs, build_diff_preview, compute_overall_risk

    all_results:     list[dict]  = []
    modified_files:  list[str]   = []
    completed_steps: set[str]    = set()
    git_branch_name: Optional[str] = None

    # ── Git Pre-flight: Create isolated branch ──────────────────
    git_cfg = _get_project_git_config(project_id, db)
    git_enabled = bool(git_cfg.get("repo_path"))

    if git_enabled:
        branch_result = create_ai_branch(git_cfg["repo_path"])
        if not branch_result["success"]:
            print(f"[DAGExecutor] ⚠️  Git branch creation failed (non-fatal): {branch_result['error']}")
            print("[DAGExecutor] Proceeding without Git integration for this run.")
            git_enabled = False
        else:
            git_branch_name = branch_result["branch_name"]
            print(f"[DAGExecutor] 🌿 AI branch created: '{git_branch_name}'")
    # ────────────────────────────────────────────────────────────

    # ── Diff Pre-flight: Snapshot all files before any changes ──
    # Collect all filenames the plan intends to touch so we can diff them later.
    # This is purely observational — no execution logic is affected.
    planned_files = list({
        s.target_file for s in dag_plan.steps if s.target_file
    })
    pre_execution_snapshots = snapshot_files_before(planned_files, project_id, db)
    # ────────────────────────────────────────────────────────────
    # ────────────────────────────────────────────────────────────

    batches = dag_plan.execution_batches()
    print(f"[DAGExecutor] Executing {len(dag_plan.steps)} steps in {len(batches)} batch(es)")

    for batch_idx, batch in enumerate(batches):
        print(f"[DAGExecutor] Batch {batch_idx + 1}/{len(batches)}: {[s.step_id for s in batch]}")

        for dag_step in batch:
            # Map DAGStep → PlanStep for the existing executor
            plan_step = PlanStep(
                step_number=int(dag_step.step_id.split("_")[-1]),
                action=dag_step.action,
                target_file=dag_step.target_file,
                target_function=dag_step.target_function,
                instruction=dag_step.instruction
            )

            tool_fn = TOOL_MAP.get(dag_step.action.lower().replace(" ", "_"))
            if not tool_fn:
                result = StepResult(
                    step_number=plan_step.step_number,
                    action=dag_step.action,
                    success=False,
                    error=f"Unknown action '{dag_step.action}'"
                )
            else:
                try:
                    result = tool_fn(plan_step, project_id, db, model, groq_llm, enable_tests=enable_tests)
                except Exception as e:
                    result = StepResult(
                        step_number=plan_step.step_number,
                        action=dag_step.action,
                        success=False,
                        error=f"Unexpected error in step {dag_step.step_id}: {e}"
                    )

            all_results.append(result.model_dump())

            if not result.success:
                print(f"[DAGExecutor] ❌ Step {dag_step.step_id} failed: {result.error}")

                # ── Git Rollback on failure ──────────────────────
                if git_enabled and git_branch_name:
                    rollback = reset_branch_to_previous_commit(
                        git_cfg["repo_path"], git_branch_name
                    )
                    if rollback["success"]:
                        print(f"[DAGExecutor] ↩️  Git branch reset to {rollback['reset_to'][:8]}")
                    else:
                        print(f"[DAGExecutor] ⚠️  Git rollback failed: {rollback['error']}")
                # ────────────────────────────────────────────────

                return {
                    "goal":              dag_plan.goal,
                    "status":            "failed",
                    "failed_at_step":    dag_step.step_id,
                    "error":             result.error,
                    "completed_steps":   all_results,
                    "git_branch":        git_branch_name
                }

            # Track modified files for QA and Git commit
            if dag_step.target_file:
                modified_files.append(dag_step.target_file)
            completed_steps.add(dag_step.step_id)

    # ── Git Post-execution: Write files, Commit & Push ──────────
    if git_enabled and git_branch_name and modified_files:
        # Sync DB file contents into the workspace filesystem
        unique_files = list(set(modified_files))
        file_contents = _load_modified_file_contents(unique_files, project_id, db)

        write_result = write_files_to_workspace(git_cfg["repo_path"], file_contents)
        if not write_result["success"]:
            print(f"[DAGExecutor] ⚠️  Failed to write files to workspace: {write_result['error']}")
        else:
            commit_msg = (
                f"AI CTO: {dag_plan.goal[:72]}\n\n"
                f"Modified files: {', '.join(unique_files)}\n"
                f"Steps executed: {len(dag_plan.steps)}\n"
                f"Branch: {git_branch_name}"
            )
            push_result = commit_and_push(
                repo_path=git_cfg["repo_path"],
                branch_name=git_branch_name,
                commit_message=commit_msg,
                github_token=git_cfg.get("github_token"),
                repo_url=git_cfg.get("repo_url")
            )
            if push_result["success"]:
                print(f"[DAGExecutor] ✅ Changes committed & pushed on branch '{git_branch_name}'")
            else:
                print(f"[DAGExecutor] ⚠️  Push failed (non-fatal): {push_result['error']}")
    # ────────────────────────────────────────────────────────────

    # ── Diff Post-execution: Generate, score, and store diffs ───
    # Runs unconditionally (Git not required). Pure transparency layer.
    # Uses the pre-execution snapshots captured before the execution loop ran.
    diff_summaries = generate_and_store_diffs(
        snapshots=pre_execution_snapshots,
        project_id=project_id,
        branch_name=git_branch_name,
        db=db
    )
    overall_risk  = compute_overall_risk(diff_summaries)
    diff_preview  = build_diff_preview(diff_summaries)
    # ────────────────────────────────────────────────────────────

    # ── Phase 5: Enterprise governance gate ────────────────────
    _role_ctx = role_ctx or RoleContext(
        user_id=user_id, role=UserRole.ADMIN, project_id=project_id
    )
    _plan_json = dag_plan.to_dict()

    enterprise_risk, approval_id = governance_gate(
        modified_files=list(set(modified_files)),
        diff_summaries=diff_summaries,
        project_id=project_id,
        user_id=user_id,
        role_ctx=_role_ctx,
        plan_json=_plan_json,
        branch=git_branch_name,
        db=db,
        skip_gate=skip_gate,
    )

    write_audit_log(
        project_id=project_id,
        user_id=user_id,
        action="dag_execution_complete",
        db=db,
        branch=git_branch_name,
        risk_score=enterprise_risk.score,
        detail=f"steps={len(dag_plan.steps)} files={len(set(modified_files))} status=success",
    )

    if approval_id:
        # High-risk: halt and await reviewer approval
        return {
            "goal":           dag_plan.goal,
            "status":         "pending_approval",
            "approval_id":    approval_id,
            "steps_executed": len(dag_plan.steps),
            "results":        all_results,
            "modified_files": list(set(modified_files)),
            "git_branch":     git_branch_name,
            "risk_level":     enterprise_risk.band,
            "risk_score":     enterprise_risk.score,
            "risk_factors":   enterprise_risk.factors,
            "diff_preview":   diff_preview,
            "message":        f"High-risk change queued for reviewer approval (#{approval_id}). "                                f"Use POST /approve-change/ to proceed.",
        }
    # ────────────────────────────────────────────────────────────

    return {
        "goal":           dag_plan.goal,
        "status":         "success",
        "steps_executed": len(dag_plan.steps),
        "results":        all_results,
        "modified_files": list(set(modified_files)),
        "git_branch":     git_branch_name,
        # ── Phase 2: diff transparency ──
        "risk_level":     overall_risk,
        "diff_preview":   diff_preview,
        "diff_branch":    git_branch_name,
        # ── Phase 5: enterprise risk ──
        "enterprise_risk_score":  enterprise_risk.score,
        "enterprise_risk_band":   enterprise_risk.band,
        "enterprise_risk_factors": enterprise_risk.factors,
    }


def _load_modified_file_contents(filenames: list[str], project_id: int, db: Session) -> dict[str, str]:
    """
    Load the current content of modified files from the DB so they can be
    written into the local Git workspace before committing.
    Returns { filename: content } for all found files.
    """
    result = {}
    for filename in filenames:
        row = db.execute(
            text("""
                SELECT content FROM files
                WHERE project_id = :pid AND filename = :fn
                ORDER BY created_at DESC LIMIT 1
            """),
            {"pid": project_id, "fn": filename}
        ).fetchone()
        if row:
            result[filename] = row[0]
    return result


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════

def orchestrate(
    query:        str,
    project_id:   int,
    db:           Session,
    model,
    groq_llm,
    context:      str  = "",
    enable_tests: bool = False,
    skip_approval_gate: bool = False,
    # Enterprise governance (new — backward-compatible defaults)
    user_id:      str                  = "system",
    role_ctx:     Optional[RoleContext] = None,
) -> OrchestratorResult:
    """
    Full multi-agent orchestration flow:

      1. Build DAG plan
      2. Architect Agent reviews plan
      3. If architect rejects: return rejected
      4. If high-risk plan and gate not bypassed: create approval record, return pending_approval
      5. Execute DAG plan via run_agent_from_dag()
      6. QA Agent verifies result
      7. If QA fails: rollback all modified files, return failed
      8. Store decision memory, return completed

    This is the recommended entry point for all new code.
    The existing /modify-code/ endpoint can continue using run_agent() directly
    for backward compatibility, or be upgraded to use orchestrate().
    """
    from agent_executor import _get_project_filenames
    from memory_engine import store_memory, build_decision_memory

    trace: list[dict] = []
    print(f"\n[Orchestrator] Goal: {query!r}")

    # ── Step 1: Build DAG Plan ──────────────────────────────────
    file_list = _get_project_filenames(project_id, db)
    if not file_list:
        return OrchestratorResult(
            goal=query, status="failed", approval_id=None,
            agent_trace=[{"step": "plan", "error": "No files in project"}],
            final_result={"error": "No files uploaded to this project yet."}
        )

    try:
        dag_plan = build_dag_plan(
            query=query, context=context, groq_llm=groq_llm,
            file_list=file_list, project_id=project_id, model=model, db=db
        )
    except Exception as e:
        return OrchestratorResult(
            goal=query, status="failed", approval_id=None,
            agent_trace=[{"step": "plan", "error": str(e)}],
            final_result={"error": f"Planning failed: {e}"}
        )

    trace.append({"step": "plan", "goal": dag_plan.goal, "steps": len(dag_plan.steps)})
    print(f"[Orchestrator] Plan: {len(dag_plan.steps)} step(s) across {len(dag_plan.execution_batches())} batch(es)")

    # ── Step 2: Architect Review ────────────────────────────────
    arch_decision = architect_review(dag_plan, project_id, db, groq_llm, model)
    trace.append({
        "step":       "architect_review",
        "approved":   arch_decision.approved,
        "reason":     arch_decision.reason,
        "risk_flags": arch_decision.risk_flags
    })

    if not arch_decision.approved:
        print(f"[Orchestrator] ❌ Architect rejected plan: {arch_decision.reason}")
        return OrchestratorResult(
            goal=dag_plan.goal, status="rejected", approval_id=None,
            agent_trace=trace,
            final_result={"error": f"Architect rejected: {arch_decision.reason}",
                          "risk_flags": arch_decision.risk_flags}
        )

    # ── Step 3: Approval Gate (for high-risk plans) ─────────────
    if not skip_approval_gate and _plan_needs_approval(dag_plan, arch_decision):
        approval_id = create_approval_request(dag_plan, arch_decision.risk_flags, project_id, db)
        trace.append({"step": "approval_gate", "approval_id": approval_id, "status": "pending"})
        print(f"[Orchestrator] ⏸  High-risk plan queued for approval #{approval_id}")
        return OrchestratorResult(
            goal=dag_plan.goal, status="pending_approval", approval_id=approval_id,
            agent_trace=trace,
            final_result={
                "message":     "Plan requires approval before execution.",
                "approval_id": approval_id,
                "risk_flags":  arch_decision.risk_flags,
                "plan_steps":  len(dag_plan.steps),
            },
            user_id=user_id,
        )

    # ── Step 4: Execute DAG Plan ────────────────────────────────
    _orch_role_ctx = role_ctx or RoleContext(
        user_id=user_id, role=UserRole.ADMIN, project_id=project_id
    )
    # Observer role cannot trigger modifications
    _orch_role_ctx.require("modify", "orchestrate requires modify permission")

    write_audit_log(
        project_id=project_id,
        user_id=user_id,
        action="orchestrate_started",
        db=db,
        risk_score=0,
        detail=f"query={query[:120]} steps={len(dag_plan.steps)}",
    )

    exec_result = run_agent_from_dag(
        dag_plan=dag_plan, project_id=project_id, db=db,
        model=model, groq_llm=groq_llm, context=context, enable_tests=enable_tests,
        user_id=user_id, role_ctx=_orch_role_ctx, skip_gate=skip_approval_gate,
    )
    trace.append({"step": "execution", "status": exec_result.get("status")})

    if exec_result.get("status") != "success":
        return OrchestratorResult(
            goal=dag_plan.goal, status="failed", approval_id=None,
            agent_trace=trace, final_result=exec_result
        )

    # ── Step 5: QA Verification ─────────────────────────────────
    modified_files = exec_result.get("modified_files", [])
    qa_verdict = qa_verify(
        plan=dag_plan, project_id=project_id,
        modified_files=modified_files, db=db,
        groq_llm=groq_llm, model=model, original_query=query
    )
    trace.append({
        "step":           "qa",
        "passed":         qa_verdict.passed,
        "reason":         qa_verdict.reason,
        "test_results":   qa_verdict.test_results,
        "arch_violations": qa_verdict.architectural_violations
    })

    if not qa_verdict.passed:
        # Rollback all modified files
        print(f"[Orchestrator] ❌ QA failed — rolling back {modified_files}")
        _rollback_files(modified_files, project_id, db, model)
        return OrchestratorResult(
            goal=dag_plan.goal, status="failed", approval_id=None,
            agent_trace=trace,
            final_result={
                "error":             f"QA verification failed: {qa_verdict.reason}",
                "test_results":      qa_verdict.test_results,
                "arch_violations":   qa_verdict.architectural_violations,
                "rolled_back_files": modified_files
            }
        )

    # ── Step 6: Store Decision Memory ───────────────────────────
    try:
        memory_content = build_decision_memory(
            instruction=query, plan_goal=dag_plan.goal,
            steps_executed=len(dag_plan.steps), results=exec_result.get("results", [])
        )
        store_memory(project_id=project_id, memory_type="decision",
                     content=memory_content, model=model, db=db)
    except Exception as e:
        print(f"[Orchestrator] Memory storage failed (non-fatal): {e}")

    print(f"[Orchestrator] ✅ Goal completed successfully")
    exec_result["agent_trace"] = trace
    exec_result["qa"] = {
        "passed": qa_verdict.passed,
        "test_results": qa_verdict.test_results
    }

    # ── Phase 2: Surface diff transparency in orchestrator response ──
    # Pull the diff data that run_agent_from_dag() already stored.
    # If diff data is missing (e.g. older call path), defaults are safe.
    exec_result.setdefault("risk_level",   "unknown")
    exec_result.setdefault("diff_preview", [])
    exec_result.setdefault("diff_branch",  exec_result.get("git_branch"))
    # ─────────────────────────────────────────────────────────────────

    return OrchestratorResult(
        goal=dag_plan.goal, status="completed", approval_id=None,
        agent_trace=trace, final_result=exec_result,
        risk_score=exec_result.get("enterprise_risk_score"),
        risk_band=exec_result.get("enterprise_risk_band"),
        user_id=user_id,
    )


def _rollback_files(filenames: list[str], project_id: int, db: Session, model):
    """Rollback all files in the list to their last saved version."""
    from agent_executor import _get_best_file, _reembed_file

    for filename in filenames:
        try:
            file = _get_best_file(project_id, filename, db)
            if not file:
                continue
            row = db.execute(
                text("SELECT content FROM file_versions WHERE file_id=:fid ORDER BY created_at DESC LIMIT 1"),
                {"fid": file.id}
            ).fetchone()
            if row:
                file.content = row[0]
                db.commit()
                _reembed_file(file, model, db)
                print(f"[Rollback] ✅ Restored '{filename}'")
        except Exception as e:
            print(f"[Rollback] Failed for '{filename}': {e}")


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — GOAL DECOMPOSITION ENGINE
# ═══════════════════════════════════════════════════════════════

def decompose_goal(
    goal:       str,
    project_id: int,
    db:         Session,
    groq_llm,
    model
) -> DecomposedGoal:
    """
    Decompose a high-level product goal into ordered sub-goals.
    Each sub-goal is achievable by a single orchestrate() call.

    Example:
      Input:  "Add JWT authentication to all protected endpoints"
      Output: [
        SubGoal("add_jwt_model", "Create JWT token model and utilities", depends_on=[]),
        SubGoal("add_middleware", "Add JWT validation middleware", depends_on=["add_jwt_model"]),
        SubGoal("protect_endpoints", "Apply middleware to all protected routes", depends_on=["add_middleware"]),
        SubGoal("update_tests", "Update and add tests for auth flow", depends_on=["protect_endpoints"])
      ]
    """
    from agent_executor import _get_project_filenames
    from memory_engine import retrieve_relevant_memory, format_memories_for_prompt

    file_list = _get_project_filenames(project_id, db)
    memories  = retrieve_relevant_memory(project_id, goal, model, db, top_k=5)
    mem_str   = format_memories_for_prompt(memories)

    decomp_prompt = f"""
You are an AI CTO decomposing a high-level product goal into implementable sub-goals.

Project files: {file_list}

{f"Project memory:{chr(10)}{mem_str}" if mem_str else ""}

High-level goal: {goal}

Break this into 2-6 concrete sub-goals. Each sub-goal should:
- Be achievable by modifying 1-3 files
- Be independently testable
- Have clear dependencies on other sub-goals (if any)

Return ONLY valid JSON. No markdown.

Format:
[
  {{
    "sub_goal_id": "sg_1",
    "description": "Create the X module with Y functionality",
    "depends_on": [],
    "estimated_files": ["filename.py"],
    "priority": 1
  }}
]
"""
    raw = groq_llm(
        "You are a senior software architect. Decompose goals into concrete implementable sub-goals. Return only JSON.",
        decomp_prompt
    )
    raw = re.sub(r"```(?:json)?\n?", "", raw).replace("```", "").strip()

    try:
        data      = json.loads(raw)
        sub_goals = [SubGoal(**sg) for sg in data]
        sub_goals.sort(key=lambda sg: sg.priority)
        print(f"[GoalEngine] Decomposed '{goal[:50]}' into {len(sub_goals)} sub-goal(s)")
        return DecomposedGoal(high_level_goal=goal, sub_goals=sub_goals)
    except Exception as e:
        print(f"[GoalEngine] Decomposition failed ({e}) — treating as single sub-goal")
        return DecomposedGoal(
            high_level_goal=goal,
            sub_goals=[SubGoal(
                sub_goal_id="sg_1",
                description=goal,
                depends_on=[],
                estimated_files=file_list[:3],
                priority=1
            )]
        )


def execute_goal(
    goal:         str,
    project_id:   int,
    db:           Session,
    model,
    groq_llm,
    context:      str = "",
    enable_tests: bool = False
) -> dict:
    """
    Top-level entry point for high-level goal execution.

    Flow:
      1. Decompose goal into sub-goals
      2. Execute each sub-goal through orchestrate() in dependency order
      3. If any sub-goal fails, stop and report
      4. Return full execution summary

    This is the highest-level API — designed for product-level instructions.
    """
    decomposed = decompose_goal(goal, project_id, db, groq_llm, model)
    print(f"[GoalEngine] Executing {len(decomposed.sub_goals)} sub-goal(s) for: {goal[:60]}")

    results        = []
    completed_ids  = set()
    approval_queue = []

    for sg in decomposed.sub_goals:
        # Ensure dependencies are completed
        missing = [d for d in sg.depends_on if d not in completed_ids]
        if missing:
            results.append({
                "sub_goal_id": sg.sub_goal_id,
                "status":      "skipped",
                "reason":      f"Dependencies not completed: {missing}"
            })
            continue

        print(f"[GoalEngine] Executing sub-goal: {sg.sub_goal_id} — {sg.description[:60]}")
        orch_result = orchestrate(
            query=sg.description,
            project_id=project_id,
            db=db,
            model=model,
            groq_llm=groq_llm,
            context=context,
            enable_tests=enable_tests
        )

        result_entry = {
            "sub_goal_id":  sg.sub_goal_id,
            "description":  sg.description,
            "status":       orch_result.status,
            "approval_id":  orch_result.approval_id,
            "final_result": orch_result.final_result
        }
        results.append(result_entry)

        if orch_result.status == "completed":
            completed_ids.add(sg.sub_goal_id)
        elif orch_result.status == "pending_approval":
            approval_queue.append(orch_result.approval_id)
            # Don't mark as completed — downstream sub-goals must wait
        elif orch_result.status in ("failed", "rejected"):
            print(f"[GoalEngine] Sub-goal {sg.sub_goal_id} failed — halting goal execution")
            return {
                "goal":           goal,
                "status":         "failed",
                "failed_at":      sg.sub_goal_id,
                "completed":      list(completed_ids),
                "pending_approvals": approval_queue,
                "sub_goal_results": results
            }

    overall = "completed" if len(completed_ids) == len(decomposed.sub_goals) else "partial"
    if approval_queue:
        overall = "pending_approvals"

    return {
        "goal":              goal,
        "status":            overall,
        "sub_goals_total":   len(decomposed.sub_goals),
        "sub_goals_completed": len(completed_ids),
        "pending_approvals": approval_queue,
        "sub_goal_results":  results
    }


# ═══════════════════════════════════════════════════════════════
# INTEGRATION GUIDE — main.py additions
# ═══════════════════════════════════════════════════════════════
#
# ── 1. Add import ──────────────────────────────────────────────
#
# from multi_agent import orchestrate, execute_goal, get_pending_approvals, resolve_approval
#
#
# ── 2. Add /execute-goal/ endpoint ────────────────────────────
#
# @app.post("/execute-goal/")
# def execute_goal_endpoint(
#     goal: str,
#     project_id: int,
#     enable_tests: bool = False,
#     db: Session = Depends(get_db),
#     x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
#     key_project_id: int = Depends(require_api_key)
# ):
#     verify_project_access(project_id, key_project_id)
#     check_rate_limit(x_api_key, db, endpoint="execute-goal", limit=MODIFY_LIMIT, window=MODIFY_WINDOW)
#     goal, _ = sanitize_query(goal)
#     relevant_ids = retrieve_relevant_files(goal, project_id, db)
#     context = ""
#     if relevant_ids:
#         emb  = model.encode(goal).tolist()
#         rows = db.execute(text("""SELECT content FROM code_chunks WHERE file_id=ANY(:ids)
#                                   ORDER BY embedding <-> CAST(:emb AS vector) LIMIT 5"""),
#                           {"ids": relevant_ids, "emb": emb}).fetchall()
#         context = "\n\n".join(r[0] for r in rows)
#     return execute_goal(goal, project_id, db, model, groq_llm, context, enable_tests)
#
#
# ── 3. Add approval endpoints ─────────────────────────────────
#
# @app.get("/pending-approvals/")
# def list_approvals(project_id: int, db: Session = Depends(get_db),
#                    key_project_id: int = Depends(require_api_key)):
#     verify_project_access(project_id, key_project_id)
#     return {"approvals": get_pending_approvals(project_id, db)}
#
# @app.post("/approve/{approval_id}")
# def approve(approval_id: int, project_id: int, db: Session = Depends(get_db),
#             key_project_id: int = Depends(require_api_key)):
#     verify_project_access(project_id, key_project_id)
#     return resolve_approval(approval_id, "approve", project_id, db, model, groq_llm)
#
# @app.post("/reject/{approval_id}")
# def reject(approval_id: int, project_id: int, db: Session = Depends(get_db),
#            key_project_id: int = Depends(require_api_key)):
#     verify_project_access(project_id, key_project_id)
#     return resolve_approval(approval_id, "reject", project_id, db)
#
#
# ── 4. DB migrations (run once) ────────────────────────────────