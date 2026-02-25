"""
enterprise_governance.py — Enterprise Safety Controls
======================================================
Phase 5 — AI CTO Enterprise Governance Layer

Responsibilities:
  1. Role-based access control (RBAC): admin | reviewer | observer
  2. Enterprise approval gate with reviewer sign-off for high-risk changes
  3. Persistent audit log: audit_log (project_id, user_id, action, branch, timestamp)
  4. Enterprise risk scoring formula (0-100, bands: low/medium/high)
  5. PR description enrichment with risk score and governance metadata

DB migrations (run once — append to existing schema):

  CREATE TABLE user_roles (
      id           SERIAL PRIMARY KEY,
      project_id   INTEGER REFERENCES projects(id) ON DELETE CASCADE,
      user_id      VARCHAR(128)  NOT NULL,
      role         VARCHAR(32)   NOT NULL CHECK (role IN ('admin','reviewer','observer')),
      granted_by   VARCHAR(128),
      granted_at   TIMESTAMPTZ   DEFAULT now(),
      UNIQUE (project_id, user_id)
  );

  CREATE TABLE audit_log (
      id           SERIAL PRIMARY KEY,
      project_id   INTEGER REFERENCES projects(id) ON DELETE CASCADE,
      user_id      VARCHAR(128),
      action       VARCHAR(128)  NOT NULL,
      branch       VARCHAR(255),
      risk_score   INTEGER       DEFAULT 0,
      detail       TEXT,
      timestamp    TIMESTAMPTZ   DEFAULT now()
  );
  CREATE INDEX idx_audit_log_project ON audit_log (project_id, timestamp DESC);

  -- Extend pending_approvals with governance fields (idempotent):
  ALTER TABLE pending_approvals
    ADD COLUMN IF NOT EXISTS risk_score   INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS reviewer_id  VARCHAR(128),
    ADD COLUMN IF NOT EXISTS reviewed_at  TIMESTAMPTZ;
"""

import re
import json
import logging
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — RBAC: ROLES & ACCESS
# ═══════════════════════════════════════════════════════════════

class UserRole(str, Enum):
    ADMIN    = "admin"      # full access: read, modify, approve, manage roles
    REVIEWER = "reviewer"   # read + approve/reject high-risk changes only
    OBSERVER = "observer"   # read-only: cannot trigger modifications or approve


# Permission matrix
_ROLE_PERMISSIONS: dict[str, set[str]] = {
    UserRole.ADMIN:    {"read", "modify", "approve", "reject", "manage_roles", "create_pr"},
    UserRole.REVIEWER: {"read", "approve", "reject"},
    UserRole.OBSERVER: {"read"},
}

# Core modules — touching any of these raises risk score
CORE_MODULES: frozenset[str] = frozenset({
    "main.py", "database.py", "models.py", "auth.py",
    "agent_executor.py", "multi_agent.py", "execution_hardening.py",
    "production_safety.py", "infra.py",
})


@dataclass
class RoleContext:
    """Resolved identity + role for one API request."""
    user_id:    str
    role:       UserRole
    project_id: int

    def can(self, permission: str) -> bool:
        return permission in _ROLE_PERMISSIONS.get(self.role, set())

    def require(self, permission: str, detail: str = ""):
        if not self.can(permission):
            msg = f"Role '{self.role}' lacks '{permission}' permission"
            if detail:
                msg += f": {detail}"
            raise HTTPException(status_code=403, detail=msg)


# ─────────────────────────────────────────────
# Role management
# ─────────────────────────────────────────────

def assign_role(
    project_id: int,
    user_id:    str,
    role:       str,
    granted_by: str,
    db:         Session,
) -> dict:
    """Assign or update a user's role on a project. Only admins should call this."""
    valid = {r.value for r in UserRole}
    if role not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role '{role}'. Must be one of: {sorted(valid)}"
        )
    try:
        db.execute(
            text("""
                INSERT INTO user_roles (project_id, user_id, role, granted_by, granted_at)
                VALUES (:pid, :uid, :role, :by, now())
                ON CONFLICT (project_id, user_id)
                DO UPDATE SET role = :role, granted_by = :by, granted_at = now()
            """),
            {"pid": project_id, "uid": user_id, "role": role, "by": granted_by},
        )
        db.commit()
        logger.info("[RBAC] %s assigned %s role=%s on project %d", granted_by, user_id, role, project_id)
        return {"project_id": project_id, "user_id": user_id, "role": role}
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Role assignment failed: {exc}")


def get_role(project_id: int, user_id: str, db: Session) -> Optional[UserRole]:
    """Look up a user's role. Returns None if unassigned."""
    try:
        row = db.execute(
            text("SELECT role FROM user_roles WHERE project_id=:pid AND user_id=:uid"),
            {"pid": project_id, "uid": user_id},
        ).fetchone()
        return UserRole(row[0]) if row else None
    except Exception:
        return None


def resolve_role_context(
    project_id:   int,
    user_id:      str,
    db:           Session,
    default_role: UserRole = UserRole.OBSERVER,
) -> RoleContext:
    """
    Resolve a RoleContext. Falls back to observer for unenrolled users
    so read-only access always works without errors.
    """
    role = get_role(project_id, user_id, db) or default_role
    return RoleContext(user_id=user_id, role=role, project_id=project_id)


def list_project_roles(project_id: int, db: Session) -> list[dict]:
    """List all role assignments for a project."""
    try:
        rows = db.execute(
            text("""
                SELECT user_id, role, granted_by, granted_at
                FROM user_roles WHERE project_id = :pid
                ORDER BY granted_at DESC
            """),
            {"pid": project_id},
        ).fetchall()
        return [{"user_id": r[0], "role": r[1], "granted_by": r[2], "granted_at": str(r[3])}
                for r in rows]
    except Exception as exc:
        logger.warning("[RBAC] list_project_roles failed: %s", exc)
        return []


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — AUDIT LOG
# ═══════════════════════════════════════════════════════════════

def write_audit_log(
    project_id:  int,
    user_id:     str,
    action:      str,
    db:          Session,
    branch:      Optional[str] = None,
    risk_score:  int           = 0,
    detail:      Optional[str] = None,
) -> None:
    """
    Persist one audit entry. Never raises — failures are logged and swallowed.
    """
    try:
        db.execute(
            text("""
                INSERT INTO audit_log
                    (project_id, user_id, action, branch, risk_score, detail, timestamp)
                VALUES
                    (:pid, :uid, :action, :branch, :risk, :detail, now())
            """),
            {
                "pid":    project_id,
                "uid":    user_id,
                "action": action[:128],
                "branch": branch,
                "risk":   risk_score,
                "detail": (detail or "")[:2000],
            },
        )
        db.commit()
        logger.info("[Audit] project=%d user=%s action=%s risk=%d", project_id, user_id, action, risk_score)
    except Exception as exc:
        logger.warning("[Audit] write failed (non-fatal): %s", exc)


def get_audit_log(
    project_id: int,
    db:         Session,
    limit:      int            = 100,
    user_id:    Optional[str]  = None,
    action:     Optional[str]  = None,
) -> list[dict]:
    """Retrieve audit log entries for a project, newest first."""
    try:
        filters: list[str] = ["project_id = :pid"]
        params: dict = {"pid": project_id, "limit": limit}
        if user_id:
            filters.append("user_id = :uid")
            params["uid"] = user_id
        if action:
            filters.append("action = :action")
            params["action"] = action

        rows = db.execute(
            text(f"""
                SELECT id, user_id, action, branch, risk_score, detail, timestamp
                FROM audit_log
                WHERE {" AND ".join(filters)}
                ORDER BY timestamp DESC
                LIMIT :limit
            """),
            params,
        ).fetchall()
        return [
            {"id": r[0], "user_id": r[1], "action": r[2], "branch": r[3],
             "risk_score": r[4], "detail": r[5], "timestamp": str(r[6])}
            for r in rows
        ]
    except Exception as exc:
        logger.warning("[Audit] get_audit_log failed: %s", exc)
        return []


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — ENTERPRISE RISK SCORING
# ═══════════════════════════════════════════════════════════════

@dataclass
class RiskScoreResult:
    """Full breakdown of the enterprise risk score."""
    score:                    int
    band:                     str        # "low" | "medium" | "high"
    factors:                  list[str]
    files_modified:           int        = 0
    signature_changes:        bool       = False
    core_modules_touched:     list[str]  = field(default_factory=list)
    circular_deps_introduced: list       = field(default_factory=list)
    high_risk_file_count:     int        = 0

    def requires_reviewer_approval(self) -> bool:
        return self.band == "high"

    def to_dict(self) -> dict:
        return {
            "score":                       self.score,
            "band":                        self.band,
            "requires_reviewer_approval":  self.requires_reviewer_approval(),
            "factors":                     self.factors,
            "files_modified":              self.files_modified,
            "signature_changes":           self.signature_changes,
            "core_modules_touched":        self.core_modules_touched,
            "circular_deps_introduced":    self.circular_deps_introduced,
            "high_risk_file_count":        self.high_risk_file_count,
        }


# Signature-change detector for diff text
_SIG_RE = re.compile(r"^[+-]\s*(def |async def |class )", re.MULTILINE)


def compute_enterprise_risk_score(
    modified_files: list[str],
    diff_summaries: list[dict],
    project_id:     int,
    db:             Session,
    check_circular: bool = True,
) -> RiskScoreResult:
    """
    Enterprise risk scoring formula (additive, capped at 100):

      +25  modifies > 3 files
      +20  function/class signature changes detected in any diff
      +20  any core module touched
      +20  introduces new circular dependency
      +15  1+ files rated "high" by diff engine (capped to avoid double-counting)

    Bands:  0-30 → low | 31-60 → medium | 61-100 → high
    High band → reviewer approval required.
    """
    score   = 0
    factors: list[str] = []
    unique_files = list(set(modified_files))

    # Factor 1: > 3 files
    if len(unique_files) > 3:
        score += 25
        factors.append(f"+25  modifies {len(unique_files)} files (threshold >3)")

    # Factor 2: signature changes (scan diff text)
    sig_changed = any(
        _SIG_RE.search(d.get("diff_text", ""))
        for d in diff_summaries
        if d.get("has_changes")
    )
    if sig_changed:
        score += 20
        factors.append("+20  function/class signature changes detected")

    # Factor 3: core modules
    core_touched = [f for f in unique_files if f in CORE_MODULES]
    if core_touched:
        score += 20
        factors.append(f"+20  core module(s) touched: {', '.join(core_touched)}")

    # Factor 4: new circular dependencies
    new_circulars: list = []
    if check_circular:
        try:
            from autonomous_ops import _build_current_snapshot
            current = _build_current_snapshot(project_id, db)
            current_circles = {tuple(sorted(c)) for c in current.get("circular_deps", [])}

            last_row = db.execute(
                text("""
                    SELECT snapshot FROM architectural_snapshots
                    WHERE project_id = :pid ORDER BY created_at DESC LIMIT 1
                """),
                {"pid": project_id},
            ).fetchone()

            if last_row:
                old_circles = {tuple(sorted(c)) for c in last_row[0].get("circular_deps", [])}
                new_circulars = [list(c) for c in (current_circles - old_circles)]
            else:
                new_circulars = [list(c) for c in current_circles]

            if new_circulars:
                score += 20
                factors.append(f"+20  new circular dependency introduced: {new_circulars[:2]}")
        except Exception as exc:
            logger.warning("[RiskScore] Circular dep check skipped (non-fatal): %s", exc)

    # Factor 5: diff-engine high-risk files
    high_risk_files = [d for d in diff_summaries if d.get("risk_level") == "high"]
    if high_risk_files:
        added = min(15, 100 - score)  # cap at 100
        if added > 0:
            score += added
            factors.append(f"+{added}  {len(high_risk_files)} file(s) flagged high-risk by diff engine")

    score = min(score, 100)

    if score <= 30:
        band = "low"
    elif score <= 60:
        band = "medium"
    else:
        band = "high"

    if not factors:
        factors.append("No elevated risk factors detected")

    result = RiskScoreResult(
        score=score,
        band=band,
        factors=factors,
        files_modified=len(unique_files),
        signature_changes=sig_changed,
        core_modules_touched=core_touched,
        circular_deps_introduced=new_circulars,
        high_risk_file_count=len(high_risk_files),
    )
    logger.info("[RiskScore] project=%d score=%d band=%s", project_id, score, band)
    return result


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — ENTERPRISE APPROVAL GATE
# ═══════════════════════════════════════════════════════════════

def create_enterprise_approval(
    project_id:   int,
    plan_json:    dict,
    risk_score:   RiskScoreResult,
    requested_by: str,
    branch:       Optional[str],
    db:           Session,
) -> int:
    """
    Create a pending approval for a high-risk change.
    Returns the approval ID. Caller should halt execution until approved.
    """
    row = db.execute(
        text("""
            INSERT INTO pending_approvals
                (project_id, plan_json, risk_flags, status, risk_score)
            VALUES
                (:pid, CAST(:plan AS jsonb), CAST(:flags AS jsonb), 'pending', :rs)
            RETURNING id
        """),
        {
            "pid":   project_id,
            "plan":  json.dumps(plan_json),
            "flags": json.dumps(risk_score.factors),
            "rs":    risk_score.score,
        },
    ).fetchone()
    db.commit()
    approval_id = row[0]

    write_audit_log(
        project_id=project_id,
        user_id=requested_by,
        action="approval_requested",
        db=db,
        branch=branch,
        risk_score=risk_score.score,
        detail=json.dumps({
            "approval_id": approval_id,
            "band":        risk_score.band,
            "factors":     risk_score.factors,
        }),
    )
    logger.info("[EnterpriseGate] Approval #%d created project=%d risk=%d(%s)",
                approval_id, project_id, risk_score.score, risk_score.band)
    return approval_id


def approve_change(
    approval_id:  int,
    project_id:   int,
    reviewer_id:  str,
    role_ctx:     RoleContext,
    db:           Session,
    model=None,
    groq_llm=None,
    context:      str = "",
) -> dict:
    """
    Enterprise POST /approve-change/ handler.

    Validates reviewer permission, marks record approved, audits,
    then executes the stored DAG plan.
    """
    role_ctx.require("approve", "only reviewer or admin roles can approve high-risk changes")

    row = db.execute(
        text("""
            SELECT plan_json, risk_flags, risk_score
            FROM pending_approvals
            WHERE id = :aid AND project_id = :pid AND status = 'pending'
        """),
        {"aid": approval_id, "pid": project_id},
    ).fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Approval #{approval_id} not found, already resolved, or wrong project.",
        )

    plan_data, risk_flags, stored_risk = row[0], row[1], row[2] or 0

    db.execute(
        text("""
            UPDATE pending_approvals
            SET status = 'approved', reviewer_id = :rev, reviewed_at = now()
            WHERE id = :aid
        """),
        {"rev": reviewer_id, "aid": approval_id},
    )
    db.commit()

    write_audit_log(
        project_id=project_id,
        user_id=reviewer_id,
        action="change_approved",
        db=db,
        risk_score=stored_risk,
        detail=json.dumps({"approval_id": approval_id, "risk_flags": risk_flags}),
    )

    # Reconstruct and execute the plan
    from multi_agent import run_agent_from_dag, DAGPlan, DAGStep
    try:
        steps    = [DAGStep(**s) for s in plan_data["steps"]]
        dag_plan = DAGPlan(goal=plan_data["goal"], steps=steps)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Plan reconstruction failed: {exc}")

    logger.info("[EnterpriseGate] Approval #%d approved by %s — executing", approval_id, reviewer_id)
    return run_agent_from_dag(
        dag_plan=dag_plan, project_id=project_id, db=db,
        model=model, groq_llm=groq_llm, context=context,
    )


def reject_change(
    approval_id: int,
    project_id:  int,
    reviewer_id: str,
    role_ctx:    RoleContext,
    reason:      str,
    db:          Session,
) -> dict:
    """Enterprise rejection handler. Requires reviewer or admin role."""
    role_ctx.require("reject", "only reviewer or admin roles can reject changes")

    row = db.execute(
        text("""
            SELECT risk_score FROM pending_approvals
            WHERE id = :aid AND project_id = :pid AND status = 'pending'
        """),
        {"aid": approval_id, "pid": project_id},
    ).fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Approval #{approval_id} not found or already resolved.",
        )

    db.execute(
        text("""
            UPDATE pending_approvals
            SET status = 'rejected', reviewer_id = :rev, reviewed_at = now()
            WHERE id = :aid
        """),
        {"rev": reviewer_id, "aid": approval_id},
    )
    db.commit()

    write_audit_log(
        project_id=project_id,
        user_id=reviewer_id,
        action="change_rejected",
        db=db,
        risk_score=row[0] or 0,
        detail=json.dumps({"approval_id": approval_id, "reason": reason[:500]}),
    )
    logger.info("[EnterpriseGate] Approval #%d rejected by %s", approval_id, reviewer_id)
    return {"status": "rejected", "approval_id": approval_id, "reason": reason}


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — PR DESCRIPTION ENRICHMENT
# ═══════════════════════════════════════════════════════════════

_RISK_EMOJI = {"low": "🟢", "medium": "🟡", "high": "🔴"}


def build_enterprise_pr_body(
    base_body:      str,
    risk_score:     RiskScoreResult,
    modified_files: list[str],
    branch_name:    str,
    requested_by:   str,
    approval_id:    Optional[int] = None,
) -> str:
    """
    Append the enterprise governance section to a PR description.
    Includes risk score, factors, file list, and approval status.
    Non-raising — returns base_body unchanged on any error.
    """
    try:
        emoji = _RISK_EMOJI.get(risk_score.band, "⚪")

        if approval_id:
            approval_status = f"✅ **Approved** (approval #{approval_id})"
        elif risk_score.requires_reviewer_approval():
            approval_status = "⚠️ **Awaiting reviewer approval** — do not merge until approved"
        else:
            approval_status = "✅ **Auto-approved** — risk score within threshold"

        factors_md = "\n".join(f"  - {f}" for f in risk_score.factors)
        files_md   = "\n".join(f"  - `{f}`" for f in sorted(set(modified_files))) or "  _(none)_"

        section = f"""

---
## 🏢 Enterprise Governance Report

| Field | Value |
|---|---|
| **Risk Score** | {emoji} **{risk_score.score}/100** — {risk_score.band.upper()} |
| **Files Modified** | {risk_score.files_modified} |
| **Signature Changes** | {"Yes ⚠️" if risk_score.signature_changes else "No"} |
| **Core Modules Touched** | {", ".join(f"`{m}`" for m in risk_score.core_modules_touched) or "None"} |
| **Circular Deps Introduced** | {"Yes ⛔" if risk_score.circular_deps_introduced else "No"} |
| **Branch** | `{branch_name}` |
| **Requested By** | `{requested_by}` |

### Risk Factors
{factors_md}

### Modified Files
{files_md}

### Approval Status
{approval_status}

---
_AI CTO Enterprise Governance · Audit trail: GET /audit-log/?project_id={{}}_
"""
        return base_body + section
    except Exception as exc:
        logger.warning("[EnterpriseGovern] PR body enrichment failed (non-fatal): %s", exc)
        return base_body


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — ORCHESTRATOR INTEGRATION HELPER
# ═══════════════════════════════════════════════════════════════

def governance_gate(
    modified_files: list[str],
    diff_summaries: list[dict],
    project_id:     int,
    user_id:        str,
    role_ctx:       RoleContext,
    plan_json:      dict,
    branch:         Optional[str],
    db:             Session,
    skip_gate:      bool = False,
) -> tuple[RiskScoreResult, Optional[int]]:
    """
    Post-execution enterprise gate. Call after run_agent_from_dag() succeeds.

    Returns (risk_score, approval_id_or_None).

    Flow:
      - Always computes risk score and writes audit log.
      - If risk is HIGH and role is not admin (and skip_gate=False):
          creates approval request → returns (risk, approval_id)
          CALLER MUST halt and wait for /approve-change/.
      - If risk is HIGH and role IS admin (or skip_gate=True):
          logs admin bypass → returns (risk, None).
      - If risk is LOW/MEDIUM:
          returns (risk, None) — no gate needed.
    """
    risk = compute_enterprise_risk_score(
        modified_files=modified_files,
        diff_summaries=diff_summaries,
        project_id=project_id,
        db=db,
    )

    write_audit_log(
        project_id=project_id,
        user_id=user_id,
        action="execution_risk_scored",
        db=db,
        branch=branch,
        risk_score=risk.score,
        detail=json.dumps(risk.to_dict()),
    )

    if risk.requires_reviewer_approval() and not skip_gate:
        if role_ctx.role == UserRole.ADMIN:
            write_audit_log(
                project_id=project_id,
                user_id=user_id,
                action="admin_high_risk_bypass",
                db=db,
                branch=branch,
                risk_score=risk.score,
                detail=f"admin bypassed reviewer gate band={risk.band}",
            )
            return risk, None

        # Requires reviewer — create pending approval and halt
        approval_id = create_enterprise_approval(
            project_id=project_id,
            plan_json=plan_json,
            risk_score=risk,
            requested_by=user_id,
            branch=branch,
            db=db,
        )
        return risk, approval_id

    return risk, None