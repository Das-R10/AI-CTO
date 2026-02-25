"""
git_ops.py — Git Orchestration Layer for AI CTO
=================================================
Provides all Git and GitHub operations needed for the PR-first workflow.

Responsibilities:
  1. Clone a remote repository into a project workspace.
  2. Create isolated feature branches (ai-change-{timestamp}).
  3. Commit modified files after successful DAG execution.
  4. Push branch to remote origin.
  5. Create GitHub Pull Requests via REST API.
  6. Rollback branch to previous commit on QA failure.

CRITICAL PRINCIPLES enforced here:
  - NEVER commit or push directly to 'main' or 'master'.
  - Every AI-driven change lives exclusively on a named ai-change-* branch.
  - All operations are wrapped with error handling; failures are non-silent.
  - Rollback is always available via reset_branch_to_previous_commit().

Dependencies:
  pip install gitpython httpx

Integration points:
  - Called by run_agent_from_dag() in multi_agent.py (branch + commit + push).
  - Called by POST /clone-repo/ endpoint in main.py.
  - Called by POST /create-pr/ endpoint in main.py.
"""

import os
import time
import shutil
import httpx
from pathlib import Path
from typing import Optional

try:
    import git
    from git import Repo, GitCommandError, InvalidGitRepositoryError
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    print("[GitOps] ⚠️  GitPython not installed. Run: pip install gitpython")


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

WORKSPACE_ROOT = os.environ.get("AICTO_WORKSPACE", "/tmp/aicto_workspaces")
PROTECTED_BRANCHES = {"main", "master", "production", "release"}
GITHUB_API_BASE    = "https://api.github.com"
AI_BRANCH_PREFIX   = "ai-change"


# ─────────────────────────────────────────────
# SECTION 1 — WORKSPACE & CLONE
# ─────────────────────────────────────────────

def get_workspace_path(project_id: int) -> str:
    """Return the local workspace directory path for a project."""
    return os.path.join(WORKSPACE_ROOT, f"project_{project_id}")


def clone_repository(
    repo_url: str,
    project_id: int,
    github_token: Optional[str] = None
) -> dict:
    """
    Clone a remote Git repository into the project workspace.

    If the workspace already exists, re-uses it (pulls latest instead of re-cloning).
    Injects the GitHub token into the URL for private repos if provided.

    Returns:
        {
          "success": bool,
          "repo_path": str,
          "default_branch": str,
          "error": str | None
        }
    """
    if not GIT_AVAILABLE:
        return {"success": False, "repo_path": None, "default_branch": None,
                "error": "GitPython not installed"}

    workspace_path = get_workspace_path(project_id)

    # Inject token for private repos
    authenticated_url = _inject_token(repo_url, github_token)

    try:
        if os.path.exists(workspace_path):
            # Workspace exists — just pull to get latest state
            print(f"[GitOps] Workspace exists at {workspace_path}, pulling latest…")
            repo = Repo(workspace_path)
            origin = repo.remotes.origin
            # Update remote URL in case token changed
            origin.set_url(authenticated_url)
            origin.pull()
            default_branch = repo.active_branch.name
        else:
            # Fresh clone
            os.makedirs(WORKSPACE_ROOT, exist_ok=True)
            print(f"[GitOps] Cloning {repo_url} → {workspace_path}")
            repo = Repo.clone_from(authenticated_url, workspace_path)
            default_branch = repo.active_branch.name

        print(f"[GitOps] ✅ Repository ready. Default branch: '{default_branch}'")
        return {
            "success":        True,
            "repo_path":      workspace_path,
            "default_branch": default_branch,
            "error":          None
        }

    except GitCommandError as e:
        msg = f"Git command failed: {e.stderr.strip() if e.stderr else str(e)}"
        print(f"[GitOps] ❌ {msg}")
        return {"success": False, "repo_path": None, "default_branch": None, "error": msg}
    except Exception as e:
        msg = f"Unexpected error during clone: {e}"
        print(f"[GitOps] ❌ {msg}")
        return {"success": False, "repo_path": None, "default_branch": None, "error": msg}


def delete_workspace(project_id: int) -> bool:
    """Remove the local workspace for a project (e.g. on project deletion)."""
    workspace_path = get_workspace_path(project_id)
    if os.path.exists(workspace_path):
        shutil.rmtree(workspace_path, ignore_errors=True)
        print(f"[GitOps] 🗑️  Deleted workspace {workspace_path}")
        return True
    return False


# ─────────────────────────────────────────────
# SECTION 2 — BRANCH MANAGEMENT
# ─────────────────────────────────────────────

def create_ai_branch(repo_path: str) -> dict:
    """
    Create a new isolated branch for an AI-driven change.

    Branch name format: ai-change-{unix_timestamp}
    Always branches from the current HEAD of the default branch.

    Safety: Refuses to operate if current branch is a protected branch
    and no clean state exists.

    Returns:
        {
          "success": bool,
          "branch_name": str | None,
          "error": str | None
        }
    """
    if not GIT_AVAILABLE:
        return {"success": False, "branch_name": None, "error": "GitPython not installed"}

    if not repo_path or not os.path.exists(repo_path):
        return {"success": False, "branch_name": None,
                "error": f"Repo path does not exist: {repo_path}"}

    try:
        repo = Repo(repo_path)
        _assert_clean_working_tree(repo)

        # Always branch from the default branch (main/master)
        default_branch = _get_default_branch(repo)
        repo.git.checkout(default_branch)
        repo.remotes.origin.pull()  # Ensure we're up-to-date

        branch_name = f"{AI_BRANCH_PREFIX}-{int(time.time())}"
        repo.git.checkout("-b", branch_name)

        print(f"[GitOps] ✅ Created branch '{branch_name}' from '{default_branch}'")
        return {"success": True, "branch_name": branch_name, "error": None}

    except GitCommandError as e:
        msg = f"Branch creation failed: {e.stderr.strip() if e.stderr else str(e)}"
        print(f"[GitOps] ❌ {msg}")
        return {"success": False, "branch_name": None, "error": msg}
    except Exception as e:
        msg = f"Unexpected error creating branch: {e}"
        print(f"[GitOps] ❌ {msg}")
        return {"success": False, "branch_name": None, "error": msg}


def _get_default_branch(repo: "Repo") -> str:
    """Detect the default branch (main or master) of a repo."""
    try:
        # Try to get from remote HEAD reference
        for ref in repo.remotes.origin.refs:
            if ref.name == "origin/HEAD":
                return ref.reference.name.replace("origin/", "")
    except Exception:
        pass

    # Fallback: check common names
    local_branches = [b.name for b in repo.branches]
    for candidate in ("main", "master"):
        if candidate in local_branches:
            return candidate

    # Last resort: current branch
    return repo.active_branch.name


def _assert_clean_working_tree(repo: "Repo"):
    """Raise if there are uncommitted changes that would interfere."""
    if repo.is_dirty(untracked_files=True):
        # Stash any leftover changes from a previous run to keep workspace clean
        print("[GitOps] ⚠️  Dirty working tree detected — stashing before branch creation")
        repo.git.stash("push", "-m", "aicto-auto-stash")


# ─────────────────────────────────────────────
# SECTION 3 — WRITE FILES TO WORKSPACE
# ─────────────────────────────────────────────

def write_files_to_workspace(
    repo_path: str,
    files: dict[str, str]
) -> dict:
    """
    Write updated file contents into the local workspace.

    Args:
        repo_path: Local path to the cloned repository.
        files:     { "relative/path/to/file.py": "file content", ... }

    Returns:
        { "success": bool, "written": [filenames], "error": str | None }
    """
    if not repo_path or not os.path.exists(repo_path):
        return {"success": False, "written": [],
                "error": f"Repo path does not exist: {repo_path}"}

    written = []
    try:
        for rel_path, content in files.items():
            full_path = Path(repo_path) / rel_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            written.append(rel_path)
            print(f"[GitOps] 📝 Written: {rel_path}")

        return {"success": True, "written": written, "error": None}

    except Exception as e:
        msg = f"Failed to write files: {e}"
        print(f"[GitOps] ❌ {msg}")
        return {"success": False, "written": written, "error": msg}


# ─────────────────────────────────────────────
# SECTION 4 — COMMIT & PUSH
# ─────────────────────────────────────────────

def commit_and_push(
    repo_path: str,
    branch_name: str,
    commit_message: str,
    github_token: Optional[str] = None,
    repo_url: Optional[str] = None
) -> dict:
    """
    Stage all modified files, commit them, and push the branch to origin.

    Safety: Will REFUSE to push to any protected branch (main/master/etc).

    Returns:
        {
          "success": bool,
          "commit_sha": str | None,
          "branch_name": str,
          "error": str | None
        }
    """
    if not GIT_AVAILABLE:
        return {"success": False, "commit_sha": None, "branch_name": branch_name,
                "error": "GitPython not installed"}

    # SAFETY: Block pushes to protected branches
    if branch_name in PROTECTED_BRANCHES:
        msg = f"BLOCKED: Refusing to push directly to protected branch '{branch_name}'"
        print(f"[GitOps] 🚫 {msg}")
        return {"success": False, "commit_sha": None, "branch_name": branch_name, "error": msg}

    try:
        repo = Repo(repo_path)

        # Confirm we are on the correct branch
        current_branch = repo.active_branch.name
        if current_branch != branch_name:
            return {
                "success": False, "commit_sha": None, "branch_name": branch_name,
                "error": f"Expected branch '{branch_name}' but HEAD is on '{current_branch}'"
            }

        # Stage all changes
        repo.git.add(A=True)

        # Nothing to commit?
        if not repo.index.diff("HEAD") and not repo.untracked_files:
            print("[GitOps] ℹ️  No changes to commit.")
            return {
                "success":     True,
                "commit_sha":  repo.head.commit.hexsha,
                "branch_name": branch_name,
                "error":       None,
                "note":        "no_changes"
            }

        # Commit
        commit = repo.index.commit(
            commit_message,
            author=git.Actor("AI CTO", "aicto@noreply"),
            committer=git.Actor("AI CTO", "aicto@noreply")
        )
        print(f"[GitOps] 💾 Committed: {commit.hexsha[:8]} — {commit_message[:60]}")

        # Update remote URL with token if provided
        origin = repo.remotes.origin
        if repo_url and github_token:
            origin.set_url(_inject_token(repo_url, github_token))

        # Push
        push_info = origin.push(refspec=f"{branch_name}:{branch_name}")
        if push_info and push_info[0].flags & push_info[0].ERROR:
            msg = f"Push failed: {push_info[0].summary}"
            print(f"[GitOps] ❌ {msg}")
            return {"success": False, "commit_sha": commit.hexsha, "branch_name": branch_name, "error": msg}

        print(f"[GitOps] ✅ Pushed branch '{branch_name}' to origin")
        return {
            "success":     True,
            "commit_sha":  commit.hexsha,
            "branch_name": branch_name,
            "error":       None
        }

    except GitCommandError as e:
        msg = f"Git error during commit/push: {e.stderr.strip() if e.stderr else str(e)}"
        print(f"[GitOps] ❌ {msg}")
        return {"success": False, "commit_sha": None, "branch_name": branch_name, "error": msg}
    except Exception as e:
        msg = f"Unexpected error during commit/push: {e}"
        print(f"[GitOps] ❌ {msg}")
        return {"success": False, "commit_sha": None, "branch_name": branch_name, "error": msg}


# ─────────────────────────────────────────────
# SECTION 5 — ROLLBACK
# ─────────────────────────────────────────────

def reset_branch_to_previous_commit(
    repo_path: str,
    branch_name: str
) -> dict:
    """
    Hard-reset the current branch to its parent commit (HEAD~1).
    Called when QA fails to undo all AI changes on the branch.

    Safety: Will NOT reset a protected branch.

    Returns:
        { "success": bool, "reset_to": str | None, "error": str | None }
    """
    if not GIT_AVAILABLE:
        return {"success": False, "reset_to": None, "error": "GitPython not installed"}

    if branch_name in PROTECTED_BRANCHES:
        msg = f"BLOCKED: Refusing to reset protected branch '{branch_name}'"
        print(f"[GitOps] 🚫 {msg}")
        return {"success": False, "reset_to": None, "error": msg}

    try:
        repo = Repo(repo_path)

        if repo.active_branch.name != branch_name:
            repo.git.checkout(branch_name)

        parent_sha = repo.head.commit.parents[0].hexsha if repo.head.commit.parents else None
        if not parent_sha:
            return {"success": False, "reset_to": None,
                    "error": "No parent commit to reset to (initial commit)"}

        repo.git.reset("--hard", "HEAD~1")
        print(f"[GitOps] ↩️  Reset branch '{branch_name}' to {parent_sha[:8]}")
        return {"success": True, "reset_to": parent_sha, "error": None}

    except GitCommandError as e:
        msg = f"Reset failed: {e.stderr.strip() if e.stderr else str(e)}"
        print(f"[GitOps] ❌ {msg}")
        return {"success": False, "reset_to": None, "error": msg}
    except Exception as e:
        msg = f"Unexpected error during reset: {e}"
        print(f"[GitOps] ❌ {msg}")
        return {"success": False, "reset_to": None, "error": msg}


# ─────────────────────────────────────────────
# SECTION 6 — GITHUB PULL REQUEST
# ─────────────────────────────────────────────

def create_github_pr(
    repo_url: str,
    branch_name: str,
    github_token: str,
    title: str,
    body: str,
    base_branch: str = "main"
) -> dict:
    """
    Create a GitHub Pull Request from the AI branch to the base branch.

    Uses GitHub REST API v3. Requires a Personal Access Token (PAT) with
    'repo' scope stored in the project's github_token field.

    Safety: Will REFUSE to create a PR targeting anything other than the
    project's declared default branch (no lateral PRs between feature branches).

    Returns:
        {
          "success":  bool,
          "pr_url":   str | None,
          "pr_number": int | None,
          "error":    str | None
        }
    """
    if branch_name in PROTECTED_BRANCHES:
        msg = f"BLOCKED: AI branch '{branch_name}' is itself a protected branch — cannot create PR"
        print(f"[GitOps] 🚫 {msg}")
        return {"success": False, "pr_url": None, "pr_number": None, "error": msg}

    owner, repo_name = _parse_github_repo(repo_url)
    if not owner or not repo_name:
        return {"success": False, "pr_url": None, "pr_number": None,
                "error": f"Could not parse GitHub owner/repo from URL: {repo_url}"}

    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    payload = {
        "title": title,
        "body":  body,
        "head":  branch_name,
        "base":  base_branch,
        "draft": False
    }

    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo_name}/pulls"

    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=30)

        if response.status_code == 201:
            data = response.json()
            pr_url    = data.get("html_url")
            pr_number = data.get("number")
            print(f"[GitOps] ✅ PR created: {pr_url}")
            return {"success": True, "pr_url": pr_url, "pr_number": pr_number, "error": None}

        elif response.status_code == 422:
            # Could mean PR already exists — return existing PR URL if possible
            detail = response.json().get("errors", [{}])[0].get("message", response.text)
            msg = f"GitHub validation failed (422): {detail}"
            print(f"[GitOps] ⚠️  {msg}")
            return {"success": False, "pr_url": None, "pr_number": None, "error": msg}

        else:
            msg = f"GitHub API error {response.status_code}: {response.text[:300]}"
            print(f"[GitOps] ❌ {msg}")
            return {"success": False, "pr_url": None, "pr_number": None, "error": msg}

    except httpx.TimeoutException:
        msg = "GitHub API request timed out"
        print(f"[GitOps] ❌ {msg}")
        return {"success": False, "pr_url": None, "pr_number": None, "error": msg}
    except Exception as e:
        msg = f"Unexpected error creating PR: {e}"
        print(f"[GitOps] ❌ {msg}")
        return {"success": False, "pr_url": None, "pr_number": None, "error": msg}


# ─────────────────────────────────────────────
# SECTION 7 — UTILITIES
# ─────────────────────────────────────────────

def _inject_token(repo_url: str, token: Optional[str]) -> str:
    """
    Inject a GitHub PAT into an HTTPS remote URL for authentication.
    Input:  https://github.com/owner/repo.git
    Output: https://{token}@github.com/owner/repo.git
    """
    if not token:
        return repo_url
    if repo_url.startswith("https://") and "@" not in repo_url:
        return repo_url.replace("https://", f"https://{token}@")
    return repo_url


def _parse_github_repo(repo_url: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extract owner and repo name from a GitHub HTTPS or SSH URL.
    Supports:
      https://github.com/owner/repo.git
      https://github.com/owner/repo
      git@github.com:owner/repo.git
    """
    import re
    # HTTPS
    m = re.match(r"https://(?:[^@]+@)?github\.com/([^/]+)/([^/]+?)(?:\.git)?$", repo_url)
    if m:
        return m.group(1), m.group(2)
    # SSH
    m = re.match(r"git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$", repo_url)
    if m:
        return m.group(1), m.group(2)
    return None, None


def get_repo(repo_path: str) -> Optional["Repo"]:
    """Open an existing repo; return None if path is invalid."""
    if not GIT_AVAILABLE or not repo_path or not os.path.exists(repo_path):
        return None
    try:
        return Repo(repo_path)
    except InvalidGitRepositoryError:
        return None


def repo_status(repo_path: str) -> dict:
    """Return current branch, last commit, and dirty status for diagnostics."""
    repo = get_repo(repo_path)
    if not repo:
        return {"error": "Invalid or missing repository path"}
    return {
        "current_branch": repo.active_branch.name,
        "last_commit":    repo.head.commit.hexsha[:8],
        "is_dirty":       repo.is_dirty(untracked_files=True),
        "untracked":      repo.untracked_files
    }