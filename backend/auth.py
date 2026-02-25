"""
auth.py — API Key Authentication Layer
=======================================
Drop this file next to main.py.

How it works:
  - Every project gets a unique API key on creation
  - All sensitive endpoints require: Header `X-API-Key: <key>`
  - The key is validated against the project it claims to own
  - Wrong key = 403. Missing key = 401. Correct key = access granted.

Integration steps (3 changes to main.py):
  1. Add to imports:     from auth import require_api_key, create_api_key
  2. Replace /create-project/ with the version below
  3. Add `api_key: str = Depends(require_api_key)` to every protected endpoint
"""

import secrets
import hashlib
from fastapi import Header, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import SessionLocal
from typing import Optional

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────

def _get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def generate_api_key() -> str:
    """Generate a secure random API key — e.g. 'aicto_a3f9bc2d...'"""
    return "aicto_" + secrets.token_hex(24)


def hash_key(raw_key: str) -> str:
    """We store only the hash, never the raw key."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


# ─────────────────────────────────────────────
# CREATE + STORE A KEY FOR A PROJECT
# ─────────────────────────────────────────────

def create_api_key(project_id: int, db: Session) -> str:
    """
    Generate a new API key for a project, store its hash,
    and return the raw key (shown to user ONCE, then gone).
    """
    raw_key    = generate_api_key()
    hashed_key = hash_key(raw_key)

    db.execute(
        text("""
            INSERT INTO api_keys (project_id, key_hash, is_active)
            VALUES (:project_id, :key_hash, true)
            ON CONFLICT (project_id)
            DO UPDATE SET key_hash = :key_hash, is_active = true
        """),
        {"project_id": project_id, "key_hash": hashed_key}
    )
    db.commit()

    return raw_key


# ─────────────────────────────────────────────
# FASTAPI DEPENDENCY — use on any endpoint
# ─────────────────────────────────────────────

# REPLACE require_api_key in auth.py:
def require_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: Session = Depends(_get_db)
) -> int:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    
    hashed = hash_key(x_api_key)
    row = db.execute(
        text("SELECT project_id FROM api_keys WHERE key_hash = :key_hash AND is_active = true"),
        {"key_hash": hashed}
    ).fetchone()

    if not row:
        raise HTTPException(status_code=403, detail="Invalid or inactive API key")
    return row[0]


def verify_project_access(claimed_project_id: int, key_project_id: int):
    """
    Call this inside any endpoint that takes a project_id param
    to ensure the key actually owns that project.
    """
    if claimed_project_id != key_project_id:
        raise HTTPException(
            status_code=403,
            detail=f"This API key does not have access to project {claimed_project_id}"
        )