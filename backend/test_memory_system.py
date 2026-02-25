"""
test_memory_system.py — Full Test Suite for Long-Term Project Memory
=====================================================================
Run with:
  pytest test_memory_system.py -v

Or directly:
  python test_memory_system.py

Environment variables (optional):
  BASE_URL=http://localhost:8000
"""

import os
import sys
import time
import requests
import pytest

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

SAMPLE_PYTHON = """\
def greet(name: str) -> str:
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b
"""


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def auth(key: str) -> dict:
    return {"X-API-Key": key}


def create_project(name: str = "mem-test") -> tuple:
    r = requests.post(f"{BASE_URL}/create-project/", params={"name": name}, timeout=10)
    r.raise_for_status()
    d = r.json()
    return d["project_id"], d["api_key"]


def upload_file(project_id: int, api_key: str, filename: str = "sample.py") -> dict:
    r = requests.post(
        f"{BASE_URL}/upload-file/",
        headers=auth(api_key),
        data={"project_id": str(project_id)},
        files={"file": (filename, SAMPLE_PYTHON.encode(), "text/plain")},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def safe_json(response: requests.Response) -> dict:
    try:
        return response.json()
    except Exception:
        return {"_raw": response.text[:300], "_status": response.status_code}


# ─────────────────────────────────────────────
# PYTEST FIXTURES
# session-scoped = created ONCE per pytest run
# ─────────────────────────────────────────────

@pytest.fixture(scope="session")
def server():
    """Verify the server is reachable before any tests run."""
    try:
        requests.get(BASE_URL, timeout=5)
    except Exception:
        pytest.skip(f"Server not reachable at {BASE_URL} — start with: uvicorn main:app --reload")


@pytest.fixture(scope="session")
def project(server):
    """
    Create one project + upload one file, shared across the whole test session.
    Returns dict with project_id, api_key, file_id.
    """
    pid, key = create_project("pytest-memory-suite")
    up = upload_file(pid, key)
    return {"project_id": pid, "api_key": key, "file_id": up.get("file_id")}


@pytest.fixture(scope="session")
def fresh_project(server):
    """
    A separate isolated project with no memories, used for safety tests.
    """
    pid, key = create_project("pytest-empty-project")
    upload_file(pid, key, "empty_test.py")
    return {"project_id": pid, "api_key": key}


# ─────────────────────────────────────────────
# SECTION 1 — DB SCHEMA
# ─────────────────────────────────────────────

class TestDBSchema:

    def test_table_exists(self):
        psycopg2 = pytest.importorskip("psycopg2")
        conn = psycopg2.connect(
            host="127.0.0.1", port=5433,
            dbname="aicto", user="postgres", password="postgres"
        )
        cur = conn.cursor()
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'project_memory'
            )
        """)
        exists = cur.fetchone()[0]
        cur.close()
        conn.close()
        assert exists, (
            "project_memory table does NOT exist.\n"
            "Run the migration first:\n"
            "  psql -U postgres -d aicto -p 5433 -f migrate_add_project_memory.sql"
        )

    def test_required_columns_exist(self):
        psycopg2 = pytest.importorskip("psycopg2")
        conn = psycopg2.connect(
            host="127.0.0.1", port=5433,
            dbname="aicto", user="postgres", password="postgres"
        )
        cur = conn.cursor()
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'project_memory'
        """)
        cols = {r[0] for r in cur.fetchall()}
        cur.close()
        conn.close()
        for expected in ["id", "project_id", "memory_type", "content", "embedding", "created_at"]:
            assert expected in cols, f"Missing column: '{expected}'"

    def test_indexes_exist(self):
        psycopg2 = pytest.importorskip("psycopg2")
        conn = psycopg2.connect(
            host="127.0.0.1", port=5433,
            dbname="aicto", user="postgres", password="postgres"
        )
        cur = conn.cursor()
        cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'project_memory'")
        indexes = [r[0] for r in cur.fetchall()]
        cur.close()
        conn.close()
        assert len(indexes) >= 1, f"Expected at least 1 index on project_memory, found: {indexes}"


# ─────────────────────────────────────────────
# SECTION 2 — memory_engine.py UNIT TESTS
# ─────────────────────────────────────────────

class TestMemoryEngineUnits:

    def test_format_empty_returns_empty_string(self):
        from memory_engine import format_memories_for_prompt
        assert format_memories_for_prompt([]) == ""

    def test_format_includes_type_tags(self):
        from memory_engine import format_memories_for_prompt
        memories = [
            {"id": 1, "memory_type": "architecture", "content": "Use Clean Architecture", "created_at": ""},
            {"id": 2, "memory_type": "decision",     "content": "Added docstring",        "created_at": ""},
        ]
        out = format_memories_for_prompt(memories)
        assert "[ARCHITECTURE]" in out
        assert "[DECISION]" in out
        assert "Use Clean Architecture" in out

    def test_build_decision_memory_has_required_keys(self):
        from memory_engine import build_decision_memory
        results = [{"output": "Modified 'sample.py' successfully (fixed in 2 attempt(s))", "success": True}]
        out = build_decision_memory("Add docstring to greet", "Improve greet docs", 1, results)
        assert len(out) > 0
        assert "INSTRUCTION" in out
        assert "FILES" in out

    def test_build_decision_memory_under_1000_chars(self):
        from memory_engine import build_decision_memory
        out = build_decision_memory("x" * 500, "y" * 300, 5, [])
        assert len(out) <= 1000, f"Expected ≤1000 chars, got {len(out)}"

    def test_build_decision_memory_no_crash_on_empty_results(self):
        from memory_engine import build_decision_memory
        out = build_decision_memory("instruction", "goal", 0, [])
        assert isinstance(out, str)


# ─────────────────────────────────────────────
# SECTION 3 — POST /add-architecture-memory/
# ─────────────────────────────────────────────

class TestAddArchitectureMemory:

    def test_happy_path(self, project):
        r = requests.post(
            f"{BASE_URL}/add-architecture-memory/",
            headers=auth(project["api_key"]),
            params={"project_id": project["project_id"], "note": "Never modify public API signatures"},
            timeout=10,
        )
        assert r.status_code == 200, f"Got {r.status_code}: {r.text[:200]}"
        d = r.json()
        assert "memory_id" in d, f"Missing memory_id in: {d}"
        assert isinstance(d["memory_id"], int)

    def test_second_note_accepted(self, project):
        r = requests.post(
            f"{BASE_URL}/add-architecture-memory/",
            headers=auth(project["api_key"]),
            params={"project_id": project["project_id"], "note": "Always use snake_case for functions"},
            timeout=10,
        )
        assert r.status_code == 200, f"Got {r.status_code}: {r.text[:200]}"

    def test_long_note_truncated_and_accepted(self, project):
        r = requests.post(
            f"{BASE_URL}/add-architecture-memory/",
            headers=auth(project["api_key"]),
            params={"project_id": project["project_id"], "note": "A" * 2000},
            timeout=10,
        )
        assert r.status_code == 200, f"Long note rejected unexpectedly: {r.text[:200]}"

    def test_missing_api_key_returns_401(self, project):
        r = requests.post(
            f"{BASE_URL}/add-architecture-memory/",
            params={"project_id": project["project_id"], "note": "should fail"},
            timeout=10,
        )
        assert r.status_code == 401

    def test_wrong_api_key_returns_403(self, project):
        r = requests.post(
            f"{BASE_URL}/add-architecture-memory/",
            headers={"X-API-Key": "aicto_totallyinvalidkey"},
            params={"project_id": project["project_id"], "note": "should fail"},
            timeout=10,
        )
        assert r.status_code == 403

    def test_cross_project_access_denied(self, project):
        other_pid, other_key = create_project("cross-project-attacker")
        r = requests.post(
            f"{BASE_URL}/add-architecture-memory/",
            headers={"X-API-Key": other_key},
            params={"project_id": project["project_id"], "note": "cross-project attack"},
            timeout=10,
        )
        assert r.status_code == 403, f"Expected 403, got {r.status_code}"

    def test_blank_note_returns_400(self, project):
        r = requests.post(
            f"{BASE_URL}/add-architecture-memory/",
            headers=auth(project["api_key"]),
            params={"project_id": project["project_id"], "note": "   "},
            timeout=10,
        )
        assert r.status_code == 400, f"Expected 400 for blank note, got {r.status_code}: {r.text[:100]}"

    def test_special_characters_accepted(self, project):
        r = requests.post(
            f"{BASE_URL}/add-architecture-memory/",
            headers=auth(project["api_key"]),
            params={"project_id": project["project_id"],
                    "note": "UTF-8 required: ñ, ü, 中文, العربية"},
            timeout=10,
        )
        assert r.status_code == 200, f"Got {r.status_code}: {r.text[:200]}"


# ─────────────────────────────────────────────
# SECTION 4 — GET /project-memory/
# ─────────────────────────────────────────────

class TestGetProjectMemory:

    def test_returns_200_with_correct_shape(self, project):
        r = requests.get(
            f"{BASE_URL}/project-memory/",
            headers=auth(project["api_key"]),
            params={"project_id": project["project_id"]},
            timeout=10,
        )
        assert r.status_code == 200, f"Got {r.status_code}: {r.text[:200]}"
        d = r.json()
        assert "memories" in d
        assert "total" in d

    def test_at_least_one_memory_present(self, project):
        r = requests.get(
            f"{BASE_URL}/project-memory/",
            headers=auth(project["api_key"]),
            params={"project_id": project["project_id"]},
            timeout=10,
        )
        assert r.json().get("total", 0) >= 1, "Expected at least 1 memory from earlier tests"

    def test_memory_entry_has_required_fields(self, project):
        r = requests.get(
            f"{BASE_URL}/project-memory/",
            headers=auth(project["api_key"]),
            params={"project_id": project["project_id"]},
            timeout=10,
        )
        memories = r.json().get("memories", [])
        assert len(memories) > 0, "No memories to inspect"
        for field in ["id", "memory_type", "content", "created_at"]:
            assert field in memories[0], f"Missing field: '{field}'"

    def test_filter_by_architecture_type(self, project):
        r = requests.get(
            f"{BASE_URL}/project-memory/",
            headers=auth(project["api_key"]),
            params={"project_id": project["project_id"], "memory_type": "architecture"},
            timeout=10,
        )
        assert r.status_code == 200
        for m in r.json().get("memories", []):
            assert m["memory_type"] == "architecture", f"Wrong type returned: {m['memory_type']}"

    def test_filter_by_decision_no_crash(self, project):
        r = requests.get(
            f"{BASE_URL}/project-memory/",
            headers=auth(project["api_key"]),
            params={"project_id": project["project_id"], "memory_type": "decision"},
            timeout=10,
        )
        assert r.status_code == 200

    def test_missing_key_returns_401(self, project):
        r = requests.get(
            f"{BASE_URL}/project-memory/",
            params={"project_id": project["project_id"]},
            timeout=10,
        )
        assert r.status_code == 401


# ─────────────────────────────────────────────
# SECTION 5 — DECISION MEMORY AUTO-CREATION
# ─────────────────────────────────────────────

class TestDecisionMemoryAutoCreation:

    def test_decision_memory_created_after_modify(self, project):
        pid = project["project_id"]
        key = project["api_key"]

        r_before = requests.get(
            f"{BASE_URL}/project-memory/",
            headers=auth(key),
            params={"project_id": pid, "memory_type": "decision"},
            timeout=10,
        )
        before = r_before.json().get("total", 0)

        r_mod = requests.post(
            f"{BASE_URL}/modify-code/",
            headers=auth(key),
            params={"project_id": pid, "query": "add a docstring to the greet function"},
            timeout=60,
        )
        assert r_mod.status_code == 200, f"modify-code failed: {r_mod.text[:300]}"

        time.sleep(1)

        r_after = requests.get(
            f"{BASE_URL}/project-memory/",
            headers=auth(key),
            params={"project_id": pid, "memory_type": "decision"},
            timeout=10,
        )
        after = r_after.json().get("total", 0)
        assert after > before, f"Decision memory did not increase (before={before}, after={after})"

    def test_decision_memory_content_is_structured(self, project):
        r = requests.get(
            f"{BASE_URL}/project-memory/",
            headers=auth(project["api_key"]),
            params={"project_id": project["project_id"], "memory_type": "decision"},
            timeout=10,
        )
        memories = r.json().get("memories", [])
        assert len(memories) > 0, "No decision memories found — run test_decision_memory_created_after_modify first"
        content = memories[0]["content"]
        assert "INSTRUCTION" in content, f"Missing INSTRUCTION in: {content[:200]}"
        assert "FILES" in content,       f"Missing FILES in: {content[:200]}"
        assert len(content) <= 1000,     f"Content too long: {len(content)} chars"


# ─────────────────────────────────────────────
# SECTION 6 — EMPTY MEMORY SAFETY
# ─────────────────────────────────────────────

class TestEmptyMemorySafety:

    def test_modify_code_works_with_zero_memories(self, fresh_project):
        r = requests.post(
            f"{BASE_URL}/modify-code/",
            headers=auth(fresh_project["api_key"]),
            params={"project_id": fresh_project["project_id"],
                    "query": "add type hints to the add function"},
            timeout=60,
        )
        assert r.status_code == 200, f"Got {r.status_code}: {r.text[:300]}"

    def test_get_memory_on_empty_project_returns_200(self, fresh_project):
        r = requests.get(
            f"{BASE_URL}/project-memory/",
            headers=auth(fresh_project["api_key"]),
            params={"project_id": fresh_project["project_id"]},
            timeout=10,
        )
        assert r.status_code == 200, f"Got {r.status_code}: {r.text[:300]}"

    def test_get_memory_returns_memories_key(self, fresh_project):
        r = requests.get(
            f"{BASE_URL}/project-memory/",
            headers=auth(fresh_project["api_key"]),
            params={"project_id": fresh_project["project_id"]},
            timeout=10,
        )
        d = safe_json(r)
        assert "memories" in d, f"Expected 'memories' key in response, got: {d}"


# ─────────────────────────────────────────────
# SECTION 7 — BACKWARD COMPATIBILITY
# ─────────────────────────────────────────────

class TestBackwardCompatibility:

    def test_list_files(self, project):
        r = requests.get(
            f"{BASE_URL}/list-files/",
            headers=auth(project["api_key"]),
            params={"project_id": project["project_id"]},
            timeout=10,
        )
        assert r.status_code == 200

    def test_quality_project(self, project):
        r = requests.get(
            f"{BASE_URL}/quality/project/",
            headers=auth(project["api_key"]),
            params={"project_id": project["project_id"]},
            timeout=10,
        )
        assert r.status_code == 200

    def test_dependency_graph(self, project):
        r = requests.get(
            f"{BASE_URL}/dependency-graph/",
            headers=auth(project["api_key"]),
            params={"project_id": project["project_id"]},
            timeout=10,
        )
        assert r.status_code == 200

    def test_get_architecture(self, project):
        r = requests.get(
            f"{BASE_URL}/get-architecture/",
            params={"project_id": project["project_id"]},
            timeout=10,
        )
        assert r.status_code == 200

    def test_rate_limit_status(self, project):
        r = requests.get(
            f"{BASE_URL}/rate-limit-status/",
            headers=auth(project["api_key"]),
            timeout=10,
        )
        assert r.status_code == 200

    def test_ask_endpoint_returns_answer(self, project):
        r = requests.post(
            f"{BASE_URL}/ask/",
            headers=auth(project["api_key"]),
            params={"project_id": project["project_id"],
                    "query": "what does the greet function do?"},
            timeout=30,
        )
        assert r.status_code == 200
        assert "answer" in r.json()

    def test_history_endpoint(self, project):
        file_id = project.get("file_id")
        if not file_id:
            pytest.skip("No file_id in project fixture")
        r = requests.get(
            f"{BASE_URL}/history/{file_id}",
            headers=auth(project["api_key"]),
            timeout=10,
        )
        assert r.status_code == 200
        assert "versions" in r.json()


# ─────────────────────────────────────────────
# SECTION 8 — MEMORY RETRIEVAL RELEVANCE
# ─────────────────────────────────────────────

class TestMemoryRetrievalRelevance:

    def test_relevant_memory_for_database_query(self):
        pytest.importorskip("sentence_transformers")
        from memory_engine import store_memory, retrieve_relevant_memory
        from sentence_transformers import SentenceTransformer
        from database import SessionLocal
        from sqlalchemy import text

        model = SentenceTransformer("BAAI/bge-small-en")
        db = SessionLocal()
        try:
            result = db.execute(
                text("INSERT INTO projects (name) VALUES ('relevance-test') RETURNING id")
            )
            db.commit()
            pid = result.fetchone()[0]

            store_memory(pid, "architecture", "Database layer must remain isolated from business logic", model, db)
            store_memory(pid, "architecture", "All API endpoints must validate input with Pydantic", model, db)

            hits = retrieve_relevant_memory(pid, "database isolation layer", model, db, top_k=1)
            assert len(hits) == 1
            assert "database" in hits[0]["content"].lower(), f"Expected DB-related result, got: {hits[0]['content']}"
        finally:
            db.close()

    def test_top_k_limit_respected(self):
        pytest.importorskip("sentence_transformers")
        from memory_engine import store_memory, retrieve_relevant_memory
        from sentence_transformers import SentenceTransformer
        from database import SessionLocal
        from sqlalchemy import text

        model = SentenceTransformer("BAAI/bge-small-en")
        db = SessionLocal()
        try:
            result = db.execute(
                text("INSERT INTO projects (name) VALUES ('topk-test') RETURNING id")
            )
            db.commit()
            pid = result.fetchone()[0]

            for i in range(5):
                store_memory(pid, "architecture", f"Rule {i}: keep things simple", model, db)

            hits = retrieve_relevant_memory(pid, "simple rules", model, db, top_k=3)
            assert len(hits) <= 3, f"Expected ≤3, got {len(hits)}"
        finally:
            db.close()

    def test_empty_project_returns_empty_list(self):
        pytest.importorskip("sentence_transformers")
        from memory_engine import retrieve_relevant_memory
        from sentence_transformers import SentenceTransformer
        from database import SessionLocal
        from sqlalchemy import text

        model = SentenceTransformer("BAAI/bge-small-en")
        db = SessionLocal()
        try:
            result = db.execute(
                text("INSERT INTO projects (name) VALUES ('empty-retrieval') RETURNING id")
            )
            db.commit()
            pid = result.fetchone()[0]

            hits = retrieve_relevant_memory(pid, "anything", model, db, top_k=5)
            assert isinstance(hits, list)
            assert len(hits) == 0
        finally:
            db.close()


# ─────────────────────────────────────────────
# SECTION 9 — ROLLBACK STILL WORKS
# ─────────────────────────────────────────────

class TestRollbackStillWorks:

    def test_history_endpoint_accessible(self, project):
        file_id = project.get("file_id")
        if not file_id:
            pytest.skip("No file_id available")
        r = requests.get(
            f"{BASE_URL}/history/{file_id}",
            headers=auth(project["api_key"]),
            timeout=10,
        )
        assert r.status_code == 200
        assert "versions" in r.json()

    def test_rollback_endpoint_responds(self, project):
        file_id = project.get("file_id")
        if not file_id:
            pytest.skip("No file_id available")

        r_hist = requests.get(
            f"{BASE_URL}/history/{file_id}",
            headers=auth(project["api_key"]),
            timeout=10,
        )
        if not r_hist.json().get("versions"):
            pytest.skip("No saved versions to roll back to yet")

        r = requests.post(
            f"{BASE_URL}/rollback/",
            headers=auth(project["api_key"]),
            params={"file_id": file_id},
            timeout=30,
        )
        assert r.status_code == 200, f"Rollback failed: {r.text[:200]}"
        assert "message" in r.json()


# ─────────────────────────────────────────────
# DIRECT RUN ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))