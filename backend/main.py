import os
from dotenv import load_dotenv
load_dotenv()
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, Project, File as DBFile
import re
from models import CodeChunk
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from groq import Groq
import json
import html
import ast
from agent_executor import run_agent
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from auth import require_api_key, create_api_key, verify_project_access
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Header
from typing import Optional
from production_safety import (
    check_rate_limit, get_rate_limit_status,
    sanitize_query, sanitize_code, validate_filename,
    MODIFY_LIMIT, MODIFY_WINDOW, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW
)
from code_quality import analyze_quality, analyze_project_quality
from architectural_awareness import build_dependency_graph
from test_engine import find_impacted_tests, run_tests, store_test
from multi_agent import orchestrate, execute_goal, get_pending_approvals, resolve_approval
from enterprise_governance import (
    UserRole, RoleContext,
    assign_role, get_role, resolve_role_context, list_project_roles,
    write_audit_log, get_audit_log,
    approve_change, reject_change,
    build_enterprise_pr_body,
    compute_enterprise_risk_score,
)
from autonomous_ops import setup_scheduler
from database import DATABASE_URL

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


model = SentenceTransformer("BAAI/bge-small-en")  # Free local embedding


from infra import build_groq_router
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
_router     = build_groq_router(groq_client, model_primary="llama-3.1-8b-instant")
groq_llm    = _router.call   # same (system_prompt, user_prompt) -> str interface


def chunk_code(content: str):
    chunks = re.split(r"\n(?=def |class )", content)
    if not chunks:
        chunks = [content]
    return chunks

@app.get("/llm-status/")
def llm_status(key_project_id: int = Depends(require_api_key)):
    return _router.status()

@app.post("/ask/")
def ask(
    query: str,
    project_id: int,
    db: Session = Depends(get_db),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    key_project_id: int = Depends(require_api_key)
):
    verify_project_access(project_id, key_project_id)
    check_rate_limit(x_api_key, db, endpoint="ask")

    query, _ = sanitize_query(query)

    query_embedding   = model.encode(query).tolist()
    relevant_file_ids = retrieve_relevant_files(query, project_id, db)

    if not relevant_file_ids:
        return {"answer": "No relevant files found in project."}

    sql = text("""
        SELECT content FROM code_chunks
        WHERE file_id = ANY(:file_ids)
        ORDER BY embedding <-> CAST(:embedding AS vector)
        LIMIT 5;
    """)
    results = db.execute(sql, {"embedding": query_embedding, "file_ids": relevant_file_ids}).fetchall()
    context = "\n\n".join([row[0] for row in results])

    system_prompt = "You are an expert AI CTO. Use ONLY the provided code context."
    user_prompt   = f"Context:\n{context}\n\nQuestion:\n{query}"
    raw_output    = groq_llm(system_prompt, user_prompt)

    return {"answer": raw_output}



# @app.post("/modify-code/")
# def modify_code(query: str, project_id: int, db: Session = Depends(get_db)):

#     # 🔥 Detect file name from query
#     project_files = db.query(DBFile).filter(DBFile.project_id == project_id).all()

#     file_to_modify = None

#     for file in project_files:
#         if file.filename.lower() in query.lower():
#             file_to_modify = file
#             break

#     # If no file explicitly mentioned → fallback to latest file
#     if not file_to_modify:
#         file_to_modify = (
#             db.query(DBFile)
#             .filter(DBFile.project_id == project_id)
#             .order_by(DBFile.id.desc())
#             .first()
#         )

#     if not file_to_modify:
#         return {"error": "No file found in project"}
#     # 🔐 Ensure file is syntactically valid before modifying
#     try:
#         ast.parse(file_to_modify.content)
#     except SyntaxError:
#         return {
#             "error": "Current file is syntactically invalid. Restore previous version before modifying."
#         }

#     # 🔎 Detect target function
#     functions = db.execute(
#         text("""
#             SELECT name FROM architectural_elements
#             WHERE file_id = :file_id
#             AND element_type IN ('function', 'method')
#         """),
#         {"file_id": file_to_modify.id}
#     ).fetchall()

#     function_names = [f[0] for f in functions]

#     target_function = None
#     for name in function_names:
#         if name.lower() in query.lower():
#             target_function = name
#             break
    



#     # 🧠 Extract function source if target detected
#     function_source = None
#     target_node = None

#     if target_function:
#         try:
#             tree = ast.parse(file_to_modify.content)

#             target_node = None

#             for node in ast.walk(tree):
#                 if isinstance(node, ast.FunctionDef):

#                     # match plain function
#                     if node.name == target_function:
#                         target_node = node
#                         function_source = ast.get_source_segment(file_to_modify.content, node)
#                         break

#                     # match method like Calculator.multiply
#                     if "." in target_function and node.name == target_function.split(".")[1]:
#                         target_node = node
#                         function_source = ast.get_source_segment(file_to_modify.content, node)
#                         break


#         except SyntaxError:
#             function_source = None

#     # 1️⃣ Embed query
#     query_embedding = model.encode(query).tolist()

#     # 2️⃣ Retrieve relevant code
#     # 2️⃣ Retrieve relevant files first
#     relevant_file_ids = retrieve_relevant_files(query, project_id, db)

#     if not relevant_file_ids:
#         return {"error": "No relevant files found."}

#     # 3️⃣ Retrieve chunks only from relevant files
#     sql = text("""
#         SELECT content
#         FROM code_chunks
#         WHERE file_id = ANY(:file_ids)
#         ORDER BY embedding <-> CAST(:embedding AS vector)
#         LIMIT 5;
#     """)

#     results = db.execute(
#         sql,
#         {
#             "embedding": query_embedding,
#             "file_ids": relevant_file_ids
#         }
#     ).fetchall()

#     context = "\n\n".join([row[0] for row in results])

#     decision = decide_action(query, context)

#     if decision["action"] == "explain":
#         return {"explanation": decision}

#     plan = generate_plan(query, context)
#     print("Execution Plan:\n", plan)    

#     # 3️⃣ Ask GPT to generate updated code
#     if target_function and function_source:
#         user_prompt = f"""
# Modify ONLY this function:

# {function_source}

# Instruction:
# {query}

# Return ONLY the updated function definition.
# """
#     else:
#         user_prompt = f"""
# Existing context:
# {context}

# Instruction:
# {query}

# Return ONLY valid Python code.
# """

#     system_prompt = "You are an expert AI CTO. Return ONLY valid Python code."

#     raw_output = groq_llm(system_prompt, user_prompt)
#     generated_code = raw_output.strip()

#     generated_code = re.sub(r"```.*?\n", "", generated_code)
#     generated_code = generated_code.replace("```", "").strip()


#     # 5️⃣ Save old version
#     db.execute(
#         text("""
#             INSERT INTO file_versions (file_id, content)
#             VALUES (:file_id, :content)
#         """),
#         {"file_id": file_to_modify.id, "content": file_to_modify.content}
#     )

#     # 6️⃣ Replace only target function if detected
#     # 6️⃣ Replace only target function if detected
#     if function_source and target_function and target_node:
#         start_line = target_node.lineno - 1
#         end_line = target_node.end_lineno

#         lines = file_to_modify.content.splitlines()

#         # Detect original indentation
#         original_line = lines[start_line]
#         indent = original_line[:len(original_line) - len(original_line.lstrip())]

#         # Normalize GPT indentation
#         generated_lines = generated_code.splitlines()

#         min_indent = None
#         for line in generated_lines:
#             stripped = line.lstrip()
#             if stripped:
#                 current_indent = len(line) - len(stripped)
#                 if min_indent is None or current_indent < min_indent:
#                     min_indent = current_indent

#         if min_indent is None:
#             min_indent = 0

#         normalized = [line[min_indent:] for line in generated_lines]

#         # Apply correct indent
#         indented_generated = [
#             indent + line if line.strip() else line
#             for line in normalized
#         ]

#         # 🔥 REBUILD FILE SAFELY
#         updated_lines = (
#             lines[:start_line] +
#             indented_generated +
#             lines[end_line:]
#         )

#         updated_file_content = "\n".join(updated_lines)

#     else:
#         updated_file_content = generated_code

#     # 🔥 SELF-HEALING SYNTAX CHECK
#     try:
#         ast.parse(updated_file_content)

#     except SyntaxError as e:
#         fix_prompt = f"""
#     The following code has syntax error:

#     {updated_file_content}

#     Error:
#     {e}

#     Fix it and return corrected Python only.
#     """

#         fixed = groq_llm("Fix syntax errors only.", fix_prompt)

#         # Clean markdown if needed
#         fixed = fixed.strip()
#         fixed = re.sub(r"```.*?\n", "", fixed)
#         fixed = fixed.replace("```", "").strip()

#         # Try parsing again
#         try:
#             ast.parse(fixed)
#             updated_file_content = fixed
#         except SyntaxError:
#             return {"error": "AI could not auto-fix syntax error."}

#     # ✅ Now safe to save
#     file_to_modify.content = updated_file_content
#     db.commit()


#     # Clear old architecture entries
#     db.execute(
#         text("DELETE FROM architectural_elements WHERE file_id = :file_id"),
#         {"file_id": file_to_modify.id}
#     )
#     db.commit()

#     # Re-extract structure
#     extract_python_structure(file_to_modify.id, updated_file_content, db)
#     extract_dependencies(file_to_modify.id, updated_file_content, db)

#     # 7️⃣ Delete old chunks
#     db.execute(
#         text("DELETE FROM code_chunks WHERE file_id = :file_id"),
#         {"file_id": file_to_modify.id}
#     )
#     db.commit()

#     # 8️⃣ Re-chunk and re-embed
#     chunks = chunk_code(updated_file_content)

#     for chunk in chunks:
#         embedding = model.encode(chunk).tolist()

#         db_chunk = CodeChunk(
#             file_id=file_to_modify.id,
#             content=chunk,
#             embedding=embedding
#         )
#         db.add(db_chunk)

#     db.commit()


    

#     return {
#         "message": "Code modified and stored successfully",
#         "generated_code": generated_code
#     }

# def decide_action(query, context):
#     system_prompt = """
# Return strictly JSON:

# {
#   "action": "modify_function" | "create_file" | "explain",
#   "target_file": "filename if mentioned else null",
#   "target_function": "function name if mentioned else null"
# }
# """

#     user_prompt = f"""
# Context:
# {context}

# Instruction:
# {query}
# """

#     response = groq_llm(system_prompt, user_prompt)

#     try:
#         return json.loads(response)
#     except:
#         return {"action": "modify_function"}

def generate_file_summary(content: str):
    system_prompt = """
You are an expert software architect.

Summarize this file in 5-8 concise technical lines.
Mention:
- Main responsibility
- Key classes
- Key functions
- External dependencies
Return plain text only.
"""

    user_prompt = f"""
Code:
{content}
"""

    return groq_llm(system_prompt, user_prompt)

# def generate_plan(query, context):
#     system_prompt = """
# You are an AI CTO.

# Create a clear step-by-step technical plan to implement the requested change.
# Return numbered steps only.
# """

#     user_prompt = f"""
# Context:
# {context}

# Instruction:
# {query}
# """

#     return groq_llm(system_prompt, user_prompt)


def extract_python_structure(file_id, content, db):
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return

    for node in tree.body:

        # ---- Detect Class ----
        if isinstance(node, ast.ClassDef):
            db.execute(
                text("""
                    INSERT INTO architectural_elements 
                    (file_id, element_type, name, signature)
                    VALUES (:file_id, 'class', :name, :signature)
                """),
                {
                    "file_id": file_id,
                    "name": node.name,
                    "signature": node.name
                }
            )

            # ---- Detect Methods Inside Class ----
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    db.execute(
                        text("""
                            INSERT INTO architectural_elements 
                            (file_id, element_type, name, signature)
                            VALUES (:file_id, 'method', :name, :signature)
                        """),
                        {
                            "file_id": file_id,
                            "name": f"{node.name}.{child.name}",
                            "signature": f"{child.name}({', '.join(arg.arg for arg in child.args.args)})"
                        }
                    )

        # ---- Detect Top-Level Functions ----
        elif isinstance(node, ast.FunctionDef):
            db.execute(
                text("""
                    INSERT INTO architectural_elements 
                    (file_id, element_type, name, signature)
                    VALUES (:file_id, 'function', :name, :signature)
                """),
                {
                    "file_id": file_id,
                    "name": node.name,
                    "signature": f"{node.name}({', '.join(arg.arg for arg in node.args.args)})"
                }
            )

    db.commit()


@app.post("/retrieve-context/")
def retrieve_context(query: str, project_id: int, db: Session = Depends(get_db)):
    query_embedding = model.encode(query).tolist()

    sql = text("""
        SELECT content
        FROM code_chunks
        WHERE file_id IN (
            SELECT id FROM files WHERE project_id = :project_id
        )
        ORDER BY embedding <-> CAST(:embedding AS vector)
        LIMIT 5;

    """)

    results = db.execute(
        sql,
        {
            "embedding": query_embedding,
            "project_id": project_id
        }
    ).fetchall()


    chunks = [row[0] for row in results]

    return {"relevant_chunks": chunks}

@app.post("/create-project/")
def create_project(name: str, db: Session = Depends(get_db)):
    '''Create a project and return its API key. Store the key — it won't be shown again.'''
    project = Project(name=name)
    db.add(project)
    db.commit()
    db.refresh(project)

    raw_key = create_api_key(project.id, db)

    return {
        "project_id": project.id,
        "name": project.name,
        "api_key": raw_key,
        "warning": "Save this key now — it will never be shown again."
    }


@app.post("/upload-file/")
def upload_file(
    project_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    key_project_id: int = Depends(require_api_key)
):

    verify_project_access(project_id, key_project_id)
    check_rate_limit(x_api_key, db, endpoint="upload-file")

    valid, err = validate_filename(file.filename)
    if not valid:
        raise HTTPException(status_code=400, detail=err)

    content = file.file.read().decode("utf-8")
    content, warnings = sanitize_code(content, file.filename)

    db_file = DBFile(project_id=project_id, filename=file.filename, content=content)
    db.add(db_file)
    db.commit()
    db.refresh(db_file)

    extract_python_structure(db_file.id, content, db)
    extract_dependencies(db_file.id, content, db)

    from models import PageIndex
    summary   = generate_file_summary(content)
    embedding = model.encode(summary).tolist()
    page_entry = PageIndex(
        project_id=project_id, file_id=db_file.id,
        filename=file.filename, summary=summary, embedding=embedding
    )
    db.add(page_entry)
    db.commit()

    chunks = chunk_code(content)
    for chunk in chunks:
        embedding = model.encode(chunk).tolist()
        db.add(CodeChunk(file_id=db_file.id, content=chunk, embedding=embedding))
    db.commit()

    quality = analyze_quality(content, file.filename)
    return {
        "message":           "File stored + embedded successfully",
        "file_id":           db_file.id,
        "quality_score":     quality.score,
        "quality_grade":     quality.grade,
        "sanitize_warnings": warnings
    }

def retrieve_relevant_files(query: str, project_id: int, db: Session):
    query_embedding = model.encode(query).tolist()

    sql = text("""
        SELECT file_id
        FROM page_index
        WHERE project_id = :project_id
        ORDER BY embedding <-> CAST(:embedding AS vector)
        LIMIT 3;
    """)

    results = db.execute(
        sql,
        {
            "embedding": query_embedding,
            "project_id": project_id
        }
    ).fetchall()

    return [row[0] for row in results]

def extract_dependencies(file_id, content, db):
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return

    # Clear old dependencies
    db.execute(
        text("DELETE FROM file_dependencies WHERE source_file_id = :file_id"),
        {"file_id": file_id}
    )
    db.commit()

    for node in ast.walk(tree):

        # 🔹 Imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                db.execute(
                    text("""
                        INSERT INTO file_dependencies 
                        (source_file_id, target, dependency_type)
                        VALUES (:file_id, :target, 'import')
                    """),
                    {
                        "file_id": file_id,
                        "target": alias.name
                    }
                )

        elif isinstance(node, ast.ImportFrom):
            module = node.module if node.module else ""
            db.execute(
                text("""
                    INSERT INTO file_dependencies 
                    (source_file_id, target, dependency_type)
                    VALUES (:file_id, :target, 'import')
                """),
                {
                    "file_id": file_id,
                    "target": module
                }
            )

        # 🔹 Function Calls
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                db.execute(
                    text("""
                        INSERT INTO file_dependencies 
                        (source_file_id, target, dependency_type)
                        VALUES (:file_id, :target, 'call')
                    """),
                    {
                        "file_id": file_id,
                        "target": node.func.id
                    }
                )

            elif isinstance(node.func, ast.Attribute):
                db.execute(
                    text("""
                        INSERT INTO file_dependencies 
                        (source_file_id, target, dependency_type)
                        VALUES (:file_id, :target, 'call')
                    """),
                    {
                        "file_id": file_id,
                        "target": node.func.attr
                    }
                )

    db.commit()

# ════════════════════════════════════════════════════════════════
#  PATCH FOR main.py  — 3 changes needed
# ════════════════════════════════════════════════════════════════

# ── CHANGE 1: Add this import near the top of main.py ──────────


# ── CHANGE 2: Replace the /modify-code/ endpoint entirely ───────
#    Delete the old @app.post("/modify-code/") function and paste this:

@app.post("/modify-code/")
def modify_code(
    query: str,
    project_id: int,
    db: Session = Depends(get_db),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    key_project_id: int = Depends(require_api_key)
):
    verify_project_access(project_id, key_project_id)
    check_rate_limit(x_api_key, db, endpoint="modify-code",
                     limit=MODIFY_LIMIT, window=MODIFY_WINDOW)

    query, sanitize_warnings = sanitize_query(query)

    relevant_file_ids = retrieve_relevant_files(query, project_id, db)
    context = ""
    if relevant_file_ids:
        query_embedding = model.encode(query).tolist()
        sql = text("""
            SELECT content FROM code_chunks
            WHERE file_id = ANY(:file_ids)
            ORDER BY embedding <-> CAST(:embedding AS vector)
            LIMIT 5;
        """)
        results = db.execute(sql, {"embedding": query_embedding, "file_ids": relevant_file_ids}).fetchall()
        context = "\n\n".join([row[0] for row in results])

    result = run_agent(
        query=query, project_id=project_id,
        db=db, model=model, groq_llm=groq_llm, context=context
    )

    if result.get("status") == "success":
        quality_reports = {}
        for f in db.query(DBFile).filter(DBFile.project_id == project_id).all():
            if f.filename.endswith(".py"):
                q = analyze_quality(f.content, f.filename)
                quality_reports[f.filename] = {"score": q.score, "grade": q.grade, "summary": q.summary}
        result["quality"] = quality_reports

    return result
# ── CHANGE 3: Add this rollback endpoint anywhere in main.py ────

@app.post("/rollback/")
def rollback_file(
    file_id: int,
    db: Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key)
):
    file = db.query(DBFile).filter(DBFile.id == file_id).first()
    if not file:
        return {"error": "File not found"}
    verify_project_access(file.project_id, key_project_id)

    row = db.execute(
        text("""
            SELECT id, content FROM file_versions
            WHERE file_id = :fid
            ORDER BY created_at DESC LIMIT 1
        """),
        {"fid": file_id}
    ).fetchone()

    if not row:
        return {"error": "No previous version found for this file."}

    version_id, previous_content = row
    file.content = previous_content
    db.commit()

    db.execute(text("DELETE FROM code_chunks WHERE file_id = :fid"), {"fid": file_id})
    db.commit()

    from models import CodeChunk
    chunks = re.split(r"\n(?=def |class )", previous_content)
    for chunk in chunks:
        if chunk.strip():
            embedding = model.encode(chunk).tolist()
            db.add(CodeChunk(file_id=file_id, content=chunk, embedding=embedding))
    db.commit()

    db.execute(text("DELETE FROM file_versions WHERE id = :vid"), {"vid": version_id})
    db.commit()

    return {"message": f"Rolled back {file.filename} to previous version."}



@app.get("/history/{file_id}")
def get_file_history(file_id: int, db: Session = Depends(get_db)):
    """List all saved versions of a file."""
    rows = db.execute(
        text("""
            SELECT id, created_at FROM file_versions
            WHERE file_id = :fid
            ORDER BY created_at DESC
        """),
        {"fid": file_id}
    ).fetchall()

    return {
        "file_id": file_id,
        "versions": [{"version_id": r[0], "saved_at": str(r[1])} for r in rows]
    }

# ════════════════════════════════════════════════════════════════
#  ADD THESE ENDPOINTS TO main.py
#  They let you inspect files, list project files, and view diffs
# ════════════════════════════════════════════════════════════════

@app.get("/get-file/")
def get_file(
    project_id: int,
    filename: str,
    db: Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key)
):
    verify_project_access(project_id, key_project_id)
    file = db.query(DBFile).filter(DBFile.project_id == project_id, DBFile.filename == filename).first()
    if not file:
        return {"error": f"File '{filename}' not found"}
    return {"file_id": file.id, "filename": file.filename, "content": file.content}



@app.get("/list-files/")
def list_files(
    project_id: int,
    db: Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key)
):
    verify_project_access(project_id, key_project_id)
    files = db.query(DBFile).filter(DBFile.project_id == project_id).all()
    return {
        "project_id": project_id,
        "files": [{"file_id": f.id, "filename": f.filename, "lines": len(f.content.splitlines())} for f in files]
    }


@app.get("/diff/{file_id}")
def get_diff(
    file_id: int,
    db: Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key)
):
    import difflib
    file = db.query(DBFile).filter(DBFile.id == file_id).first()
    if not file:
        return {"error": "File not found"}
    verify_project_access(file.project_id, key_project_id)

    row = db.execute(
        text("SELECT content FROM file_versions WHERE file_id = :fid ORDER BY created_at DESC LIMIT 1"),
        {"fid": file_id}
    ).fetchone()
    if not row:
        return {"error": "No previous version to diff against"}

    diff = list(difflib.unified_diff(
        row[0].splitlines(keepends=True),
        file.content.splitlines(keepends=True),
        fromfile=f"{file.filename} (before)",
        tofile=f"{file.filename} (after)",
        lineterm=""
    ))
    added   = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
    return {
        "filename": file.filename,
        "lines_added": added,
        "lines_removed": removed,
        "diff": "\n".join(diff) if diff else "No changes detected"
    }

@app.get("/get-architecture/")
def get_architecture(project_id: int, db: Session = Depends(get_db)):
    """
    Show all classes, functions, and methods extracted from
    every file in the project.
    """
    files = db.query(DBFile).filter(DBFile.project_id == project_id).all()
    if not files:
        return {"error": "No files in project"}

    result = {}
    for f in files:
        elements = db.execute(
            text("""
                SELECT element_type, name, signature
                FROM architectural_elements
                WHERE file_id = :fid
                ORDER BY element_type, name
            """),
            {"fid": f.id}
        ).fetchall()

        result[f.filename] = [
            {"type": e[0], "name": e[1], "signature": e[2]}
            for e in elements
        ]

    return {"project_id": project_id, "architecture": result}

from architectural_awareness import build_dependency_graph



@app.get("/quality/")
def get_file_quality(
    project_id: int,
    filename: str,
    db: Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key)
):
    verify_project_access(project_id, key_project_id)
    file = db.query(DBFile).filter(
        DBFile.project_id == project_id,
        DBFile.filename == filename
    ).first()
    if not file:
        return {"error": f"File '{filename}' not found"}
    return analyze_quality(file.content, file.filename).to_dict()


@app.get("/quality/project/")
def get_project_quality(
    project_id: int,
    db: Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key)
):
    verify_project_access(project_id, key_project_id)
    return analyze_project_quality(project_id, db)


@app.get("/rate-limit-status/")
def rate_limit_status(
    db: Session = Depends(get_db),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    key_project_id: int = Depends(require_api_key)
):
    return get_rate_limit_status(x_api_key, db)


@app.get("/dependency-graph/")
def dependency_graph(
    project_id: int,
    db: Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key)
):
    verify_project_access(project_id, key_project_id)
    return build_dependency_graph(project_id, db)

# ─────────────────────────────────────────────
# LONG-TERM PROJECT MEMORY ENDPOINTS
# ─────────────────────────────────────────────

from memory_engine import store_memory, retrieve_relevant_memory


@app.post("/add-architecture-memory/")
def add_architecture_memory(
    project_id: int,
    note: str,
    db: Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key)
):
    """
    Manually store an architectural constraint or note for a project.

    Examples:
      - "This project follows Clean Architecture"
      - "Never modify public API signatures"
      - "Database layer must remain isolated"

    Stored as memory_type='architecture' and used to guide future planning.
    """
    verify_project_access(project_id, key_project_id)

    memory_id = store_memory(
        project_id=project_id,
        memory_type="architecture",
        content=note,
        model=model,
        db=db
    )

    if memory_id is None:
        raise HTTPException(status_code=400, detail="Failed to store memory. Check that the note is non-empty.")

    return {
        "message":   "Architecture memory stored successfully",
        "memory_id": memory_id,
        "note":      note[:200]
    }


@app.get("/project-memory/")
def get_project_memory(
    project_id: int,
    memory_type: Optional[str] = None,
    db: Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key)
):
    """
    Inspect all stored memory entries for a project.
    Optionally filter by memory_type: architecture | decision | constraint | style
    """
    verify_project_access(project_id, key_project_id)

    query_sql = """
        SELECT id, memory_type, content, created_at
        FROM project_memory
        WHERE project_id = :project_id
    """
    params = {"project_id": project_id}

    if memory_type:
        query_sql += " AND memory_type = :memory_type"
        params["memory_type"] = memory_type

    query_sql += " ORDER BY created_at DESC"

    rows = db.execute(text(query_sql), params).fetchall()

    return {
        "project_id": project_id,
        "total":      len(rows),
        "memories": [
            {
                "id":          row[0],
                "memory_type": row[1],
                "content":     row[2],
                "created_at":  str(row[3])
            }
            for row in rows
        ]
    }

# ── BACKGROUND JOB ENDPOINTS ────────────────────────────────────

from job_system import create_job, get_job_status
from infra import dispatch_persistent_job

@app.post("/modify-code-async/")
def modify_code_async(
    query: str,
    project_id: int,
    db: Session = Depends(get_db),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    key_project_id: int = Depends(require_api_key)
):
    """Dispatch a modify-code job in background. Returns job_id immediately."""
    verify_project_access(project_id, key_project_id)
    check_rate_limit(x_api_key, db, endpoint="modify-code",
                     limit=MODIFY_LIMIT, window=MODIFY_WINDOW)

    query, _ = sanitize_query(query)
    job_id   = create_job(project_id, query, db)

    def _task(job_id, query, project_id, x_api_key):
        task_db = SessionLocal()
        try:
            relevant_ids = retrieve_relevant_files(query, project_id, task_db)
            context = ""
            if relevant_ids:
                emb = model.encode(query).tolist()
                sql = text("""SELECT content FROM code_chunks WHERE file_id = ANY(:ids)
                              ORDER BY embedding <-> CAST(:emb AS vector) LIMIT 5""")
                rows = task_db.execute(sql, {"ids": relevant_ids, "emb": emb}).fetchall()
                context = "\n\n".join(r[0] for r in rows)

            result = run_agent(
                query=query, project_id=project_id,
                db=task_db, model=model, groq_llm=groq_llm, context=context
            )

            if result.get("status") == "success":
                quality = {}
                for f in task_db.query(DBFile).filter(DBFile.project_id == project_id).all():
                    if f.filename.endswith(".py"):
                        q = analyze_quality(f.content, f.filename)
                        quality[f.filename] = {"score": q.score, "grade": q.grade}
                result["quality"] = quality

            return result
        finally:
            task_db.close()

    ispatch_persistent_job(job_id, _task, query=query, project_id=project_id, x_api_key=x_api_key)

    return {"job_id": job_id, "status": "pending", "message": "Job dispatched. Poll /job-status/{job_id}"}


@app.get("/job-status/{job_id}")
def job_status(
    job_id: int,
    db: Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key)
):
    job = get_job_status(job_id, db)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    verify_project_access(job["project_id"], key_project_id)
    return job


# ── PROJECT HEALTH DASHBOARD ─────────────────────────────────────

@app.get("/project-health/")
def project_health(
    project_id: int,
    db: Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key)
):
    verify_project_access(project_id, key_project_id)

    from datetime import datetime, timedelta, timezone

    # Quality
    quality_data = analyze_project_quality(project_id, db)
    overall_score = quality_data.get("overall_score", 0)
    overall_grade = quality_data.get("overall_grade", "F")

    # Test pass rate
    test_rows = db.execute(
        text("SELECT pass_rate FROM project_tests WHERE project_id=:pid"),
        {"pid": project_id}
    ).fetchall()
    avg_pass_rate = (sum(r[0] for r in test_rows if r[0] is not None) / len(test_rows)) if test_rows else None

    # Memory count
    memory_count = db.execute(
        text("SELECT COUNT(*) FROM project_memory WHERE project_id=:pid"),
        {"pid": project_id}
    ).scalar()

    # Files modified in last 24h
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    recent_files = db.execute(
        text("""
            SELECT DISTINCT f.filename FROM file_versions fv
            JOIN files f ON f.id = fv.file_id
            WHERE f.project_id=:pid AND fv.created_at >= :cutoff
        """),
        {"pid": project_id, "cutoff": cutoff}
    ).fetchall()

    # Open jobs
    open_jobs = db.execute(
        text("SELECT COUNT(*) FROM agent_jobs WHERE project_id=:pid AND status IN ('pending','running')"),
        {"pid": project_id}
    ).scalar()

    # Dependency risk (count of files with 5+ cross-file dependencies)
    dep_graph = build_dependency_graph(project_id, db)
    high_risk_files = [f for f, data in dep_graph.items() if len(data.get("imported_by", [])) >= 5]

    return {
        "project_id":         project_id,
        "quality_score":      overall_score,
        "quality_grade":      overall_grade,
        "test_pass_rate":     round(avg_pass_rate, 2) if avg_pass_rate is not None else "no tests",
        "test_files_count":   len(test_rows),
        "memory_entries":     memory_count,
        "files_modified_24h": [r[0] for r in recent_files],
        "open_jobs":          open_jobs,
        "dependency_risk": {
            "high_risk_files": high_risk_files,
            "risk_level": "high" if len(high_risk_files) >= 3 else "medium" if high_risk_files else "low"
        }
    }


# ── TEST ENDPOINTS ───────────────────────────────────────────────


@app.get("/tests/")
def list_project_tests(
    project_id: int,
    db: Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key)
):
    verify_project_access(project_id, key_project_id)
    rows = db.execute(
        text("SELECT id, filename, pass_rate, last_run_at FROM project_tests WHERE project_id=:pid ORDER BY last_run_at DESC"),
        {"pid": project_id}
    ).fetchall()
    return {"project_id": project_id, "tests": [
        {"test_id": r[0], "filename": r[1], "pass_rate": r[2], "last_run_at": str(r[3])} for r in rows
    ]}


@app.get("/global-lessons/")
def get_global_lessons(
    query: str,
    db: Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key)
):
    from global_memory import retrieve_global_lessons
    lessons = retrieve_global_lessons(query, model, db)
    return {"query": query, "lessons": lessons}


#
#
# ── 2. Add /execute-goal/ endpoint ────────────────────────────
#
@app.post("/execute-goal/")
def execute_goal_endpoint(
    goal: str,
    project_id: int,
    enable_tests: bool = False,
    db: Session = Depends(get_db),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    key_project_id: int = Depends(require_api_key)
):
    verify_project_access(project_id, key_project_id)
    check_rate_limit(x_api_key, db, endpoint="execute-goal", limit=MODIFY_LIMIT, window=MODIFY_WINDOW)
    goal, _ = sanitize_query(goal)
    relevant_ids = retrieve_relevant_files(goal, project_id, db)
    context = ""
    if relevant_ids:
        emb  = model.encode(goal).tolist()
        rows = db.execute(text("""SELECT content FROM code_chunks WHERE file_id=ANY(:ids)
                                  ORDER BY embedding <-> CAST(:emb AS vector) LIMIT 5"""),
                          {"ids": relevant_ids, "emb": emb}).fetchall()
        context = "\n\n".join(r[0] for r in rows)
    return execute_goal(goal, project_id, db, model, groq_llm, context, enable_tests)


@app.get("/pending-approvals/")
def list_approvals(project_id: int, db: Session = Depends(get_db),
                   key_project_id: int = Depends(require_api_key)):
    verify_project_access(project_id, key_project_id)
    return {"approvals": get_pending_approvals(project_id, db)}

@app.post("/approve/{approval_id}")
def approve(approval_id: int, project_id: int, db: Session = Depends(get_db),
            key_project_id: int = Depends(require_api_key)):
    verify_project_access(project_id, key_project_id)
    return resolve_approval(approval_id, "approve", project_id, db, model, groq_llm)

@app.post("/reject/{approval_id}")
def reject(approval_id: int, project_id: int, db: Session = Depends(get_db),
           key_project_id: int = Depends(require_api_key)):
    verify_project_access(project_id, key_project_id)
    return resolve_approval(approval_id, "reject", project_id, db)


# ─────────────────────────────────────────────────────────────────────────────
# ENTERPRISE GOVERNANCE ENDPOINTS  (Phase 5 — Enterprise Safety Controls)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/approve-change/")
def approve_change_endpoint(
    approval_id:  int,
    project_id:   int,
    reviewer_id:  str,
    db:           Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key),
):
    """
    Enterprise approval gate — reviewer or admin approves a high-risk change.

    Validates that:
      - The caller's role is reviewer or admin on this project.
      - The approval request is in 'pending' state.
      - The approval belongs to this project.

    On approval: executes the stored DAG plan, writes audit log, returns result.

    This supersedes the simpler POST /approve/{approval_id} with role enforcement
    and full audit trail.
    """
    verify_project_access(project_id, key_project_id)
    role_ctx = resolve_role_context(project_id, reviewer_id, db, default_role=UserRole.OBSERVER)
    return approve_change(
        approval_id=approval_id,
        project_id=project_id,
        reviewer_id=reviewer_id,
        role_ctx=role_ctx,
        db=db,
        model=model,
        groq_llm=groq_llm,
    )


@app.post("/reject-change/")
def reject_change_endpoint(
    approval_id:  int,
    project_id:   int,
    reviewer_id:  str,
    reason:       str = "Rejected by reviewer",
    db:           Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key),
):
    """Enterprise rejection: reviewer or admin rejects a high-risk change."""
    verify_project_access(project_id, key_project_id)
    role_ctx = resolve_role_context(project_id, reviewer_id, db, default_role=UserRole.OBSERVER)
    return reject_change(
        approval_id=approval_id,
        project_id=project_id,
        reviewer_id=reviewer_id,
        role_ctx=role_ctx,
        reason=reason,
        db=db,
    )


@app.post("/assign-role/")
def assign_role_endpoint(
    project_id:  int,
    target_user: str,
    role:        str,
    granted_by:  str,
    db:          Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key),
):
    """
    Assign a role (admin | reviewer | observer) to a user on a project.
    Caller must be an admin on this project.
    """
    verify_project_access(project_id, key_project_id)
    # Verify the caller (granted_by) is an admin
    caller_ctx = resolve_role_context(project_id, granted_by, db, default_role=UserRole.OBSERVER)
    caller_ctx.require("manage_roles", "only admins can assign roles")
    result = assign_role(
        project_id=project_id, user_id=target_user,
        role=role, granted_by=granted_by, db=db,
    )
    write_audit_log(
        project_id=project_id, user_id=granted_by,
        action="role_assigned", db=db,
        detail=f"target={target_user} role={role}",
    )
    return result


@app.get("/roles/")
def list_roles_endpoint(
    project_id: int,
    db: Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key),
):
    """List all role assignments for a project. Any authenticated user can view."""
    verify_project_access(project_id, key_project_id)
    return {"project_id": project_id, "roles": list_project_roles(project_id, db)}


@app.get("/audit-log/")
def audit_log_endpoint(
    project_id:  int,
    user_id:     Optional[str] = None,
    action:      Optional[str] = None,
    limit:       int           = 100,
    db:          Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key),
):
    """
    Retrieve the enterprise audit log for a project.
    Supports filtering by user_id and action type.
    Returns newest entries first.

    Recorded actions include:
      orchestrate_started, dag_execution_complete, execution_risk_scored,
      approval_requested, change_approved, change_rejected, admin_high_risk_bypass,
      role_assigned, create_pr
    """
    verify_project_access(project_id, key_project_id)
    entries = get_audit_log(
        project_id=project_id, db=db,
        limit=min(limit, 500), user_id=user_id, action=action,
    )
    return {"project_id": project_id, "count": len(entries), "entries": entries}

# ── 1. Startup event ──────────────────────────────────────────
#


@app.on_event("startup")
async def startup():
    setup_scheduler(DATABASE_URL)
from autonomous_ops import (
    get_code_evolution_summary, detect_drift,
    take_architectural_snapshot, get_function_history
)

@app.get("/evolution/")
def code_evolution(project_id: int, db: Session = Depends(get_db),
                   key_project_id: int = Depends(require_api_key)):
    verify_project_access(project_id, key_project_id)
    return get_code_evolution_summary(project_id, db)

@app.get("/drift/")
def drift_report(project_id: int, db: Session = Depends(get_db),
                 key_project_id: int = Depends(require_api_key)):
    verify_project_access(project_id, key_project_id)
    report = detect_drift(project_id, db)
    return {
        "risk_level":      report.risk_level,
        "violations":      report.new_violations,
        "coupling_delta":  report.coupling_delta,
        "circular_deps":   report.circular_deps,
        "volatile_modules": report.volatile_modules
    }

@app.post("/snapshot/")
def take_snapshot(project_id: int, db: Session = Depends(get_db),
                  key_project_id: int = Depends(require_api_key)):
    verify_project_access(project_id, key_project_id)
    snap_id = take_architectural_snapshot(project_id, db)
    return {"snapshot_id": snap_id}

@app.get("/function-history/")
def function_history(project_id: int, function_name: str,
                     db: Session = Depends(get_db),
                     key_project_id: int = Depends(require_api_key)):
    verify_project_access(project_id, key_project_id)
    return {"function": function_name, "history": get_function_history(function_name, project_id, db)}


# ─────────────────────────────────────────────────────────────────────────────
# GIT INTEGRATION ENDPOINTS  (Phase 1 — Git Workflow)
# ─────────────────────────────────────────────────────────────────────────────
#
# These endpoints add the PR-first Git workflow to AI CTO.
# All AI-driven changes land on isolated ai-change-* branches.
# Direct commits to main/master are permanently blocked at the git_ops layer.
#
# Endpoints:
#   POST /clone-repo/   — Clone a remote repo into the project workspace
#   POST /create-pr/    — Create a GitHub Pull Request from an ai-change branch
#   GET  /git-status/   — Inspect the current state of the project workspace
# ─────────────────────────────────────────────────────────────────────────────

from git_ops import (
    clone_repository, create_github_pr, repo_status,
    get_workspace_path, PROTECTED_BRANCHES
)
from diff_engine import get_diffs_for_branch, compute_overall_risk


@app.post("/clone-repo/")
def clone_repo(
    project_id:   int,
    repo_url:     str,
    github_token: Optional[str] = None,
    db: Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key)
):
    """
    Clone a remote Git repository into the project's local workspace.

    After cloning:
      - Stores repo_path and repo_url in the projects table.
      - Optionally stores a GitHub PAT (github_token) for private repos
        and PR creation. Store securely — it enables write access.

    This is the entry point for Git-enabled projects. Once called,
    run_agent_from_dag() will automatically create branches and push
    changes after successful executions.

    Args:
        project_id:   Target project.
        repo_url:     HTTPS remote URL (e.g. https://github.com/owner/repo.git).
        github_token: Optional GitHub PAT with 'repo' scope.
                      Required for: private repos, pushing branches, creating PRs.
    """
    verify_project_access(project_id, key_project_id)

    result = clone_repository(
        repo_url=repo_url,
        project_id=project_id,
        github_token=github_token
    )

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])

    # Persist Git config to the project record
    db.execute(
        text("""
            UPDATE projects
            SET repo_url      = :repo_url,
                repo_path     = :repo_path,
                github_token  = :github_token,
                default_branch = :default_branch
            WHERE id = :pid
        """),
        {
            "repo_url":       repo_url,
            "repo_path":      result["repo_path"],
            "github_token":   github_token,
            "default_branch": result["default_branch"],
            "pid":            project_id
        }
    )
    db.commit()

    return {
        "message":        "Repository cloned and workspace initialized.",
        "project_id":     project_id,
        "repo_url":       repo_url,
        "repo_path":      result["repo_path"],
        "default_branch": result["default_branch"],
        "git_enabled":    True,
        "note": (
            "All future AI-driven changes will be committed to isolated "
            "ai-change-* branches. Use POST /create-pr/ to open a Pull Request."
        )
    }


@app.post("/create-pr/")
def create_pr(
    project_id:  int,
    branch_name: str,
    title:       Optional[str] = None,
    body:        Optional[str] = None,
    db: Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key)
):
    """
    Create a GitHub Pull Request from an ai-change-* branch to the project's
    default branch (main/master).

    Safety:
      - Refuses to create a PR if branch_name is a protected branch name.
      - Requires the project to have a github_token configured via /clone-repo/.
      - Requires the project to have a repo_url configured.

    Args:
        project_id:  Target project.
        branch_name: The ai-change-* branch produced by run_agent_from_dag().
        title:       PR title (auto-generated if omitted).
        body:        PR description (auto-generated if omitted).

    Returns:
        { pr_url, pr_number, branch_name, base_branch }
    """
    verify_project_access(project_id, key_project_id)

    # Block PRs from protected branches (shouldn't happen but enforce defensively)
    if branch_name in PROTECTED_BRANCHES:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot create a PR from protected branch '{branch_name}'. "
                   "AI changes must originate from an ai-change-* branch."
        )

    # Load Git config for this project
    row = db.execute(
        text("SELECT repo_url, github_token, default_branch FROM projects WHERE id = :pid"),
        {"pid": project_id}
    ).fetchone()

    if not row or not row[0]:
        raise HTTPException(
            status_code=400,
            detail="Project has no repository configured. Call POST /clone-repo/ first."
        )

    repo_url, github_token, default_branch = row[0], row[1], row[2] or "main"

    if not github_token:
        raise HTTPException(
            status_code=400,
            detail=(
                "No GitHub token configured for this project. "
                "Re-call POST /clone-repo/ with a github_token to enable PR creation."
            )
        )

    # Auto-generate PR title/body if not provided
    pr_title = title or f"AI CTO: Changes from branch {branch_name}"
    pr_body  = body or (
        f"## AI CTO — Automated Change\n\n"
        f"**Branch:** `{branch_name}`  \n"
        f"**Target:** `{default_branch}`  \n\n"
        f"This Pull Request was generated by AI CTO after successful DAG execution "
        f"and QA verification.\n\n"
        f"Please review the diff carefully before merging.\n\n"
        f"---\n_Generated by AI CTO · Never auto-merged · Human approval required_"
    )

    result = create_github_pr(
        repo_url=repo_url,
        branch_name=branch_name,
        github_token=github_token,
        title=pr_title,
        body=pr_body,
        base_branch=default_branch
    )

    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])

    return {
        "message":     "Pull Request created successfully.",
        "pr_url":      result["pr_url"],
        "pr_number":   result["pr_number"],
        "branch_name": branch_name,
        "base_branch": default_branch,
        "project_id":  project_id,
        "note":        "Review and approve the PR in GitHub. AI CTO never merges automatically."
    }


@app.get("/git-status/")
def git_status(
    project_id: int,
    db: Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key)
):
    """
    Return the current Git workspace status for a project.
    Useful for debugging branch state, detecting dirty working trees,
    and confirming which commit is currently checked out.
    """
    verify_project_access(project_id, key_project_id)

    row = db.execute(
        text("SELECT repo_path, repo_url, default_branch FROM projects WHERE id = :pid"),
        {"pid": project_id}
    ).fetchone()

    if not row or not row[0]:
        return {
            "project_id": project_id,
            "git_enabled": False,
            "message": "No repository configured. Call POST /clone-repo/ to enable Git integration."
        }

    status = repo_status(row[0])
    return {
        "project_id":     project_id,
        "git_enabled":    True,
        "repo_url":       row[1],
        "default_branch": row[2] or "main",
        "workspace_path": row[0],
        **status
    }

# ─────────────────────────────────────────────────────────────────────────────
# DIFF VISUALIZATION ENDPOINTS  (Phase 2 — Change-Diff Transparency Layer)
# ─────────────────────────────────────────────────────────────────────────────
#
# These endpoints expose the diffs stored by diff_engine.py for human review.
# They are read-only — they never touch files, execution state, or the Git repo.
#
# Endpoints:
#   GET /diff/{project_id}/{branch_name}   — Full diffs for a branch
#   GET /diff/{project_id}/{branch_name}/summary — Metadata only (no diff text)
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/diff/{project_id}/{branch_name}")
def get_branch_diffs(
    project_id:  int,
    branch_name: str,
    preview:     bool = False,
    db: Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key)
):
    """
    Retrieve all change diffs for a specific ai-change-* branch.

    Returns the full unified diff text for every file modified on that branch,
    along with per-file risk scoring and an aggregated overall risk level.

    Use this endpoint to review what AI CTO changed before approving a PR.

    Args:
        project_id:  Target project.
        branch_name: The ai-change-* branch (returned in run_agent_from_dag response).
        preview:     If true, truncate each diff to the first 200 lines.
                     Useful for quick review in UIs. Default: false (full diff).

    Response shape:
        {
          "project_id":    int,
          "branch_name":   str,
          "overall_risk":  "low" | "medium" | "high",
          "total_files":   int,
          "total_added":   int,
          "total_removed": int,
          "diffs": [
            {
              "filename":      str,
              "diff_text":     str,   // unified diff (or truncated if preview=true)
              "lines_added":   int,
              "lines_removed": int,
              "risk_level":    "low" | "medium" | "high",
              "created_at":    str
            },
            ...
          ]
        }
    """
    verify_project_access(project_id, key_project_id)

    diffs = get_diffs_for_branch(
        project_id=project_id,
        branch_name=branch_name,
        db=db,
        preview_only=preview
    )

    if not diffs:
        return {
            "project_id":   project_id,
            "branch_name":  branch_name,
            "overall_risk": "low",
            "total_files":  0,
            "total_added":  0,
            "total_removed": 0,
            "diffs":        [],
            "message":      (
                "No diffs found for this branch. This may mean the branch has not been "
                "executed yet, or the project does not have diff tracking enabled."
            )
        }

    overall_risk  = compute_overall_risk(diffs)
    total_added   = sum(d.get("lines_added", 0)   for d in diffs)
    total_removed = sum(d.get("lines_removed", 0) for d in diffs)

    return {
        "project_id":    project_id,
        "branch_name":   branch_name,
        "overall_risk":  overall_risk,
        "total_files":   len(diffs),
        "total_added":   total_added,
        "total_removed": total_removed,
        "diffs":         diffs
    }


@app.get("/diff/{project_id}/{branch_name}/summary")
def get_branch_diff_summary(
    project_id:  int,
    branch_name: str,
    db: Session = Depends(get_db),
    key_project_id: int = Depends(require_api_key)
):
    """
    Return diff metadata only — no diff text body.

    Useful for dashboards and approval UIs that need to display risk level
    and change scope without downloading the full diff payload.

    Response shape:
        {
          "project_id":    int,
          "branch_name":   str,
          "overall_risk":  "low" | "medium" | "high",
          "total_files":   int,
          "total_added":   int,
          "total_removed": int,
          "files": [
            { "filename": str, "lines_added": int, "lines_removed": int, "risk_level": str }
          ]
        }
    """
    verify_project_access(project_id, key_project_id)

    diffs = get_diffs_for_branch(
        project_id=project_id,
        branch_name=branch_name,
        db=db,
        preview_only=False
    )

    overall_risk  = compute_overall_risk(diffs)
    total_added   = sum(d.get("lines_added", 0)   for d in diffs)
    total_removed = sum(d.get("lines_removed", 0) for d in diffs)

    return {
        "project_id":    project_id,
        "branch_name":   branch_name,
        "overall_risk":  overall_risk,
        "total_files":   len(diffs),
        "total_added":   total_added,
        "total_removed": total_removed,
        "files": [
            {
                "filename":      d["filename"],
                "lines_added":   d["lines_added"],
                "lines_removed": d["lines_removed"],
                "risk_level":    d["risk_level"],
                "created_at":    d["created_at"]
            }
            for d in diffs
        ]
    }