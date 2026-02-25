from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy import Float
from database import Base
from pgvector.sqlalchemy import Vector


class CodeChunk(Base):
    __tablename__ = "code_chunks"

    id         = Column(Integer, primary_key=True, index=True)
    file_id    = Column(Integer, ForeignKey("files.id"))
    content    = Column(Text, nullable=False)
    embedding  = Column(Vector(384))  # 384 for BGE-small
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Project(Base):
    __tablename__ = "projects"

    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # ── Git integration (Phase 1) ──────────────────────────────
    repo_url       = Column(String, nullable=True)          # remote origin URL
    repo_path      = Column(String, nullable=True)          # local workspace path (cloned copy)
    github_token   = Column(String, nullable=True)          # PAT for private repos + PR creation
    default_branch = Column(String, nullable=True, default="main")  # "main" or "master"
    # ───────────────────────────────────────────────────────────

    files    = relationship("File",          back_populates="project")
    memories = relationship("ProjectMemory", back_populates="project")


class File(Base):
    __tablename__ = "files"

    id         = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    filename   = Column(String, nullable=False)
    content    = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    project = relationship("Project", back_populates="files")


class ProjectMemory(Base):
    """
    Long-term memory for a project.
    Stores architectural constraints, past decisions, style rules, and constraints.
    Searchable via vector similarity using BGE-small embeddings.
    memory_type allowed values: "architecture" | "decision" | "constraint" | "style"
    """
    __tablename__ = "project_memory"

    id          = Column(Integer, primary_key=True, index=True)
    project_id  = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    memory_type = Column(String, nullable=False, index=True)
    content     = Column(Text, nullable=False)
    embedding   = Column(Vector(384))
    created_at  = Column(DateTime(timezone=True), server_default=func.now())

    project = relationship("Project", back_populates="memories")


class PageIndex(Base):
    __tablename__ = "page_index"

    id         = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    file_id    = Column(Integer, ForeignKey("files.id"))
    filename   = Column(String, nullable=False)
    summary    = Column(Text, nullable=False)
    embedding  = Column(Vector(384))
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class FileDependency(Base):
    __tablename__ = "file_dependencies"

    id              = Column(Integer, primary_key=True, index=True)
    source_file_id  = Column(Integer, ForeignKey("files.id"))
    target          = Column(String, nullable=False)   # module or function name
    dependency_type = Column(String, nullable=False)   # "import" or "call"
    created_at      = Column(DateTime(timezone=True), server_default=func.now())


class ArchitecturalElement(Base):
    __tablename__ = "architectural_elements"

    id           = Column(Integer, primary_key=True, index=True)
    file_id      = Column(Integer, ForeignKey("files.id"))
    element_type = Column(String, nullable=False)   # class | function | method
    name         = Column(String, nullable=False)
    signature    = Column(Text, nullable=True)
    created_at   = Column(DateTime(timezone=True), server_default=func.now())


class FileVersion(Base):
    __tablename__ = "file_versions"

    id         = Column(Integer, primary_key=True, index=True)
    file_id    = Column(Integer, ForeignKey("files.id"))
    content    = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ApiKey(Base):
    __tablename__ = "api_keys"

    id         = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), unique=True, nullable=False)
    key_hash   = Column(String(64), nullable=False)
    is_active  = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ChangeDiff(Base):
    """
    Stores unified diffs for every file modified by AI CTO on an ai-change-* branch.
    Populated by diff_engine.generate_and_store_diffs() AFTER execution, BEFORE commit.
    Never written to by the execution pipeline itself — purely a transparency record.

    Unique constraint on (project_id, branch_name, filename) ensures that re-runs
    on the same branch upsert rather than duplicate.
    """
    __tablename__ = "change_diffs"

    id            = Column(Integer, primary_key=True, index=True)
    project_id    = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    branch_name   = Column(String,  nullable=False, index=True)   # e.g. ai-change-1718000000
    filename      = Column(String,  nullable=False)
    diff_text     = Column(Text,    nullable=False)                # full unified diff
    risk_level    = Column(String,  nullable=False, default="low") # low | medium | high
    lines_added   = Column(Integer, nullable=False, default=0)
    lines_removed = Column(Integer, nullable=False, default=0)
    created_at    = Column(DateTime(timezone=True), server_default=func.now())

    project = relationship("Project")