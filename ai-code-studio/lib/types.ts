// ===== lib/types.ts =====
// All types derived directly from schemas.py and main.py response shapes.
// Every field name and type matches exactly what the backend returns.

export interface Project {
  project_id: number;   // int from DB
  api_key: string;
  name: string;
}

// GET /list-files/ → { project_id, files: FileItem[] }
export interface FileItem {
  file_id: number;      // int from DB
  filename: string;
  lines: number;        // line count from DB
}

// GET /get-file/ → FileContent
export interface FileContent {
  file_id: number;
  filename: string;
  content: string;
}

// POST /upload-file/ → UploadResult
export interface UploadResult {
  message: string;
  file_id: number;
  quality_score: number;
  quality_grade: string;        // "A" | "B" | "C" | "D" | "F"
  sanitize_warnings: string[];
}

// POST /modify-code/ → ModifyResult
// This is the raw run_agent() return shape from agent_executor.py
// Success path: { goal, status: "success", steps_executed, results }
// Failure path: { goal, status: "failed", failed_at_step, error, completed_steps }
export interface StepResult {
  step_number: number;
  action: string;
  success: boolean;
  output?: string;
  error?: string;
  file_id?: number;       // present when a file was modified
  filename?: string;
}

export interface ModifyResult {
  goal?: string;
  status: "success" | "failed" | string;
  steps_executed?: number;
  results?: StepResult[];         // present on success
  failed_at_step?: number;        // present on failure
  error?: string;                 // present on failure
  completed_steps?: StepResult[]; // present on failure
  quality?: Record<string, { score: number; grade: string; summary?: string }>;
}

// GET /diff/{file_id} → DiffResult
// The backend returns a unified diff string, NOT split original/modified
export interface DiffResult {
  filename: string;
  lines_added: number;
  lines_removed: number;
  diff: string;   // unified diff text, e.g. "--- a/file\n+++ b/file\n@@ ... @@\n..."
}

// GET /history/{file_id}
export interface HistoryEntry {
  version_id: number;
  saved_at: string;
}

// Frontend-only chat message shape
export interface ChatMessage {
  role: "user" | "ai";
  content: string;
  timestamp: Date;
  file_id?: number;   // set when AI response references a modified file
}