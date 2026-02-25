// ===== lib/api.ts =====
// Central API layer — all calls to the FastAPI backend go through here.
// Authentication: every request sends X-API-Key header.
// Call setApiKey() immediately after createProject() and on studio page load.

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// In-memory API key — set once after project creation, restored from sessionStorage
let _apiKey = "";

/** Call this immediately after createProject() and on studio page load. */
export function setApiKey(key: string) {
  _apiKey = key;
}

/** Core fetch wrapper — injects auth header, throws on non-2xx */
async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": _apiKey,
      ...(options.headers || {}),
    },
  });

  if (!res.ok) {
    let msg = `Request failed: ${res.status}`;
    try {
      const body = await res.json();
      msg = body?.detail || JSON.stringify(body) || msg;
    } catch {
      msg = (await res.text()) || msg;
    }
    throw new Error(msg);
  }

  return res.json();
}

// ─────────────────────────────────────────────
// PROJECT
// POST /create-project/?name=...
// Returns: { project_id: int, name, api_key, warning }
// ─────────────────────────────────────────────
export async function createProject(name: string) {
  return request<{ project_id: number; name: string; api_key: string; warning: string }>(
    `/create-project/?name=${encodeURIComponent(name)}`,
    { method: "POST" }
  );
}

// ─────────────────────────────────────────────
// FILES
// POST /upload-file/  (multipart/form-data)
// project_id sent as form field (integer)
// X-API-Key still required as header
// ─────────────────────────────────────────────
export async function uploadFile(project_id: number, file: File) {
  const form = new FormData();
  form.append("file", file);
  form.append("project_id", String(project_id)); // FastAPI Form(...) expects string in multipart

  const res = await fetch(`${BASE_URL}/upload-file/`, {
    method: "POST",
    headers: {
      "X-API-Key": _apiKey,
      // DO NOT set Content-Type here — browser sets it with correct boundary
    },
    body: form,
  });

  if (!res.ok) {
    let msg = `Upload failed: ${res.status}`;
    try { msg = (await res.json())?.detail || msg; } catch { /* ignore */ }
    throw new Error(msg);
  }

  return res.json() as Promise<{
    message: string;
    file_id: number;
    quality_score: number;
    quality_grade: string;
    sanitize_warnings: string[];
  }>;
}

// GET /list-files/?project_id=...
// Returns: { project_id, files: [{ file_id, filename, lines }] }
export async function listFiles(project_id: number) {
  return request<{ project_id: number; files: import("./types").FileItem[] }>(
    `/list-files/?project_id=${project_id}`
  );
}

// GET /get-file/?project_id=...&filename=...
// Returns: { file_id, filename, content }
export async function getFile(project_id: number, filename: string) {
  return request<import("./types").FileContent>(
    `/get-file/?project_id=${project_id}&filename=${encodeURIComponent(filename)}`
  );
}

// ─────────────────────────────────────────────
// MODIFY CODE
// POST /modify-code/?query=...&project_id=...
// Both are QUERY PARAMS — backend uses FastAPI query param parsing, not body
// Returns run_agent() result: { goal, status, steps_executed, results[] }
//   OR on failure: { goal, status, failed_at_step, error, completed_steps }
// ─────────────────────────────────────────────
export async function modifyCode(project_id: number, query: string) {
  return request<import("./types").ModifyResult>(
    `/modify-code/?query=${encodeURIComponent(query)}&project_id=${project_id}`,
    { method: "POST" }
  );
}

// ─────────────────────────────────────────────
// DIFF
// GET /diff/{file_id}
// Returns: { filename, lines_added, lines_removed, diff }
// `diff` is a unified diff string (not original/modified split)
// ─────────────────────────────────────────────
export async function getDiff(file_id: number) {
  return request<import("./types").DiffResult>(`/diff/${file_id}`);
}

// ─────────────────────────────────────────────
// ROLLBACK
// POST /rollback/?file_id=...
// file_id is a QUERY PARAM — not a JSON body
// Returns: { message: string }
// ─────────────────────────────────────────────
export async function rollback(file_id: number) {
  return request<{ message: string }>(
    `/rollback/?file_id=${file_id}`,
    { method: "POST" }
  );
}

// GET /history/{file_id}
export async function getHistory(file_id: number) {
  return request<{ file_id: number; versions: import("./types").HistoryEntry[] }>(
    `/history/${file_id}`
  );
}

// GET /history/{file_id} — alias used by InsightsPanel
export async function getFileHistory(file_id: number) {
  return request<{ file_id: number; versions: import("./types").HistoryEntry[] }>(
    `/history/${file_id}`
  );
}

// GET /quality/project/?project_id=...
// Returns: { overall_score, overall_grade, files: { [filename]: { score, grade, summary } } }
export async function getProjectQuality(project_id: number) {
  return request<Record<string, { score: number; grade: string; summary?: string }>>(
    `/quality/project/?project_id=${project_id}`
  );
}

// GET /project-health/?project_id=...
export async function getProjectHealth(project_id: number) {
  return request<{
    quality_score: number;
    quality_grade: string;
    test_pass_rate: number | string;
    test_files_count: number;
    memory_entries: number;
    files_modified_24h: string[];
    open_jobs: number;
    dependency_risk: { risk_level: string; high_risk_files: string[] };
  }>(`/project-health/?project_id=${project_id}`);
}

// GET /pending-approvals/?project_id=...
export async function getPendingApprovals(project_id: number) {
  return request<{ approvals: Array<{ id: number; risk_flags: Record<string, unknown>; status: string; created_at: string }> }>(
    `/pending-approvals/?project_id=${project_id}`
  );
}

// POST /approve/{approval_id}?project_id=...
export async function approveChange(approval_id: number, project_id: number) {
  return request<{ message: string }>(
    `/approve/${approval_id}?project_id=${project_id}`,
    { method: "POST" }
  );
}

// POST /reject/{approval_id}?project_id=...
export async function rejectChange(approval_id: number, project_id: number) {
  return request<{ message: string }>(
    `/reject/${approval_id}?project_id=${project_id}`,
    { method: "POST" }
  );
}