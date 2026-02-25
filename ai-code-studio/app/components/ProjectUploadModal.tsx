"use client";
// ===== app/components/ProjectUploadModal.tsx =====
// Handles entire project upload: folder drag-drop, ZIP extraction,
// file filtering, tree preview, and batched upload to /upload-file/.

import { useCallback, useRef, useState } from "react";
import { uploadFile } from "@/lib/api";

// ─── Config ───────────────────────────────────────────────────────────────────

const ALLOWED_EXTENSIONS = new Set([
  // Web / frontend
  "ts", "tsx", "js", "jsx", "mjs", "cjs",
  "html", "htm", "css", "scss", "sass", "less",
  "vue", "svelte",
  // Backend
  "py", "go", "rs", "java", "kt", "rb", "php",
  "cs", "cpp", "c", "h", "hpp",
  "sh", "bash", "zsh",
  // Config / data
  "json", "yaml", "yml", "toml", "env", "ini",
  "xml", "graphql", "gql", "prisma", "sql",
  // Docs
  "md", "mdx", "txt",
  // Build / infra
  "dockerfile", "tf", "hcl",
]);

const IGNORED_DIRS = new Set([
  "node_modules", ".git", ".next", "__pycache__", ".venv", "venv",
  "env", "dist", "build", ".cache", "coverage", ".nyc_output",
  "target", ".gradle", "vendor", ".idea", ".vscode", "out",
  ".turbo", ".svelte-kit", ".nuxt", "storybook-static",
]);

const IGNORED_PATTERNS = [
  /\.DS_Store$/, /Thumbs\.db$/, /\.log$/, /\.lock$/,
  /package-lock\.json$/, /yarn\.lock$/, /pnpm-lock\.yaml$/,
  /\.min\.(js|css)$/, /\.map$/, /\.pyc$/, /\.egg-info/,
];

const MAX_FILE_SIZE = 500 * 1024; // 500 KB per file
const MAX_FILES = 200;

// ─── Types ────────────────────────────────────────────────────────────────────

interface TreeFile {
  path: string;      // relative path e.g. "src/components/App.tsx"
  file: File;
  size: number;
  skipped: boolean;
  skipReason?: string;
}

type UploadStatus = "idle" | "parsing" | "preview" | "uploading" | "done" | "error";

interface FileUploadState {
  path: string;
  status: "pending" | "uploading" | "done" | "error";
  error?: string;
  fileId?: number;
  qualityGrade?: string;
}

interface Props {
  projectId: number;
  onClose: () => void;
  onUploaded: () => void;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function shouldSkipPath(path: string): { skip: boolean; reason?: string } {
  const parts = path.split("/");
  for (const part of parts) {
    if (IGNORED_DIRS.has(part)) return { skip: true, reason: `ignored dir: ${part}` };
  }
  for (const pat of IGNORED_PATTERNS) {
    if (pat.test(path)) return { skip: true, reason: "ignored pattern" };
  }
  return { skip: false };
}

function shouldSkipFile(file: File, path: string): { skip: boolean; reason?: string } {
  const pathCheck = shouldSkipPath(path);
  if (pathCheck.skip) return pathCheck;

  const ext = path.split(".").pop()?.toLowerCase() ?? "";
  const base = path.split("/").pop()?.toLowerCase() ?? "";

  // Files without extensions that we allow explicitly
  const allowedNoExt = ["dockerfile", "makefile", "procfile", "gemfile", "rakefile"];
  if (!ext && !allowedNoExt.includes(base)) {
    return { skip: true, reason: "no extension" };
  }

  if (ext && !ALLOWED_EXTENSIONS.has(ext) && !allowedNoExt.includes(base)) {
    return { skip: true, reason: `unsupported type: .${ext}` };
  }

  if (file.size > MAX_FILE_SIZE) {
    return { skip: true, reason: `too large (${(file.size / 1024).toFixed(0)} KB)` };
  }

  return { skip: false };
}

async function extractFromZip(zipFile: File): Promise<TreeFile[]> {
  // Dynamic import of JSZip — if not available, we surface an error
  try {
    const JSZip = (await import("jszip")).default;
    const zip = await JSZip.loadAsync(zipFile);
    const results: TreeFile[] = [];

    for (const [relativePath, entry] of Object.entries(zip.files)) {
      if (entry.dir) continue;

      // Strip top-level folder if all files share one
      const cleanPath = relativePath.replace(/^[^/]+\//, "");
      if (!cleanPath) continue;

      const skipResult = shouldSkipPath(relativePath);
      const ext = cleanPath.split(".").pop()?.toLowerCase() ?? "";
      const base = cleanPath.split("/").pop()?.toLowerCase() ?? "";
      const allowedNoExt = ["dockerfile", "makefile", "procfile", "gemfile"];
      const typeAllowed = ALLOWED_EXTENSIONS.has(ext) || allowedNoExt.includes(base);

      if (skipResult.skip || !typeAllowed) {
        results.push({ path: cleanPath, file: new File([], cleanPath), size: 0, skipped: true, skipReason: skipResult.reason ?? `unsupported: .${ext}` });
        continue;
      }

      const content = await entry.async("uint8array");
      if (content.length > MAX_FILE_SIZE) {
        results.push({ path: cleanPath, file: new File([], cleanPath), size: content.length, skipped: true, skipReason: "too large" });
        continue;
      }

      const file = new File([content], cleanPath.split("/").pop()!, { type: "text/plain" });
      results.push({ path: cleanPath, file, size: content.length, skipped: false });
    }

    return results;
  } catch {
    throw new Error("Failed to read ZIP. Make sure jszip is installed: npm install jszip");
  }
}

function extractFromFileList(fileList: FileList | File[]): TreeFile[] {
  const results: TreeFile[] = [];
  const files = Array.from(fileList);

  for (const file of files) {
    // webkitRelativePath is set when using folder picker
    const relativePath = (file as { webkitRelativePath?: string }).webkitRelativePath || file.name;
    // Normalize: strip top-level folder name
    const parts = relativePath.split("/");
    const cleanPath = parts.length > 1 ? parts.slice(1).join("/") : relativePath;

    const skipResult = shouldSkipFile(file, relativePath);
    results.push({
      path: cleanPath || file.name,
      file,
      size: file.size,
      skipped: skipResult.skip,
      skipReason: skipResult.reason,
    });
  }

  return results;
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: FileUploadState["status"] }) {
  const configs = {
    pending:   { color: "#3d4a5c", icon: "○" },
    uploading: { color: "#e3b341", icon: "◎" },
    done:      { color: "#3fb950", icon: "✓" },
    error:     { color: "#f85149", icon: "✗" },
  };
  const { color, icon } = configs[status];
  return <span style={{ color, fontSize: 11, flexShrink: 0, width: 14, textAlign: "center" }}>{icon}</span>;
}

function GradeBadge({ grade }: { grade: string }) {
  const colors: Record<string, string> = { A: "#3fb950", B: "#79c0ff", C: "#e3b341", D: "#f85149", F: "#f85149" };
  return (
    <span style={{ color: colors[grade] || "#8b949e", fontSize: 10, fontWeight: 700, marginLeft: 4 }}>
      {grade}
    </span>
  );
}

function ProgressBar({ value, total }: { value: number; total: number }) {
  const pct = total > 0 ? Math.round((value / total) * 100) : 0;
  return (
    <div style={{ height: 3, background: "#1e2430", borderRadius: 2, overflow: "hidden", width: "100%" }}>
      <div
        style={{
          height: "100%",
          width: `${pct}%`,
          background: "linear-gradient(90deg, #388bfd, #3fb950)",
          borderRadius: 2,
          transition: "width 0.2s ease",
        }}
      />
    </div>
  );
}

function FileTree({ files }: { files: TreeFile[] }) {
  const included = files.filter(f => !f.skipped);
  const skipped = files.filter(f => f.skipped);

  // Group by directory
  const dirs = new Map<string, TreeFile[]>();
  for (const f of included) {
    const parts = f.path.split("/");
    const dir = parts.length > 1 ? parts[0] : "(root)";
    if (!dirs.has(dir)) dirs.set(dir, []);
    dirs.get(dir)!.push(f);
  }

  return (
    <div style={{ fontSize: 11, fontFamily: '"JetBrains Mono", monospace' }}>
      {Array.from(dirs.entries()).map(([dir, dirFiles]) => (
        <div key={dir} style={{ marginBottom: 6 }}>
          <div style={{ color: "#79c0ff", marginBottom: 2, fontSize: 10, letterSpacing: "0.05em" }}>
            📁 {dir}/
          </div>
          {dirFiles.map((f) => {
            const name = f.path.split("/").pop()!;
            const ext = name.split(".").pop()?.toLowerCase() ?? "";
            const extColors: Record<string, string> = {
              ts: "#3178c6", tsx: "#61dafb", js: "#f7df1e", jsx: "#61dafb",
              py: "#3572A5", go: "#00ADD8", rs: "#dea584", json: "#cbcb41",
              css: "#563d7c", html: "#e34c26", md: "#83a598", yaml: "#e3b341",
            };
            return (
              <div key={f.path} style={{ display: "flex", gap: 6, padding: "1px 0 1px 14px", color: "#8b949e", alignItems: "center" }}>
                <span style={{ color: extColors[ext] ?? "#6e8098", fontSize: 9 }}>■</span>
                <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{name}</span>
                <span style={{ marginLeft: "auto", color: "#2d3a4a", flexShrink: 0, fontSize: 10 }}>
                  {(f.size / 1024).toFixed(0)}k
                </span>
              </div>
            );
          })}
        </div>
      ))}

      {skipped.length > 0 && (
        <div style={{ marginTop: 8, padding: "6px 8px", background: "#0d1017", borderRadius: 4, border: "1px solid #1e2430" }}>
          <div style={{ color: "#3d4a5c", fontSize: 10, marginBottom: 4 }}>
            ⊘ {skipped.length} files skipped
          </div>
          {skipped.slice(0, 5).map((f) => (
            <div key={f.path} style={{ color: "#2d3a4a", fontSize: 10, padding: "1px 0" }}>
              {f.path.split("/").pop()} — {f.skipReason}
            </div>
          ))}
          {skipped.length > 5 && (
            <div style={{ color: "#2d3a4a", fontSize: 10 }}>…and {skipped.length - 5} more</div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Main Modal ───────────────────────────────────────────────────────────────

export default function ProjectUploadModal({ projectId, onClose, onUploaded }: Props) {
  const [status, setStatus] = useState<UploadStatus>("idle");
  const [dragging, setDragging] = useState(false);
  const [treeFiles, setTreeFiles] = useState<TreeFile[]>([]);
  const [uploadStates, setUploadStates] = useState<FileUploadState[]>([]);
  const [errorMsg, setErrorMsg] = useState("");
  const [doneCount, setDoneCount] = useState(0);
  const [errorCount, setErrorCount] = useState(0);

  const folderInputRef = useRef<HTMLInputElement>(null);
  const zipInputRef = useRef<HTMLInputElement>(null);

  const includedFiles = treeFiles.filter(f => !f.skipped);

  // ── Parsing ──────────────────────────────────────────────────────────────

  async function processFiles(files: TreeFile[]) {
    const capped = files.slice(0, MAX_FILES + 50); // allow parse of more, we'll warn
    setTreeFiles(capped);
    setStatus("preview");
  }

  async function handleFolderInput(e: React.ChangeEvent<HTMLInputElement>) {
    const fileList = e.target.files;
    if (!fileList || fileList.length === 0) return;
    setStatus("parsing");
    setErrorMsg("");
    try {
      const files = extractFromFileList(fileList);
      await processFiles(files);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to read folder");
      setStatus("error");
    }
    e.target.value = "";
  }

  async function handleZipInput(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setStatus("parsing");
    setErrorMsg("");
    try {
      const files = await extractFromZip(file);
      await processFiles(files);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to extract ZIP");
      setStatus("error");
    }
    e.target.value = "";
  }

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    setStatus("parsing");
    setErrorMsg("");

    try {
      const items = e.dataTransfer.items;
      const allFiles: File[] = [];

      async function readEntry(entry: FileSystemEntry, prefix = ""): Promise<void> {
        if (entry.isFile) {
          const fileEntry = entry as FileSystemFileEntry;
          await new Promise<void>((res) => {
            fileEntry.file((f) => {
              Object.defineProperty(f, "webkitRelativePath", { value: prefix + f.name });
              allFiles.push(f);
              res();
            });
          });
        } else if (entry.isDirectory) {
          const dirEntry = entry as FileSystemDirectoryEntry;
          const dirName = prefix + dirEntry.name + "/";
          const reader = dirEntry.createReader();
          await new Promise<void>((res) => {
            function readBatch() {
              reader.readEntries(async (entries) => {
                if (entries.length === 0) { res(); return; }
                for (const child of entries) await readEntry(child, dirName);
                readBatch();
              });
            }
            readBatch();
          });
        }
      }

      if (items && items.length > 0) {
        const entries: FileSystemEntry[] = [];
        for (let i = 0; i < items.length; i++) {
          const entry = items[i].webkitGetAsEntry?.();
          if (entry) entries.push(entry);
        }
        for (const entry of entries) await readEntry(entry);
      }

      if (allFiles.length === 0) {
        // Fallback: maybe it's a ZIP
        const droppedFile = e.dataTransfer.files[0];
        if (droppedFile?.name.endsWith(".zip")) {
          const files = await extractFromZip(droppedFile);
          await processFiles(files);
          return;
        }
        throw new Error("No files found. Try using the folder or ZIP buttons.");
      }

      const files = extractFromFileList(allFiles);
      await processFiles(files);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Drop failed");
      setStatus("error");
    }
  }, []);

  // ── Uploading ─────────────────────────────────────────────────────────────

  async function startUpload() {
    const toUpload = includedFiles.slice(0, MAX_FILES);
    const states: FileUploadState[] = toUpload.map(f => ({ path: f.path, status: "pending" }));
    setUploadStates(states);
    setStatus("uploading");
    setDoneCount(0);
    setErrorCount(0);

    let done = 0;
    let errors = 0;

    // Upload with concurrency of 3
    const CONCURRENCY = 3;
    let idx = 0;

    async function worker() {
      while (idx < toUpload.length) {
        const i = idx++;
        const treeFile = toUpload[i];

        setUploadStates(prev => {
          const next = [...prev];
          next[i] = { ...next[i], status: "uploading" };
          return next;
        });

        try {
          // Create a properly named File object
          const namedFile = new File([treeFile.file], treeFile.path.split("/").pop()!, {
            type: treeFile.file.type || "text/plain",
          });

          const result = await uploadFile(projectId, namedFile);

          done++;
          setDoneCount(d => d + 1);
          setUploadStates(prev => {
            const next = [...prev];
            next[i] = { ...next[i], status: "done", fileId: result.file_id, qualityGrade: result.quality_grade };
            return next;
          });
        } catch (err) {
          errors++;
          setErrorCount(e => e + 1);
          setUploadStates(prev => {
            const next = [...prev];
            next[i] = { ...next[i], status: "error", error: err instanceof Error ? err.message : "upload failed" };
            return next;
          });
        }
      }
    }

    await Promise.all(Array.from({ length: CONCURRENCY }, () => worker()));

    setStatus("done");
    onUploaded();
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 1000,
        background: "rgba(0,0,0,0.75)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        backdropFilter: "blur(4px)",
      }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        style={{
          width: "min(680px, 95vw)",
          maxHeight: "85vh",
          display: "flex",
          flexDirection: "column",
          background: "#0d1117",
          border: "1px solid #21262d",
          borderRadius: 10,
          fontFamily: '"JetBrains Mono", monospace',
          overflow: "hidden",
          boxShadow: "0 24px 64px rgba(0,0,0,0.6)",
        }}
      >
        {/* ── Header ── */}
        <div style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "14px 18px",
          borderBottom: "1px solid #21262d",
          background: "#080b0f",
          flexShrink: 0,
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontSize: 16 }}>🐉</span>
            <div>
              <div style={{ color: "#e6edf3", fontSize: 13, fontWeight: 600 }}>Upload Project</div>
              <div style={{ color: "#3d4a5c", fontSize: 10, marginTop: 1 }}>
                Drop your entire frontend + backend codebase
              </div>
            </div>
          </div>
          <button
            onClick={onClose}
            style={{
              background: "none", border: "none", cursor: "pointer",
              color: "#484f58", fontSize: 18, lineHeight: 1, padding: "2px 6px",
              borderRadius: 4,
            }}
            onMouseEnter={e => (e.currentTarget.style.color = "#8b949e")}
            onMouseLeave={e => (e.currentTarget.style.color = "#484f58")}
          >×</button>
        </div>

        {/* ── Body ── */}
        <div style={{ flex: 1, overflowY: "auto", padding: 18 }}>

          {/* ── IDLE: Drop zone ── */}
          {(status === "idle" || status === "error") && (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {/* Drop area */}
              <div
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={handleDrop}
                style={{
                  border: `2px dashed ${dragging ? "#388bfd" : "#21262d"}`,
                  borderRadius: 8,
                  padding: "36px 24px",
                  textAlign: "center",
                  background: dragging ? "rgba(56,139,253,0.06)" : "#080b0f",
                  transition: "all 0.15s",
                  cursor: "default",
                }}
              >
                <div style={{ fontSize: 32, marginBottom: 10 }}>📂</div>
                <div style={{ color: "#c9d1d9", fontSize: 13, marginBottom: 6 }}>
                  Drag & drop your project folder or ZIP
                </div>
                <div style={{ color: "#484f58", fontSize: 11, marginBottom: 20, lineHeight: 1.6 }}>
                  All code files are uploaded. node_modules, .git,<br/>
                  build artifacts and binary files are automatically excluded.
                </div>
                <div style={{ display: "flex", gap: 10, justifyContent: "center", flexWrap: "wrap" }}>
                  <button
                    onClick={() => folderInputRef.current?.click()}
                    style={{
                      padding: "8px 18px",
                      background: "#1c2128",
                      border: "1px solid #30363d",
                      borderRadius: 6,
                      color: "#c9d1d9",
                      fontSize: 12,
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                      fontFamily: "inherit",
                    }}
                    onMouseEnter={e => (e.currentTarget.style.borderColor = "#388bfd")}
                    onMouseLeave={e => (e.currentTarget.style.borderColor = "#30363d")}
                  >
                    📁 Select Folder
                  </button>
                  <button
                    onClick={() => zipInputRef.current?.click()}
                    style={{
                      padding: "8px 18px",
                      background: "#1c2128",
                      border: "1px solid #30363d",
                      borderRadius: 6,
                      color: "#c9d1d9",
                      fontSize: 12,
                      cursor: "pointer",
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                      fontFamily: "inherit",
                    }}
                    onMouseEnter={e => (e.currentTarget.style.borderColor = "#388bfd")}
                    onMouseLeave={e => (e.currentTarget.style.borderColor = "#30363d")}
                  >
                    🗜 Upload ZIP
                  </button>
                </div>
                <input ref={folderInputRef} type="file" style={{ display: "none" }}
                  // @ts-expect-error webkitdirectory is non-standard
                  webkitdirectory="" mozdirectory="" directory=""
                  multiple onChange={handleFolderInput} />
                <input ref={zipInputRef} type="file" style={{ display: "none" }}
                  accept=".zip" onChange={handleZipInput} />
              </div>

              {/* Supported types hint */}
              <div style={{
                padding: "10px 14px",
                background: "#080b0f",
                border: "1px solid #1e2430",
                borderRadius: 6,
                fontSize: 10,
                color: "#3d4a5c",
                lineHeight: 1.7,
              }}>
                <span style={{ color: "#484f58" }}>Supported: </span>
                {".ts .tsx .js .jsx .py .go .rs .java .css .html .json .yaml .md .sql .sh …"}
                <br/>
                <span style={{ color: "#484f58" }}>Ignored: </span>
                {"node_modules/ .git/ __pycache__/ dist/ build/ *.lock *.min.js *.map"}
                <br/>
                <span style={{ color: "#484f58" }}>Limits: </span>
                {`max ${MAX_FILES} files · max ${MAX_FILE_SIZE / 1024}KB per file`}
              </div>

              {errorMsg && (
                <div style={{ color: "#f85149", fontSize: 11, padding: "8px 12px", background: "rgba(248,81,73,0.08)", border: "1px solid rgba(248,81,73,0.2)", borderRadius: 6 }}>
                  ✗ {errorMsg}
                </div>
              )}
            </div>
          )}

          {/* ── PARSING ── */}
          {status === "parsing" && (
            <div style={{ textAlign: "center", padding: "40px 0", color: "#484f58", fontSize: 12 }}>
              <Spinner size={24} color="#388bfd" />
              <div style={{ marginTop: 14 }}>Scanning project structure…</div>
            </div>
          )}

          {/* ── PREVIEW ── */}
          {status === "preview" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {/* Stats bar */}
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {[
                  { label: "to upload", value: Math.min(includedFiles.length, MAX_FILES), color: "#3fb950" },
                  { label: "skipped", value: treeFiles.filter(f => f.skipped).length, color: "#484f58" },
                  { label: "total size", value: `${(includedFiles.reduce((s, f) => s + f.size, 0) / 1024).toFixed(0)} KB`, color: "#79c0ff" },
                ].map(s => (
                  <div key={s.label} style={{
                    flex: 1, minWidth: 100, padding: "8px 12px",
                    background: "#080b0f", border: "1px solid #1e2430",
                    borderRadius: 6, textAlign: "center",
                  }}>
                    <div style={{ fontSize: 18, color: s.color, fontWeight: 700 }}>{s.value}</div>
                    <div style={{ fontSize: 10, color: "#3d4a5c", marginTop: 2 }}>{s.label}</div>
                  </div>
                ))}
              </div>

              {includedFiles.length > MAX_FILES && (
                <div style={{ color: "#e3b341", fontSize: 11, padding: "6px 10px", background: "rgba(227,179,65,0.08)", border: "1px solid rgba(227,179,65,0.2)", borderRadius: 6 }}>
                  ⚠ {includedFiles.length} files detected — only the first {MAX_FILES} will be uploaded.
                </div>
              )}

              {/* File tree */}
              <div style={{
                maxHeight: 300,
                overflowY: "auto",
                background: "#080b0f",
                border: "1px solid #1e2430",
                borderRadius: 6,
                padding: "10px 12px",
              }}>
                <FileTree files={treeFiles} />
              </div>
            </div>
          )}

          {/* ── UPLOADING ── */}
          {(status === "uploading" || status === "done") && (
            <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
              {/* Overall progress */}
              <div style={{ padding: "12px 14px", background: "#080b0f", border: "1px solid #1e2430", borderRadius: 6 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8, fontSize: 11 }}>
                  <span style={{ color: "#c9d1d9" }}>
                    {status === "done" ? "Upload complete" : "Uploading project…"}
                  </span>
                  <span style={{ color: "#484f58" }}>
                    {doneCount + errorCount} / {uploadStates.length}
                  </span>
                </div>
                <ProgressBar value={doneCount + errorCount} total={uploadStates.length} />
                <div style={{ display: "flex", gap: 16, marginTop: 8, fontSize: 10 }}>
                  <span style={{ color: "#3fb950" }}>✓ {doneCount} done</span>
                  {errorCount > 0 && <span style={{ color: "#f85149" }}>✗ {errorCount} failed</span>}
                </div>
              </div>

              {/* Per-file list */}
              <div style={{
                maxHeight: 280,
                overflowY: "auto",
                background: "#080b0f",
                border: "1px solid #1e2430",
                borderRadius: 6,
                padding: "6px 0",
              }}>
                {uploadStates.map((f) => (
                  <div key={f.path} style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    padding: "4px 12px",
                    fontSize: 11,
                    borderBottom: "1px solid #0d1117",
                  }}>
                    <StatusBadge status={f.status} />
                    <span style={{ flex: 1, color: "#8b949e", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {f.path}
                    </span>
                    {f.qualityGrade && <GradeBadge grade={f.qualityGrade} />}
                    {f.error && (
                      <span style={{ color: "#f85149", fontSize: 10, flexShrink: 0 }}>
                        {f.error.slice(0, 40)}
                      </span>
                    )}
                    {f.status === "uploading" && <MiniSpinner />}
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>

        {/* ── Footer ── */}
        <div style={{
          borderTop: "1px solid #21262d",
          padding: "12px 18px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          background: "#080b0f",
          flexShrink: 0,
          gap: 10,
        }}>
          <div style={{ fontSize: 10, color: "#3d4a5c" }}>
            {status === "preview" && `${Math.min(includedFiles.length, MAX_FILES)} files ready`}
            {status === "uploading" && "uploading with concurrency 3…"}
            {status === "done" && `${doneCount} files uploaded successfully`}
          </div>

          <div style={{ display: "flex", gap: 8 }}>
            {status !== "uploading" && (
              <button
                onClick={status === "done" ? onClose : () => { setStatus("idle"); setTreeFiles([]); setErrorMsg(""); }}
                style={{
                  padding: "7px 16px",
                  background: "transparent",
                  border: "1px solid #30363d",
                  borderRadius: 6,
                  color: "#8b949e",
                  fontSize: 12,
                  cursor: "pointer",
                  fontFamily: "inherit",
                }}
              >
                {status === "done" ? "Close" : "Cancel"}
              </button>
            )}

            {status === "preview" && (
              <button
                onClick={startUpload}
                disabled={includedFiles.length === 0}
                style={{
                  padding: "7px 20px",
                  background: includedFiles.length === 0 ? "#1c2128" : "#388bfd",
                  border: "none",
                  borderRadius: 6,
                  color: includedFiles.length === 0 ? "#484f58" : "#fff",
                  fontSize: 12,
                  cursor: includedFiles.length === 0 ? "not-allowed" : "pointer",
                  fontFamily: "inherit",
                  fontWeight: 600,
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                }}
              >
                ↑ Upload {Math.min(includedFiles.length, MAX_FILES)} Files
              </button>
            )}

            {status === "done" && doneCount > 0 && (
              <div style={{ display: "flex", alignItems: "center", gap: 6, color: "#3fb950", fontSize: 12 }}>
                ✓ Project uploaded successfully
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Micro components ─────────────────────────────────────────────────────────

function Spinner({ size = 16, color = "#388bfd" }: { size?: number; color?: string }) {
  return (
    <div style={{ display: "flex", justifyContent: "center", alignItems: "center" }}>
      <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={2}
        style={{ animation: "spin 1s linear infinite" }}>
        <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
        <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
      </svg>
    </div>
  );
}

function MiniSpinner() {
  return (
    <svg width={10} height={10} viewBox="0 0 24 24" fill="none" stroke="#e3b341" strokeWidth={2.5}
      style={{ animation: "spin 0.8s linear infinite", flexShrink: 0 }}>
      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
    </svg>
  );
}