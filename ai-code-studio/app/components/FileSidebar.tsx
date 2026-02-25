"use client";
// ===== app/components/FileSidebar.tsx =====

import { useEffect, useState, useCallback } from "react";
import { listFiles, getFile } from "@/lib/api";
import { FileItem, FileContent } from "@/lib/types";
import UploadPanel from "./UploadPanel";

interface Props {
  projectId: number;
  onFileSelect?: (file: FileContent) => void;
  selectedFileId?: number;
}

// Tiny file-type icon
function FileIcon({ name }: { name: string }) {
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  const colors: Record<string, string> = {
    py: "#3572A5", ts: "#3178c6", tsx: "#3178c6", js: "#f7df1e",
    jsx: "#61dafb", go: "#00ADD8", rs: "#dea584", json: "#cbcb41",
    md: "#83a598", css: "#563d7c", html: "#e34c26", sh: "#89e051",
    sql: "#e38c00",
  };
  const color = colors[ext] ?? "#8b949e";
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" style={{ flexShrink: 0 }}>
      <path d="M2 1h5.5L10 3.5V11H2V1z" stroke={color} strokeWidth="0.8" fill={color} fillOpacity="0.12" />
      <path d="M7 1v3h3" stroke={color} strokeWidth="0.8" fill="none" />
    </svg>
  );
}

export default function FileSidebar({ projectId, onFileSelect, selectedFileId }: Props) {
  const [files, setFiles] = useState<FileItem[]>([]);
  const [loadingId, setLoadingId] = useState<number | null>(null);

  const refresh = useCallback(async () => {
    try {
      const { files: f } = await listFiles(projectId);
      setFiles(f);
    } catch { /* silent */ }
  }, [projectId]);

  useEffect(() => { refresh(); }, [refresh]);

  async function selectFile(f: FileItem) {
    setLoadingId(f.file_id);
    try {
      const content = await getFile(projectId, f.filename);
      onFileSelect?.(content);
    } catch { /* ignore */ }
    finally { setLoadingId(null); }
  }

  return (
    <aside style={{
      width: "200px",
      minWidth: "200px",
      display: "flex",
      flexDirection: "column",
      background: "#0d1017",
      borderRight: "1px solid #1e2430",
      fontFamily: '"JetBrains Mono", monospace',
      overflow: "hidden",
    }}>
      {/* Upload zone */}
      <div style={{ borderBottom: "1px solid #1e2430" }}>
        <div style={{ padding: "8px 12px 2px", fontSize: "9px", color: "#3d4a5c", letterSpacing: "0.12em" }}>
          UPLOAD
        </div>
        <UploadPanel projectId={projectId} onUploaded={refresh} />
      </div>

      {/* File tree */}
      <div style={{ flex: 1, overflowY: "auto", paddingTop: 4 }}>
        <div style={{ padding: "6px 12px 4px", fontSize: "9px", color: "#3d4a5c", letterSpacing: "0.12em", position: "sticky", top: 0, background: "#0d1017" }}>
          FILES
        </div>

        {files.length === 0 ? (
          <p style={{ padding: "4px 12px", fontSize: "11px", color: "#2d3a4a" }}>no files yet</p>
        ) : (
          files.map((f) => {
            const active = selectedFileId === f.file_id;
            const isLoading = loadingId === f.file_id;
            return (
              <button
                key={f.file_id}
                onClick={() => selectFile(f)}
                style={{
                  width: "100%",
                  textAlign: "left",
                  display: "flex",
                  alignItems: "center",
                  gap: 7,
                  padding: "5px 12px",
                  fontSize: "11px",
                  fontFamily: "inherit",
                  background: active ? "rgba(56,139,253,0.08)" : "transparent",
                  color: active ? "#79c0ff" : "#6e8098",
                  borderLeft: active ? "2px solid #388bfd" : "2px solid transparent",
                  border: "none",
                  cursor: "pointer",
                  transition: "all 0.12s",
                  overflow: "hidden",
                  whiteSpace: "nowrap",
                  textOverflow: "ellipsis",
                }}
                onMouseEnter={(e) => { if (!active) e.currentTarget.style.color = "#8b9eb8"; }}
                onMouseLeave={(e) => { if (!active) e.currentTarget.style.color = "#6e8098"; }}
              >
                {isLoading
                  ? <span style={{ color: "#3d4a5c" }}>loading…</span>
                  : <>
                      <FileIcon name={f.filename} />
                      <span style={{ overflow:"hidden", textOverflow:"ellipsis" }}>{f.filename}</span>
                      <span style={{ marginLeft:"auto", fontSize:"9px", color:"#3d4a5c", flexShrink:0 }}>{f.lines}</span>
                    </>
                }
              </button>
            );
          })
        )}
      </div>
    </aside>
  );
}