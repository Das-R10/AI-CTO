"use client";
// ===== app/components/UploadPanel.tsx =====
// Single-file upload panel with a "Upload Project" button
// that opens the ProjectUploadModal for full codebase ingestion.

import { useRef, useState } from "react";
import { uploadFile } from "@/lib/api";
import { UploadResult } from "@/lib/types";
import ProjectUploadModal from "./ProjectUploadModal";

interface Props {
  projectId: number;
  onUploaded: () => void;
}

export default function UploadPanel({ projectId, onUploaded }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState("");
  const [showProjectModal, setShowProjectModal] = useState(false);

  async function handleFile(file: File) {
    setUploading(true);
    setError("");
    setResult(null);
    try {
      const res = await uploadFile(projectId, file);
      setResult(res);
      onUploaded();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  function onInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
    e.target.value = "";
  }

  const gradeColor: Record<string, string> = {
    A: "#3fb950", B: "#79c0ff", C: "#e3b341", D: "#f85149", F: "#f85149",
  };

  return (
    <>
      <div className="p-3 space-y-2">
        {/* ── Upload Project button ── */}
        <button
          onClick={() => setShowProjectModal(true)}
          style={{
            width: "100%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 6,
            padding: "7px 10px",
            background: "linear-gradient(135deg, #0d2340 0%, #102a50 100%)",
            border: "1px solid #1f3858",
            borderRadius: 6,
            color: "#79c0ff",
            fontSize: 11,
            fontFamily: '"JetBrains Mono", monospace',
            cursor: "pointer",
            transition: "all 0.15s",
            fontWeight: 600,
            letterSpacing: "0.03em",
          }}
          onMouseEnter={e => {
            e.currentTarget.style.borderColor = "#388bfd";
            e.currentTarget.style.color = "#a5d6ff";
          }}
          onMouseLeave={e => {
            e.currentTarget.style.borderColor = "#1f3858";
            e.currentTarget.style.color = "#79c0ff";
          }}
        >
          <span style={{ fontSize: 13 }}>📂</span>
          Upload Entire Project
        </button>

        {/* ── Single file drop zone ── */}
        <div
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className="rounded-lg flex items-center justify-center cursor-pointer text-xs transition-all"
          style={{
            border: `1px dashed ${dragging ? "#388bfd" : "#21262d"}`,
            background: dragging ? "#1f3858" : "transparent",
            color: "#484f58",
            height: "40px",
            userSelect: "none",
            fontSize: 11,
            fontFamily: '"JetBrains Mono", monospace',
          }}
        >
          {uploading ? (
            <span className="flex items-center gap-2">
              <Spinner size={11} /> uploading…
            </span>
          ) : (
            <span>↑ drop single file or click</span>
          )}
          <input
            ref={inputRef}
            type="file"
            className="hidden"
            onChange={onInputChange}
          />
        </div>

        {result && (
          <div
            className="rounded p-2 text-xs space-y-1"
            style={{ background: "#1c2128", border: "1px solid #21262d" }}
          >
            <div className="flex items-center justify-between">
              <span className="font-mono truncate" style={{ color: "#e6edf3", fontSize: 10 }}>
                file_id: {result.file_id}
              </span>
              <span
                className="font-mono font-bold ml-2"
                style={{ color: gradeColor[result.quality_grade] || "#8b949e" }}
              >
                {result.quality_grade}
              </span>
            </div>
            <div style={{ color: "#484f58", fontSize: 10 }}>
              quality: {result.quality_score}/100
            </div>
            {result.sanitize_warnings.length > 0 && (
              <ul className="space-y-0.5">
                {result.sanitize_warnings.map((w, i) => (
                  <li key={i} style={{ color: "#e3b341", fontSize: 10 }}>⚠ {w}</li>
                ))}
              </ul>
            )}
          </div>
        )}

        {error && (
          <p style={{ color: "#f85149", fontSize: 10 }}>{error}</p>
        )}
      </div>

      {/* ── Project Upload Modal ── */}
      {showProjectModal && (
        <ProjectUploadModal
          projectId={projectId}
          onClose={() => setShowProjectModal(false)}
          onUploaded={() => {
            setShowProjectModal(false);
            onUploaded();
          }}
        />
      )}
    </>
  );
}

function Spinner({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth={2} className="animate-spin">
      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
    </svg>
  );
}
