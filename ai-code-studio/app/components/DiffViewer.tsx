"use client";
// ===== app/components/DiffViewer.tsx =====
// GET /diff/{file_id} → { filename, lines_added, lines_removed, diff }
// diff is a unified diff string. POST /rollback/?file_id=... uses query param.

import { useEffect, useState } from "react";
import { getDiff, rollback } from "@/lib/api";
import { DiffResult } from "@/lib/types";

interface Props {
  fileId: number | null;
  onRolledBack?: () => void;
}

interface DiffLine {
  type: "add" | "remove" | "context" | "header";
  content: string;
}

function parseUnifiedDiff(diffText: string): DiffLine[] {
  if (!diffText || diffText === "No changes detected") return [];
  return diffText.split("\n").map((line): DiffLine => {
    if (line.startsWith("+++") || line.startsWith("---") || line.startsWith("@@"))
      return { type: "header", content: line };
    if (line.startsWith("+")) return { type: "add", content: line.slice(1) };
    if (line.startsWith("-")) return { type: "remove", content: line.slice(1) };
    return { type: "context", content: line.startsWith(" ") ? line.slice(1) : line };
  });
}

export default function DiffViewer({ fileId, onRolledBack }: Props) {
  const [diff, setDiff] = useState<DiffResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState<{ msg: string; ok: boolean } | null>(null);

  useEffect(() => {
    if (!fileId) return;
    setLoading(true);
    setDiff(null);
    getDiff(fileId)
      .then(setDiff)
      .catch(() => setDiff(null))
      .finally(() => setLoading(false));
  }, [fileId]);

  function showToast(msg: string, ok: boolean) {
    setToast({ msg, ok });
    setTimeout(() => setToast(null), 3000);
  }

  async function handleRollback() {
    if (!fileId) return;
    try {
      const res = await rollback(fileId);
      showToast(res.message || "rolled back", true);
      setDiff(null);
      onRolledBack?.();
    } catch (e: unknown) {
      showToast(e instanceof Error ? e.message : "rollback failed", false);
    }
  }

  function handleApprove() {
    showToast("changes committed ✓", true);
    setDiff(null);
  }

  if (!fileId) return <Empty text="─  awaiting diff" />;
  if (loading) return <Empty text="loading diff…" spin />;
  if (!diff) return <Empty text="no diff available" />;

  const noDiff = !diff.diff || diff.diff === "No changes detected";
  const parsed = noDiff ? [] : parseUnifiedDiff(diff.diff);

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      height: "100%",
      background: "#080b0f",
      fontFamily: '"JetBrains Mono", "Fira Code", monospace',
      position: "relative",
    }}>
      {/* Stats bar */}
      <div style={{
        padding: "6px 14px",
        borderBottom: "1px solid #1e2430",
        display: "flex",
        alignItems: "center",
        gap: 10,
        background: "#0a0d12",
        flexShrink: 0,
      }}>
        <span style={{ fontSize:"10px", color:"#3d4a5c", letterSpacing:"0.1em" }}>CHANGES</span>
        <span style={{ marginLeft:4, fontSize:"11px", color:"#6e8098", overflow:"hidden", textOverflow:"ellipsis", whiteSpace:"nowrap" }}>
          {diff.filename}
        </span>
        <div style={{ marginLeft:"auto", display:"flex", gap:10, flexShrink:0 }}>
          <span style={{ fontSize:"11px", color:"#3fb950" }}>+{diff.lines_added}</span>
          <span style={{ fontSize:"11px", color:"#f85149" }}>−{diff.lines_removed}</span>
        </div>
      </div>

      {/* Diff lines */}
      <div style={{ flex:1, overflowY:"auto", padding:"8px 0" }}>
        {noDiff ? (
          <div style={{ padding:"12px 16px", fontSize:"12px", color:"#3d4a5c" }}>no changes detected</div>
        ) : (
          parsed.map((line, i) => {
            const isAdd = line.type === "add";
            const isRm  = line.type === "remove";
            const isHdr = line.type === "header";
            return (
              <div
                key={i}
                style={{
                  display: "flex",
                  alignItems: "flex-start",
                  padding: "0 0",
                  background: isAdd ? "rgba(63,185,80,0.07)" : isRm ? "rgba(248,81,73,0.07)" : "transparent",
                  borderLeft: isAdd ? "2px solid rgba(63,185,80,0.4)" : isRm ? "2px solid rgba(248,81,73,0.4)" : "2px solid transparent",
                }}
              >
                {/* gutter */}
                <span style={{
                  width: 20,
                  textAlign: "center",
                  flexShrink: 0,
                  fontSize: "11px",
                  lineHeight: "20px",
                  color: isAdd ? "#3fb950" : isRm ? "#f85149" : isHdr ? "#388bfd" : "#2d3a4a",
                  userSelect: "none",
                }}>
                  {isAdd ? "+" : isRm ? "−" : isHdr ? "·" : " "}
                </span>
                {/* content */}
                <span style={{
                  flex: 1,
                  fontSize: "11px",
                  lineHeight: "20px",
                  color: isAdd ? "#aff5b4" : isRm ? "#ffa198" : isHdr ? "#79c0ff" : "#8b949e",
                  whiteSpace: "pre",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  fontFamily: "inherit",
                }}>
                  {line.content || " "}
                </span>
              </div>
            );
          })
        )}
      </div>

      {/* Action bar */}
      <div style={{
        display: "flex",
        gap: 8,
        padding: "10px 14px",
        borderTop: "1px solid #1e2430",
        background: "#0a0d12",
        flexShrink: 0,
      }}>
        <button
          onClick={handleApprove}
          style={{
            flex: 1,
            padding: "7px 0",
            background: "rgba(63,185,80,0.08)",
            border: "1px solid rgba(63,185,80,0.25)",
            borderRadius: 6,
            color: "#3fb950",
            fontSize: "11px",
            fontFamily: "inherit",
            cursor: "pointer",
            letterSpacing: "0.04em",
            transition: "all 0.15s",
          }}
          onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(63,185,80,0.15)"; }}
          onMouseLeave={(e) => { e.currentTarget.style.background = "rgba(63,185,80,0.08)"; }}
        >
          ✓  approve &amp; commit
        </button>
        <button
          onClick={handleRollback}
          style={{
            flex: 1,
            padding: "7px 0",
            background: "rgba(248,81,73,0.06)",
            border: "1px solid rgba(248,81,73,0.2)",
            borderRadius: 6,
            color: "#f85149",
            fontSize: "11px",
            fontFamily: "inherit",
            cursor: "pointer",
            letterSpacing: "0.04em",
            transition: "all 0.15s",
          }}
          onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(248,81,73,0.13)"; }}
          onMouseLeave={(e) => { e.currentTarget.style.background = "rgba(248,81,73,0.06)"; }}
        >
          ↩  rollback
        </button>
      </div>

      {/* Toast */}
      {toast && (
        <div style={{
          position: "absolute",
          bottom: 72,
          right: 14,
          padding: "6px 12px",
          background: toast.ok ? "rgba(63,185,80,0.12)" : "rgba(248,81,73,0.12)",
          border: `1px solid ${toast.ok ? "rgba(63,185,80,0.3)" : "rgba(248,81,73,0.3)"}`,
          borderRadius: 6,
          color: toast.ok ? "#3fb950" : "#f85149",
          fontSize: "11px",
          fontFamily: "inherit",
          zIndex: 50,
          animation: "fadeIn 0.15s ease",
        }}>
          {toast.msg}
        </div>
      )}
    </div>
  );
}

function Empty({ text, spin }: { text: string; spin?: boolean }) {
  return (
    <div style={{
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      height: "100%",
      gap: 8,
      color: "#2d3a4a",
      fontSize: "11px",
      fontFamily: '"JetBrains Mono", monospace',
      background: "#080b0f",
    }}>
      {spin && (
        <svg width={12} height={12} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} style={{ animation:"spin 1s linear infinite" }}>
          <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
          <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
        </svg>
      )}
      {text}
    </div>
  );
}