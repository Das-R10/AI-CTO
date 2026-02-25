"use client";
// ===== app/components/ChatPanel.tsx =====
// Terminal-style chat bar pinned at the bottom of the studio.
// Sends POST /modify-code/?query=...&project_id=... (query params, no body)

import { useEffect, useRef, useState } from "react";
import { modifyCode } from "@/lib/api";
import { ChatMessage, ModifyResult } from "@/lib/types";

interface Props {
  projectId: number;
  onModified: (result: ModifyResult) => void;
}

type TermLine =
  | { kind: "input"; text: string; ts: string }
  | { kind: "output"; text: string; ok: boolean; ts: string }
  | { kind: "system"; text: string }
  | { kind: "thinking" };

function now() {
  return new Date().toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export default function TerminalChat({ projectId, onModified }: Props) {
  const [lines, setLines] = useState<TermLine[]>([
    { kind: "system", text: "Dragon AI Studio  —  type a prompt and press Enter to modify code" },
    { kind: "system", text: `session  project_id=${projectId}` },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<string[]>([]);
  const [histIdx, setHistIdx] = useState(-1);
  const [expanded, setExpanded] = useState(true);

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [lines]);

  async function handleSubmit() {
    const query = input.trim();
    if (!query || loading) return;

    setHistory((h) => [query, ...h.slice(0, 49)]);
    setHistIdx(-1);
    setInput("");
    setLines((l) => [...l, { kind: "input", text: query, ts: now() }]);
    setLoading(true);
    setLines((l) => [...l, { kind: "thinking" }]);

    try {
      const result = await modifyCode(projectId, query);

      let summary = "";
      if (result.status === "success") {
        const filesMod = result.results
          ?.filter((r) => r.success && r.filename)
          .map((r) => r.filename)
          .filter(Boolean);
        summary = filesMod?.length
          ? `✓  modified: ${filesMod.join(", ")}`
          : `✓  ${result.goal || query}`;
      } else {
        summary = result.error
          ? `✗  failed at step ${result.failed_at_step}: ${result.error}`
          : `✗  status: ${result.status}`;
      }

      setLines((l) => [
        ...l.filter((x) => x.kind !== "thinking"),
        { kind: "output", text: summary, ok: result.status === "success", ts: now() },
      ]);

      onModified({ ...result, results: result.results });
    } catch (e: unknown) {
      setLines((l) => [
        ...l.filter((x) => x.kind !== "thinking"),
        { kind: "output", text: `✗  ${e instanceof Error ? e.message : "unknown error"}`, ok: false, ts: now() },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") { e.preventDefault(); handleSubmit(); return; }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      const next = Math.min(histIdx + 1, history.length - 1);
      setHistIdx(next);
      setInput(history[next] ?? "");
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      const next = Math.max(histIdx - 1, -1);
      setHistIdx(next);
      setInput(next === -1 ? "" : history[next]);
      return;
    }
    if (e.key === "l" && e.ctrlKey) {
      e.preventDefault();
      setLines([{ kind: "system", text: "terminal cleared" }]);
      return;
    }
  }

  const COLLAPSED_H = 44;
  const EXPANDED_H = 220;

  return (
    <div
      style={{
        flexShrink: 0,
        height: expanded ? EXPANDED_H : COLLAPSED_H,
        transition: "height 0.2s cubic-bezier(0.16,1,0.3,1)",
        borderTop: "1px solid #1e2430",
        background: "#080b0f",
        display: "flex",
        flexDirection: "column",
        fontFamily: '"JetBrains Mono", "Fira Code", "Cascadia Code", monospace',
        fontSize: "12px",
      }}
      onClick={() => inputRef.current?.focus()}
    >
      {/* ── Terminal title bar ── */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0 12px",
          height: 28,
          borderBottom: expanded ? "1px solid #1e2430" : "none",
          background: "#0a0d12",
          flexShrink: 0,
          cursor: "pointer",
          userSelect: "none",
        }}
        onClick={(e) => { e.stopPropagation(); setExpanded((v) => !v); }}
      >
        <div style={{ display:"flex", alignItems:"center", gap:6 }}>
          {/* traffic dots */}
          {["#f85149","#e3b341","#3fb950"].map((c) => (
            <div key={c} style={{ width:8, height:8, borderRadius:"50%", background:c, opacity:0.7 }} />
          ))}
          <span style={{ color:"#3d4a5c", fontSize:"10px", letterSpacing:"0.12em", marginLeft:4 }}>TERMINAL</span>
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:12 }}>
          {loading && (
            <span style={{ color:"#3fb950", fontSize:"10px", display:"flex", alignItems:"center", gap:4 }}>
              <SpinnerIcon size={10} /> running
            </span>
          )}
          <span style={{ color:"#3d4a5c", fontSize:"10px" }}>{expanded ? "▾" : "▸"}</span>
        </div>
      </div>

      {/* ── Scrollable history ── */}
      {expanded && (
        <div
          style={{
            flex: 1,
            overflowY: "auto",
            padding: "6px 0 2px",
            scrollbarWidth: "thin",
            scrollbarColor: "#1e2430 transparent",
          }}
        >
          {lines.map((line, i) => {
            if (line.kind === "thinking") {
              return (
                <div key={i} style={{ display:"flex", alignItems:"center", gap:6, padding:"1px 16px", color:"#3fb950" }}>
                  <span style={{ color:"#1e2430" }}>{"  "}</span>
                  <ThinkingDots />
                </div>
              );
            }
            if (line.kind === "system") {
              return (
                <div key={i} style={{ padding:"1px 16px", color:"#2d3a4a", fontSize:"11px" }}>
                  # {line.text}
                </div>
              );
            }
            if (line.kind === "input") {
              return (
                <div key={i} style={{ display:"flex", gap:8, padding:"1px 16px" }}>
                  <span style={{ color:"#3d4a5c", flexShrink:0 }}>{line.ts}</span>
                  <Prompt />
                  <span style={{ color:"#c9d1d9", wordBreak:"break-all" }}>{line.text}</span>
                </div>
              );
            }
            if (line.kind === "output") {
              return (
                <div key={i} style={{ display:"flex", gap:8, padding:"1px 16px 3px" }}>
                  <span style={{ color:"#3d4a5c", flexShrink:0 }}>{line.ts}</span>
                  <span style={{ color:"#3d4a5c", flexShrink:0 }}>{"  "}</span>
                  <span style={{ color: line.ok ? "#3fb950" : "#f85149", wordBreak:"break-all" }}>{line.text}</span>
                </div>
              );
            }
            return null;
          })}
          <div ref={bottomRef} />
        </div>
      )}

      {/* ── Input row ── */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "0 12px",
          height: 44,
          flexShrink: 0,
          borderTop: expanded ? "1px solid #1e2430" : "none",
        }}
      >
        <Prompt />
        <input
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
          placeholder={loading ? "waiting for response…" : "describe the change you want…"}
          spellCheck={false}
          autoComplete="off"
          style={{
            flex: 1,
            background: "transparent",
            border: "none",
            outline: "none",
            color: loading ? "#3d4a5c" : "#c9d1d9",
            fontFamily: "inherit",
            fontSize: "12px",
            caretColor: "#3fb950",
          }}
        />
        <span style={{ color:"#2d3a4a", fontSize:"10px", flexShrink:0 }}>↑↓ history · ctrl+l clear</span>
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Prompt() {
  return (
    <span style={{ display:"flex", alignItems:"center", gap:2, flexShrink:0 }}>
      <span style={{ color:"#388bfd" }}>dragon</span>
      <span style={{ color:"#3d4a5c" }}>@</span>
      <span style={{ color:"#7b6fff" }}>studio</span>
      <span style={{ color:"#3d4a5c" }}>{"  ❯"}</span>
    </span>
  );
}

function ThinkingDots() {
  return (
    <span style={{ display:"flex", gap:3, alignItems:"center" }}>
      {[0,1,2].map((i) => (
        <span
          key={i}
          style={{
            display: "inline-block",
            width: 5,
            height: 5,
            borderRadius: "50%",
            background: "#3fb950",
            opacity: 0.7,
            animation: `termPulse 1.2s ease-in-out ${i * 0.2}s infinite`,
          }}
        />
      ))}
      <style>{`@keyframes termPulse { 0%,80%,100%{transform:scale(0.6);opacity:0.3} 40%{transform:scale(1);opacity:1} }`}</style>
    </span>
  );
}

function SpinnerIcon({ size = 12 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} style={{ animation:"spin 1s linear infinite" }}>
      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
    </svg>
  );
}