"use client";

import { FileContent } from "@/lib/types";
import Editor from "@monaco-editor/react";

interface Props {
  file: FileContent | null;
}

function guessLanguage(filename: string): string {
  const ext = filename.split(".").pop()?.toLowerCase() || "";
  const map: Record<string, string> = {
    ts: "typescript", tsx: "typescript", js: "javascript", jsx: "javascript",
    py: "python", rs: "rust", go: "go", java: "java", cpp: "cpp", c: "c",
    cs: "csharp", html: "html", css: "css", json: "json", md: "markdown",
    yaml: "yaml", yml: "yaml", sh: "shell", sql: "sql",
  };
  return map[ext] || "plaintext";
}

export default function CodePreview({ file }: Props) {
  if (!file) {
    return (
      <div className="flex-1 flex items-center justify-center text-sm"
        style={{ color: "#484f58", background: "#0d1117" }}>
        select a file to preview
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{
        display: "flex", alignItems: "center", padding: "6px 16px", flexShrink: 0,
        background: "#1c2128", borderBottom: "1px solid #30363d",
        color: "#8b949e", fontSize: 12,
        fontFamily: '"JetBrains Mono", monospace',
      }}>
        <span style={{ color: "#e6edf3" }}>{file.filename}</span>
      </div>
      <div style={{ flex: 1 }}>
        <Editor
          height="100%"
          language={guessLanguage(file.filename)}
          value={file.content}
          theme="vs-dark"
          options={{
            readOnly: true,
            fontSize: 13,
            fontFamily: '"JetBrains Mono", "Fira Code", monospace',
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            lineNumbers: "on",
            renderLineHighlight: "none",
            padding: { top: 12, bottom: 12 },
          }}
        />
      </div>
    </div>
  );
}