"use client";
// ===== app/components/HeaderBar.tsx =====

import { Project } from "@/lib/types";

interface Props {
  project: Project | null;
}

export default function HeaderBar({ project }: Props) {
  return (
    <header style={{
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      padding: "0 16px",
      height: 40,
      background: "#070a0e",
      borderBottom: "1px solid #1e2430",
      flexShrink: 0,
      fontFamily: '"JetBrains Mono", "Fira Code", monospace',
    }}>
      {/* Left: brand + breadcrumb */}
      <div style={{ display:"flex", alignItems:"center", gap:10 }}>
        {/* Logo mark */}
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <rect width="16" height="16" rx="4" fill="#388bfd" fillOpacity="0.15" />
          <path d="M3 8h4m0 0L5.5 5.5M7 8 5.5 10.5M9 6l3 2-3 2" stroke="#388bfd" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span style={{ fontSize:"11px", color:"#388bfd", letterSpacing:"0.1em" }}>DRAGON</span>
        <span style={{ color:"#1e2430", fontSize:"11px" }}>/</span>
        <span style={{ fontSize:"11px", color:"#7b6fff", letterSpacing:"0.06em" }}>STUDIO</span>

        {project && (
          <>
            <span style={{ color:"#1e2430", fontSize:"11px" }}>/</span>
            <span style={{ fontSize:"11px", color:"#6e8098" }}>{project.name}</span>
          </>
        )}
      </div>

      {/* Right: project id badge */}
      {project && (
        <div style={{ display:"flex", alignItems:"center", gap:6 }}>
          <span style={{ fontSize:"10px", color:"#3d4a5c" }}>id</span>
          <code style={{
            fontSize:"10px",
            padding:"2px 8px",
            background:"rgba(255,255,255,0.03)",
            border:"1px solid #1e2430",
            borderRadius:4,
            color:"#3d4a5c",
          }}>
            {project.project_id}
          </code>
        </div>
      )}
    </header>
  );
}