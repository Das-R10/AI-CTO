"use client";
// ===== app/studio/page.tsx =====

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { setApiKey } from "@/lib/api";
import { Project, FileContent, ModifyResult } from "@/lib/types";
import HeaderBar from "../components/HeaderBar";
import FileSidebar from "../components/FileSidebar";
import ChatPanel from "../components/ChatPanel";
import InsightsPanel from "../components/InsightsPanel";
import CodePreview from "../components/CodePreview";

export default function StudioPage() {
  const router = useRouter();
  const [project, setProject] = useState<Project | null>(null);
  const [activeFileId, setActiveFileId] = useState<number | null>(null);
  const [activeFile, setActiveFile] = useState<FileContent | null>(null);
  const [diffFileId, setDiffFileId] = useState<number | null>(null);

  useEffect(() => {
    const stored = sessionStorage.getItem("project");
    if (!stored) {
      router.replace("/");
      return;
    }
    const p: Project = JSON.parse(stored);
    setApiKey(p.api_key);
    setProject(p);
  }, [router]);

  function onFileSelect(file: FileContent) {
    setActiveFileId(file.file_id);
    setActiveFile(file);
  }

  function onModified(result: ModifyResult) {
    // Try to get file_id directly from agent step results
    const modifiedStep = result.results?.find((r) => r.success && r.file_id);
    if (modifiedStep?.file_id) {
      setDiffFileId(modifiedStep.file_id);
      return;
    }

    // Fallback: use the currently selected file in the sidebar
    if (activeFileId) {
      setDiffFileId(activeFileId);
      return;
    }

    setDiffFileId(null);
  }

  if (!project) {
    return (
      <div
        className="flex items-center justify-center h-screen text-xs font-mono"
        style={{ background: "#0d1117", color: "#484f58" }}
      >
        loading…
      </div>
    );
  }

  return (
    <div
      className="flex flex-col h-screen overflow-hidden"
      style={{ background: "#0d1117" }}
    >
      <HeaderBar project={project} />

      <div className="flex flex-1 overflow-hidden">
        {/* LEFT: File Sidebar */}
        <FileSidebar
          projectId={project.project_id}
          onFileSelect={onFileSelect}
          selectedFileId={activeFileId ?? undefined}
        />

        {/* CENTER: Code Preview + Terminal at bottom */}
        <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
          {/* Code preview fills remaining space */}
          <div className="flex-1 overflow-hidden">
            <CodePreview file={activeFile} />
          </div>
          {/* Terminal pinned at bottom */}
          <ChatPanel
            projectId={project.project_id}
            onModified={onModified}
          />
        </div>

        {/* RIGHT: Diff Viewer */}
        <div
          className="overflow-hidden relative"
          style={{
            width: "420px",
            minWidth: "320px",
            borderLeft: "1px solid #30363d",
          }}
        >
          <InsightsPanel
            projectId={project.project_id}
            fileId={diffFileId}
            onRolledBack={() => setDiffFileId(null)}
          />
        </div>
      </div>
    </div>
  );
}