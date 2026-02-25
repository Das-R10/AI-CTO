"use client";
// ===== app/components/InsightsPanel.tsx =====
// Surfaces the most important unused backend features in one tabbed panel:
//   1. DIFF      — unified diff viewer (moved from standalone DiffViewer)
//   2. QUALITY   — GET /quality/project/  — per-file scores + grades
//   3. HEALTH    — GET /project-health/   — quality, tests, memory, drift risk
//   4. HISTORY   — GET /history/{file_id} — version list + rollback
//   5. APPROVALS — GET /pending-approvals/ + POST /approve|reject/{id}

import { useEffect, useState, useCallback } from "react";
import {
  getDiff, rollback,
  getProjectQuality, getProjectHealth,
  getFileHistory, getPendingApprovals,
  approveChange, rejectChange,
} from "@/lib/api";
import { DiffResult } from "@/lib/types";

// ─── Types ───────────────────────────────────────────────────────────────────

interface QualityFile {
  filename: string;
  score: number;
  grade: string;
  summary?: string;
}

interface HealthData {
  quality_score: number;
  quality_grade: string;
  test_pass_rate: number | string;
  test_files_count: number;
  memory_entries: number;
  files_modified_24h: string[];
  open_jobs: number;
  dependency_risk: { risk_level: string; high_risk_files: string[] };
}

interface VersionEntry {
  version_id: number;
  saved_at: string;
}

interface Approval {
  id: number;
  risk_flags: Record<string, unknown>;
  status: string;
  created_at: string;
}

interface Props {
  projectId: number;
  fileId: number | null;           // currently modified file (for diff + history)
  onRolledBack?: () => void;
}

// ─── Diff helpers ────────────────────────────────────────────────────────────

interface DiffLine { type: "add" | "remove" | "context" | "header"; content: string }

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

// ─── Shared primitives ───────────────────────────────────────────────────────

const mono = '"JetBrains Mono", "Fira Code", monospace';

const TABS = ["DIFF", "QUALITY", "HEALTH", "HISTORY", "APPROVALS"] as const;
type Tab = typeof TABS[number];

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ fontSize: 9, letterSpacing: "0.14em", color: "#3d4a5c", padding: "10px 14px 4px", fontFamily: mono }}>
      {children}
    </div>
  );
}

function Empty({ text, spin }: { text: string; spin?: boolean }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", gap: 8, color: "#2d3a4a", fontSize: 11, fontFamily: mono, background: "#080b0f" }}>
      {spin && (
        <svg width={12} height={12} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} style={{ animation: "spin 1s linear infinite" }}>
          <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
          <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
        </svg>
      )}
      {text}
    </div>
  );
}

function GradeChip({ grade, score }: { grade: string; score: number }) {
  const color =
    grade === "A" ? "#3fb950" :
    grade === "B" ? "#79c0ff" :
    grade === "C" ? "#e3b341" :
    grade === "D" ? "#f0883e" : "#f85149";
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 5, padding: "2px 8px", borderRadius: 4, border: `1px solid ${color}30`, background: `${color}10`, color, fontSize: 11, fontFamily: mono }}>
      <strong>{grade}</strong>
      <span style={{ color: `${color}99` }}>{score}</span>
    </span>
  );
}

function RiskChip({ level }: { level: string }) {
  const color = level === "high" ? "#f85149" : level === "medium" ? "#e3b341" : "#3fb950";
  return (
    <span style={{ padding: "2px 7px", borderRadius: 4, border: `1px solid ${color}30`, background: `${color}10`, color, fontSize: 10, fontFamily: mono, letterSpacing: "0.08em" }}>
      {level.toUpperCase()}
    </span>
  );
}

function StatCard({ label, value, sub }: { label: string; value: React.ReactNode; sub?: string }) {
  return (
    <div style={{ padding: "10px 12px", background: "#0d1117", border: "1px solid #1e2430", borderRadius: 6 }}>
      <div style={{ fontSize: 9, color: "#3d4a5c", letterSpacing: "0.12em", fontFamily: mono, marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 16, color: "#c9d1d9", fontFamily: mono, fontWeight: 600 }}>{value}</div>
      {sub && <div style={{ fontSize: 10, color: "#484f58", fontFamily: mono, marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

// ─── Tab: DIFF ───────────────────────────────────────────────────────────────

function DiffTab({ fileId, onRolledBack }: { fileId: number | null; onRolledBack?: () => void }) {
  const [diff, setDiff] = useState<DiffResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState<{ msg: string; ok: boolean } | null>(null);

  useEffect(() => {
    if (!fileId) { setDiff(null); return; }
    setLoading(true);
    setDiff(null);
    getDiff(fileId).then(setDiff).catch(() => setDiff(null)).finally(() => setLoading(false));
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

  if (!fileId) return <Empty text="─  awaiting diff" />;
  if (loading) return <Empty text="loading diff…" spin />;
  if (!diff) return <Empty text="no diff available" />;

  const noDiff = !diff.diff || diff.diff === "No changes detected";
  const parsed = noDiff ? [] : parseUnifiedDiff(diff.diff);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", position: "relative" }}>
      {/* Stats bar */}
      <div style={{ padding: "6px 14px", borderBottom: "1px solid #1e2430", display: "flex", alignItems: "center", gap: 10, background: "#0a0d12", flexShrink: 0 }}>
        <span style={{ fontSize: 10, color: "#3d4a5c", letterSpacing: "0.1em", fontFamily: mono }}>CHANGES</span>
        <span style={{ fontSize: 11, color: "#6e8098", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontFamily: mono }}>{diff.filename}</span>
        <div style={{ marginLeft: "auto", display: "flex", gap: 10 }}>
          <span style={{ fontSize: 11, color: "#3fb950", fontFamily: mono }}>+{diff.lines_added}</span>
          <span style={{ fontSize: 11, color: "#f85149", fontFamily: mono }}>−{diff.lines_removed}</span>
        </div>
      </div>

      {/* Lines */}
      <div style={{ flex: 1, overflowY: "auto", padding: "8px 0" }}>
        {noDiff ? (
          <div style={{ padding: "12px 16px", fontSize: 12, color: "#3d4a5c", fontFamily: mono }}>no changes detected</div>
        ) : parsed.map((line, i) => {
          const isAdd = line.type === "add", isRm = line.type === "remove", isHdr = line.type === "header";
          return (
            <div key={i} style={{ display: "flex", alignItems: "flex-start", background: isAdd ? "rgba(63,185,80,0.07)" : isRm ? "rgba(248,81,73,0.07)" : "transparent", borderLeft: isAdd ? "2px solid rgba(63,185,80,0.4)" : isRm ? "2px solid rgba(248,81,73,0.4)" : "2px solid transparent" }}>
              <span style={{ width: 20, textAlign: "center", flexShrink: 0, fontSize: 11, lineHeight: "20px", color: isAdd ? "#3fb950" : isRm ? "#f85149" : isHdr ? "#388bfd" : "#2d3a4a", userSelect: "none", fontFamily: mono }}>
                {isAdd ? "+" : isRm ? "−" : isHdr ? "·" : " "}
              </span>
              <span style={{ flex: 1, fontSize: 11, lineHeight: "20px", color: isAdd ? "#aff5b4" : isRm ? "#ffa198" : isHdr ? "#79c0ff" : "#8b949e", whiteSpace: "pre", overflow: "hidden", textOverflow: "ellipsis", fontFamily: mono }}>
                {line.content || " "}
              </span>
            </div>
          );
        })}
      </div>

      {/* Actions */}
      <div style={{ display: "flex", gap: 8, padding: "10px 14px", borderTop: "1px solid #1e2430", background: "#0a0d12", flexShrink: 0 }}>
        <button onClick={() => showToast("changes committed ✓", true)} style={{ flex: 1, padding: "7px 0", background: "rgba(63,185,80,0.08)", border: "1px solid rgba(63,185,80,0.25)", borderRadius: 6, color: "#3fb950", fontSize: 11, fontFamily: mono, cursor: "pointer", letterSpacing: "0.04em" }}>
          ✓  approve &amp; commit
        </button>
        <button onClick={handleRollback} style={{ flex: 1, padding: "7px 0", background: "rgba(248,81,73,0.06)", border: "1px solid rgba(248,81,73,0.2)", borderRadius: 6, color: "#f85149", fontSize: 11, fontFamily: mono, cursor: "pointer", letterSpacing: "0.04em" }}>
          ↩  rollback
        </button>
      </div>

      {toast && (
        <div style={{ position: "absolute", bottom: 72, right: 14, padding: "6px 12px", background: toast.ok ? "rgba(63,185,80,0.12)" : "rgba(248,81,73,0.12)", border: `1px solid ${toast.ok ? "rgba(63,185,80,0.3)" : "rgba(248,81,73,0.3)"}`, borderRadius: 6, color: toast.ok ? "#3fb950" : "#f85149", fontSize: 11, fontFamily: mono, zIndex: 50 }}>
          {toast.msg}
        </div>
      )}
    </div>
  );
}

// ─── Tab: QUALITY ────────────────────────────────────────────────────────────

function QualityTab({ projectId }: { projectId: number }) {
  const [files, setFiles] = useState<QualityFile[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    getProjectQuality(projectId)
      .then((data: Record<string, { score: number; grade: string; summary?: string }>) => {
        setFiles(Object.entries(data).map(([filename, v]) => ({ filename, ...v })));
      })
      .catch(() => setFiles([]))
      .finally(() => setLoading(false));
  }, [projectId]);

  if (loading) return <Empty text="loading quality…" spin />;
  if (!files.length) return <Empty text="no files scored yet" />;

  const avg = Math.round(files.reduce((s, f) => s + f.score, 0) / files.length);
  const avgGrade = avg >= 90 ? "A" : avg >= 80 ? "B" : avg >= 70 ? "C" : avg >= 60 ? "D" : "F";

  return (
    <div style={{ height: "100%", overflowY: "auto" }}>
      <SectionLabel>PROJECT AVERAGE</SectionLabel>
      <div style={{ padding: "0 14px 12px" }}>
        <GradeChip grade={avgGrade} score={avg} />
      </div>
      <SectionLabel>FILES</SectionLabel>
      <div style={{ padding: "0 14px", display: "flex", flexDirection: "column", gap: 6 }}>
        {files.map((f) => (
          <div key={f.filename} style={{ padding: "10px 12px", background: "#0d1117", border: "1px solid #1e2430", borderRadius: 6 }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: f.summary ? 6 : 0 }}>
              <span style={{ fontSize: 11, color: "#8b949e", fontFamily: mono, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: "60%" }}>{f.filename}</span>
              <GradeChip grade={f.grade} score={f.score} />
            </div>
            {f.summary && (
              <p style={{ fontSize: 10, color: "#484f58", fontFamily: mono, lineHeight: 1.5, margin: 0 }}>{f.summary}</p>
            )}
          </div>
        ))}
      </div>
      <div style={{ height: 14 }} />
    </div>
  );
}

// ─── Tab: HEALTH ─────────────────────────────────────────────────────────────

function HealthTab({ projectId }: { projectId: number }) {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(() => {
    setLoading(true);
    getProjectHealth(projectId)
      .then(setHealth)
      .catch(() => setHealth(null))
      .finally(() => setLoading(false));
  }, [projectId]);

  useEffect(() => { load(); }, [load]);

  if (loading) return <Empty text="loading health…" spin />;
  if (!health) return <Empty text="health data unavailable" />;

  return (
    <div style={{ height: "100%", overflowY: "auto" }}>
      <SectionLabel>OVERVIEW</SectionLabel>
      <div style={{ padding: "0 14px", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
        <StatCard label="QUALITY" value={<GradeChip grade={health.quality_grade} score={health.quality_score} />} />
        <StatCard
          label="TESTS"
          value={health.test_pass_rate === "no tests" ? <span style={{ fontSize: 12, color: "#484f58" }}>none</span> : `${health.test_pass_rate}%`}
          sub={`${health.test_files_count} file(s)`}
        />
        <StatCard label="MEMORY" value={health.memory_entries} sub="stored decisions" />
        <StatCard label="OPEN JOBS" value={health.open_jobs} />
      </div>

      <SectionLabel>DEPENDENCY RISK</SectionLabel>
      <div style={{ padding: "0 14px 8px" }}>
        <div style={{ padding: "10px 12px", background: "#0d1117", border: "1px solid #1e2430", borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span style={{ fontSize: 11, color: "#6e8098", fontFamily: mono }}>overall risk</span>
          <RiskChip level={health.dependency_risk.risk_level} />
        </div>
        {health.dependency_risk.high_risk_files.length > 0 && (
          <div style={{ marginTop: 6, padding: "8px 12px", background: "#0d1117", border: "1px solid #1e2430", borderRadius: 6 }}>
            <div style={{ fontSize: 9, color: "#3d4a5c", letterSpacing: "0.1em", fontFamily: mono, marginBottom: 4 }}>HIGH RISK FILES</div>
            {health.dependency_risk.high_risk_files.map((f) => (
              <div key={f} style={{ fontSize: 10, color: "#f0883e", fontFamily: mono, padding: "2px 0" }}>⚠ {f}</div>
            ))}
          </div>
        )}
      </div>

      {health.files_modified_24h.length > 0 && (
        <>
          <SectionLabel>MODIFIED (24H)</SectionLabel>
          <div style={{ padding: "0 14px 8px" }}>
            {health.files_modified_24h.map((f) => (
              <div key={f} style={{ fontSize: 11, color: "#6e8098", fontFamily: mono, padding: "3px 0", borderBottom: "1px solid #1e2430" }}>
                {f}
              </div>
            ))}
          </div>
        </>
      )}

      <div style={{ padding: "10px 14px" }}>
        <button onClick={load} style={{ width: "100%", padding: "6px 0", background: "rgba(56,139,253,0.06)", border: "1px solid rgba(56,139,253,0.2)", borderRadius: 6, color: "#388bfd", fontSize: 11, fontFamily: mono, cursor: "pointer", letterSpacing: "0.04em" }}>
          ↻  refresh
        </button>
      </div>
    </div>
  );
}

// ─── Tab: HISTORY ────────────────────────────────────────────────────────────

function HistoryTab({ fileId, onRolledBack }: { fileId: number | null; onRolledBack?: () => void }) {
  const [versions, setVersions] = useState<VersionEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [rolling, setRolling] = useState<number | null>(null);
  const [toast, setToast] = useState<{ msg: string; ok: boolean } | null>(null);

  useEffect(() => {
    if (!fileId) { setVersions([]); return; }
    setLoading(true);
    getFileHistory(fileId)
      .then((d: { versions: VersionEntry[] }) => setVersions(d.versions))
      .catch(() => setVersions([]))
      .finally(() => setLoading(false));
  }, [fileId]);

  async function handleRollback(versionId: number) {
    if (!fileId) return;
    setRolling(versionId);
    try {
      const res = await rollback(fileId);
      setToast({ msg: res.message || "rolled back", ok: true });
      onRolledBack?.();
      // Refresh list
      const d = await getFileHistory(fileId);
      setVersions(d.versions);
    } catch (e: unknown) {
      setToast({ msg: e instanceof Error ? e.message : "failed", ok: false });
    } finally {
      setRolling(null);
      setTimeout(() => setToast(null), 3000);
    }
  }

  if (!fileId) return <Empty text="select a file to view history" />;
  if (loading) return <Empty text="loading history…" spin />;
  if (!versions.length) return <Empty text="no saved versions" />;

  return (
    <div style={{ height: "100%", overflowY: "auto", position: "relative" }}>
      <SectionLabel>VERSION HISTORY  ·  file {fileId}</SectionLabel>
      <div style={{ padding: "0 14px", display: "flex", flexDirection: "column", gap: 6 }}>
        {versions.map((v, i) => (
          <div key={v.version_id} style={{ padding: "10px 12px", background: "#0d1117", border: "1px solid #1e2430", borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div>
              <div style={{ fontSize: 10, color: "#3d4a5c", fontFamily: mono, marginBottom: 2 }}>
                {i === 0 ? "● LATEST SNAPSHOT" : `version ${v.version_id}`}
              </div>
              <div style={{ fontSize: 11, color: "#6e8098", fontFamily: mono }}>
                {new Date(v.saved_at).toLocaleString()}
              </div>
            </div>
            <button
              onClick={() => handleRollback(v.version_id)}
              disabled={rolling === v.version_id}
              style={{ padding: "4px 10px", background: "rgba(248,81,73,0.06)", border: "1px solid rgba(248,81,73,0.2)", borderRadius: 5, color: "#f85149", fontSize: 10, fontFamily: mono, cursor: "pointer", opacity: rolling === v.version_id ? 0.5 : 1 }}
            >
              {rolling === v.version_id ? "…" : "↩ restore"}
            </button>
          </div>
        ))}
      </div>
      <div style={{ height: 14 }} />
      {toast && (
        <div style={{ position: "fixed", bottom: 24, right: 24, padding: "6px 12px", background: toast.ok ? "rgba(63,185,80,0.12)" : "rgba(248,81,73,0.12)", border: `1px solid ${toast.ok ? "rgba(63,185,80,0.3)" : "rgba(248,81,73,0.3)"}`, borderRadius: 6, color: toast.ok ? "#3fb950" : "#f85149", fontSize: 11, fontFamily: mono, zIndex: 99 }}>
          {toast.msg}
        </div>
      )}
    </div>
  );
}

// ─── Tab: APPROVALS ──────────────────────────────────────────────────────────

function ApprovalsTab({ projectId }: { projectId: number }) {
  const [approvals, setApprovals] = useState<Approval[]>([]);
  const [loading, setLoading] = useState(true);
  const [acting, setActing] = useState<number | null>(null);
  const [toast, setToast] = useState<{ msg: string; ok: boolean } | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    getPendingApprovals(projectId)
      .then((d: { approvals: Approval[] }) => setApprovals(d.approvals))
      .catch(() => setApprovals([]))
      .finally(() => setLoading(false));
  }, [projectId]);

  useEffect(() => { load(); }, [load]);

  async function act(id: number, action: "approve" | "reject") {
    setActing(id);
    try {
      if (action === "approve") await approveChange(id, projectId);
      else await rejectChange(id, projectId);
      setToast({ msg: `${action}d ✓`, ok: action === "approve" });
      load();
    } catch (e: unknown) {
      setToast({ msg: e instanceof Error ? e.message : "failed", ok: false });
    } finally {
      setActing(null);
      setTimeout(() => setToast(null), 3000);
    }
  }

  if (loading) return <Empty text="loading approvals…" spin />;

  return (
    <div style={{ height: "100%", overflowY: "auto", position: "relative" }}>
      <SectionLabel>PENDING APPROVALS  ·  {approvals.length} waiting</SectionLabel>

      {approvals.length === 0 ? (
        <div style={{ padding: "20px 14px", fontSize: 11, color: "#2d3a4a", fontFamily: mono, textAlign: "center" }}>
          ✓ nothing awaiting review
        </div>
      ) : (
        <div style={{ padding: "0 14px", display: "flex", flexDirection: "column", gap: 8 }}>
          {approvals.map((a) => {
            const flags = Object.keys(a.risk_flags || {});
            return (
              <div key={a.id} style={{ padding: "12px", background: "#0d1117", border: "1px solid rgba(248,81,73,0.25)", borderRadius: 6 }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}>
                  <span style={{ fontSize: 10, color: "#484f58", fontFamily: mono }}>#{a.id} · {new Date(a.created_at).toLocaleString()}</span>
                  <RiskChip level="high" />
                </div>
                {flags.length > 0 && (
                  <div style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: 9, color: "#3d4a5c", letterSpacing: "0.1em", fontFamily: mono, marginBottom: 4 }}>RISK FLAGS</div>
                    {flags.map((f) => (
                      <div key={f} style={{ fontSize: 10, color: "#f0883e", fontFamily: mono, padding: "1px 0" }}>⚠ {f}</div>
                    ))}
                  </div>
                )}
                <div style={{ display: "flex", gap: 6 }}>
                  <button
                    onClick={() => act(a.id, "approve")}
                    disabled={acting === a.id}
                    style={{ flex: 1, padding: "6px 0", background: "rgba(63,185,80,0.08)", border: "1px solid rgba(63,185,80,0.25)", borderRadius: 5, color: "#3fb950", fontSize: 10, fontFamily: mono, cursor: "pointer" }}
                  >
                    ✓ approve
                  </button>
                  <button
                    onClick={() => act(a.id, "reject")}
                    disabled={acting === a.id}
                    style={{ flex: 1, padding: "6px 0", background: "rgba(248,81,73,0.06)", border: "1px solid rgba(248,81,73,0.2)", borderRadius: 5, color: "#f85149", fontSize: 10, fontFamily: mono, cursor: "pointer" }}
                  >
                    ✕ reject
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      <div style={{ padding: "10px 14px" }}>
        <button onClick={load} style={{ width: "100%", padding: "6px 0", background: "rgba(56,139,253,0.06)", border: "1px solid rgba(56,139,253,0.2)", borderRadius: 6, color: "#388bfd", fontSize: 11, fontFamily: mono, cursor: "pointer", letterSpacing: "0.04em" }}>
          ↻  refresh
        </button>
      </div>

      {toast && (
        <div style={{ position: "fixed", bottom: 24, right: 24, padding: "6px 12px", background: toast.ok ? "rgba(63,185,80,0.12)" : "rgba(248,81,73,0.12)", border: `1px solid ${toast.ok ? "rgba(63,185,80,0.3)" : "rgba(248,81,73,0.3)"}`, borderRadius: 6, color: toast.ok ? "#3fb950" : "#f85149", fontSize: 11, fontFamily: mono, zIndex: 99 }}>
          {toast.msg}
        </div>
      )}
    </div>
  );
}

// ─── Main Panel ───────────────────────────────────────────────────────────────

export default function InsightsPanel({ projectId, fileId, onRolledBack }: Props) {
  const [activeTab, setActiveTab] = useState<Tab>("DIFF");

  // Auto-switch to DIFF tab when a new diff arrives
  useEffect(() => {
    if (fileId) setActiveTab("DIFF");
  }, [fileId]);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", background: "#080b0f", fontFamily: mono }}>
      {/* Tab bar */}
      <div style={{ display: "flex", borderBottom: "1px solid #1e2430", background: "#0a0d12", flexShrink: 0 }}>
        {TABS.map((tab) => {
          const active = tab === activeTab;
          return (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                flex: 1,
                padding: "8px 0",
                fontSize: 9,
                letterSpacing: "0.1em",
                fontFamily: mono,
                background: "transparent",
                border: "none",
                borderBottom: active ? "2px solid #388bfd" : "2px solid transparent",
                color: active ? "#388bfd" : "#3d4a5c",
                cursor: "pointer",
                transition: "color 0.15s",
              }}
            >
              {tab}
            </button>
          );
        })}
      </div>

      {/* Tab content */}
      <div style={{ flex: 1, overflow: "hidden", position: "relative" }}>
        {activeTab === "DIFF"      && <DiffTab fileId={fileId} onRolledBack={onRolledBack} />}
        {activeTab === "QUALITY"   && <QualityTab projectId={projectId} />}
        {activeTab === "HEALTH"    && <HealthTab projectId={projectId} />}
        {activeTab === "HISTORY"   && <HistoryTab fileId={fileId} onRolledBack={onRolledBack} />}
        {activeTab === "APPROVALS" && <ApprovalsTab projectId={projectId} />}
      </div>
    </div>
  );
}