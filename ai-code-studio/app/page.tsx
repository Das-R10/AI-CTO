"use client";
import React, { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { createProject, setApiKey } from "@/lib/api";
import "./dragon.css";

// ─── Ambient orb canvas ───────────────────────────────────────────────────────
function AmbientCanvas() {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let W = (canvas.width = window.innerWidth);
    let H = (canvas.height = window.innerHeight);

    const onResize = () => {
      W = canvas.width = window.innerWidth;
      H = canvas.height = window.innerHeight;
    };
    window.addEventListener("resize", onResize);

    // Three slow drifting orbs
    const orbs = [
      { x: W * 0.18, y: H * 0.25, r: 420, vx: 0.08, vy: 0.05, color: "74,144,226" },
      { x: W * 0.78, y: H * 0.55, r: 380, vx: -0.06, vy: 0.07, color: "99,91,255" },
      { x: W * 0.52, y: H * 0.82, r: 300, vx: 0.04, vy: -0.06, color: "20,184,166" },
    ];

    let raf: number;
    const draw = () => {
      ctx.clearRect(0, 0, W, H);
      for (const o of orbs) {
        o.x += o.vx;
        o.y += o.vy;
        // Soft boundary bounce
        if (o.x < -o.r || o.x > W + o.r) o.vx *= -1;
        if (o.y < -o.r || o.y > H + o.r) o.vy *= -1;

        const g = ctx.createRadialGradient(o.x, o.y, 0, o.x, o.y, o.r);
        g.addColorStop(0, `rgba(${o.color},0.13)`);
        g.addColorStop(1, `rgba(${o.color},0)`);
        ctx.beginPath();
        ctx.arc(o.x, o.y, o.r, 0, Math.PI * 2);
        ctx.fillStyle = g;
        ctx.fill();
      }
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
    };
  }, []);

  return (
    <canvas
      ref={ref}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 0,
        pointerEvents: "none",
      }}
    />
  );
}

// ─── Noise SVG overlay ────────────────────────────────────────────────────────
function NoiseOverlay() {
  return (
    <svg
      style={{
        position: "fixed",
        inset: 0,
        width: "100%",
        height: "100%",
        zIndex: 1,
        pointerEvents: "none",
        opacity: 0.028,
        mixBlendMode: "overlay",
      }}
    >
      <filter id="noise">
        <feTurbulence type="fractalNoise" baseFrequency="0.65" numOctaves="3" stitchTiles="stitch" />
        <feColorMatrix type="saturate" values="0" />
      </filter>
      <rect width="100%" height="100%" filter="url(#noise)" />
    </svg>
  );
}

// ─── Grid lines ───────────────────────────────────────────────────────────────
function GridLines() {
  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 1,
        pointerEvents: "none",
        backgroundImage: `
          linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px)
        `,
        backgroundSize: "80px 80px",
        maskImage: "radial-gradient(ellipse 80% 80% at 50% 40%, black 20%, transparent 100%)",
      }}
    />
  );
}

// ─── Types ────────────────────────────────────────────────────────────────────
interface ModalProps {
  onClose: () => void;
  onSuccess: () => void;
}

// ─── Create Project Modal ─────────────────────────────────────────────────────
function CreateModal({ onClose, onSuccess }: ModalProps) {
  const [name, setName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setTimeout(() => inputRef.current?.focus(), 80);
  }, []);

  async function handleCreate() {
    if (!name.trim() || loading) return;
    setLoading(true);
    setError("");
    try {
      const data = await createProject(name.trim());
      setApiKey(data.api_key);
      sessionStorage.setItem(
        "project",
        JSON.stringify({ project_id: data.project_id, api_key: data.api_key, name: data.name })
      );
      onSuccess();
      router.push("/studio");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to create project");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 100,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "rgba(0,0,0,0.7)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        padding: "1rem",
        animation: "fadeIn 0.15s ease",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          width: "100%",
          maxWidth: "420px",
          background: "rgba(13,17,23,0.95)",
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: "16px",
          padding: "36px",
          boxShadow: "0 32px 80px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.04) inset",
          animation: "slideUp 0.2s cubic-bezier(0.16,1,0.3,1)",
        }}
      >
        {/* Header */}
        <div style={{ marginBottom: "28px" }}>
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: "8px",
              background: "rgba(74,144,226,0.1)",
              border: "1px solid rgba(74,144,226,0.2)",
              borderRadius: "6px",
              padding: "4px 10px",
              marginBottom: "16px",
            }}
          >
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#4a90e2", display: "inline-block" }} />
            <span style={{ color: "#4a90e2", fontSize: "11px", fontFamily: "monospace", letterSpacing: "0.08em" }}>
              NEW PROJECT
            </span>
          </div>
          <h2
            style={{
              margin: 0,
              fontSize: "22px",
              fontWeight: 600,
              color: "#f0f6fc",
              fontFamily: "'DM Sans', system-ui, sans-serif",
              letterSpacing: "-0.02em",
            }}
          >
            Name your project
          </h2>
          <p style={{ margin: "8px 0 0", color: "#6e7681", fontSize: "14px", fontFamily: "system-ui, sans-serif" }}>
            You can change this later from your settings.
          </p>
        </div>

        {/* Input */}
        <div style={{ marginBottom: "20px" }}>
          <label
            style={{
              display: "block",
              marginBottom: "8px",
              fontSize: "13px",
              color: "#8b949e",
              fontFamily: "system-ui, sans-serif",
              fontWeight: 500,
            }}
          >
            Project name
          </label>
          <input
            ref={inputRef}
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleCreate()}
            placeholder="e.g. my-api-service"
            style={{
              width: "100%",
              padding: "10px 14px",
              background: "rgba(255,255,255,0.04)",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: "8px",
              color: "#f0f6fc",
              fontSize: "14px",
              fontFamily: "monospace",
              outline: "none",
              boxSizing: "border-box",
              transition: "border-color 0.15s",
            }}
            onFocus={(e) => (e.target.style.borderColor = "rgba(74,144,226,0.6)")}
            onBlur={(e) => (e.target.style.borderColor = "rgba(255,255,255,0.1)")}
          />
          {error && (
            <p style={{ margin: "8px 0 0", color: "#f85149", fontSize: "12px", fontFamily: "system-ui" }}>
              {error}
            </p>
          )}
        </div>

        {/* Actions */}
        <div style={{ display: "flex", gap: "10px" }}>
          <button
            onClick={onClose}
            style={{
              flex: 1,
              padding: "10px",
              background: "transparent",
              border: "1px solid rgba(255,255,255,0.1)",
              borderRadius: "8px",
              color: "#8b949e",
              fontSize: "14px",
              fontFamily: "system-ui, sans-serif",
              cursor: "pointer",
              transition: "all 0.15s",
              fontWeight: 500,
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.background = "rgba(255,255,255,0.05)";
              e.currentTarget.style.color = "#e6edf3";
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.background = "transparent";
              e.currentTarget.style.color = "#8b949e";
            }}
          >
            Cancel
          </button>
          <button
            onClick={handleCreate}
            disabled={!name.trim() || loading}
            style={{
              flex: 2,
              padding: "10px",
              background: loading || !name.trim() ? "rgba(74,144,226,0.3)" : "#4a90e2",
              border: "none",
              borderRadius: "8px",
              color: loading || !name.trim() ? "rgba(255,255,255,0.4)" : "#fff",
              fontSize: "14px",
              fontFamily: "system-ui, sans-serif",
              cursor: loading || !name.trim() ? "not-allowed" : "pointer",
              fontWeight: 600,
              transition: "all 0.15s",
              letterSpacing: "-0.01em",
            }}
            onMouseOver={(e) => {
              if (!loading && name.trim()) e.currentTarget.style.background = "#5a9fe8";
            }}
            onMouseOut={(e) => {
              if (!loading && name.trim()) e.currentTarget.style.background = "#4a90e2";
            }}
          >
            {loading ? "Creating…" : "Create project →"}
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Feature card data ────────────────────────────────────────────────────────
const FEATURES = [
  {
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 20h9" /><path d="M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4L16.5 3.5z" />
      </svg>
    ),
    title: "AI-Powered Edits",
    desc: "Describe changes in plain English. The agent plans, executes, and validates each modification step by step.",
    accent: "#4a90e2",
  },
  {
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="16 18 22 12 16 6" /><polyline points="8 6 2 12 8 18" />
      </svg>
    ),
    title: "Live Diff Viewer",
    desc: "Every change rendered as a unified diff. Approve commits instantly or roll back to any previous version.",
    accent: "#63f5c0",
  },
  {
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" /><polyline points="14 2 14 8 20 8" /><line x1="16" y1="13" x2="8" y2="13" /><line x1="16" y1="17" x2="8" y2="17" /><polyline points="10 9 9 9 8 9" />
      </svg>
    ),
    title: "Quality Grading",
    desc: "Each upload is scored and graded automatically — catching issues before the agent ever touches your code.",
    accent: "#f0a14e",
  },
  {
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      </svg>
    ),
    title: "Safe by Default",
    desc: "Isolated projects with API-key auth. Rollback is always one click away — nothing is ever permanently lost.",
    accent: "#b97cff",
  },
  {
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="3" /><path d="M19.07 4.93A10 10 0 0012 2v10" /><path d="M4.93 4.93A10 10 0 002 12h10" />
      </svg>
    ),
    title: "Multi-File Projects",
    desc: "Upload, manage, and edit any number of source files within a single project workspace.",
    accent: "#4a90e2",
  },
  {
    icon: (
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
      </svg>
    ),
    title: "Version History",
    desc: "Full audit trail of every change. Browse, compare, and restore any past state of your codebase.",
    accent: "#63f5c0",
  },
];

const SOCIAL_PROOF = [
  { stat: "10x", label: "Faster refactors" },
  { stat: "<2s", label: "Avg. response time" },
  { stat: "100%", label: "Rollback coverage" },
  { stat: "∞", label: "File history" },
];

// ─── Main page ────────────────────────────────────────────────────────────────
export default function LandingPage() {
  const router = useRouter();
  const [modal, setModal] = useState(false);
  const [navScrolled, setNavScrolled] = useState(false);
  const [hoveredFeature, setHoveredFeature] = useState<number | null>(null);

  useEffect(() => {
    const onScroll = () => setNavScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  function handleCTA() {
    const stored = sessionStorage.getItem("project");
    if (stored) {
      const p = JSON.parse(stored);
      setApiKey(p.api_key);
      router.push("/studio");
    } else {
      setModal(true);
    }
  }

  return (
    <>
      <AmbientCanvas />
      <NoiseOverlay />
      <GridLines />

      {/* ── Nav ──────────────────────────────────────────────── */}
      <nav
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          zIndex: 50,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0 32px",
          height: "60px",
          background: navScrolled ? "rgba(6,8,16,0.85)" : "transparent",
          backdropFilter: navScrolled ? "blur(20px)" : "none",
          borderBottom: navScrolled ? "1px solid rgba(255,255,255,0.06)" : "1px solid transparent",
          transition: "all 0.3s",
        }}
      >
        {/* Logo */}
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <svg width="26" height="26" viewBox="0 0 26 26" fill="none">
            <rect width="26" height="26" rx="7" fill="#4a90e2" />
            <path d="M8 13h4m0 0l-2-3m2 3l-2 3M14 10l4 3-4 3" stroke="white" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span
            style={{
              fontSize: "15px",
              fontWeight: 700,
              color: "#f0f6fc",
              letterSpacing: "-0.02em",
              fontFamily: "'DM Sans', sans-serif",
            }}
          >
            DragonAI
          </span>
        </div>

        {/* Nav links */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "28px",
            fontSize: "14px",
            color: "#6e7681",
            fontWeight: 500,
          }}
        >
          {["Features", "Docs", "Changelog", "Pricing"].map((l) => (
            <a
              key={l}
              href="#"
              style={{ transition: "color 0.15s" }}
              onMouseOver={(e) => (e.currentTarget.style.color = "#e6edf3")}
              onMouseOut={(e) => (e.currentTarget.style.color = "#6e7681")}
            >
              {l}
            </a>
          ))}
        </div>

        {/* Right side */}
        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <button
            onClick={handleCTA}
            className="ghost-btn"
            style={{ padding: "8px 16px", fontSize: "13px" }}
          >
            Sign in
          </button>
          <button
            onClick={handleCTA}
            className="cta-btn"
            style={{ padding: "8px 16px", fontSize: "13px" }}
          >
            Get started free
          </button>
        </div>
      </nav>

      {/* ── Hero ─────────────────────────────────────────────── */}
      <section
        style={{
          position: "relative",
          zIndex: 10,
          minHeight: "100vh",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          padding: "120px 24px 80px",
          textAlign: "center",
        }}
      >
        {/* Badge */}
        <div
          className="hero-badge"
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "8px",
            background: "rgba(74,144,226,0.08)",
            border: "1px solid rgba(74,144,226,0.22)",
            borderRadius: "100px",
            padding: "5px 14px 5px 8px",
            marginBottom: "28px",
          }}
        >
          <span
            style={{
              background: "#4a90e2",
              color: "#fff",
              fontSize: "10px",
              fontWeight: 700,
              letterSpacing: "0.06em",
              padding: "2px 8px",
              borderRadius: "100px",
              fontFamily: "monospace",
            }}
          >
            NEW
          </span>
          <span style={{ color: "#79b8ff", fontSize: "13px", fontWeight: 500 }}>
            Introducing AI rollback & version history →
          </span>
        </div>

        {/* Headline */}
        <h1
          className="hero-h1"
          style={{
            fontSize: "clamp(42px, 7vw, 76px)",
            fontWeight: 700,
            lineHeight: 1.05,
            letterSpacing: "-0.04em",
            color: "#f0f6fc",
            maxWidth: "800px",
            marginBottom: "20px",
          }}
        >
          Your codebase,{" "}
          <span
            style={{
              background: "linear-gradient(135deg, #4a90e2 0%, #7b6fff 50%, #63f5c0 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
            }}
          >
            rewritten by AI
          </span>
        </h1>

        {/* Sub */}
        <p
          className="hero-sub"
          style={{
            fontSize: "clamp(16px, 2.2vw, 20px)",
            color: "#6e7681",
            maxWidth: "540px",
            lineHeight: 1.65,
            marginBottom: "36px",
            fontWeight: 400,
          }}
        >
          DragonAI lets you modify, refactor, and review code changes in natural language — with live diffs, quality grading, and one-click rollback.
        </p>

        {/* CTA group */}
        <div
          className="hero-cta"
          style={{ display: "flex", alignItems: "center", gap: "12px", flexWrap: "wrap", justifyContent: "center", marginBottom: "56px" }}
        >
          <button onClick={handleCTA} className="cta-btn" style={{ fontSize: "15px", padding: "13px 28px" }}>
            Start for free
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M5 12h14M12 5l7 7-7 7" />
            </svg>
          </button>
          <button className="ghost-btn" style={{ fontSize: "15px" }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
            Watch demo
          </button>
        </div>

        {/* Social proof */}
        <div
          className="hero-proof"
          style={{
            display: "flex",
            alignItems: "center",
            gap: "10px",
            color: "#6e7681",
            fontSize: "13px",
            marginBottom: "80px",
          }}
        >
          {/* Avatars */}
          <div style={{ display: "flex" }}>
            {["#4a90e2", "#7b6fff", "#63f5c0", "#f0a14e", "#f85149"].map((c, i) => (
              <div
                key={i}
                style={{
                  width: 28,
                  height: 28,
                  borderRadius: "50%",
                  background: c,
                  border: "2px solid #060810",
                  marginLeft: i === 0 ? 0 : -8,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "11px",
                  color: "#fff",
                  fontWeight: 700,
                }}
              >
                {["A", "B", "C", "D", "E"][i]}
              </div>
            ))}
          </div>
          <span>
            Trusted by <strong style={{ color: "#8b949e" }}>500+</strong> developers
          </span>
          <span style={{ color: "#30363d" }}>·</span>
          <span>⭐ 4.9 / 5</span>
        </div>

        {/* Terminal mockup */}
        <div
          className="terminal"
          style={{ width: "100%", maxWidth: "640px", textAlign: "left" }}
        >
          <div className="terminal-bar">
            <div className="dot" style={{ background: "#f85149" }} />
            <div className="dot" style={{ background: "#e3b341" }} />
            <div className="dot" style={{ background: "#3fb950" }} />
            <span style={{ color: "#484f58", fontSize: "12px", marginLeft: "8px" }}>
              dragon-studio — modify-code
            </span>
          </div>
          <div className="terminal-body">
            <div style={{ color: "#484f58" }}>$ dragon modify --query</div>
            <div style={{ color: "#e6edf3", marginTop: "4px" }}>
              <span style={{ color: "#63f5c0" }}>❯</span>{" "}
              <span style={{ color: "#f0a14e" }}>"Refactor auth.py to use async/await and add rate limiting"</span>
            </div>
            <div style={{ color: "#484f58", marginTop: "12px" }}>Planning steps…</div>
            <div style={{ color: "#3fb950", marginTop: "4px" }}>✓ Analyzed 3 files (312 lines)</div>
            <div style={{ color: "#3fb950" }}>✓ Generated refactored auth.py</div>
            <div style={{ color: "#3fb950" }}>✓ Quality grade: <span style={{ fontWeight: 700 }}>A</span> (96/100)</div>
            <div style={{ color: "#3fb950" }}>✓ Diff ready — +48 / -31 lines</div>
            <div style={{ color: "#e6edf3", marginTop: "12px" }}>
              <span style={{ color: "#484f58" }}>Approve changes? </span>
              <span style={{ color: "#4a90e2" }}>[Y/n]</span>
              <span style={{ color: "#e6edf3" }}> Y</span>
            </div>
            <div style={{ color: "#3fb950", marginTop: "4px" }}>✓ Committed. Rollback available anytime.</div>
          </div>
        </div>
      </section>

      {/* ── Stats strip ──────────────────────────────────────── */}
      <section
        className="stats-row"
        style={{
          position: "relative",
          zIndex: 10,
          borderTop: "1px solid rgba(255,255,255,0.05)",
          borderBottom: "1px solid rgba(255,255,255,0.05)",
          background: "rgba(255,255,255,0.015)",
          padding: "40px 24px",
        }}
      >
        <div
          style={{
            maxWidth: "800px",
            margin: "0 auto",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: "40px",
            flexWrap: "wrap",
          }}
        >
          {SOCIAL_PROOF.map((s, i) => (
            <React.Fragment key={s.stat}>
              {i > 0 && <div className="stat-divider" />}
              <div style={{ textAlign: "center" }}>
                <div
                  style={{
                    fontSize: "32px",
                    fontWeight: 700,
                    letterSpacing: "-0.03em",
                    color: "#f0f6fc",
                    lineHeight: 1,
                    marginBottom: "4px",
                  }}
                >
                  {s.stat}
                </div>
                <div style={{ fontSize: "13px", color: "#6e7681", fontWeight: 500 }}>{s.label}</div>
              </div>
            </React.Fragment>
          ))}
        </div>
      </section>

      {/* ── Features grid ────────────────────────────────────── */}
      <section
        className="features"
        id="features"
        style={{
          position: "relative",
          zIndex: 10,
          padding: "100px 24px",
          maxWidth: "1100px",
          margin: "0 auto",
        }}
      >
        {/* Section heading */}
        <div style={{ textAlign: "center", marginBottom: "64px" }}>
          <div
            style={{
              display: "inline-block",
              fontSize: "11px",
              fontWeight: 600,
              letterSpacing: "0.12em",
              color: "#4a90e2",
              textTransform: "uppercase",
              marginBottom: "12px",
              fontFamily: "monospace",
            }}
          >
            Everything you need
          </div>
          <h2
            style={{
              fontSize: "clamp(28px, 4vw, 44px)",
              fontWeight: 700,
              letterSpacing: "-0.03em",
              color: "#f0f6fc",
              maxWidth: "480px",
              margin: "0 auto 14px",
              lineHeight: 1.15,
            }}
          >
            Built for serious code workflows
          </h2>
          <p style={{ color: "#6e7681", fontSize: "16px", maxWidth: "420px", margin: "0 auto", lineHeight: 1.6 }}>
            From first upload to production-ready patch — DragonAI handles the full loop.
          </p>
        </div>

        {/* Grid */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
            gap: "16px",
          }}
        >
          {FEATURES.map((f, i) => (
            <div
              key={f.title}
              className="feature-card"
              style={{
                borderColor: hoveredFeature === i ? `rgba(${f.accent === "#4a90e2" ? "74,144,226" : f.accent === "#63f5c0" ? "99,245,192" : f.accent === "#f0a14e" ? "240,161,78" : "185,124,255"},0.25)` : "rgba(255,255,255,0.07)",
              }}
              onMouseEnter={() => setHoveredFeature(i)}
              onMouseLeave={() => setHoveredFeature(null)}
            >
              {/* Icon */}
              <div
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  width: "38px",
                  height: "38px",
                  borderRadius: "9px",
                  background: `rgba(${f.accent === "#4a90e2" ? "74,144,226" : f.accent === "#63f5c0" ? "99,245,192" : f.accent === "#f0a14e" ? "240,161,78" : "185,124,255"},0.1)`,
                  color: f.accent,
                  marginBottom: "16px",
                }}
              >
                {f.icon}
              </div>
              <h3
                style={{
                  fontSize: "15px",
                  fontWeight: 600,
                  color: "#f0f6fc",
                  marginBottom: "8px",
                  letterSpacing: "-0.01em",
                }}
              >
                {f.title}
              </h3>
              <p style={{ fontSize: "14px", color: "#6e7681", lineHeight: 1.65 }}>{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Bottom CTA ───────────────────────────────────────── */}
      <section
        style={{
          position: "relative",
          zIndex: 10,
          padding: "80px 24px 120px",
          textAlign: "center",
        }}
      >
        {/* Glowing divider */}
        <div
          style={{
            width: "1px",
            height: "80px",
            background: "linear-gradient(180deg, transparent, rgba(74,144,226,0.6), transparent)",
            margin: "0 auto 60px",
          }}
        />

        <h2
          style={{
            fontSize: "clamp(28px, 4vw, 48px)",
            fontWeight: 700,
            letterSpacing: "-0.03em",
            color: "#f0f6fc",
            marginBottom: "16px",
            lineHeight: 1.1,
          }}
        >
          Ready to ship better code, faster?
        </h2>
        <p style={{ color: "#6e7681", fontSize: "17px", marginBottom: "36px", maxWidth: "400px", margin: "0 auto 36px", lineHeight: 1.6 }}>
          Create a free project in seconds. No credit card required.
        </p>
        <button onClick={handleCTA} className="cta-btn" style={{ fontSize: "16px", padding: "14px 32px" }}>
          Get started free →
        </button>
      </section>

      {/* ── Footer ───────────────────────────────────────────── */}
      <footer
        style={{
          position: "relative",
          zIndex: 10,
          borderTop: "1px solid rgba(255,255,255,0.05)",
          padding: "28px 32px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: "12px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <svg width="18" height="18" viewBox="0 0 26 26" fill="none">
            <rect width="26" height="26" rx="7" fill="#4a90e2" />
            <path d="M8 13h4m0 0l-2-3m2 3l-2 3M14 10l4 3-4 3" stroke="white" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span style={{ fontSize: "13px", color: "#484f58", fontWeight: 500 }}>
            DragonAI © {new Date().getFullYear()}
          </span>
        </div>
        <div style={{ display: "flex", gap: "20px" }}>
          {["Privacy", "Terms", "Docs", "Status"].map((l) => (
            <a
              key={l}
              href="#"
              style={{ fontSize: "13px", color: "#484f58", transition: "color 0.15s" }}
              onMouseOver={(e) => (e.currentTarget.style.color = "#8b949e")}
              onMouseOut={(e) => (e.currentTarget.style.color = "#484f58")}
            >
              {l}
            </a>
          ))}
        </div>
      </footer>

      {/* ── Modal ────────────────────────────────────────────── */}
      {modal && (
        <CreateModal onClose={() => setModal(false)} onSuccess={() => setModal(false)} />
      )}
    </>
  );
}