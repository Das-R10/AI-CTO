# 🧠 AI CTO — Autonomous Code Execution & Governance Engine

> Not a chatbot.  
> A correctness-gated, mutation-tested, sandboxed AI engineering system.

AI CTO is a production-oriented backend that autonomously plans, modifies, validates, and governs code changes — using strict correctness gates, mutation-tested specifications, behavioral contracts, security sandboxing, architectural awareness, and long-term project memory.

---

## 🚀 Core Philosophy

Most AI coding systems stop when:

> “Code runs without throwing an exception.”

AI CTO enforces:

> “Code must satisfy intent, survive mutation testing, pass deterministic contracts, clear adversarial review, and respect architecture — or it is rejected.”

No warnings. No soft passes. No best-effort persistence.

**Fail closed. Commit only on full validation.**

---

# 🏗️ System Architecture

```
User Instruction
      ↓
Planner (Execution DAG)
      ↓
Tool Execution (modify / create / refactor)
      ↓
Self-Correction Loop (runtime)
      ↓
Layer 2 — Postcondition Validation (detinistic)
      ↓
Layer 3 — Mutation-Anchored Spec Tests
      ↓
Layer 4 — Adversarial Critic Pass
      ↓
Security Scan + Sandbox
      ↓
Commit (only if ALL pass)
```

---

# 🧩 Major Components

---

## 1️⃣ Planner Engine

- Generates structured JSON execution plans
- Enforces atomic steps
- Uses project memory
- Respects architectural constraints
- Prevents public API mutation unless requested

File: `agent_executor.py`

---

## 2️⃣ Self-Correction Engine

Executes:

```
Generate → Execute → Catch → Analyze → Fix → Retry
```

- Runtime subprocess execution
- Timeout protection
- Traceback parsing
- Iterative repair loop
- Max retry enforcement

File: `self_correction.py`

---

## 3️⃣ Behavioral Validator (Deterministic)

Layer 2 correctness gate.

- Contracts derived from instruction
- Deterministic subprocess execution
- Probes with diverse input generation
- Flags None returns
- Flags type mismatches
- Flags structural contract violations

File: `correction_validator.py`

---

## 4️⃣ Mutation-Anchored Spec Engine

Breaks circular LLM bias.

### The Problem
If the same model generates both:
- The fix
- The tests

They share bias.

### The Solution

1. Tests generated BEFORE fix.
2. Tests validated against deliberate mutations.
3. Weak tests discarded.
4. Fix must pass only mutation-killing tests.

File: `test_spec_engine.py`

---

## 5️⃣ Hardened Execution Sandbox

Security-first execution layer:

- Dangerous import blocking
- os.system detection
- shell=True blocking
- Memory limits (RLIMIT_AS)
- CPU limits (RLIMIT_CPU)
- Process limits (anti-fork-bomb)
- Network isolation (`unshare --net`)
- File write restrictions
- Audit logging
- Safe mode (static-only execution)

File: `execution_hardening.py`

---

## 6️⃣ LLM Router (Failover + Circuit Breaker)

- Multi-provider routing
- Exponential backoff
- Circuit breaker
- Failure tracking
- Fallback models
- Health metrics

File: `infra.py`

---

## 7️⃣ Diff Engine (Change Transparency)

Before commit:
- Snapshot original files
- Generate unified diffs
- Risk score changes:
  - Public interface changes → HIGH
  - Large diffs → HIGH
  - Body edits → MEDIUM

Stored in `change_diffs` table.

File: `diff_engine.py`

---

## 8️⃣ Long-Term Project Memory

Vector-searchable persistent memory:

- Architecture decisions
- Constraints
- Past failures
- Style preferences
- Decision logs

Memory types:
- `architecture`
- `decision`
- `constraint`
- `style`

File: `memory_engine.py`

---

## 9️⃣ Autonomous Operations Layer

Background intelligence:

- Code evolution timeline
- Change velocity analysis
- Performance baseline capture
- Regression detection
- Behavioral diff snapshots
- Architectural drift detection
- Memory decay engine
- APScheduler automation

File: `autonomous_ops.py`

---

## 🔐 Correctness Gates (Strict Policy)

A change is accepted only if:

- ✅ Self-correction succeeds
- ✅ Postcondition passes
- ✅ Spec suite is trustworthy
- ✅ Fix passes locked spec tests
- ✅ Critic affirms intent
- ✅ Security scan passes
- ✅ No sandbox violations

Otherwise:

```
❌ Reject
❌ Rollback
❌ No persistence
```

---

# 🧪 Circular Dependency Prevention

AI CTO structurally prevents:

> “Tests written to pass the fix.”

By enforcing:

- Temporal separation
- Mutation oracle
- Kill-rate threshold
- Locked test suites

No trustworthy spec → No commit.

---

# 🛡️ Enterprise Governance

Supports:

- Role-based approvals
- Risk scoring
- Audit logging
- PR body generation
- Change approval flow
- High-risk change escalation

File: `enterprise_governance.py`

---

# 📊 API Endpoints

Key routes:

| Endpoint | Purpose |
|----------|---------|
| `/modify-code/` | Autonomous modification |
| `/ask/` | Context-aware code Q&A |
| `/project-health/` | Quality + dependency risk |
| `/quality/` | File quality metrics |
| `/dependency-graph/` | Coupling analysis |
| `/diff/{project}/{branch}` | Change inspection |
| `/rate-limit-status/` | Rate monitoring |
| `/llm-status/` | Router health |
| `/execute-goal/` | Multi-agent orchestration |

---

# 🗃️ Database Requirements

- PostgreSQL
- pgvector extension
- Tables:
  - projects
  - files
  - file_versions
  - project_memory
  - change_diffs
  - agent_jobs
  - performance_baselines
  - architectural_snapshots
  - drift_reports
  - rate_limit_log
  - scheduled_job_log

---

# ⚙️ Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup environment variables
export DATABASE_URL=postgresql://...
export GROQ_API_KEY=...

# 3. Enable pgvector
CREATE EXTENSION vector;

# 4. Run server
uvicorn main:app --reload
```

---

# 🧠 Design Goals

- Deterministic correctness gates
- Mutation-tested intent validation
- Structural bias prevention
- Fail-closed persistence
- Enterprise safety model
- Transparent diff tracking
- Long-term learning memory
- Secure execution sandbox
- Autonomous scheduled intelligence

---

# 🧩 Why This Is Different

Most AI code systems:
- Validate syntax
- Maybe run tests
- Persist on “no error”

AI CTO:
- Validates behavioral contracts
- Validates intent via mutation oracle
- Validates via adversarial critic
- Rejects weak spec suites
- Blocks None-return silent corruption
- Blocks dangerous execution
- Blocks weak validation passes

This is not “AI coding help.”  
This is an autonomous engineering control system.

---

# 🛠️ Roadmap

- Formal acceptance policy engine
- Static type diff validator
- Semantic regression snapshots
- Multi-model adversarial critic
- Automatic spec coverage scoring
- Formalized correctness state machine

---

# 🏁 Final Statement

AI CTO enforces:

> Code must prove correctness structurally — not just run successfully.

This repository demonstrates a correctness-gated autonomous software governance engine built for real production use.

---

## Author

Reyan Das  
AI Systems Engineering

---

## License

MIT License
