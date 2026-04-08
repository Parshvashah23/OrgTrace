
# 🏢 OrgTrace — Organizational Memory RL Environment

> **Submitted to the Meta OpenEnv Hackathon 2025**

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-compliant-4A90D9?style=for-the-badge)](https://openenv.ai)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=for-the-badge&logo=docker)](https://hub.docker.com)
[![HF Space](https://img.shields.io/badge/🤗%20HF%20Space-Live-yellow?style=for-the-badge)](https://huggingface.co/spaces/Parshva06/orgtrace)

---

## 📖 Table of Contents

1. [Environment Description & Motivation](#-environment-description--motivation)
2. [Architecture Overview](#-architecture-overview)
3. [Task Descriptions](#-task-descriptions)
4. [Observation Space](#-observation-space)
5. [Action Space](#-action-space)
6. [Reward Function](#-reward-function)
7. [Corpus & World Model](#-corpus--world-model)
8. [Baseline Scores](#-baseline-scores)
9. [Setup & Usage](#-setup--usage)
10. [API Reference](#-api-reference)
11. [Inference Script](#-inference-script)
12. [Project Structure](#-project-structure)
13. [OpenEnv Compliance](#-openenv-compliance)
14. [Design Principles](#-design-principles)

---

## 🌍 Environment Description & Motivation

### The Organizational Memory Problem

Every growing company accumulates **communication debt** — a silent, compounding liability:

- **Decisions** are made in Slack threads or ad-hoc meetings and never written down. Teams re-litigate them months later, wasting weeks of engineering time.
- **Commitments** slip between channels. An engineer promises to fix a security issue in a Zoom call. No one writes it down. The issue ships to production six months later.
- **Knowledge** walks out the door. When a senior engineer leaves, their understanding of three critical systems lives only in their head. The team spends weeks reconstructing context from 18 months of messages.

These are not edge cases. Studies estimate that knowledge loss from employee turnover costs Fortune 500 companies over **$31.5 billion per year**. Yet no RL benchmark exists to train agents that can detect and resolve these failure modes.

**OrgTrace** fills this gap.

### What the Agent Must Do

An AI agent is placed in the role of an **organizational memory analyst** at *Meridian Labs*, a fictional 40-person B2B SaaS startup. The agent receives a corpus of ~600 realistic messages (emails, Slack threads, meeting notes, document comments) spanning 60 simulated days, plus an organizational graph.

The agent must:
1. **Retrieve** relevant messages using keyword search and thread tracing
2. **Reason** causally about decision chains and implied commitments
3. **Tag** structured facts (decisions, commitments, causal links)
4. **Draft** structured artifacts
5. **Submit** a structured answer for automated scoring

### Why This Is Hard

| Challenge | Description |
|-----------|-------------|
| **Multi-hop reasoning** | Root causes can be 6–8 messages removed from the observable symptom |
| **Implicit signals** | Commitments are implied, not stated as `TODO:`. Decisions use natural language hedging |
| **Large context** | ~600 messages across 60 days, 7 communication channels, 40 people, 3 projects |
| **No oracle retrieval** | The agent must learn *what to search for*, not just how to read retrieved docs |
| **Structured grounding** | Answers must reference real `message_id` values — hallucinated IDs are penalized |

### Why This Benchmark Is Unique

OrgTrace is the **first RL benchmark** that:
- Tests agents on the *full pipeline* from retrieval → reasoning → structured generation
- Uses **realistic multi-channel communication** (not sanitized Q&A pairs)
- Provides **dense intermediate reward** at every agent action
- Requires **three qualitatively different skills** within a single environment
- Is fully synthetic, reproducible, and self-contained — zero external dependencies

---

## 🏗️ Architecture Overview

```
╔═══════════════════════════════════════════════════════════════════╗
║                         AGENT LOOP                                ║
║   Observation → LLM/Agent → Action JSON → env.step() → Reward     ║
╚══════════════════════════════════╦════════════════════════════════╝
                                   │
            ╔══════════════════════▼══════════════════════╗
            ║             OrgMemoryEnv Core               ║
            ║                                             ║
            ║  ┌──────────────────┐  ┌────────────────┐  ║
            ║  │  State Machine   │  │  Reward Engine │  ║
            ║  │  - task_id       │  │  - Relevance   │  ║
            ║  │  - step_counter  │  │    weights     │  ║
            ║  │  - agent_state   │  │  - Penalty     │  ║
            ║  │  - done flag     │  │    detection   │  ║
            ║  └────────┬─────────┘  └────────────────┘  ║
            ║           │                                  ║
            ║  ┌────────▼─────────────────────────────┐   ║
            ║  │          Retrieval Engine             │   ║
            ║  │  BM25 keyword search (rank-bm25)      │   ║
            ║  │  Thread tracing (forward/backward)    │   ║
            ║  │  Filters: person, date, project,      │   ║
            ║  │           channel, top_k              │   ║
            ║  └────────┬─────────────────────────────┘   ║
            ║           │                                  ║
            ║  ┌────────▼─────────────────────────────┐   ║
            ║  │       Synthetic Message Corpus        │   ║
            ║  │  ~600 messages across 60 sim-days     │   ║
            ║  │  5 channels: email, slack, meeting,   │   ║
            ║  │  doc_comments, slack topic channels   │   ║
            ║  │  40 personas, 3 projects, 7 teams     │   ║
            ║  └──────────────────────────────────────┘   ║
            ║                                             ║
            ║  ┌──────────────────────────────────────┐   ║
            ║  │       Terminal Graders (on submit)   │   ║
            ║  │  G1: Decision Archaeology grader     │   ║
            ║  │  G2: Commitment Detection grader     │   ║
            ║  │  G3: Knowledge Recovery grader       │   ║
            ║  └──────────────────────────────────────┘   ║
            ╚══════════════════════╦══════════════════════╝
                                   │
            ╔══════════════════════▼══════════════════════╗
            ║       FastAPI Server (Docker, Port 7860)    ║
            ║                                             ║
            ║  POST /reset       POST /step              ║
            ║  GET  /state       GET  /tasks              ║
            ║  GET  /validate    GET  /health             ║
            ║  GET  /                                     ║
            ╚═════════════════════════════════════════════╝
```

---

## 📋 Task Descriptions

OrgTrace contains three tasks of increasing difficulty, each testing a distinct reasoning capability. All tasks operate over the **same 60-day message corpus** but require different strategies.

---

### Task 1 · Decision Archaeology 🔍
**Difficulty: Medium** | **Max Steps: 20** | **Reward Range: 0.0 – 1.0**

#### Description

The company's Q2 roadmap currently excludes the OAuth migration. The agent's job is to trace back through the organization's full communication history to answer:

- What was the **original decision** that led to this exclusion?
- What is the **full causal chain** of messages from that root decision to the observable outcome?
- **Who is accountable** for this decision?

The correct answer spans a **6-hop decision chain** hidden across emails, a Slack thread, a meeting note, and a document comment — none of which explicitly say "we are dropping OAuth". The agent must reconstruct the chain from indirect signals.

#### Why It's Hard

- The root cause is buried ~45 days before the visible symptom
- No single message is a smoking gun — the chain must be assembled
- Several red herrings (similar-sounding messages) will mislead naive retrieval

#### Grading

| Component | Weight | Scoring Method |
|-----------|--------|---------------|
| Root decision identified correctly | 0.40 | Exact `message_id` match |
| Decision chain hops correct | 0.45 | Partial credit (+0.075/hop) up to 6 hops |
| Accountable person correct | 0.15 | `person_id` match |

**Penalties:**
- Wrong person blamed: **-0.10** off final score

---

### Task 2 · Commitment Detection 🎯
**Difficulty: Hard** | **Max Steps: 35** | **Reward Range: 0.0 – 1.0**

#### Description

Over the 60-day simulation period, employees made **40 commitments** — promises, action items, and agreed deliverables — across all communication channels. **15 of these were silently dropped**: never followed up on, never closed, never escalated.

The agent must:
1. Surface **all 15 dropped commitments** (with precision — don't flag resolved ones)
2. Identify the **committer** (person who made the promise) and the **source message**
3. Assign a **risk level** (`low` / `medium` / `high` / `critical`) based on organizational impact
4. Propose a **resolution plan** for each

#### Why It's Hard

- Commitments are implied, not explicit (`"I'll get to that"`, `"will handle it"`, `"let's circle back"`)
- Resolved commitments must be distinguished from dropped ones by tracking follow-up signals
- Risk ranking requires holistic understanding of the organizational context
- 40-commitment haystack with only 15 dropped = significant signal-to-noise challenge

#### Grading

| Component | Weight | Scoring Method |
|-----------|--------|---------------|
| Recall (TP / 15 dropped) | 0.40 | Partial credit per correctly identified dropped commitment |
| Precision (TP / all flagged) | 0.30 | Penalizes flagging resolved commitments |
| Risk ranking accuracy | 0.20 | Spearman rank correlation (ρ) vs. ground truth |
| Resolution plan quality | 0.10 | Presence and specificity of plan |

**Penalties:**
- Each resolved commitment flagged as dropped: **-0.05**

---

### Task 3 · Knowledge Recovery 📚
**Difficulty: Expert** | **Max Steps: 50** | **Reward Range: 0.0 – 1.0**

#### Description

*Sofia Reyes*, Meridian Labs' Senior Engineer, is **leaving next week**. She owns:
- 3 critical internal systems (auth gateway, data pipeline, billing service)
- 11 key architectural decisions over the past 18 months
- Mentorship relationships with 6 junior engineers
- 7 open items and unresolved ownership questions

The agent must construct a **complete knowledge transfer document** covering everything Sofia knows, by mining her 18 months of communication footprint in the corpus.

#### Why It's Hard

- Sofia's knowledge is distributed across 60+ messages; no single thread has the full picture
- Technical context must be inferred from architectural discussions, not explicit docs
- Open items must be distinguished from closed ones by tracking resolution signals
- The actionability of the document is scored by an LLM judge — not just factual coverage

#### Grading

| Component | Weight | Scoring Method |
|-----------|--------|---------------|
| System coverage (3 systems) | 0.25 | Partial credit per system correctly documented |
| Decision/ownership accuracy | 0.25 | F1 score vs. ground truth manifest |
| Relationship completeness | 0.20 | Coverage of collaborators and dependencies |
| Temporal accuracy | 0.15 | Events placed correctly within ±5 simulation days |
| Actionability (LLM-as-judge) | 0.15 | Scored by GPT-4o on usefulness and completeness |

**Penalties:**
- Confidently wrong factual claim: **-0.10** per instance

---

## 👁️ Observation Space

Every call to `step()` or `reset()` returns a structured **Observation** object. All fields are typed Pydantic models serializable to JSON.

```
Observation
├── task_id          : str                  # "decision_archaeology" | "commitment_detection" | "knowledge_recovery"
├── current_step     : int                  # Current step number (1-indexed)
├── max_steps        : int                  # Maximum steps for this task (20 / 35 / 50)
├── query            : str                  # Task-specific investigation prompt for the agent
├── visible_messages : List[Message]        # 20 seed messages (stratified by channel & project)
├── org_graph        : OrgGraph             # Full organizational structure
├── project_states   : List[ProjectState]  # Current status of 3 active projects
├── retrieved_context: List[Message]        # All messages retrieved by agent so far
└── action_history   : List[Action]         # All actions taken in current episode
```

### Message Schema

```python
class Message(BaseModel):
    message_id    : str            # Unique ID (e.g., "msg_0042")
    timestamp     : datetime       # ISO 8601 datetime (within 60 simulated days)
    sender_id     : str            # Person ID (e.g., "P07")
    sender_name   : str            # Full name (e.g., "Sofia Reyes")
    sender_email  : str            # Email address
    recipient_ids : List[str]      # List of recipient person IDs
    recipient_names: List[str]     # Recipient names
    channel       : str            # "email" | "slack" | "slack:#channel" | "meeting_notes" | "doc_comment"
    subject       : Optional[str]  # Email subject line (if email)
    body          : str            # Full message body
    thread_id     : Optional[str]  # Thread ID for traceable conversations
    project_tag   : Optional[str]  # "atlas" | "beacon" | "chronos"
    day           : int            # Simulation day (1–60)
```

### OrgGraph Schema

```python
class OrgGraph(BaseModel):
    people         : Dict[str, PersonDict]  # 40 personas (id → details)
    teams          : Dict[str, List[str]]   # 7 teams (team_name → person_id list)
    reports_to     : Dict[str, Optional[str]] # Manager hierarchy
    works_with     : Dict[str, List[str]]   # Collaboration links derived from projects
    project_members: Dict[str, List[str]]   # Project → team member IDs
```

### Initial Observation Details

- **Visible messages**: 20 seed messages selected at start, stratified to ensure:
  - ≥3 messages from each channel type
  - ≥1 message referencing each active project
  - All drawn from simulation days 1–30 (early corpus only)
- **Retrieved context**: Empty at episode start; grows as agent uses `retrieve_messages` and `trace_thread`

---

## 🎮 Action Space

All actions are discrete JSON objects. The agent emits exactly one action per step. An action has three fields:

```json
{
  "action_type": "<one of 7 types>",
  "parameters": { ... },
  "reasoning": "The agent's chain-of-thought for this action"
}
```

### Action Reference

#### `retrieve_messages` — BM25 Keyword Search

```json
{
  "action_type": "retrieve_messages",
  "parameters": {
    "query": "OAuth migration security decision",
    "person_id": "P07",
    "date_from": "2024-01-01",
    "date_to": "2024-02-15",
    "project_id": "atlas",
    "channel": "email",
    "top_k": 10
  },
  "reasoning": "Searching for early decision-making around OAuth"
}
```
- **Required:** `query`
- **Optional:** `person_id`, `date_from`, `date_to`, `project_id`, `channel`, `top_k` (default 10)
- **Returns:** Up to `top_k` ranked messages sorted by BM25 relevance

---

#### `trace_thread` — Conversation Thread Tracing

```json
{
  "action_type": "trace_thread",
  "parameters": {
    "thread_id": "thread_0018",
    "direction": "both"
  },
  "reasoning": "Following the email thread to see all replies"
}
```
- **Required:** `thread_id`
- **Optional:** `direction` — `"forward"` | `"backward"` | `"both"` (default `"both"`)
- **Returns:** All messages in the thread, sorted by timestamp

---

#### `tag_decision` — Mark a Decision Point

```json
{
  "action_type": "tag_decision",
  "parameters": {
    "message_id": "msg_0122",
    "decision_text": "Decided to deprioritize OAuth in favor of shipping search",
    "accountable_person_id": "P03"
  },
  "reasoning": "This message explicitly deprioritizes OAuth"
}
```
- **Required:** `message_id`, `decision_text`
- **Optional:** `accountable_person_id`
- **Returns:** Confirmation + intermediate reward based on correctness

---

#### `tag_commitment` — Mark an Implied Commitment

```json
{
  "action_type": "tag_commitment",
  "parameters": {
    "message_id": "msg_0245",
    "committer_id": "P11",
    "commitment_text": "Will follow up on the API rate limit fix by end of sprint",
    "risk_level": "high"
  },
  "reasoning": "Marcus explicitly committed to this fix"
}
```
- **Required:** `message_id`, `committer_id`, `commitment_text`
- **Optional:** `risk_level` — `"low"` | `"medium"` | `"high"` | `"critical"`
- **Returns:** Confirmation + intermediate reward

---

#### `link_cause_effect` — Causal Link

```json
{
  "action_type": "link_cause_effect",
  "parameters": {
    "cause_message_id": "msg_0031",
    "effect_message_id": "msg_0178",
    "explanation": "The security incident (cause) led to halting the OAuth work (effect)"
  },
  "reasoning": "These two messages are causally linked"
}
```
- **Required:** `cause_message_id`, `effect_message_id`
- **Optional:** `explanation`

---

#### `draft_artifact` — Write Document Section

```json
{
  "action_type": "draft_artifact",
  "parameters": {
    "section": "critical_systems",
    "content": "Sofia owns the Auth Gateway, Data Pipeline, and Billing Service...",
    "artifact_type": "knowledge_transfer"
  },
  "reasoning": "Documenting Sofia's system ownership"
}
```
- **Required:** `section`, `content`
- **Optional:** `artifact_type`

---

#### `submit` — Final Answer Submission

```json
{
  "action_type": "submit",
  "parameters": {
    "answer": { ... }
  },
  "reasoning": "I have sufficient evidence to submit"
}
```

Submission schemas differ by task. See [Task Descriptions](#-task-descriptions) for formats.

---

## 🏆 Reward Function

OrgTrace provides **dense intermediate rewards** at every step to avoid sparse reward pathology, plus a terminal reward from the automated grader.

### Intermediate Rewards (per step)

| Event | Reward |
|-------|--------|
| Retrieving a message in the top-50% relevance tier | +0.05 max (scaled by relevance weight) |
| Correct `tag_decision` (matches ground truth decision list) | +0.05 |
| Correct `tag_commitment` (matches ground truth dropped commitment) | +0.05 |
| Valid causal link that matches a ground truth edge | +0.03 |
| Drafting an artifact section (effort signal) | +0.01 |

### Penalties (per step)

| Event | Penalty |
|-------|---------|
| Exact repeated query (identical parameters) | -0.02 |
| Using a `message_id` that doesn't exist in corpus | -0.05 |
| Using a `thread_id` that doesn't exist in corpus | -0.05 |
| Using a `person_id` that doesn't exist in org graph | -0.03 |
| Episode timeout (no `submit` before max_steps) | -0.10 |

### Terminal Reward (on `submit`)

The `submit` action invokes the task-specific grader and returns the **graded score** (0.0–1.0) which becomes the dominant component of the final score. Intermediate rewards serve as a gradient signal; the grader score is the primary evaluation metric.

---

## 🌐 Corpus & World Model

### Meridian Labs — The Simulated Organization

| Property | Value |
|----------|-------|
| Company type | B2B SaaS startup |
| Simulation window | 60 days |
| number of messages | ~600 |
| Personas (employees) | 40 |
| Teams | 7 (Engineering, Product, Design, Sales, Marketing, Data, Ops) |
| Active Projects | 3 (Atlas, Beacon, Chronos) |

### Seeded Ground Truth

The corpus contains **fully deterministic, reproducible ground truth** — all seeded facts are fixed in `data/seeds.py` and keyed in `data/generated/ground_truth.json` (hidden from the agent):

| Ground Truth Category | Count |
|----------------------|-------|
| Decision seeds (with multi-hop chains) | 3 primary, 7 total decision nodes |
| Total commitments seeded | 40 |
| Dropped (not followed up) commitments | 15 |
| Sofia's critical systems | 3 |
| Sofia's key decisions | 11 |

### Communication Channels

| Channel | Description |
|---------|-------------|
| `email` | Formal inter-team email threads |
| `slack` | Direct messages and informal discussions |
| `slack:#channel` | Topic-specific Slack channels (e.g., `#product`, `#infra`) |
| `meeting_notes` | Structured notes from standup, sprint, and all-hands meetings |
| `doc_comment` | Comments on shared documents and specs |

---

## 📊 Baseline Scores

Scores obtained using `gpt-4o-mini` via the baseline inference script (`baseline/run_baseline.py`), averaged over 3 seeds (42, 123, 999):

| Task | Mean Score | Std Dev | Difficulty | Notes |
|------|-----------|---------|-----------|-------|
| Decision Archaeology | **0.51** | ±0.047 | Medium | Struggles with correct root message identification |
| Commitment Detection | **0.41** | ±0.062 | Hard | High recall but lower precision; misses implied commitments |
| Knowledge Recovery | **0.33** | ±0.058 | Expert | Temporal placement and open-item tracking are weak points |

### Baseline Strategy

The baseline agent uses a **zero-shot prompting** approach with:
- BM25 retrieval driven by task-specific query keywords
- No fine-tuning or task-specific prompt optimization
- Chain-of-thought reasoning in the `reasoning` field of each action
- Forced `submit` 2 steps before the step limit

These scores establish a **meaningful floor** — there is substantial headroom for improvement via better prompting, retrieval strategies, or fine-tuned models.

---

## 🚀 Setup & Usage

### Prerequisites

- Python 3.11+
- Docker (for containerized deployment)
- `pip install -r requirements.txt`

### Option 1: Local Python

```bash
# 1. Clone the repo
git clone https://github.com/Parshvasha23/OrgTrace.git
cd OrgTrace

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the synthetic corpus (takes ~10 seconds)
python scripts/generate_corpus.py --seed 42

# 4. Start the FastAPI server
uvicorn app:app --host 0.0.0.0 --port 7860

# 5. Verify it's running
curl http://localhost:7860/
curl http://localhost:7860/tasks
```

### Option 2: Docker

```bash
# Build (corpus is pre-generated at build time)
docker build -t orgtrace .

# Run
docker run -p 7860:7860 orgtrace

# Verify
curl http://localhost:7860/health
curl http://localhost:7860/tasks
curl http://localhost:7860/validate
```

### Option 3: Hugging Face Space (Live)

The environment is deployed and live at:

```
https://Parshva06-orgtrace.hf.space
```

No setup required — call the API directly.

### Running the Baseline Agent

```bash
# Set your LLM API key
export GROQ_API_KEY=your_key_here         # Groq (recommended, fast & free)
# or
export OPENAI_API_KEY=your_key_here       # OpenAI

# Run all 3 tasks × 3 seeds (9 total episodes)
python -m baseline.run_baseline

# Run a single task with a specific seed
python -m baseline.run_baseline --task decision_archaeology --seed 42

# Run silently (no per-step output)
python -m baseline.run_baseline --quiet
```

### Running the Evaluation Inference Script

```bash
# Set mandatory environment variables
export HF_TOKEN=your_hf_token
export API_BASE_URL=https://router.huggingface.co/v1     # or any OpenAI-compatible endpoint
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

# Optional
export ORG_TRACE_TASK=decision_archaeology  # Select specific task (default: decision_archaeology)
export LOCAL_IMAGE_NAME=orgtrace            # If using Docker image directly

# Run inference (produces structured STDOUT logs)
python inference.py
```

**Expected STDOUT format:**

```
[START] task=decision_archaeology env=orgtrace model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={...} reward=0.03 done=false error=null
[STEP] step=2 action={...} reward=0.05 done=false error=null
...
[END] success=true steps=12 score=0.58 rewards=0.03,0.05,...
```

---

## 📡 API Reference

Base URL: `https://Parshva06-orgtrace.hf.space` (or `http://localhost:7860`)

---

### `GET /`
Health check.
```json
{ "status": "ok", "env": "orgtrace" }
```

---

### `GET /health`
Detailed health check.
```json
{ "status": "healthy", "version": "1.0.0" }
```

---

### `GET /tasks`
List all available tasks.
```json
["decision_archaeology", "commitment_detection", "knowledge_recovery"]
```

---

### `GET /validate`
OpenEnv compliance check. Returns which spec requirements are satisfied.

---

### `POST /reset`
Initialize a new episode.

**Request (JSON body):**
```json
{
  "task_id": "decision_archaeology",
  "seed": 42,
  "session_id": "agent-run-001"
}
```

**Also supports query params:**
```bash
curl -X POST "http://localhost:7860/reset?task_id=decision_archaeology&session_id=test"
```

**Response:**
```json
{
  "observation": {
    "task_id": "decision_archaeology",
    "current_step": 0,
    "max_steps": 20,
    "query": "Our Q2 roadmap currently excludes the OAuth migration...",
    "visible_messages": [ ... ],
    "org_graph": { ... },
    "project_states": [ ... ],
    "retrieved_context": [],
    "action_history": []
  },
  "session_id": "agent-run-001",
  "task_id": "decision_archaeology",
  "max_steps": 20
}
```

---

### `POST /step`
Execute one agent action.

**Request:**
```json
{
  "action_type": "retrieve_messages",
  "parameters": { "query": "OAuth security decision", "top_k": 10 },
  "reasoning": "Looking for the root decision around OAuth",
  "session_id": "agent-run-001"
}
```

**Response:**
```json
{
  "observation": { ... },
  "reward": {
    "step_score": 0.035,
    "cumulative_score": 0.035,
    "component_scores": {},
    "penalty": 0.0,
    "done": false,
    "feedback": "Retrieved 8 messages (3 newly relevant)"
  },
  "done": false,
  "info": { "retrieved_count": 8 }
}
```

---

### `GET /state`
Full internal debug state.
```
GET /state?session_id=agent-run-001
```

---

## 🔬 Inference Script

The `inference.py` script in the root directory is the **evaluator-compliant** inference entry point. It follows the exact STDOUT format required by the Meta OpenEnv Hackathon.

**Required Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | *(required)* | Your Hugging Face API key |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | OpenAI-compatible API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `LOCAL_IMAGE_NAME` | *(optional)* | Docker image name if using local deployment |
| `ORG_TRACE_TASK` | `decision_archaeology` | Task to run |

**STDOUT Format:**
```
[START] task=<task_name> env=orgtrace model=<model_name>
[STEP]  step=<n> action=<action_json> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
```

---

## 📁 Project Structure

```
OrgTrace/
├── data/
│   ├── personas.py            # 40 synthetic employee personas with full profiles
│   ├── seeds.py               # Seeded ground truth: decisions, commitments, knowledge profile of Sofia
│   ├── templates.py           # Realistic communication style templates
│   └── generated/             # Created at build/run time (gitignored)
│       ├── corpus.json        # ~600 messages — the agent's observable world
│       └── ground_truth.json  # Hidden ground truth for graders
│
├── org_env/                   # Core environment package
│   ├── __init__.py
│   ├── email_triage_env.py    # OrgTraceEnv alias for compatibility
│   ├── models.py              # All Pydantic models (Observation, Action, Reward, etc.)
│   ├── org_memory_env.py      # Core OrgMemoryEnv class
│   ├── retrieval.py           # BM25 retrieval engine + thread tracing
│   ├── reward.py              # Intermediate reward computation logic
│   └── graders/
│       ├── decision_archaeology.py   # Task 1 grader
│       ├── commitment_detection.py   # Task 2 grader
│       └── knowledge_recovery.py     # Task 3 grader
│
├── baseline/
│   ├── run_baseline.py        # Full baseline inference agent (Groq/OpenAI)
│   └── results.csv            # Populated after baseline run
│
├── scripts/
│   └── generate_corpus.py    # CLI wrapper for the corpus generator
│
├── app.py                    # FastAPI server (OpenEnv-compliant)
├── inference.py              # ⭐ Hackathon evaluation entrypoint
├── generator.py              # Synthetic corpus generator (60-day simulation)
├── openenv.yaml              # OpenEnv specification file
├── Dockerfile                # Container build configuration
├── requirements.txt          # Python dependencies
├── ping_check.py             # Space verification utility
└── README.md                 # This file
```

---

## ✅ OpenEnv Compliance

OrgTrace is fully compliant with the OpenEnv specification:

| Requirement | Status |
|-------------|--------|
| `openenv.yaml` present and valid | ✅ |
| `POST /reset` endpoint | ✅ |
| `POST /step` endpoint | ✅ |
| `GET /state` endpoint | ✅ |
| `GET /tasks` endpoint | ✅ |
| `GET /validate` endpoint | ✅ |
| Typed Pydantic models for all IO | ✅ |
| Reward range [0.0, 1.0] | ✅ |
| 3+ tasks with automated graders | ✅ (3 tasks) |
| Reproducible with seed | ✅ |
| Docker deployment on port 7860 | ✅ |
| `inference.py` in root directory | ✅ |
| Structured STDOUT logs (START/STEP/END) | ✅ |
| OpenAI client for all LLM calls | ✅ |

---

## 🔒 Design Principles

1. **No real data** — All personas, messages, projects, and events are fully synthetic. No privacy concerns.
2. **Ground truth always exists** — Every seeded fact has a concrete answer in the hidden manifest. Grading is objective and reproducible.
3. **Dense gradient signal** — Intermediate rewards at every step prevent sparse reward problems and allow agents to learn from exploration.
4. **OpenEnv-first** — Designed as a drop-in compatible environment for any OpenEnv agent framework from day one.
5. **Deterministic reproducibility** — Same `seed` always produces identical corpus, episodes, and scores.
6. **Scalable difficulty** — Three tasks of increasing cognitive complexity test different agent capabilities without requiring new infrastructure.
7. **Hallucination-resistant** — Agents are penalized for using IDs that don't exist, encouraging grounded retrieval over confabulation.

---

## 🙏 Acknowledgements

Built for the **Meta OpenEnv Hackathon 2025**. The OrgTrace environment is inspired by real organizational memory challenges experienced at high-growth software companies.

---

*OrgTrace — because the decisions that shaped your company are hiding in your inbox.*
