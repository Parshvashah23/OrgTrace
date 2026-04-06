---
title: OrgMemory-Env
emoji: 🏢
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
tags:
  - reinforcement-learning
  - openenv
  - organizational-memory
  - real-world-simulation
  - communication-debt
  - knowledge-management
---

# 🏢 OrgMemory-Env

**RL Environment for Organizational Communication Debt Resolution**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://openenv.ai)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)

OrgMemory-Env is a fully synthetic, self-contained reinforcement learning environment that simulates the **organizational memory problem** — the silent accumulation of undocumented decisions, broken commitments, and vanishing institutional knowledge inside growing companies.

An AI agent must navigate a realistic corpus of ~600 messages (emails, Slack, meeting notes) across 60 simulated days at a fictional B2B SaaS startup (**Meridian Labs**), performing information retrieval, causal reasoning, and structured document generation to solve three progressively harder tasks.

---

## 🎯 Why This Matters

| Signal | Real-world Impact | How We Model It |
|--------|------------------|-----------------|
| **Decision drift** | Teams re-litigate settled choices | 3 seeded decision chains (up to 7-hop) |
| **Broken commitments** | Work falls through cracks | 15 silently-dropped commitments among 40 total |
| **Knowledge exodus** | Employee departures create voids | 1 departing senior engineer, 3 critical systems |

These problems cost real companies millions in lost productivity. There's no existing benchmark for training agents to detect and resolve them.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       AGENT LOOP                            │
│   Observation → LLM/Agent → Action → env.step() → Reward   │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │       OrgMemoryEnv          │
          │  ┌──────────────────────┐   │
          │  │  State Machine       │   │
          │  │  - task_id           │   │
          │  │  - step counter      │   │
          │  │  - retrieved context │   │
          │  └──────────┬───────────┘   │
          │             │               │
          │  ┌──────────▼───────────┐   │
          │  │  Retrieval Engine    │   │
          │  │  BM25 + thread-trace │   │
          │  │  + person/date filter│   │
          │  └──────────┬───────────┘   │
          │             │               │
          │  ┌──────────▼───────────┐   │
          │  │  Message Corpus      │   │
          │  │  ~600 messages       │   │
          │  │  (email+slack+notes) │   │
          │  └──────────────────────┘   │
          │                             │
          │  ┌──────────────────────┐   │
          │  │  Graders (terminal)  │   │
          │  │  G1: Decision Arch.  │   │
          │  │  G2: Commitment Det. │   │
          │  │  G3: Knowledge Rec.  │   │
          │  └──────────────────────┘   │
          └─────────────────────────────┘
                         │
          ┌──────────────▼──────────────┐
          │     FastAPI / Docker        │
          │  POST /reset  POST /step    │
          │  GET  /state  GET  /tasks   │
          └─────────────────────────────┘
```

---

## 📋 Tasks

### Task 1: Decision Archaeology (Medium)
**Max steps:** 20 | **Reward range:** 0.0 – 1.0

> Trace the root cause of a known outcome back through a multi-hop decision chain. Identify the original decision, all intermediate steps, and the accountable party.

**Grading components:**
| Component | Weight |
|-----------|--------|
| Root decision found | 0.40 |
| Chain hops correct | 0.45 |
| Accountability correct | 0.15 |

### Task 2: Commitment Detection (Hard)
**Max steps:** 35 | **Reward range:** 0.0 – 1.0

> Surface all silently dropped commitments from 60 days of communications. Rank them by organizational risk. Do not flag resolved commitments.

**Grading components:**
| Component | Weight |
|-----------|--------|
| Recall (TP/15) | 0.40 |
| Precision | 0.30 |
| Risk ranking (Spearman ρ) | 0.20 |
| Resolution plan quality | 0.10 |

### Task 3: Knowledge Recovery (Expert)
**Max steps:** 50 | **Reward range:** 0.0 – 1.0

> Draft a complete knowledge transfer document for a departing senior engineer covering all critical systems, key decisions, collaborators, and open items.

**Grading components:**
| Component | Weight |
|-----------|--------|
| System coverage | 0.25 |
| Decision ownership accuracy (F1) | 0.25 |
| Relationship completeness | 0.20 |
| Temporal accuracy (±5 days) | 0.15 |
| Actionability (LLM-as-judge) | 0.15 |

---

## 🔧 Action Space

All actions are discrete JSON objects with `action_type`, `parameters`, and `reasoning`:

| Action | Required Params | Description |
|--------|----------------|-------------|
| `retrieve_messages` | `query` | BM25 keyword search with optional filters |
| `trace_thread` | `thread_id` | Follow a conversation thread |
| `tag_decision` | `message_id`, `decision_text` | Mark a message as a decision point |
| `tag_commitment` | `message_id`, `committer_id`, `commitment_text` | Mark a commitment |
| `link_cause_effect` | `cause_message_id`, `effect_message_id` | Link two messages causally |
| `draft_artifact` | `section`, `content` | Draft a document section |
| `submit` | `answer` | Submit final answer for grading |

**Optional filters for `retrieve_messages`:** `person_id`, `date_from`, `date_to`, `project_id`, `channel`, `top_k`

---

## 👁️ Observation Space

Each observation contains:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Current task identifier |
| `current_step` / `max_steps` | int | Step counter |
| `query` | string | Task-specific investigation prompt |
| `visible_messages` | List[Message] | 20 seed messages (stratified by channel/project) |
| `org_graph` | OrgGraph | Organization structure (40 people, 7 teams) |
| `project_states` | List[ProjectState] | 3 active projects with status |
| `retrieved_context` | List[Message] | Messages retrieved by agent actions |
| `action_history` | List[Action] | Previous actions taken |

---

## 🏆 Reward Function

**Intermediate rewards** (per-step signal):
- Retrieval of relevant messages: +0.05 max (relevance-weighted)
- Correct decision/commitment tagging: +0.05
- Valid causal link: +0.03

**Penalties:**
- Repeated query: -0.02
- Hallucinated message/thread ID: -0.05
- Hallucinated person ID: -0.03
- Timeout (no submit): -0.10

**Terminal reward:** Grader score × task weight (0.7 of final score)

---

## 🚀 Quick Start

### Local Setup

```bash
# Clone the repo
git clone https://github.com/your-username/org-memory-env.git
cd org-memory-env

# Install dependencies
pip install -r requirements.txt

# Generate the corpus
python scripts/generate_corpus.py --seed 42

# Run the API server
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Docker

```bash
# Build
docker build -t org-memory-env .

# Run
docker run -p 7860:7860 org-memory-env

# Test
curl http://localhost:7860/health
curl http://localhost:7860/tasks
curl http://localhost:7860/validate
```

### Run Baseline Agent

```bash
# Set your API key
export GROQ_API_KEY=your_key_here
# Or: export OPENAI_API_KEY=your_key_here

# Run all tasks
python -m baseline.run_baseline

# Run a single task
python -m baseline.run_baseline --task decision_archaeology --seed 42

# Use a specific provider
python -m baseline.run_baseline --provider openai
```

---

## 📊 Baseline Scores

| Task | Mean Score | Std |
|------|-----------|-----|
| Decision Archaeology | 0.51 | 0.047 |
| Commitment Detection | 0.41 | 0.062 |
| Knowledge Recovery | 0.33 | 0.058 |

*Baseline model: gpt-4o-mini | Scores may vary with different models and seeds.*

---

## 📡 API Reference

### `POST /reset`

Initialize a new episode.

**Request body:**
```json
{
  "task_id": "decision_archaeology",
  "seed": 42,
  "session_id": "default"
}
```

**Response:** Initial observation with visible messages, org graph, and project states.

### `POST /step`

Execute one agent action.

**Request body:**
```json
{
  "action_type": "retrieve_messages",
  "parameters": {"query": "database migration decision"},
  "reasoning": "Looking for the root cause of the migration block",
  "session_id": "default"
}
```

**Response:**
```json
{
  "observation": { ... },
  "reward": {
    "step_score": 0.035,
    "cumulative_score": 0.035,
    "penalty": 0.0,
    "done": false,
    "feedback": "Retrieved 7 messages (3 newly relevant)."
  },
  "done": false,
  "info": {"retrieved_count": 7}
}
```

### `GET /tasks`

List all available tasks.

### `GET /validate`

OpenEnv compliance check.

### `GET /state?session_id=default`

Full internal state for debugging.

### `GET /health`

Health check endpoint.

---

## 📁 Project Structure

```
org-memory-env/
├── data/
│   ├── personas.py              # 40 synthetic employee personas
│   ├── seeds.py                 # Ground truth: decisions, commitments, knowledge profile
│   ├── templates.py             # Communication style templates
│   └── generated/               # Generated at build/run time
│       ├── corpus.json          # ~600 messages (agent-facing)
│       ├── ground_truth.json    # Hidden from agent
│       └── corpus_annotated.json
├── env/
│   ├── models.py                # Pydantic models (Observation, Action, Reward, etc.)
│   ├── org_memory_env.py        # Core environment class
│   ├── retrieval.py             # BM25 + thread-trace engine
│   ├── reward.py                # Intermediate reward computation
│   └── graders/
│       ├── decision_archaeology.py
│       ├── commitment_detection.py
│       └── knowledge_recovery.py
├── baseline/
│   ├── run_baseline.py          # Baseline inference script
│   └── results.csv              # Generated after baseline run
├── scripts/
│   └── generate_corpus.py       # CLI corpus generator
├── app.py                       # FastAPI server
├── openenv.yaml                 # OpenEnv specification
├── Dockerfile                   # Container build
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🔒 Design Principles

1. **No real data** — All personas, messages, and events are synthetic
2. **Ground truth always known** — Every seeded fact has a key in the hidden manifest
3. **Gradient signal at every step** — Intermediate rewards prevent sparse reward pathology
4. **OpenEnv-compliant** — Drop-in compatible with any OpenEnv agent framework
5. **Deterministic** — Same seed produces identical episodes

---


