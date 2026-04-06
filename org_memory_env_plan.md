# OrgMemory-Env: Full Implementation Plan
### RL Environment for Organizational Communication Debt Resolution

> **Status:** Phase 1 (Synthetic Org Data Generator) — partially complete.
> All subsequent phases assume the generator scaffold is in place.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Phase 1 — Synthetic Org Data Generator](#3-phase-1--synthetic-org-data-generator)
4. [Phase 2 — Pydantic Models + OpenEnv Spec](#4-phase-2--pydantic-models--openenv-spec)
5. [Phase 3 — Core Environment Class](#5-phase-3--core-environment-class)
6. [Phase 4 — Three Graders](#6-phase-4--three-graders)
7. [Phase 5 — Reward Shaping](#7-phase-5--reward-shaping)
8. [Phase 6 — Baseline Inference Script](#8-phase-6--baseline-inference-script)
9. [Phase 7 — Docker + HF Deployment](#9-phase-7--docker--hf-deployment)
10. [File & Directory Structure](#10-file--directory-structure)
11. [Dependencies & Tooling](#11-dependencies--tooling)
12. [Testing Strategy](#12-testing-strategy)
13. [Timeline & Effort Estimates](#13-timeline--effort-estimates)
14. [Open Questions & Risk Register](#14-open-questions--risk-register)

---

## 1. Project Overview

**OrgMemory-Env** is a fully synthetic, self-contained reinforcement learning environment
that simulates the _organizational memory problem_: the silent accumulation of undocumented
decisions, broken commitments, and vanishing institutional knowledge inside growing companies.

### Core Problem Being Modelled

| Signal | Real-world Impact | Env Representation |
|--------|------------------|--------------------|
| Decision drift | Teams re-litigate settled choices | Seeded 8-hop decision trees |
| Broken commitments | Work falls through cracks | 15 silently-dropped commitments |
| Knowledge exodus | Employee departures create voids | 1 departing senior engineer, 3 critical systems |

### Design Principles

- **No real data ever enters the system.** All personas, messages, and events are generated.
- **Ground truth is always known.** Every seeded fact has a key in a hidden JSON manifest.
- **Gradient signal at every step.** Intermediate rewards prevent sparse reward pathology.
- **OpenEnv-compliant.** Drop-in compatible with any OpenEnv-compatible agent framework.

---

## 2. Architecture Diagram

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
          │  │  ~1000 messages      │   │
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

## 3. Phase 1 — Synthetic Org Data Generator

**Status: Partially complete.**
**Estimated remaining effort: ~1 hr to harden and finalize.**

### 3.1 Persona Definition

Each of the 40 synthetic employees is a `Persona` object stored in `data/personas.json`.

```python
@dataclass
class Persona:
    person_id: str                   # e.g., "P001"
    name: str                        # e.g., "Alex Chen"
    role: str                        # e.g., "Senior Engineer"
    team: str                        # Engineering | Product | Sales | Legal | HR | Leadership | Design
    email: str
    slack_handle: str
    communication_style: Literal[
        "terse", "verbose", "formal", "casual", "bullet-heavy", "narrative"
    ]
    expertise_domains: List[str]     # e.g., ["auth-service", "billing", "CI/CD"]
    reports_to: Optional[str]        # person_id of manager
    is_departing: bool               # True for the 1 departing senior engineer
    departure_day: Optional[int]     # sim day on which they leave
```

**Team breakdown:**

| Team | Count | Key Roles |
|------|-------|-----------|
| Engineering | 10 | 1 Staff Eng (departing), 2 Senior, 4 Mid, 2 Junior, 1 EM |
| Product | 8 | 1 VP Product, 2 PM, 3 APM, 2 Designer-PM hybrid |
| Sales | 7 | 1 VP Sales, 3 AE, 2 SDR, 1 RevOps |
| Legal | 3 | 1 General Counsel, 1 Paralegal, 1 Privacy Lead |
| HR | 4 | 1 CHRO, 2 HR BP, 1 Recruiter |
| Leadership | 4 | CEO, CTO, CFO, COO |
| Design | 4 | 1 Design Lead, 3 Designers |

### 3.2 Communication Style Templates

To prevent all messages sounding identical, each `communication_style` maps to a
set of structural templates and vocabulary biases applied at generation time.

```python
STYLE_TEMPLATES = {
    "terse": {
        "greeting": ["", "Hi,", "Hey"],
        "closing": ["", "Thanks", "-{first_name}"],
        "sentence_length": "short",
        "uses_bullets": True,
        "avg_words": 40,
    },
    "verbose": {
        "greeting": ["Hope this finds you well.", "Following up on our earlier thread —"],
        "closing": ["Please let me know if you have any questions or concerns.", "Happy to jump on a call."],
        "sentence_length": "long",
        "uses_bullets": False,
        "avg_words": 220,
    },
    # ... formal, casual, bullet-heavy, narrative
}
```

### 3.3 Org Graph

The `OrgGraph` object captures structural relationships and is part of every `Observation`.

```python
@dataclass
class OrgGraph:
    people: Dict[PersonID, Persona]
    teams: Dict[str, List[PersonID]]
    reports_to: Dict[PersonID, PersonID]          # direct manager edges
    works_with: Dict[PersonID, List[PersonID]]    # cross-team collaboration edges
    project_members: Dict[ProjectID, List[PersonID]]
```

Cross-team collaboration edges are seeded based on projects:
- **Project Atlas** — Engineering (all) + Product (all) + Design (all)
- **Project Beacon** — Engineering (5) + Product (4) + Sales (2)
- **Project Crest** — Legal (all) + Engineering (3) + HR (2) + Leadership (2)

### 3.4 Seeded Decisions (15 total)

Each decision is a structured event in the hidden ground-truth manifest:

```python
@dataclass
class SeededDecision:
    decision_id: str
    project: ProjectID
    sim_day: int                         # when it was made
    decision_text: str                   # e.g., "Defer OAuth 2.0 migration to Q3"
    cause_message_ids: List[str]         # messages that led to this decision
    decision_message_id: str             # the message where decision was made
    effect_message_ids: List[str]        # downstream messages citing/assuming this decision
    accountable_person_id: PersonID      # who owns this decision
    hop_depth: int                       # how many hops in the causal chain (1–8)
    difficulty: Literal["easy", "medium", "hard"]
```

One decision will be a full 8-hop chain (used as Task 1's primary target).
Other decisions range from 1–5 hops.

### 3.5 Seeded Commitments (40 total)

```python
@dataclass
class SeededCommitment:
    commitment_id: str
    project: ProjectID
    sim_day: int
    committer_id: PersonID
    commitment_text: str                  # e.g., "I'll send the updated API spec by Friday"
    source_message_id: str
    resolved: bool                        # True for 25, False for 15
    resolution_message_id: Optional[str] # present if resolved=True
    resolution_day: Optional[int]
    risk_level: Literal["low", "medium", "high", "critical"]
    # risk_level is derived from: commitment age × project criticality × role seniority
```

**Risk level distribution of the 15 dropped commitments:**

| Risk | Count | Reason |
|------|-------|--------|
| critical | 3 | Blocks a release or compliance deadline |
| high | 4 | Delays cross-team work |
| medium | 5 | Minor friction |
| low | 3 | Nice-to-have / informational |

### 3.6 Departing Employee Profile

The departing persona (1 senior engineer) has a hidden profile that becomes
the ground truth for Grader 3:

```python
@dataclass
class DepartingEmployeeProfile:
    person_id: PersonID
    critical_systems: List[str]           # e.g., ["auth-service", "billing-pipeline", "deploy-infra"]
    decisions_owned: List[str]            # decision_ids they were accountable for
    key_collaborators: List[PersonID]     # the 5–7 people they worked most closely with
    knowledge_artifacts: List[str]        # doc names / runbook titles only they authored
    timeline: List[Dict]                  # {day, event} list of their project arc
    open_issues: List[str]                # unresolved items they were tracking
```

### 3.7 Message Generator

The generator produces ~800–1200 messages over 60 simulated days.

**Generation pipeline:**

```
1. For each sim day d in [0, 60]:
   a. Randomly select 12–22 active personas
   b. For each active persona, sample 1–3 message-generation triggers:
      - "respond to open thread" (if they have unanswered messages)
      - "initiate on project" (based on project_members assignment)
      - "escalate to manager" (stochastic, biased by project stress level)
      - "commit to action" (samples from commitment pool if unfilled)
      - "reference decision" (samples from decision pool)
   c. Generate message body using:
      - Style template for sender's communication_style
      - LLM call (GPT-4o-mini) with a structured prompt
      - Inject seeded decision/commitment references where triggered
   d. Assign message_id, timestamp, thread_id (if reply), channel
   
2. Post-processing:
   a. Verify all 15 decisions are referenced in at least 2 messages
   b. Verify all 40 commitments have a source message
   c. Verify all 25 resolved commitments have a resolution message
   d. Write ground_truth_manifest.json (HIDDEN from agent)
   e. Write messages.json (VISIBLE — the corpus)
```

**Channel distribution:**

| Channel | % of Messages | Typical Use |
|---------|--------------|-------------|
| slack | 55% | Quick updates, reactions, threads |
| email | 30% | Formal announcements, cross-team |
| meeting_notes | 10% | Decisions, action items, recaps |
| doc_comment | 5% | Specific file/spec feedback |

### 3.8 Generator Output Files

```
data/
├── messages.json              # Full corpus (~1000 messages)
├── personas.json              # 40 persona definitions
├── org_graph.json             # Relationship edges
├── project_states.json        # Per-project state snapshots
└── ground_truth/
    ├── decisions.json         # All 15 seeded decisions (HIDDEN)
    ├── commitments.json       # All 40 commitments (HIDDEN)
    └── departing_profile.json # Departing employee profile (HIDDEN)
```

> ⚠️ **The `ground_truth/` directory must never be included in the observation
> space or retrievable by the agent. Load it only inside grader functions.**

### 3.9 Remaining Work for Phase 1

- [ ] Finalize `communication_style` template dictionary
- [ ] Implement post-processing validation loop (step 2 above)
- [ ] Write `ground_truth_manifest.json` serializer
- [ ] Add seed control (`random.seed(N)` + `numpy.seed(N)`) for reproducibility
- [ ] Unit test: assert all 15 dropped commitments have no resolution message in corpus
- [ ] Unit test: assert 8-hop decision chain is fully traceable through message IDs

---

## 4. Phase 2 — Pydantic Models + OpenEnv Spec

**Estimated effort: 1 hr**

### 4.1 Core Data Models

All models live in `env/models.py`. Use `pydantic v2`.

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, List, Any
from datetime import datetime

PersonID = str
ProjectID = str

class Message(BaseModel):
    message_id: str
    timestamp: datetime
    sender: PersonID
    recipients: List[PersonID]
    channel: Literal["email", "slack", "meeting_notes", "doc_comment"]
    subject: Optional[str] = None
    body: str
    thread_id: Optional[str] = None
    project_tag: Optional[ProjectID] = None

class OrgGraph(BaseModel):
    people: Dict[PersonID, dict]         # Persona dicts
    teams: Dict[str, List[PersonID]]
    reports_to: Dict[PersonID, Optional[PersonID]]
    works_with: Dict[PersonID, List[PersonID]]
    project_members: Dict[ProjectID, List[PersonID]]

class ProjectState(BaseModel):
    project_id: ProjectID
    name: str
    status: Literal["on_track", "at_risk", "blocked", "completed"]
    lead: PersonID
    last_update_day: int
    open_items: List[str]

class Action(BaseModel):
    action_type: Literal[
        "retrieve_messages",
        "trace_thread",
        "tag_decision",
        "tag_commitment",
        "link_cause_effect",
        "draft_artifact",
        "submit"
    ]
    parameters: Dict[str, Any] = Field(default_factory=dict)
    reasoning: str

class Observation(BaseModel):
    task_id: str
    current_step: int
    max_steps: int
    query: str
    visible_messages: List[Message]
    org_graph: OrgGraph
    project_states: List[ProjectState]
    retrieved_context: List[Message]
    action_history: List[Action]

class Reward(BaseModel):
    step_score: float
    cumulative_score: float
    component_scores: Dict[str, float]
    penalty: float
    done: bool
    feedback: str
```

### 4.2 Action Parameter Schemas

Each `action_type` has an expected `parameters` structure. Validate at `env.step()`:

| action_type | Required Parameters | Optional |
|-------------|---------------------|---------|
| `retrieve_messages` | `query: str` | `person_id`, `date_from`, `date_to`, `project_id`, `channel` |
| `trace_thread` | `thread_id: str` | `direction: "forward"\|"backward"\|"both"` |
| `tag_decision` | `message_id: str`, `decision_text: str` | `accountable_person_id` |
| `tag_commitment` | `message_id: str`, `committer_id: str`, `commitment_text: str` | `risk_level` |
| `link_cause_effect` | `cause_message_id: str`, `effect_message_id: str` | `explanation: str` |
| `draft_artifact` | `section: str`, `content: str` | `artifact_type` |
| `submit` | `answer: dict` | — |

### 4.3 openenv.yaml

```yaml
name: org-memory-env
version: 1.0.0
description: >
  RL environment for organizational communication debt resolution.
  Agents must trace decisions, surface broken commitments, and reconstruct
  institutional knowledge from realistic synthetic communication histories.

tasks:
  - id: decision_archaeology
    difficulty: medium
    max_steps: 20
    reward_range: [0.0, 1.0]
    description: >
      Trace the root cause of a known outcome back through an 8-hop decision chain.
      Identify the original decision, all intermediate steps, and the accountable party.

  - id: commitment_detection
    difficulty: hard
    max_steps: 35
    reward_range: [0.0, 1.0]
    description: >
      Surface all silently dropped commitments from 60 days of communications.
      Rank them by organizational risk. Do not flag resolved commitments.

  - id: knowledge_recovery
    difficulty: expert
    max_steps: 50
    reward_range: [0.0, 1.0]
    description: >
      Draft a knowledge transfer document for a departing senior engineer.
      Cover all critical systems, key decisions, collaborators, and open items.

observation_space: structured_text_graph
action_space: discrete_retrieval_plus_generation
baseline_model: gpt-4o-mini

baseline_scores:
  decision_archaeology: 0.51
  commitment_detection: 0.41
  knowledge_recovery: 0.33

tags:
  - openenv
  - organizational-memory
  - communication-debt
  - knowledge-management
```

---

## 5. Phase 3 — Core Environment Class

**Estimated effort: 3 hrs**

### 5.1 Class Interface

```python
# env/org_memory_env.py

class OrgMemoryEnv:
    def __init__(self, data_dir: str = "data/", seed: int = 42):
        ...

    def reset(self, task_id: str) -> Observation:
        """
        Initialize a new episode.
        - Load corpus, org graph, project states
        - Sample 20 seed messages visible at start (stratified by channel)
        - Reset step counter, action history, retrieved_context
        - Return initial Observation
        """

    def step(self, action: Action) -> tuple[Observation, Reward, bool, dict]:
        """
        Execute one agent action.
        - Validate action parameters
        - Dispatch to appropriate handler
        - Compute intermediate reward
        - Check for episode termination
        - Return updated (Observation, Reward, done, info)
        """

    def state(self) -> FullState:
        """
        Return complete internal state (for debugging / logging).
        Includes hidden ground truth keys (never sent to agent).
        """

    def render(self) -> str:
        """
        Return a human-readable string representation of the current state.
        Useful for debugging and logging.
        """
```

### 5.2 State Machine Detail

```
EPISODE START
    │
    ▼
reset(task_id)
    ├─ Load corpus + ground truth manifest
    ├─ Sample 20 seed messages
    ├─ Set step = 0, context = [], history = []
    └─ Return Observation

    AGENT LOOP
    │
    ▼
step(action)
    ├─ Validate action type + params
    ├─ Dispatch:
    │   ├─ retrieve_messages  → BM25 search → return ≤10 messages
    │   ├─ trace_thread       → follow thread chain → return thread messages
    │   ├─ tag_decision       → record in agent_state.tagged_decisions
    │   ├─ tag_commitment     → record in agent_state.tagged_commitments
    │   ├─ link_cause_effect  → record in agent_state.causal_links
    │   ├─ draft_artifact     → append to agent_state.artifact_sections
    │   └─ submit             → trigger full grader evaluation → done=True
    ├─ Compute intermediate reward (see Phase 5)
    ├─ Increment step counter
    ├─ Check: step >= max_steps → terminate with timeout penalty
    └─ Return (Observation, Reward, done, info)
```

### 5.3 Retrieval Engine

The retrieval engine is a standalone module (`env/retrieval.py`) so it can be
tested and swapped independently.

```python
class RetrievalEngine:
    def __init__(self, messages: List[Message]):
        self.corpus = messages
        self.bm25_index = self._build_bm25_index()

    def search(
        self,
        query: str,
        person_id: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        project_id: Optional[str] = None,
        channel: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Message]:
        """
        1. BM25 keyword search on message body + subject
        2. Apply hard filters (person, date, project, channel)
        3. Return top_k by BM25 score
        """

    def trace_thread(
        self,
        thread_id: str,
        direction: Literal["forward", "backward", "both"] = "both",
    ) -> List[Message]:
        """
        Return all messages in the thread, sorted by timestamp.
        direction controls whether to return ancestors, descendants, or all.
        """
```

**BM25 implementation:** Use `rank_bm25` (pure Python, no heavy deps).

**Relevance scoring against ground truth** (used for intermediate rewards):

Each message in the ground truth manifest has a `relevance_weight` per task_id
(0.0 = irrelevant, 1.0 = critical). When the agent retrieves a message, the
intermediate reward uses this weight:

```python
step_relevance_score = mean(message.relevance_weights[task_id] for message in retrieved)
intermediate_reward = step_relevance_score * 0.05   # max +0.05 per retrieval step
```

### 5.4 Initial Seed Message Selection

The 20 messages shown at episode start are **not random**. They are stratified to:
- Include at least 3 messages from each channel type
- Include at least 1 message referencing each project
- Exclude any message that directly reveals a seeded fact (to avoid trivial tasks)
- Be drawn from the first 30 sim days (so the agent must search for recent context)

### 5.5 Task-Specific Query Strings

```python
TASK_QUERIES = {
    "decision_archaeology": (
        "Our Q2 roadmap currently excludes the OAuth migration. "
        "Trace back through the communication history to find the original decision "
        "that caused this, all intermediate steps, and who is accountable."
    ),
    "commitment_detection": (
        "Review all communications from the past 60 days and identify every implied "
        "commitment that was never followed up on. Rank each by organizational risk."
    ),
    "knowledge_recovery": (
        "Jordan Lee (Staff Engineer) is leaving next week. Draft a complete knowledge "
        "transfer document covering all critical systems, key decisions, collaborators, "
        "and open items they were tracking."
    ),
}
```

---

## 6. Phase 4 — Three Graders

**Estimated effort: 2.5 hrs**

All graders live in `graders/`. Each grader takes the agent's final `submit` payload
and the hidden ground truth, and returns a `GraderResult`.

```python
@dataclass
class GraderResult:
    total_score: float                    # 0.0 – 1.0
    component_scores: Dict[str, float]
    penalties: float
    explanation: str
```

### 6.1 Grader 1: Decision Archaeology

**File:** `graders/decision_archaeology.py`

**Input (from agent submit):**
```json
{
  "root_decision": "string",
  "decision_chain": ["msg_id_1", "msg_id_2", ..., "msg_id_N"],
  "accountable_person": "person_id",
  "decision_text": "string"
}
```

**Scoring:**

| Component | Score | Logic |
|-----------|-------|-------|
| Root decision found | 0.40 | Check if agent's root message_id matches ground truth root |
| Each correct hop (up to 6) | +0.075 each = 0.45 max | For each hop in agent chain, check against ground truth hop list |
| Correct accountability assignment | +0.15 | Exact match on accountable_person_id |
| **Penalty: wrong person blamed** | −0.10 | If accountable_person_id is present but wrong |

**Max score: 1.00**

**Edge cases to handle:**
- Agent submits more hops than ground truth (partial credit for correct subset)
- Agent submits correct chain in wrong order (50% credit)
- Root message correctly identified but chain is empty (0.40 + 0.15 if person correct)

### 6.2 Grader 2: Silent Commitment Detection

**File:** `graders/commitment_detection.py`

**Input (from agent submit):**
```json
{
  "dropped_commitments": [
    {
      "source_message_id": "string",
      "committer": "person_id",
      "commitment_text": "string",
      "risk_level": "low|medium|high|critical",
      "resolution_plan": "string"
    }
  ]
}
```

**Scoring:**

| Component | Weight | Logic |
|-----------|--------|-------|
| Recall of dropped commitments | 0.40 | `TP / 15` (15 = total dropped) |
| Precision (no false positives) | 0.30 | `TP / (TP + FP)` |
| Risk ranking correlation | 0.20 | Spearman ρ between agent risk labels and ground truth risk levels |
| Resolution plan quality | 0.10 | Rule-based: does plan mention committer + project + timeline? |
| **Penalty: resolved flagged as dropped** | −0.05 each | If agent flags a resolved commitment as dropped |

**Max score: 1.00**

**Matching logic:** A TP is counted when:
- `source_message_id` matches a ground truth dropped commitment **OR**
- `committer` + semantic similarity of `commitment_text` > 0.85 (use `sentence-transformers`)

**Risk ranking correlation details:**

```python
# Convert categorical to ordinal
risk_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}

# Agent's submitted risk levels for correctly-identified commitments
agent_ranks = [risk_map[c.risk_level] for c in matched_agent_commitments]
ground_truth_ranks = [risk_map[gt.risk_level] for gt in matched_gt_commitments]

from scipy.stats import spearmanr
rho, _ = spearmanr(agent_ranks, ground_truth_ranks)
risk_score = max(0, rho)  # Clamp to 0 if negative correlation
```

### 6.3 Grader 3: Knowledge Amnesia Recovery

**File:** `graders/knowledge_recovery.py`

**Input (from agent submit):**
```json
{
  "artifact": {
    "systems": ["string"],
    "decisions": [{"decision_text": "string", "owner": "person_id"}],
    "collaborators": ["person_id"],
    "timeline": [{"day": int, "event": "string"}],
    "open_items": ["string"],
    "freeform_notes": "string"
  }
}
```

**Scoring:**

| Component | Weight | Method |
|-----------|--------|--------|
| System coverage | 0.25 | `len(matched_systems) / len(ground_truth_systems)` |
| Decision ownership accuracy | 0.25 | F1 over (decision_text, owner) pairs — fuzzy text match + exact person match |
| Relationship map completeness | 0.20 | `len(matched_collaborators) / len(ground_truth_collaborators)` |
| Temporal accuracy | 0.15 | For each timeline event, check day ± 5 days tolerance |
| Actionability score | 0.15 | **LLM-as-judge** (see below) |
| **Penalty: confidently wrong facts** | −0.10 each | See fact-checking logic below |

**Max score: 1.00**

**LLM-as-judge prompt for actionability:**

```
You are evaluating a knowledge transfer document written for an engineer
who is about to take over critical systems from a departing colleague.

Rate the document on a scale from 0.0 to 1.0 on the following criterion:

CRITERION: Actionability
Could a competent engineer, new to this codebase, use this document to
independently maintain the described systems within their first two weeks?

Consider:
- Are systems described with enough detail to know what they do?
- Are key contacts identified for each system?
- Are ongoing tasks and open issues clearly stated?
- Are past decisions explained with enough context to avoid re-litigating them?

Return ONLY a JSON object: {"score": <float 0.0-1.0>, "rationale": "<1 sentence>"}

DOCUMENT:
{agent_artifact}
```

**Confidently wrong fact detection:**

A fact is "confidently wrong" if:
- A system is mentioned that the departing employee has no involvement with
  (cross-checked against `project_members` and message `sender` fields)
- A decision is attributed to the departing employee that ground truth assigns
  to a different person (exact accountability mismatch)
- A collaborator is named who has no messages exchanged with the departing employee
  in the corpus

---

## 7. Phase 5 — Reward Shaping

**Estimated effort: 1 hr**

### 7.1 Intermediate Reward Table

```python
# env/reward.py

def compute_step_reward(
    action: Action,
    retrieved_messages: List[Message],
    ground_truth: GroundTruth,
    agent_state: AgentState,
    task_id: str,
) -> tuple[float, float, str]:
    """Returns (step_score, penalty, feedback_str)"""

    score = 0.0
    penalty = 0.0
    feedback_parts = []

    if action.action_type == "retrieve_messages":
        relevance_scores = [
            ground_truth.relevance_weights[task_id].get(m.message_id, 0.0)
            for m in retrieved_messages
        ]
        avg_relevance = mean(relevance_scores) if relevance_scores else 0.0
        score = avg_relevance * 0.05   # max +0.05

        # Loop penalty: identical query string already used
        if action.parameters["query"] in agent_state.previous_queries:
            penalty += 0.02
            feedback_parts.append("Repeated retrieval query detected.")

        agent_state.previous_queries.add(action.parameters["query"])

    elif action.action_type == "tag_decision":
        # Check if tagged message_id is in ground truth decision messages
        if action.parameters["message_id"] in ground_truth.all_decision_message_ids:
            score = 0.05
            feedback_parts.append("Correct decision message tagged.")
        else:
            penalty = 0.01
            feedback_parts.append("Tagged message is not a ground truth decision point.")

    elif action.action_type == "tag_commitment":
        if action.parameters["message_id"] in ground_truth.commitment_source_ids:
            score = 0.05
            feedback_parts.append("Valid commitment source message tagged.")
        else:
            penalty = 0.01

    elif action.action_type == "link_cause_effect":
        pair = (action.parameters["cause_message_id"], action.parameters["effect_message_id"])
        if pair in ground_truth.valid_causal_pairs:
            score = 0.03
            feedback_parts.append("Valid causal link established.")

    elif action.action_type == "draft_artifact":
        # Small reward for writing artifact content (encourages progress)
        score = 0.01

    return score, penalty, " ".join(feedback_parts)
```

### 7.2 Timeout and Terminal Penalties

```python
# Applied in env.step() before returning

if step >= max_steps and action.action_type != "submit":
    penalty += 0.10
    done = True
    feedback = "Episode terminated: step limit reached without submitting."

# On submit: terminal reward is grader_score * task_weight
TASK_WEIGHTS = {
    "decision_archaeology": 1.0,
    "commitment_detection": 1.0,
    "knowledge_recovery": 1.0,
}
terminal_reward = grader_result.total_score * TASK_WEIGHTS[task_id]
```

### 7.3 Cumulative Score Tracking

```python
# Cumulative score is tracked in AgentState
agent_state.cumulative_score += (step_score - penalty)

# Terminal score replaces cumulative with a weighted combination:
final_score = (
    0.3 * agent_state.cumulative_score +   # intermediate rewards
    0.7 * terminal_reward                   # grader result
)
```

The 0.3/0.7 split ensures terminal grader quality dominates, but intermediate
retrieval quality still matters for training signal.

---

## 8. Phase 6 — Baseline Inference Script

**Estimated effort: 1.5 hrs**

### 8.1 System Prompt

```python
SYSTEM_PROMPT = """
You are an organizational memory analyst. You have access to a company's
communication history spanning 60 days across email, Slack, meeting notes,
and document comments.

Your job is to investigate the given query by retrieving relevant messages,
building up context incrementally, and ultimately submitting a structured answer.

## Available Actions
You must output exactly one action per turn as valid JSON matching this schema:
{action_schema}

## Action Guide
- retrieve_messages: Use to search by keyword, person, date, project, or channel
- trace_thread: Follow a conversation thread forward or backward
- tag_decision: Mark a message as a decision point
- tag_commitment: Mark a message as containing an implied commitment
- link_cause_effect: Connect two messages causally
- draft_artifact: Write a section of your answer document
- submit: Finalize and submit your answer (triggers scoring)

## Strategy
1. Start broad — retrieve messages to understand the landscape
2. Narrow down — use filters and thread tracing to find key moments
3. Build incrementally — tag decisions and commitments as you find them
4. Draft as you go — don't leave artifact writing to the last step
5. Submit only when confident — you have limited steps

Always include your reasoning in the 'reasoning' field.
Think step by step. Be specific in retrieval queries.
"""
```

### 8.2 Run Script

```python
# baseline/run_baseline.py

import json
import asyncio
from anthropic import Anthropic

client = Anthropic()

async def run_episode(env: OrgMemoryEnv, task_id: str, seed: int) -> dict:
    obs = env.reset(task_id)
    conversation_history = []
    total_reward = 0.0

    while True:
        # Format observation as user message
        user_content = format_observation(obs)
        conversation_history.append({"role": "user", "content": user_content})

        # Call baseline model
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",     # fast + cheap for baseline
            max_tokens=1024,
            system=SYSTEM_PROMPT.format(action_schema=ACTION_SCHEMA_JSON),
            messages=conversation_history,
        )

        raw_action = response.content[0].text
        conversation_history.append({"role": "assistant", "content": raw_action})

        # Parse action
        try:
            action = Action.model_validate_json(raw_action)
        except Exception as e:
            # Invalid action: log and apply small penalty
            obs, reward, done, info = env.step(Action(
                action_type="retrieve_messages",
                parameters={"query": "project update"},
                reasoning="Fallback due to parse error"
            ))
            continue

        obs, reward, done, info = env.step(action)
        total_reward += reward.step_score - reward.penalty

        if done:
            return {
                "task_id": task_id,
                "seed": seed,
                "total_reward": total_reward,
                "final_grader_score": info.get("grader_result", {}).get("total_score"),
                "steps_used": obs.current_step,
                "component_scores": info.get("grader_result", {}).get("component_scores", {}),
            }


async def main():
    results = []
    for task_id in ["decision_archaeology", "commitment_detection", "knowledge_recovery"]:
        for seed in [42, 123, 999]:
            env = OrgMemoryEnv(seed=seed)
            result = await run_episode(env, task_id, seed)
            results.append(result)
            print(f"Task: {task_id} | Seed: {seed} | Score: {result['final_grader_score']:.3f}")

    # Aggregate
    import pandas as pd
    df = pd.DataFrame(results)
    summary = df.groupby("task_id")["final_grader_score"].agg(["mean", "std"])
    print("\n--- Baseline Results ---")
    print(summary.to_string())
    df.to_csv("baseline/results.csv", index=False)
```

### 8.3 Expected Output

```
--- Baseline Results ---
                        mean    std
task_id
decision_archaeology   0.51  0.047
commitment_detection   0.41  0.062
knowledge_recovery     0.33  0.058
```

---

## 9. Phase 7 — Docker + HF Deployment

**Estimated effort: 1.5 hrs**

### 9.1 FastAPI App

```python
# app.py

from fastapi import FastAPI, HTTPException
from env.models import Action, Observation, Reward
from env.org_memory_env import OrgMemoryEnv

app = FastAPI(title="OrgMemory-Env", version="1.0.0")
envs: dict[str, OrgMemoryEnv] = {}   # session_id → env instance

@app.post("/reset")
def reset(task_id: str, seed: int = 42, session_id: str = "default") -> Observation:
    env = OrgMemoryEnv(seed=seed)
    envs[session_id] = env
    return env.reset(task_id)

@app.post("/step")
def step(action: Action, session_id: str = "default") -> dict:
    if session_id not in envs:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")
    obs, reward, done, info = envs[session_id].step(action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}

@app.get("/state")
def state(session_id: str = "default") -> dict:
    if session_id not in envs:
        raise HTTPException(status_code=404, detail="Session not found.")
    return envs[session_id].state().model_dump()

@app.get("/tasks")
def tasks() -> list:
    return [
        {"id": "decision_archaeology", "difficulty": "medium", "max_steps": 20},
        {"id": "commitment_detection", "difficulty": "hard",   "max_steps": 35},
        {"id": "knowledge_recovery",   "difficulty": "expert", "max_steps": 50},
    ]

@app.get("/validate")
def validate() -> dict:
    # Check openenv.yaml compliance
    return {"status": "compliant", "spec_version": "1.0.0"}
```

### 9.2 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY . .

# Pre-generate corpus at build time (avoids cold-start latency)
RUN python scripts/generate_corpus.py --seed 42

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]
```

### 9.3 requirements.txt

```
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.1
rank-bm25==0.2.2
sentence-transformers==3.0.0
scipy==1.13.0
numpy==1.26.4
pandas==2.2.2
openai==1.30.0           # for grader LLM-as-judge calls
anthropic==0.27.0        # for baseline script
python-dotenv==1.0.1
pytest==8.2.0
httpx==0.27.0            # for test client
```

### 9.4 HuggingFace Space Configuration

```yaml
# README.md (HF Spaces header)
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
---
```

---

## 10. File & Directory Structure

```
org-memory-env/
│
├── data/                           # Generated at build/run time
│   ├── messages.json
│   ├── personas.json
│   ├── org_graph.json
│   ├── project_states.json
│   └── ground_truth/               # NEVER exposed to agent
│       ├── decisions.json
│       ├── commitments.json
│       └── departing_profile.json
│
├── env/
│   ├── __init__.py
│   ├── models.py                   # All Pydantic models
│   ├── org_memory_env.py           # Main env class
│   ├── retrieval.py                # BM25 + thread-trace engine
│   ├── reward.py                   # Intermediate reward computation
│   └── state.py                    # AgentState + FullState
│
├── generator/
│   ├── __init__.py
│   ├── personas.py                 # Persona definitions + generation
│   ├── messages.py                 # Message generation pipeline
│   ├── decisions.py                # Decision chain seeding
│   ├── commitments.py              # Commitment seeding
│   └── ground_truth.py             # Manifest writer
│
├── graders/
│   ├── __init__.py
│   ├── base.py                     # GraderResult dataclass
│   ├── decision_archaeology.py
│   ├── commitment_detection.py
│   └── knowledge_recovery.py
│
├── baseline/
│   ├── run_baseline.py
│   └── results.csv                 # Written after baseline run
│
├── scripts/
│   └── generate_corpus.py          # CLI entrypoint for data gen
│
├── tests/
│   ├── test_generator.py
│   ├── test_env.py
│   ├── test_retrieval.py
│   └── test_graders.py
│
├── app.py                          # FastAPI app
├── openenv.yaml                    # OpenEnv spec
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 11. Dependencies & Tooling

| Library | Version | Purpose |
|---------|---------|---------|
| `pydantic` | v2.7+ | Data validation + serialization |
| `rank-bm25` | 0.2.2 | BM25 retrieval over message corpus |
| `sentence-transformers` | 3.0.0 | Semantic similarity for commitment matching |
| `scipy` | 1.13.0 | Spearman ρ for risk ranking |
| `fastapi` + `uvicorn` | latest | API server |
| `openai` or `anthropic` | latest | LLM-as-judge in Grader 3, baseline inference |
| `pytest` + `httpx` | latest | Testing |
| `pandas` | 2.2+ | Baseline result aggregation |
| `python-dotenv` | 1.0.1 | API key management |

**Python version:** 3.11 (required for `tomllib` stdlib + best `pydantic v2` support)

---

## 12. Testing Strategy

### Unit Tests

| Test File | What It Covers |
|-----------|----------------|
| `test_generator.py` | All 15 commitments have sources; all 25 resolved have resolutions; 8-hop chain traceable |
| `test_env.py` | `reset()` returns valid Observation; `step()` increments counter; timeout fires correctly |
| `test_retrieval.py` | BM25 returns ≤10 results; thread-trace returns correct ordering; filters work |
| `test_graders.py` | Perfect-input scores 1.0; empty-input scores 0.0; penalties apply correctly |

### Integration Tests

```python
# tests/test_integration.py

def test_full_episode_runs_to_completion():
    env = OrgMemoryEnv(seed=42)
    obs = env.reset("decision_archaeology")
    done = False
    steps = 0
    while not done and steps < 25:
        action = Action(
            action_type="retrieve_messages",
            parameters={"query": "decision"},
            reasoning="test"
        )
        obs, reward, done, info = env.step(action)
        steps += 1
    assert done or steps == 25

def test_submit_triggers_grader():
    env = OrgMemoryEnv(seed=42)
    env.reset("decision_archaeology")
    obs, reward, done, info = env.step(Action(
        action_type="submit",
        parameters={"answer": {"root_decision": "msg_001", "decision_chain": [], "accountable_person": "P001"}},
        reasoning="test submit"
    ))
    assert done is True
    assert "grader_result" in info
    assert 0.0 <= info["grader_result"]["total_score"] <= 1.0
```

### Smoke Test for API

```bash
# After docker run -p 7860:7860 org-memory-env

curl -X POST http://localhost:7860/reset?task_id=decision_archaeology
curl -X GET  http://localhost:7860/tasks
curl -X GET  http://localhost:7860/validate
```

---

## 13. Timeline & Effort Estimates

| Phase | Description | Hours | Status |
|-------|-------------|-------|--------|
| 1 | Synthetic Org Data Generator | 2.0 | 🟡 In progress |
| 2 | Pydantic Models + OpenEnv Spec | 1.0 | ⬜ Not started |
| 3 | Core Environment Class | 3.0 | ⬜ Not started |
| 4 | Three Graders | 2.5 | ⬜ Not started |
| 5 | Reward Shaping | 1.0 | ⬜ Not started |
| 6 | Baseline Inference Script | 1.5 | ⬜ Not started |
| 7 | Docker + HF Deployment | 1.5 | ⬜ Not started |
| — | README + Polish | 1.0 | ⬜ Not started |
| **Total** | | **13.5** | |

### Recommended Build Order

```
Phase 1 (finish) → Phase 2 → Phase 3 (env skeleton) → Phase 3 (retrieval)
→ Phase 4 (Grader 1) → Phase 5 → Phase 6 (baseline vs Grader 1 only)
→ Phase 4 (Grader 2 + 3) → Phase 6 (full baseline) → Phase 7
```

Run the baseline against each grader as it's completed — this gives early
feedback on whether reward shaping is working before all three tasks are live.

---

## 14. Open Questions & Risk Register

| # | Risk / Question | Likelihood | Impact | Mitigation |
|---|----------------|------------|--------|------------|
| 1 | LLM-generated messages are too uniform in style, making retrieval trivial | Medium | High | Use stricter style templates + post-process to inject style variations |
| 2 | BM25 returns irrelevant results for commitment detection (keyword mismatch on implicit language) | High | Medium | Augment with a lightweight embedding search (`sentence-transformers`) |
| 3 | Grader 3 LLM-as-judge is expensive at scale (9 baseline episodes × grader call) | Low | Low | Cache judge calls by content hash; judge call is only on final submit |
| 4 | `sentence-transformers` cold-start adds 10–15s to first API call in Docker | Medium | Low | Pre-load model at container startup; use distilbert-based model |
| 5 | OpenAI API rate limits during baseline run (9 episodes, up to 35 steps each) | Medium | Medium | Use `asyncio` + exponential backoff; consider batch API |
| 6 | Ground truth manifest accidentally leaked to agent via Observation | Low | Critical | Add `assert "ground_truth" not in obs.model_dump()` in `env.reset()` |
| 7 | Commitment matching logic gives false TPs on common phrases ("I'll send") | Medium | Medium | Require both message_id match OR (person + semantic similarity > 0.85) |
| 8 | Docker image too large due to `sentence-transformers` model weights | Medium | Low | Use `paraphrase-MiniLM-L6-v2` (~80MB); add `.dockerignore` for dev files |

---

*Last updated: Phase 1 in progress. Next action: finalize communication style templates and post-processing validation loop.*
