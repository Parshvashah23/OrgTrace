"""
OrgMemory-Env: Pydantic Models
All core data models for the RL environment.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional, Dict, List, Any
from datetime import datetime


# Type aliases
PersonID = str
ProjectID = str
MessageID = str
ThreadID = str


class Message(BaseModel):
    """A single communication message in the corpus."""
    message_id: str
    timestamp: datetime
    sender_id: PersonID
    sender_name: str
    sender_email: str
    recipient_ids: List[PersonID]
    recipient_names: List[str]
    channel: str  # "email", "slack", "slack:#channel-name", "meeting_notes", "doc_comment"
    subject: Optional[str] = None
    body: str
    thread_id: Optional[ThreadID] = None
    project_tag: Optional[ProjectID] = None
    day: int  # Simulation day (1-60)
    index: Optional[int] = None  # Sequential index in corpus


class Persona(BaseModel):
    """A synthetic employee persona."""
    id: PersonID
    name: str
    email: str
    team: str
    role: str
    seniority: int  # 1=junior, 2=mid, 3=senior, 4=lead/manager, 5=exec
    comm_style: str
    expertise: List[str]
    reports_to: Optional[PersonID] = None
    direct_reports: List[PersonID] = Field(default_factory=list)
    slack_handle: str = ""
    departure_day: Optional[int] = None


class OrgGraph(BaseModel):
    """Organizational relationship graph."""
    people: Dict[PersonID, Dict[str, Any]]  # Persona dicts
    teams: Dict[str, List[PersonID]]
    reports_to: Dict[PersonID, Optional[PersonID]]
    works_with: Dict[PersonID, List[PersonID]]
    project_members: Dict[ProjectID, List[PersonID]]


class ProjectState(BaseModel):
    """Current state of a project."""
    project_id: ProjectID
    name: str
    description: str
    status: Literal["on_track", "at_risk", "blocked", "completed"]
    owner_id: PersonID
    lead_eng_id: PersonID
    last_update_day: int
    open_items: List[str] = Field(default_factory=list)


# Action types and parameter specs
ACTION_PARAM_SPECS = {
    "retrieve_messages": {
        "required": ["query"],
        "optional": ["person_id", "date_from", "date_to", "project_id", "channel", "top_k"],
    },
    "trace_thread": {
        "required": ["thread_id"],
        "optional": ["direction"],  # "forward", "backward", "both"
    },
    "tag_decision": {
        "required": ["message_id", "decision_text"],
        "optional": ["accountable_person_id"],
    },
    "tag_commitment": {
        "required": ["message_id", "committer_id", "commitment_text"],
        "optional": ["risk_level"],
    },
    "link_cause_effect": {
        "required": ["cause_message_id", "effect_message_id"],
        "optional": ["explanation"],
    },
    "draft_artifact": {
        "required": ["section", "content"],
        "optional": ["artifact_type"],
    },
    "submit": {
        "required": ["answer"],
        "optional": [],
    },
}


class Action(BaseModel):
    """An agent action in the environment."""
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
    reasoning: str = ""

    @field_validator("parameters")
    @classmethod
    def validate_parameters(cls, v, info):
        action_type = info.data.get("action_type")
        if action_type and action_type in ACTION_PARAM_SPECS:
            spec = ACTION_PARAM_SPECS[action_type]
            for req in spec["required"]:
                if req not in v:
                    raise ValueError(f"Missing required parameter '{req}' for action '{action_type}'")
        return v


class TaggedDecision(BaseModel):
    """A decision tagged by the agent."""
    message_id: MessageID
    decision_text: str
    accountable_person_id: Optional[PersonID] = None
    tagged_at_step: int


class TaggedCommitment(BaseModel):
    """A commitment tagged by the agent."""
    message_id: MessageID
    committer_id: PersonID
    commitment_text: str
    risk_level: Optional[Literal["low", "medium", "high", "critical"]] = None
    tagged_at_step: int


class CausalLink(BaseModel):
    """A cause-effect link identified by the agent."""
    cause_message_id: MessageID
    effect_message_id: MessageID
    explanation: Optional[str] = None
    linked_at_step: int


class ArtifactSection(BaseModel):
    """A section of a drafted artifact."""
    section: str
    content: str
    artifact_type: Optional[str] = None
    drafted_at_step: int


class AgentState(BaseModel):
    """Internal state tracking agent actions and discoveries."""
    tagged_decisions: List[TaggedDecision] = Field(default_factory=list)
    tagged_commitments: List[TaggedCommitment] = Field(default_factory=list)
    causal_links: List[CausalLink] = Field(default_factory=list)
    artifact_sections: List[ArtifactSection] = Field(default_factory=list)
    retrieved_message_ids: List[MessageID] = Field(default_factory=list)


class Observation(BaseModel):
    """The observation returned to the agent at each step."""
    task_id: str
    current_step: int
    max_steps: int
    query: str  # Task-specific query/prompt
    visible_messages: List[Message]
    org_graph: OrgGraph
    project_states: List[ProjectState]
    retrieved_context: List[Message]
    action_history: List[Action]


class Reward(BaseModel):
    """Reward information returned at each step."""
    step_score: float
    cumulative_score: float
    component_scores: Dict[str, float] = Field(default_factory=dict)
    penalty: float = 0.0
    done: bool = False
    feedback: str = ""


class FullState(BaseModel):
    """Complete internal state (for debugging/logging)."""
    task_id: str
    current_step: int
    max_steps: int
    agent_state: AgentState
    observation: Observation
    cumulative_reward: float
    done: bool
    ground_truth_loaded: bool = False


# Grading models

class GraderResult(BaseModel):
    """Result from a grader evaluation."""
    total_score: float  # 0.0 - 1.0
    component_scores: Dict[str, float]
    penalties: float = 0.0
    explanation: str = ""


# Task 1: Decision Archaeology submission
class DecisionArchaeologySubmission(BaseModel):
    """Agent submission for Task 1."""
    root_decision: str  # message_id of the root
    decision_chain: List[MessageID]  # ordered chain of message IDs
    accountable_person: PersonID
    decision_text: str


# Task 2: Commitment Detection submission
class DroppedCommitmentEntry(BaseModel):
    """A single dropped commitment entry."""
    source_message_id: MessageID
    committer: PersonID
    commitment_text: str
    risk_level: Literal["low", "medium", "high", "critical"]
    resolution_plan: str = ""


class CommitmentDetectionSubmission(BaseModel):
    """Agent submission for Task 2."""
    dropped_commitments: List[DroppedCommitmentEntry]


# Task 3: Knowledge Recovery submission
class DecisionEntry(BaseModel):
    """A decision entry for knowledge transfer."""
    decision_text: str
    owner: PersonID


class TimelineEntry(BaseModel):
    """A timeline entry for knowledge transfer."""
    day: int
    event: str


class KnowledgeArtifact(BaseModel):
    """Knowledge transfer artifact content."""
    systems: List[str]
    decisions: List[DecisionEntry]
    collaborators: List[PersonID]
    timeline: List[TimelineEntry]
    open_items: List[str]
    freeform_notes: str = ""


class KnowledgeRecoverySubmission(BaseModel):
    """Agent submission for Task 3."""
    artifact: KnowledgeArtifact
