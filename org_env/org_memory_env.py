"""
OrgMemory-Env: Core Environment Class
Main RL environment for organizational communication debt resolution.
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from .models import (
    Message, OrgGraph, ProjectState, Action, Observation, Reward,
    FullState, AgentState, TaggedDecision, TaggedCommitment,
    CausalLink, ArtifactSection, GraderResult,
)
from .retrieval import RetrievalEngine
from .reward import (
    RewardState, build_relevance_weights, compute_step_reward,
    compute_timeout_penalty, compute_terminal_reward, REWARD_CONFIG,
    build_valid_ids, ValidIds,
)


# Task configurations
TASK_CONFIG = {
    "decision_archaeology": {
        "max_steps": 20,
        "query": (
            "Our Q2 roadmap currently excludes the OAuth migration. "
            "Trace back through the communication history to find the original decision "
            "that caused this, all intermediate steps, and who is accountable."
        ),
    },
    "commitment_detection": {
        "max_steps": 35,
        "query": (
            "Review all communications from the past 60 days and identify every implied "
            "commitment that was never followed up on. Rank each by organizational risk."
        ),
    },
    "knowledge_recovery": {
        "max_steps": 50,
        "query": (
            "Sofia Reyes (Sr Engineer) is leaving next week. Draft a complete knowledge "
            "transfer document covering all critical systems, key decisions, collaborators, "
            "and open items they were tracking."
        ),
    },
}


class OrgMemoryEnv:
    """
    RL Environment for Organizational Communication Debt Resolution.

    This environment simulates the organizational memory problem where agents must:
    - Trace complex decision chains (Task 1: Decision Archaeology)
    - Surface silently dropped commitments (Task 2: Commitment Detection)
    - Reconstruct institutional knowledge (Task 3: Knowledge Recovery)
    """

    def __init__(self, data_dir: str = "data/generated/", seed: int = 42):
        """
        Initialize the environment.

        Args:
            data_dir: Directory containing corpus.json and ground_truth.json
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.seed = seed
        self.rng = random.Random(seed)

        # Load data
        self._load_data()

        # Initialize retrieval engine
        self.retrieval = RetrievalEngine(self.corpus)

        # Build relevance weights for reward computation
        self.relevance_weights = build_relevance_weights(self.corpus, self.ground_truth)

        # Build valid IDs for hallucination detection
        self.valid_ids = build_valid_ids(self.corpus, self.personas)

        # Episode state (initialized in reset)
        self.task_id: Optional[str] = None
        self.current_step: int = 0
        self.max_steps: int = 0
        self.done: bool = True
        self.cumulative_reward: float = 0.0

        self.agent_state: Optional[AgentState] = None
        self.reward_state: Optional[RewardState] = None
        self.visible_messages: List[Dict] = []
        self.retrieved_context: List[Dict] = []
        self.action_history: List[Action] = []

    def _load_data(self):
        """Load corpus, ground truth, and build org graph."""
        # Load corpus
        corpus_path = self.data_dir / "corpus.json"
        with open(corpus_path, "r") as f:
            self.corpus = json.load(f)

        # Load ground truth (for grading)
        gt_path = self.data_dir / "ground_truth.json"
        with open(gt_path, "r") as f:
            self.ground_truth = json.load(f)

        # Load personas from data module
        import sys
        root_dir = str(self.data_dir.parent.parent)
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        
        try:
            from data.personas import PERSONAS, PERSONA_BY_ID
            self.personas = PERSONAS
            self.persona_by_id = {p.id: p for p in PERSONAS}
        except ImportError as e:
            raise ImportError(f"Failed to import personas from data.personas. Root dir: {root_dir}. Error: {e}")

        # Load projects
        try:
            from data.seeds import PROJECTS, PROJECT_BY_ID
            self.projects = PROJECTS
            self.project_by_id = {p.id: p for p in PROJECTS}
        except ImportError as e:
            raise ImportError(f"Failed to import seeds from data.seeds. Root dir: {root_dir}. Error: {e}")

        # Build org graph
        self._build_org_graph()

        # Build project states
        self._build_project_states()

    def _build_org_graph(self) -> OrgGraph:
        """Build the organizational graph from personas."""
        people = {}
        teams: Dict[str, List[str]] = {}
        reports_to: Dict[str, Optional[str]] = {}
        works_with: Dict[str, List[str]] = {}

        for p in self.personas:
            people[p.id] = {
                "id": p.id,
                "name": p.name,
                "email": p.email,
                "team": p.team.value if hasattr(p.team, 'value') else str(p.team),
                "role": p.role,
                "seniority": p.seniority,
                "expertise": p.expertise,
            }

            team_name = p.team.value if hasattr(p.team, 'value') else str(p.team)
            if team_name not in teams:
                teams[team_name] = []
            teams[team_name].append(p.id)

            reports_to[p.id] = p.reports_to

        # Build works_with from projects
        project_members: Dict[str, List[str]] = {}
        for proj in self.projects:
            project_members[proj.id] = proj.team_ids
            for pid in proj.team_ids:
                if pid not in works_with:
                    works_with[pid] = []
                for other_pid in proj.team_ids:
                    if other_pid != pid and other_pid not in works_with[pid]:
                        works_with[pid].append(other_pid)

        self.org_graph = OrgGraph(
            people=people,
            teams=teams,
            reports_to=reports_to,
            works_with=works_with,
            project_members=project_members,
        )

        return self.org_graph

    def _build_project_states(self):
        """Build current project state snapshots."""
        self.project_states = []
        for proj in self.projects:
            # Get latest status from status_changes
            latest_status = "on_track"
            latest_day = 0
            for change in proj.status_changes:
                if change["day"] > latest_day:
                    latest_day = change["day"]
                    latest_status = change["status"]

            self.project_states.append(ProjectState(
                project_id=proj.id,
                name=proj.name,
                description=proj.description,
                status=latest_status,
                owner_id=proj.owner_id,
                lead_eng_id=proj.lead_eng_id,
                last_update_day=latest_day,
                open_items=[],
            ))

    def _select_seed_messages(self, n: int = 20) -> List[Dict]:
        """
        Select initial seed messages for the episode.

        Selection criteria:
        - At least 3 messages from each channel type
        - At least 1 message referencing each project
        - Drawn from first 30 sim days
        - Excludes messages that directly reveal seeded facts
        """
        candidates = [m for m in self.corpus if m.get("day", 0) <= 30]

        # Separate by channel
        by_channel: Dict[str, List[Dict]] = {}
        for m in candidates:
            channel = m.get("channel", "").split(":")[0]  # normalize slack channels
            if channel not in by_channel:
                by_channel[channel] = []
            by_channel[channel].append(m)

        selected = []
        selected_ids = set()

        # Ensure at least 3 from each channel type
        for channel, msgs in by_channel.items():
            self.rng.shuffle(msgs)
            for msg in msgs[:3]:
                if msg["message_id"] not in selected_ids:
                    selected.append(msg)
                    selected_ids.add(msg["message_id"])

        # Ensure at least 1 from each project
        for proj in self.projects:
            proj_msgs = [m for m in candidates if m.get("project_tag") == proj.id]
            proj_msgs = [m for m in proj_msgs if m["message_id"] not in selected_ids]
            if proj_msgs:
                msg = self.rng.choice(proj_msgs)
                selected.append(msg)
                selected_ids.add(msg["message_id"])

        # Fill remaining slots
        remaining = [m for m in candidates if m["message_id"] not in selected_ids]
        self.rng.shuffle(remaining)
        while len(selected) < n and remaining:
            selected.append(remaining.pop())

        # Sort by timestamp
        selected.sort(key=lambda m: m.get("timestamp", ""))

        return selected

    def reset(self, task_id: str) -> Observation:
        """
        Initialize a new episode.

        Args:
            task_id: One of "decision_archaeology", "commitment_detection", "knowledge_recovery"

        Returns:
            Initial observation
        """
        if task_id not in TASK_CONFIG:
            raise ValueError(f"Unknown task_id: {task_id}. Must be one of {list(TASK_CONFIG.keys())}")

        self.task_id = task_id
        config = TASK_CONFIG[task_id]

        self.current_step = 0
        self.max_steps = config["max_steps"]
        self.done = False
        self.cumulative_reward = 0.0

        # Reset agent state and reward state
        self.agent_state = AgentState()
        self.reward_state = RewardState()

        # Select seed messages
        self.visible_messages = self._select_seed_messages(n=20)
        self.retrieved_context = []
        self.action_history = []

        return self._make_observation(config["query"])

    def _make_observation(self, query: str) -> Observation:
        """Create an observation object."""
        # Convert message dicts to Message models
        visible = [self._dict_to_message(m) for m in self.visible_messages]
        context = [self._dict_to_message(m) for m in self.retrieved_context]

        return Observation(
            task_id=self.task_id,
            current_step=self.current_step,
            max_steps=self.max_steps,
            query=query,
            visible_messages=visible,
            org_graph=self.org_graph,
            project_states=self.project_states,
            retrieved_context=context,
            action_history=self.action_history.copy(),
        )

    def _dict_to_message(self, d: Dict) -> Message:
        """Convert a message dict to a Message model."""
        return Message(
            message_id=d["message_id"],
            timestamp=datetime.fromisoformat(d["timestamp"]) if isinstance(d["timestamp"], str) else d["timestamp"],
            sender_id=d["sender_id"],
            sender_name=d["sender_name"],
            sender_email=d["sender_email"],
            recipient_ids=d["recipient_ids"],
            recipient_names=d["recipient_names"],
            channel=d["channel"],
            subject=d.get("subject"),
            body=d["body"],
            thread_id=d.get("thread_id"),
            project_tag=d.get("project_tag"),
            day=d.get("day", 0),
            index=d.get("index"),
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        """
        Execute one agent action.

        Args:
            action: The action to execute

        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self.current_step += 1
        self.action_history.append(action)

        # Dispatch action and collect retrieved messages for reward computation
        step_score = 0.0
        penalty = 0.0
        feedback = ""
        info = {}
        retrieved_messages: List[Dict] = []

        try:
            if action.action_type == "retrieve_messages":
                result, retrieved_messages = self._handle_retrieve(action.parameters)
                info["retrieved_count"] = result.get("count", 0)

            elif action.action_type == "trace_thread":
                result, retrieved_messages = self._handle_trace_thread(action.parameters)
                info["thread_count"] = result.get("count", 0)

            elif action.action_type == "tag_decision":
                result = self._handle_tag_decision(action.parameters)

            elif action.action_type == "tag_commitment":
                result = self._handle_tag_commitment(action.parameters)

            elif action.action_type == "link_cause_effect":
                result = self._handle_link_cause_effect(action.parameters)

            elif action.action_type == "draft_artifact":
                result = self._handle_draft_artifact(action.parameters)

            elif action.action_type == "submit":
                result = self._handle_submit(action.parameters)
                info["grader_result"] = result.get("grader_result")
                self.done = True

                # Terminal reward from grader
                step_score = compute_terminal_reward(result.get("score", 0.0), self.task_id)
                penalty = result.get("penalty", 0.0)
                feedback = result.get("feedback", "")

            else:
                result = {"score": 0.0, "penalty": 0.0, "feedback": f"Unknown action: {action.action_type}"}

            # Compute intermediate reward using reward module (except for submit which uses grader)
            if action.action_type != "submit":
                step_score, penalty, feedback = compute_step_reward(
                    action_type=action.action_type,
                    parameters=action.parameters,
                    retrieved_messages=retrieved_messages,
                    ground_truth=self.ground_truth,
                    reward_state=self.reward_state,
                    task_id=self.task_id,
                    relevance_weights=self.relevance_weights,
                    valid_ids=self.valid_ids,
                )

        except Exception as e:
            penalty = 0.02
            feedback = f"Action error: {str(e)}"
            info["error"] = str(e)

        # Check for timeout
        if self.current_step >= self.max_steps and not self.done:
            self.done = True
            timeout_penalty, timeout_feedback = compute_timeout_penalty()
            penalty += timeout_penalty
            feedback += f" {timeout_feedback}"

        # Compute reward
        self.cumulative_reward += step_score - penalty
        reward = Reward(
            step_score=step_score,
            cumulative_score=self.cumulative_reward,
            penalty=penalty,
            done=self.done,
            feedback=feedback,
        )

        # Make observation
        query = TASK_CONFIG[self.task_id]["query"]
        observation = self._make_observation(query)

        return observation, reward, self.done, info

    def _handle_retrieve(self, params: Dict) -> Tuple[Dict, List[Dict]]:
        """Handle retrieve_messages action."""
        query = params.get("query", "")
        results = self.retrieval.search(
            query=query,
            person_id=params.get("person_id"),
            date_from=params.get("date_from"),
            date_to=params.get("date_to"),
            project_id=params.get("project_id"),
            channel=params.get("channel"),
            top_k=params.get("top_k", 10),
        )

        # Add to retrieved context (avoid duplicates)
        existing_ids = {m["message_id"] for m in self.retrieved_context}
        new_messages = [m for m in results if m["message_id"] not in existing_ids]
        self.retrieved_context.extend(new_messages)

        # Track retrieved message IDs in agent state
        for m in results:
            if m["message_id"] not in self.agent_state.retrieved_message_ids:
                self.agent_state.retrieved_message_ids.append(m["message_id"])

        return {
            "count": len(results),
            "new_count": len(new_messages),
        }, results

    def _handle_trace_thread(self, params: Dict) -> Tuple[Dict, List[Dict]]:
        """Handle trace_thread action."""
        thread_id = params.get("thread_id")
        direction = params.get("direction", "both")

        results = self.retrieval.trace_thread(thread_id, direction)

        # Add to retrieved context
        existing_ids = {m["message_id"] for m in self.retrieved_context}
        new_messages = [m for m in results if m["message_id"] not in existing_ids]
        self.retrieved_context.extend(new_messages)

        # Track retrieved message IDs
        for m in results:
            if m["message_id"] not in self.agent_state.retrieved_message_ids:
                self.agent_state.retrieved_message_ids.append(m["message_id"])

        return {
            "count": len(results),
            "new_count": len(new_messages),
        }, results

    def _handle_tag_decision(self, params: Dict) -> Dict:
        """Handle tag_decision action."""
        tagged = TaggedDecision(
            message_id=params["message_id"],
            decision_text=params["decision_text"],
            accountable_person_id=params.get("accountable_person_id"),
            tagged_at_step=self.current_step,
        )
        self.agent_state.tagged_decisions.append(tagged)
        return {}

    def _handle_tag_commitment(self, params: Dict) -> Dict:
        """Handle tag_commitment action."""
        tagged = TaggedCommitment(
            message_id=params["message_id"],
            committer_id=params["committer_id"],
            commitment_text=params["commitment_text"],
            risk_level=params.get("risk_level"),
            tagged_at_step=self.current_step,
        )
        self.agent_state.tagged_commitments.append(tagged)
        return {}

    def _handle_link_cause_effect(self, params: Dict) -> Dict:
        """Handle link_cause_effect action."""
        link = CausalLink(
            cause_message_id=params["cause_message_id"],
            effect_message_id=params["effect_message_id"],
            explanation=params.get("explanation"),
            linked_at_step=self.current_step,
        )
        self.agent_state.causal_links.append(link)
        return {}

    def _handle_draft_artifact(self, params: Dict) -> Dict:
        """Handle draft_artifact action."""
        section = ArtifactSection(
            section=params["section"],
            content=params["content"],
            artifact_type=params.get("artifact_type"),
            drafted_at_step=self.current_step,
        )
        self.agent_state.artifact_sections.append(section)
        return {}

    def _handle_submit(self, params: Dict) -> Dict:
        """Handle submit action - triggers final grading."""
        answer = params.get("answer", {})

        # Import and run the appropriate grader
        if self.task_id == "decision_archaeology":
            from .graders.decision_archaeology import grade_decision_archaeology
            result = grade_decision_archaeology(answer, self.ground_truth)

        elif self.task_id == "commitment_detection":
            from .graders.commitment_detection import grade_commitment_detection
            result = grade_commitment_detection(answer, self.ground_truth)

        elif self.task_id == "knowledge_recovery":
            from .graders.knowledge_recovery import grade_knowledge_recovery
            result = grade_knowledge_recovery(answer, self.ground_truth, self.corpus)

        else:
            result = GraderResult(
                total_score=0.0,
                component_scores={},
                explanation="Unknown task",
            )

        return {
            "score": result.total_score,
            "penalty": result.penalties,
            "feedback": result.explanation,
            "grader_result": result.model_dump(),
        }

    def state(self) -> FullState:
        """Return complete internal state (for debugging/logging)."""
        query = TASK_CONFIG.get(self.task_id, {}).get("query", "")
        return FullState(
            task_id=self.task_id or "",
            current_step=self.current_step,
            max_steps=self.max_steps,
            agent_state=self.agent_state or AgentState(),
            observation=self._make_observation(query),
            cumulative_reward=self.cumulative_reward,
            done=self.done,
            ground_truth_loaded=True,
        )

    def render(self) -> str:
        """Return a human-readable string representation of the current state."""
        lines = [
            "=" * 60,
            f"OrgMemory-Env | Task: {self.task_id}",
            f"Step: {self.current_step}/{self.max_steps} | Done: {self.done}",
            f"Cumulative Reward: {self.cumulative_reward:.4f}",
            "-" * 60,
            f"Visible Messages: {len(self.visible_messages)}",
            f"Retrieved Context: {len(self.retrieved_context)}",
            f"Actions Taken: {len(self.action_history)}",
            "-" * 60,
            "Agent State:",
            f"  Tagged Decisions: {len(self.agent_state.tagged_decisions) if self.agent_state else 0}",
            f"  Tagged Commitments: {len(self.agent_state.tagged_commitments) if self.agent_state else 0}",
            f"  Causal Links: {len(self.agent_state.causal_links) if self.agent_state else 0}",
            f"  Artifact Sections: {len(self.agent_state.artifact_sections) if self.agent_state else 0}",
        ]

        # Add reward state info
        if self.reward_state:
            lines.extend([
                "-" * 60,
                "Reward State:",
                f"  Previous Queries: {len(self.reward_state.previous_queries)}",
                f"  Retrieved Messages: {len(self.reward_state.retrieved_message_ids)}",
                f"  Tagged Decisions: {len(self.reward_state.tagged_decision_ids)}",
                f"  Tagged Commitments: {len(self.reward_state.tagged_commitment_ids)}",
                f"  Linked Pairs: {len(self.reward_state.linked_pairs)}",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)

    @property
    def available_tasks(self) -> List[str]:
        """Return list of available task IDs."""
        return list(TASK_CONFIG.keys())

    def get_reward_config(self) -> Dict[str, float]:
        """Return the current reward configuration."""
        return REWARD_CONFIG.copy()
