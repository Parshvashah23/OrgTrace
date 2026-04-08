"""
OrgMemory-Env: Reward Shaping Module
Computes intermediate and terminal rewards for agent actions.

Reward Structure:
- Intermediate rewards encourage exploration of relevant information
- Terminal rewards from graders evaluate final submissions
- Penalties discourage unproductive behavior (loops, timeouts)
"""

from typing import Dict, List, Any, Set, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class RewardState:
    """Tracks state needed for reward computation across steps."""
    previous_queries: Set[str] = field(default_factory=set)
    retrieved_message_ids: Set[str] = field(default_factory=set)
    tagged_decision_ids: Set[str] = field(default_factory=set)
    tagged_commitment_ids: Set[str] = field(default_factory=set)
    linked_pairs: Set[Tuple[str, str]] = field(default_factory=set)

    def reset(self):
        """Reset state for new episode."""
        self.previous_queries.clear()
        self.retrieved_message_ids.clear()
        self.tagged_decision_ids.clear()
        self.tagged_commitment_ids.clear()
        self.linked_pairs.clear()


# Task weights for terminal rewards
TASK_WEIGHTS = {
    "decision_archaeology": 1.0,
    "commitment_detection": 1.0,
    "knowledge_recovery": 1.0,
}

# Intermediate reward values
REWARD_CONFIG = {
    # Retrieval rewards
    "retrieve_relevant_max": 0.05,      # Max reward per retrieval step
    "retrieve_duplicate_penalty": 0.01,  # Penalty for duplicate retrieval

    # Loop penalties
    "repeated_query_penalty": 0.02,      # Penalty for identical query

    # Tagging rewards
    "tag_decision_correct": 0.05,        # Reward for correct decision tag
    "tag_decision_wrong": 0.01,          # Penalty for wrong decision tag
    "tag_commitment_correct": 0.05,      # Reward for correct commitment tag
    "tag_commitment_wrong": 0.01,        # Penalty for wrong commitment tag

    # Causal linking rewards
    "link_correct_adjacent": 0.03,       # Reward for correct adjacent link
    "link_correct_chain": 0.02,          # Reward for correct but non-adjacent link
    "link_wrong": 0.005,                 # Small penalty for wrong link

    # Artifact drafting
    "draft_artifact_base": 0.01,         # Base reward for drafting content

    # Timeout penalty
    "timeout_penalty": 0.10,             # Penalty when step limit reached without submit

    # Destructive action penalties (hallucinations)
    "hallucinated_message_id": 0.05,     # Using a message_id that doesn't exist
    "hallucinated_thread_id": 0.05,      # Using a thread_id that doesn't exist
    "hallucinated_person_id": 0.03,      # Using a person_id that doesn't exist
}


@dataclass
class ValidIds:
    """Container for all valid IDs in the corpus."""
    message_ids: Set[str] = field(default_factory=set)
    thread_ids: Set[str] = field(default_factory=set)
    person_ids: Set[str] = field(default_factory=set)


def build_valid_ids(corpus: List[Dict[str, Any]], personas: List[Any] = None) -> ValidIds:
    """
    Build sets of all valid IDs from the corpus.

    Args:
        corpus: List of message dicts
        personas: Optional list of persona objects

    Returns:
        ValidIds containing all valid message_ids, thread_ids, and person_ids
    """
    valid = ValidIds()

    for msg in corpus:
        if msg.get("message_id"):
            valid.message_ids.add(msg["message_id"])
        if msg.get("thread_id"):
            valid.thread_ids.add(msg["thread_id"])
        if msg.get("sender_id"):
            valid.person_ids.add(msg["sender_id"])
        for rid in msg.get("recipient_ids", []):
            valid.person_ids.add(rid)

    # Add persona IDs if provided
    if personas:
        for p in personas:
            if hasattr(p, 'id'):
                valid.person_ids.add(p.id)

    return valid


def check_hallucinations(
    action_type: str,
    parameters: Dict[str, Any],
    valid_ids: ValidIds,
) -> Tuple[float, List[str]]:
    """
    Check for hallucinated (non-existent) IDs in action parameters.

    Args:
        action_type: The type of action
        parameters: Action parameters
        valid_ids: Container of all valid IDs

    Returns:
        Tuple of (total_penalty, list of feedback messages)
    """
    penalty = 0.0
    feedback = []

    # Check message_id parameters
    message_id_params = ["message_id", "cause_message_id", "effect_message_id", "source_message_id"]
    for param in message_id_params:
        if param in parameters:
            msg_id = parameters[param]
            if msg_id and msg_id not in valid_ids.message_ids:
                penalty += REWARD_CONFIG["hallucinated_message_id"]
                feedback.append(f"Hallucinated {param}: '{msg_id}' does not exist.")

    # Check thread_id parameter
    if "thread_id" in parameters:
        thread_id = parameters["thread_id"]
        if thread_id and thread_id not in valid_ids.thread_ids:
            penalty += REWARD_CONFIG["hallucinated_thread_id"]
            feedback.append(f"Hallucinated thread_id: '{thread_id}' does not exist.")

    # Check person_id parameters
    person_id_params = ["person_id", "committer_id", "accountable_person_id"]
    for param in person_id_params:
        if param in parameters:
            person_id = parameters[param]
            if person_id and person_id not in valid_ids.person_ids:
                penalty += REWARD_CONFIG["hallucinated_person_id"]
                feedback.append(f"Hallucinated {param}: '{person_id}' does not exist.")

    return penalty, feedback


def build_relevance_weights(
    corpus: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """
    Build relevance weight mapping for each message per task.

    Returns:
        Dict[task_id, Dict[message_id, weight]]
        where weight is 0.0 (irrelevant) to 1.0 (critical)
    """
    relevance = {
        "decision_archaeology": {},
        "commitment_detection": {},
        "knowledge_recovery": {},
    }

    # Task 1: Decision Archaeology
    # Relevant messages are those in decision chains
    decisions = ground_truth.get("decisions", {})
    for dec_id, dec in decisions.items():
        msg_ids = dec.get("message_ids", [])
        chain_length = len(msg_ids)

        for i, msg_id in enumerate(msg_ids):
            # Root messages are most relevant, decreasing along chain
            weight = 1.0 - (i * 0.1)  # 1.0, 0.9, 0.8, ...
            weight = max(0.3, weight)  # Floor at 0.3

            # Boost for D01 (primary decision)
            if dec_id == "D01":
                weight = min(weight * 1.2, 1.0)

            current = relevance["decision_archaeology"].get(msg_id, 0)
            relevance["decision_archaeology"][msg_id] = max(current, weight)

    # Task 2: Commitment Detection
    # Relevant messages contain commitments (especially dropped ones)
    commitments = ground_truth.get("commitments", {})
    dropped_ids = set(ground_truth.get("dropped_commitment_ids", []))

    for cid, comm in commitments.items():
        msg_id = comm.get("message_id")
        if msg_id:
            # Dropped commitments are more relevant
            if cid in dropped_ids:
                risk = comm.get("risk_level", "medium")
                risk_weights = {"critical": 1.0, "high": 0.85, "medium": 0.7, "low": 0.5}
                weight = risk_weights.get(risk, 0.7)
            else:
                weight = 0.3  # Resolved commitments less relevant

            relevance["commitment_detection"][msg_id] = weight

    # Task 3: Knowledge Recovery
    # Relevant messages involve Sofia (P06) or relate to her systems/decisions
    sofia_profile = ground_truth.get("sofia_knowledge_profile", {})
    sofia_id = sofia_profile.get("person_id", "P06")
    sofia_decisions = set(sofia_profile.get("decisions_owned", []))

    for msg in corpus:
        msg_id = msg.get("message_id")

        weight = 0.0

        # Messages sent by Sofia are highly relevant
        if msg.get("sender_id") == sofia_id:
            weight = 0.9

        # Messages sent TO Sofia
        elif sofia_id in msg.get("recipient_ids", []):
            weight = 0.7

        # Messages about projects Sofia is on
        sofia_projects = {"atlas"}  # Sofia's primary project
        if msg.get("project_tag") in sofia_projects:
            weight = max(weight, 0.5)

        # Messages part of decisions Sofia owns
        for dec_id in sofia_decisions:
            dec = decisions.get(dec_id, {})
            if msg_id in dec.get("message_ids", []):
                weight = max(weight, 0.95)
                break

        if weight > 0:
            relevance["knowledge_recovery"][msg_id] = weight

    return relevance


def compute_step_reward(
    action_type: str,
    parameters: Dict[str, Any],
    retrieved_messages: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
    reward_state: RewardState,
    task_id: str,
    relevance_weights: Optional[Dict[str, Dict[str, float]]] = None,
    valid_ids: Optional[ValidIds] = None,
) -> Tuple[float, float, str]:
    """
    Compute intermediate reward for a single step.

    Args:
        action_type: The type of action taken
        parameters: Action parameters
        retrieved_messages: Messages retrieved by this action (if any)
        ground_truth: Full ground truth data
        reward_state: Mutable state tracking previous actions
        task_id: Current task identifier
        relevance_weights: Pre-computed relevance weights per task
        valid_ids: Container of valid IDs for hallucination detection

    Returns:
        Tuple of (step_score, penalty, feedback_string)
    """
    score = 0.0
    penalty = 0.0
    feedback_parts = []

    # Check for hallucinated IDs (destructive action penalty)
    if valid_ids is not None:
        hallucination_penalty, hallucination_feedback = check_hallucinations(
            action_type, parameters, valid_ids
        )
        penalty += hallucination_penalty
        feedback_parts.extend(hallucination_feedback)

    if action_type == "retrieve_messages":
        action_score, action_penalty, feedback = _reward_retrieve(
            parameters, retrieved_messages, ground_truth,
            reward_state, task_id, relevance_weights
        )
        score += action_score
        penalty += action_penalty
        feedback_parts.append(feedback)

    elif action_type == "trace_thread":
        action_score, action_penalty, feedback = _reward_trace_thread(
            parameters, retrieved_messages, ground_truth,
            reward_state, task_id, relevance_weights
        )
        score += action_score
        penalty += action_penalty
        feedback_parts.append(feedback)

    elif action_type == "tag_decision":
        action_score, action_penalty, feedback = _reward_tag_decision(
            parameters, ground_truth, reward_state
        )
        score += action_score
        penalty += action_penalty
        feedback_parts.append(feedback)

    elif action_type == "tag_commitment":
        action_score, action_penalty, feedback = _reward_tag_commitment(
            parameters, ground_truth, reward_state
        )
        score += action_score
        penalty += action_penalty
        feedback_parts.append(feedback)

    elif action_type == "link_cause_effect":
        action_score, action_penalty, feedback = _reward_link_cause_effect(
            parameters, ground_truth, reward_state
        )
        score += action_score
        penalty += action_penalty
        feedback_parts.append(feedback)

    elif action_type == "draft_artifact":
        score = REWARD_CONFIG["draft_artifact_base"]
        feedback_parts.append("Artifact section drafted.")

    elif action_type == "submit":
        # Terminal reward handled separately by graders
        feedback_parts.append("Submission received.")

    return score, penalty, " ".join(feedback_parts)


def _reward_retrieve(
    parameters: Dict[str, Any],
    retrieved_messages: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
    reward_state: RewardState,
    task_id: str,
    relevance_weights: Optional[Dict[str, Dict[str, float]]],
) -> Tuple[float, float, str]:
    """Compute reward for retrieve_messages action."""
    query = parameters.get("query", "")
    score = 0.0
    penalty = 0.0
    feedback_parts = []

    # Check for repeated query
    if query and query in reward_state.previous_queries:
        penalty += REWARD_CONFIG["repeated_query_penalty"]
        feedback_parts.append("Repeated query detected.")
    else:
        reward_state.previous_queries.add(query)

    # Compute relevance-weighted score
    if retrieved_messages and relevance_weights:
        task_weights = relevance_weights.get(task_id, {})

        new_relevant = 0
        total_relevance = 0.0

        for msg in retrieved_messages:
            msg_id = msg.get("message_id")
            weight = task_weights.get(msg_id, 0.0)
            total_relevance += weight

            # Track new vs. duplicate retrievals
            if msg_id not in reward_state.retrieved_message_ids:
                reward_state.retrieved_message_ids.add(msg_id)
                if weight > 0:
                    new_relevant += 1
            else:
                penalty += REWARD_CONFIG["retrieve_duplicate_penalty"]

        # Average relevance score
        if retrieved_messages:
            avg_relevance = total_relevance / len(retrieved_messages)
            score = avg_relevance * REWARD_CONFIG["retrieve_relevant_max"]

        feedback_parts.append(f"Retrieved {len(retrieved_messages)} messages ({new_relevant} newly relevant).")

    return score, penalty, " ".join(feedback_parts)


def _reward_trace_thread(
    parameters: Dict[str, Any],
    retrieved_messages: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
    reward_state: RewardState,
    task_id: str,
    relevance_weights: Optional[Dict[str, Dict[str, float]]],
) -> Tuple[float, float, str]:
    """Compute reward for trace_thread action."""
    # Similar to retrieve but without query loop checking
    score = 0.0
    penalty = 0.0
    feedback_parts = []

    if retrieved_messages and relevance_weights:
        task_weights = relevance_weights.get(task_id, {})

        new_relevant = 0
        total_relevance = 0.0

        for msg in retrieved_messages:
            msg_id = msg.get("message_id")
            weight = task_weights.get(msg_id, 0.0)
            total_relevance += weight

            if msg_id not in reward_state.retrieved_message_ids:
                reward_state.retrieved_message_ids.add(msg_id)
                if weight > 0:
                    new_relevant += 1

        if retrieved_messages:
            avg_relevance = total_relevance / len(retrieved_messages)
            score = avg_relevance * REWARD_CONFIG["retrieve_relevant_max"]

        feedback_parts.append(f"Traced thread: {len(retrieved_messages)} messages ({new_relevant} newly relevant).")

    return score, penalty, " ".join(feedback_parts)


def _reward_tag_decision(
    parameters: Dict[str, Any],
    ground_truth: Dict[str, Any],
    reward_state: RewardState,
) -> Tuple[float, float, str]:
    """Compute reward for tag_decision action."""
    msg_id = parameters.get("message_id", "")
    score = 0.0
    penalty = 0.0
    feedback = ""

    # Check if this message is in any ground truth decision chain
    decisions = ground_truth.get("decisions", {})
    all_decision_msg_ids = set()

    for dec in decisions.values():
        all_decision_msg_ids.update(dec.get("message_ids", []))

    if msg_id in all_decision_msg_ids:
        if msg_id not in reward_state.tagged_decision_ids:
            score = REWARD_CONFIG["tag_decision_correct"]
            reward_state.tagged_decision_ids.add(msg_id)
            feedback = "Correct decision message tagged."
        else:
            feedback = "Decision message already tagged."
    else:
        penalty = REWARD_CONFIG["tag_decision_wrong"]
        feedback = "Tagged message is not a ground truth decision point."

    return score, penalty, feedback


def _reward_tag_commitment(
    parameters: Dict[str, Any],
    ground_truth: Dict[str, Any],
    reward_state: RewardState,
) -> Tuple[float, float, str]:
    """Compute reward for tag_commitment action."""
    msg_id = parameters.get("message_id", "")
    score = 0.0
    penalty = 0.0
    feedback = ""

    # Check if this message is a ground truth commitment source
    commitments = ground_truth.get("commitments", {})
    dropped_ids = set(ground_truth.get("dropped_commitment_ids", []))

    commitment_msg_ids = {c.get("message_id") for c in commitments.values() if c.get("message_id")}
    dropped_msg_ids = {
        commitments[cid].get("message_id")
        for cid in dropped_ids
        if commitments.get(cid, {}).get("message_id")
    }

    if msg_id in commitment_msg_ids:
        if msg_id not in reward_state.tagged_commitment_ids:
            reward_state.tagged_commitment_ids.add(msg_id)

            if msg_id in dropped_msg_ids:
                score = REWARD_CONFIG["tag_commitment_correct"]
                feedback = "Correctly tagged a dropped commitment."
            else:
                score = REWARD_CONFIG["tag_commitment_correct"] * 0.5
                feedback = "Tagged a resolved commitment (partial credit)."
        else:
            feedback = "Commitment already tagged."
    else:
        penalty = REWARD_CONFIG["tag_commitment_wrong"]
        feedback = "Tagged message is not a commitment source."

    return score, penalty, feedback


def _reward_link_cause_effect(
    parameters: Dict[str, Any],
    ground_truth: Dict[str, Any],
    reward_state: RewardState,
) -> Tuple[float, float, str]:
    """Compute reward for link_cause_effect action."""
    cause_id = parameters.get("cause_message_id", "")
    effect_id = parameters.get("effect_message_id", "")
    score = 0.0
    penalty = 0.0
    feedback = ""

    pair = (cause_id, effect_id)

    if pair in reward_state.linked_pairs:
        feedback = "This causal link was already recorded."
        return score, penalty, feedback

    reward_state.linked_pairs.add(pair)

    # Check if this is a valid causal link in any decision chain
    decisions = ground_truth.get("decisions", {})

    for dec in decisions.values():
        msg_ids = dec.get("message_ids", [])

        if cause_id in msg_ids and effect_id in msg_ids:
            cause_idx = msg_ids.index(cause_id)
            effect_idx = msg_ids.index(effect_id)

            if effect_idx == cause_idx + 1:
                # Adjacent and correct order
                score = REWARD_CONFIG["link_correct_adjacent"]
                feedback = "Valid adjacent causal link established."
            elif effect_idx > cause_idx:
                # Correct order but not adjacent
                score = REWARD_CONFIG["link_correct_chain"]
                feedback = "Valid causal link (not adjacent)."
            else:
                # Wrong order
                penalty = REWARD_CONFIG["link_wrong"]
                feedback = "Causal link has incorrect order."
            return score, penalty, feedback

    # Neither message in any decision chain, or not both in same chain
    penalty = REWARD_CONFIG["link_wrong"]
    feedback = "Causal link not found in any decision chain."

    return score, penalty, feedback


def compute_timeout_penalty() -> Tuple[float, str]:
    """Compute penalty for episode timeout."""
    return REWARD_CONFIG["timeout_penalty"], "Episode terminated: step limit reached without submitting."


def compute_terminal_reward(
    grader_score: float,
    task_id: str,
) -> float:
    """
    Compute terminal reward from grader score.

    Args:
        grader_score: Score from the grader (0.0-1.0)
        task_id: Task identifier

    Returns:
        Weighted terminal reward
    """
    weight = TASK_WEIGHTS.get(task_id, 1.0)
    return grader_score * weight
