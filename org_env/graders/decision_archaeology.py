"""
Grader 1: Decision Archaeology
Evaluates agent's ability to trace decision chains.

Scoring:
- Root decision found: 0.40
- Each correct hop (up to 6): +0.075 each = 0.45 max
- Correct accountability: +0.15
- Penalty: wrong person blamed: -0.10

Edge cases:
- Agent submits more hops than ground truth: partial credit for correct subset
- Agent submits correct chain in wrong order: 50% credit
- Root message correctly identified but chain is empty: 0.40 + 0.15 if person correct
"""

from typing import Dict, Any, List, Optional, Set
from ..models import GraderResult


def _compute_chain_order_score(
    submitted_chain: List[str],
    gt_chain: List[str],
) -> tuple[float, str]:
    """
    Compute score for chain ordering.

    Returns:
        Tuple of (score multiplier, explanation)
        - 1.0 if perfect order
        - 0.5 if correct messages but wrong order
        - proportional credit for partial matches
    """
    if not submitted_chain:
        return 0.0, "Empty chain submitted."

    if not gt_chain:
        return 0.0, "No ground truth chain available."

    # Find messages that are in both chains
    submitted_set = set(submitted_chain)
    gt_set = set(gt_chain)
    common = submitted_set & gt_set

    if not common:
        return 0.0, "No matching messages in chain."

    # Check if order is correct for common messages
    # Extract positions of common messages in both chains
    sub_positions = {msg: i for i, msg in enumerate(submitted_chain) if msg in common}
    gt_positions = {msg: i for i, msg in enumerate(gt_chain) if msg in common}

    # Check pairwise ordering
    common_list = list(common)
    correct_pairs = 0
    total_pairs = 0

    for i, msg1 in enumerate(common_list):
        for msg2 in common_list[i+1:]:
            total_pairs += 1
            # Check if relative order is preserved
            sub_order = sub_positions[msg1] < sub_positions[msg2]
            gt_order = gt_positions[msg1] < gt_positions[msg2]
            if sub_order == gt_order:
                correct_pairs += 1

    if total_pairs == 0:
        order_score = 1.0
    else:
        order_score = correct_pairs / total_pairs

    # Perfect order gets full credit, otherwise 50% base + proportional bonus
    if order_score == 1.0:
        return 1.0, "Chain order is correct."
    elif order_score >= 0.5:
        return 0.5 + (order_score - 0.5) * 0.5, f"Chain partially ordered ({order_score:.0%} pairs correct)."
    else:
        return 0.5, "Chain messages found but in wrong order (50% credit)."


def _find_best_matching_decision(
    submission: Dict[str, Any],
    decisions: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """
    Find the decision that best matches the submission.

    Uses heuristics like:
    - Message ID overlap between submission chain and decision chains
    - Text similarity in decision_text
    """
    sub_root = submission.get("root_decision", "")
    sub_chain = set(submission.get("decision_chain", []))
    sub_text = submission.get("decision_text", "").lower()

    best_match = None
    best_score = 0

    for dec_id, dec in decisions.items():
        score = 0
        gt_messages = set(dec.get("message_ids", []))

        # Check root match
        if sub_root and gt_messages and sub_root == list(gt_messages)[0]:
            score += 50
        elif sub_root in gt_messages:
            score += 25

        # Check chain overlap
        overlap = len(sub_chain & gt_messages)
        score += overlap * 10

        # Check text similarity
        gt_title = dec.get("title", "").lower()
        if gt_title and sub_text:
            common_words = set(gt_title.split()) & set(sub_text.split())
            score += len(common_words) * 2

        if score > best_score:
            best_score = score
            best_match = dec_id

    return best_match


def grade_decision_archaeology(
    submission: Dict[str, Any],
    ground_truth: Dict[str, Any],
    target_decision_id: str = "D01",
) -> GraderResult:
    """
    Grade the decision archaeology task submission.

    Expected submission format:
    {
        "root_decision": str,           # message_id of the root
        "decision_chain": [str, ...],   # ordered chain of message IDs
        "accountable_person": str,      # person_id
        "decision_text": str
    }
    """
    component_scores = {}
    penalties = 0.0
    explanations = []

    decisions = ground_truth.get("decisions", {})

    # Try to find the best matching decision if target not specified or not found
    if target_decision_id not in decisions:
        target_decision_id = _find_best_matching_decision(submission, decisions)

    if not target_decision_id or target_decision_id not in decisions:
        # Default to D01 if no match found
        target_decision_id = "D01"

    target_decision = decisions.get(target_decision_id, {})

    if not target_decision:
        return GraderResult(
            total_score=0.0,
            component_scores={},
            penalties=0.0,
            explanation="Ground truth decision not found.",
        )

    gt_message_ids = target_decision.get("message_ids", [])
    gt_root = gt_message_ids[0] if gt_message_ids else None
    gt_accountable = target_decision.get("accountable_person_id")
    gt_decision_maker = target_decision.get("final_decision_maker_id")

    # Extract submission data
    sub_root = submission.get("root_decision")
    sub_chain = submission.get("decision_chain", [])
    sub_accountable = submission.get("accountable_person")

    # Normalize: if root is provided but not in chain, prepend it
    if sub_root and sub_chain and sub_root not in sub_chain:
        sub_chain = [sub_root] + sub_chain

    # ═══════════════════════════════════════════════════════════════════════════
    # Score: Root decision found (0.40)
    # ═══════════════════════════════════════════════════════════════════════════
    if sub_root and gt_root and sub_root == gt_root:
        component_scores["root_decision_found"] = 0.40
        explanations.append("Root decision correctly identified.")
    elif sub_root and sub_root in gt_message_ids:
        # Partial credit: found a message in the chain but not the root
        position = gt_message_ids.index(sub_root)
        # More credit for messages closer to the root
        partial_credit = 0.40 * (1 - position / len(gt_message_ids)) * 0.5
        component_scores["root_decision_found"] = partial_credit
        explanations.append(f"Found message at position {position} in chain (not root).")
    else:
        component_scores["root_decision_found"] = 0.0
        explanations.append("Root decision not found or incorrect.")

    # ═══════════════════════════════════════════════════════════════════════════
    # Score: Chain hops (up to 0.45)
    # ═══════════════════════════════════════════════════════════════════════════
    max_hops = min(6, len(gt_message_ids) - 1)

    if sub_chain:
        # Count correct messages in chain
        correct_positions = 0
        correct_presence = 0

        for i, msg_id in enumerate(sub_chain):
            if msg_id in gt_message_ids:
                correct_presence += 1
                gt_pos = gt_message_ids.index(msg_id)
                if i < len(gt_message_ids) and msg_id == gt_message_ids[i]:
                    correct_positions += 1

        # Check ordering
        order_multiplier, order_explanation = _compute_chain_order_score(sub_chain, gt_message_ids)

        # Base score from correct presence
        presence_ratio = min(correct_presence / (max_hops + 1), 1.0) if max_hops > 0 else 0

        # Position score (correct messages in correct positions)
        position_ratio = correct_positions / (max_hops + 1) if max_hops > 0 else 0

        # Combined score: weighted average with order multiplier
        if position_ratio == 1.0:
            # Perfect chain
            hop_score = 0.45
            explanations.append(f"Perfect chain: all {correct_positions} hops correct.")
        elif order_multiplier < 1.0 and correct_presence > 0:
            # Correct messages but wrong order: 50% credit per spec
            hop_score = presence_ratio * 0.45 * order_multiplier
            explanations.append(f"Chain: {correct_presence}/{max_hops+1} messages found. {order_explanation}")
        else:
            # Partial credit for correct positions
            hop_score = position_ratio * 0.45
            explanations.append(f"Chain hops: {correct_positions}/{max_hops+1} in correct position.")

        component_scores["chain_hops_correct"] = hop_score
    else:
        component_scores["chain_hops_correct"] = 0.0
        explanations.append("No decision chain submitted.")

    # ═══════════════════════════════════════════════════════════════════════════
    # Score: Accountability (0.15)
    # ═══════════════════════════════════════════════════════════════════════════
    if sub_accountable:
        if sub_accountable == gt_accountable:
            component_scores["accountability_correct"] = 0.15
            explanations.append("Accountable person correctly identified.")
        elif sub_accountable == gt_decision_maker:
            # Partial credit: identified decision maker instead of accountable person
            component_scores["accountability_correct"] = 0.075
            explanations.append("Identified decision maker (not accountable person).")
            penalties += 0.05  # Smaller penalty
        else:
            component_scores["accountability_correct"] = 0.0
            penalties += 0.10  # Wrong person blamed
            explanations.append(f"Wrong person blamed. Expected {gt_accountable}, got {sub_accountable}.")
    else:
        component_scores["accountability_correct"] = 0.0
        explanations.append("Accountable person not specified.")

    # ═══════════════════════════════════════════════════════════════════════════
    # Compute total
    # ═══════════════════════════════════════════════════════════════════════════
    total_score = sum(component_scores.values()) - penalties
    total_score = max(0.0, min(1.0, total_score))

    return GraderResult(
        total_score=total_score,
        component_scores=component_scores,
        penalties=penalties,
        explanation=" ".join(explanations),
    )
