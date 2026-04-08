"""
Grader 2: Silent Commitment Detection
Evaluates agent's ability to surface dropped commitments.

Scoring:
- Recall of dropped commitments: 0.40 (TP / 15)
- Precision: 0.30 (TP / (TP + FP))
- Risk ranking correlation: 0.20 (Spearman rho)
- Resolution plan quality: 0.10
- Penalty: resolved flagged as dropped: -0.05 each

Matching logic:
- source_message_id matches ground truth dropped commitment OR
- committer + semantic similarity of commitment_text > 0.85
"""

from typing import Dict, Any, List, Set, Optional, Tuple
from ..models import GraderResult

try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

HAS_SENTENCE_TRANSFORMERS = False
_EMBEDDING_MODEL = None

def _try_import_sentence_transformers():
    """Lazy import sentence_transformers to avoid import-time errors."""
    global HAS_SENTENCE_TRANSFORMERS
    try:
        from sentence_transformers import SentenceTransformer, util
        HAS_SENTENCE_TRANSFORMERS = True
        return SentenceTransformer, util
    except (ImportError, OSError):
        HAS_SENTENCE_TRANSFORMERS = False
        return None, None


def _get_embedding_model():
    """Lazy load the sentence transformer model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        SentenceTransformer, _ = _try_import_sentence_transformers()
        if SentenceTransformer is not None:
            try:
                _EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                pass
    return _EMBEDDING_MODEL


def _compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity between two texts.

    Uses sentence-transformers if available, falls back to word overlap.
    """
    if not text1 or not text2:
        return 0.0

    model = _get_embedding_model()

    if model is not None:
        try:
            _, util = _try_import_sentence_transformers()
            if util is not None:
                embeddings = model.encode([text1, text2], convert_to_tensor=True)
                similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
                return max(0.0, similarity)
        except Exception:
            pass

    # Fallback: Jaccard similarity on words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def _match_commitment(
    submitted_item: Dict[str, Any],
    gt_commitments: Dict[str, Dict[str, Any]],
    dropped_ids: Set[str],
    similarity_threshold: float = 0.85,
) -> Tuple[Optional[str], bool, bool]:
    """
    Try to match a submitted commitment to ground truth.

    Returns:
        Tuple of (matched_commitment_id, is_dropped, is_semantic_match)
    """
    msg_id = submitted_item.get("source_message_id", "")
    committer = submitted_item.get("committer", "")
    text = submitted_item.get("commitment_text", "")

    # First try: exact message_id match
    for cid, c in gt_commitments.items():
        if c.get("message_id") == msg_id:
            return cid, cid in dropped_ids, False

    # Second try: committer + semantic similarity
    if committer and text:
        best_match = None
        best_similarity = 0.0

        for cid, c in gt_commitments.items():
            if c.get("speaker_id") == committer:
                gt_text = c.get("raw_phrase", "")
                similarity = _compute_text_similarity(text, gt_text)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = cid

        if best_match and best_similarity >= similarity_threshold:
            return best_match, best_match in dropped_ids, True

    return None, False, False


def _evaluate_resolution_plan(
    plan: str,
    committer: str,
    gt_commitment: Dict[str, Any],
    personas: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Evaluate quality of resolution plan.

    Checks for:
    - Mention of committer (by name or ID)
    - Mention of project
    - Timeline/deadline mentioned
    """
    if not plan:
        return 0.0

    plan_lower = plan.lower()
    score = 0.0

    # Check committer mention
    committer_mentioned = False
    if committer:
        if committer.lower() in plan_lower:
            committer_mentioned = True
        # Try to resolve committer name if personas provided
        if personas and committer in personas:
            committer_name = personas[committer].get("name", "").lower()
            if committer_name and committer_name in plan_lower:
                committer_mentioned = True

    if committer_mentioned:
        score += 1/3

    # Check project mention
    project_id = gt_commitment.get("project_id", "")
    if project_id and project_id.lower() in plan_lower:
        score += 1/3

    # Check timeline mention
    timeline_keywords = [
        "week", "day", "tomorrow", "monday", "tuesday", "wednesday",
        "thursday", "friday", "asap", "immediately", "urgent",
        "by end of", "deadline", "due", "schedule", "meeting",
        "sync", "follow up", "follow-up"
    ]
    if any(kw in plan_lower for kw in timeline_keywords):
        score += 1/3

    return score


def grade_commitment_detection(
    submission: Dict[str, Any],
    ground_truth: Dict[str, Any],
    similarity_threshold: float = 0.85,
) -> GraderResult:
    """
    Grade the commitment detection task submission.

    Expected submission format:
    {
        "dropped_commitments": [
            {
                "source_message_id": str,
                "committer": str,
                "commitment_text": str,
                "risk_level": "low" | "medium" | "high" | "critical",
                "resolution_plan": str
            },
            ...
        ]
    }
    """
    component_scores = {}
    penalties = 0.0
    explanations = []

    # Get ground truth commitments
    gt_commitments = ground_truth.get("commitments", {})
    dropped_ids = set(ground_truth.get("dropped_commitment_ids", []))

    total_dropped = len(dropped_ids)  # Should be 15

    # Build lookup for GT commitments by message_id
    gt_by_message_id = {}
    gt_dropped_message_ids = set()
    gt_resolved_message_ids = set()

    for cid, c in gt_commitments.items():
        msg_id = c.get("message_id")
        if msg_id:
            gt_by_message_id[msg_id] = (cid, c)
            if cid in dropped_ids:
                gt_dropped_message_ids.add(msg_id)
            else:
                gt_resolved_message_ids.add(msg_id)

    # Process submission
    submitted = submission.get("dropped_commitments", [])

    true_positives = 0
    false_positives = 0
    resolved_flagged = 0
    semantic_matches = 0

    matched_commitment_ids = set()
    matched_agent_risks = []
    matched_gt_risks = []

    risk_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}

    resolution_plan_scores = []

    for item in submitted:
        matched_cid, is_dropped, is_semantic = _match_commitment(
            item, gt_commitments, dropped_ids, similarity_threshold
        )

        if matched_cid:
            if matched_cid in matched_commitment_ids:
                # Already matched this commitment - count as false positive
                false_positives += 1
                continue

            matched_commitment_ids.add(matched_cid)

            if is_dropped:
                true_positives += 1
                if is_semantic:
                    semantic_matches += 1

                # Collect risk levels for correlation
                agent_risk = item.get("risk_level", "medium")
                gt_commitment = gt_commitments.get(matched_cid, {})
                gt_risk = gt_commitment.get("risk_level", "medium")

                matched_agent_risks.append(risk_map.get(agent_risk, 2))
                matched_gt_risks.append(risk_map.get(gt_risk, 2))

                # Evaluate resolution plan
                plan = item.get("resolution_plan", "")
                plan_score = _evaluate_resolution_plan(
                    plan, item.get("committer", ""), gt_commitment
                )
                resolution_plan_scores.append(plan_score)

            else:
                # Matched a resolved commitment
                false_positives += 1
                resolved_flagged += 1

        else:
            # No match found
            false_positives += 1

    # ═══════════════════════════════════════════════════════════════════════════
    # Calculate scores
    # ═══════════════════════════════════════════════════════════════════════════

    # Recall: TP / 15
    recall = true_positives / total_dropped if total_dropped > 0 else 0
    component_scores["recall"] = recall * 0.40
    explanations.append(f"Recall: {true_positives}/{total_dropped} dropped commitments found")
    if semantic_matches > 0:
        explanations.append(f"({semantic_matches} via semantic matching).")
    else:
        explanations.append(".")

    # Precision: TP / (TP + FP)
    total_submitted = true_positives + false_positives
    precision = true_positives / total_submitted if total_submitted > 0 else 0
    component_scores["precision"] = precision * 0.30
    explanations.append(f"Precision: {true_positives}/{total_submitted} correct.")

    # Risk ranking correlation (Spearman rho)
    if len(matched_agent_risks) >= 3:
        if HAS_SCIPY:
            try:
                rho, _ = spearmanr(matched_agent_risks, matched_gt_risks)
                # Handle NaN
                if rho != rho:
                    rho = 0.0
                rho = max(0, rho)  # Clamp negative correlations to 0
                component_scores["risk_ranking"] = rho * 0.20
                explanations.append(f"Risk ranking correlation: {rho:.2f}.")
            except Exception:
                component_scores["risk_ranking"] = 0.0
                explanations.append("Could not compute risk correlation.")
        else:
            # Fallback: percentage of exact matches
            exact_matches = sum(1 for a, g in zip(matched_agent_risks, matched_gt_risks) if a == g)
            approx_rho = exact_matches / len(matched_agent_risks)
            component_scores["risk_ranking"] = approx_rho * 0.20
            explanations.append(f"Risk ranking (approx): {approx_rho:.2f}.")
    elif len(matched_agent_risks) > 0:
        # Some matches but not enough for correlation
        exact_matches = sum(1 for a, g in zip(matched_agent_risks, matched_gt_risks) if a == g)
        partial_score = exact_matches / len(matched_agent_risks) * 0.5  # Half credit
        component_scores["risk_ranking"] = partial_score * 0.20
        explanations.append(f"Risk ranking (limited data): {exact_matches}/{len(matched_agent_risks)} exact matches.")
    else:
        component_scores["risk_ranking"] = 0.0
        explanations.append("No matches for risk ranking.")

    # Resolution plan quality
    if resolution_plan_scores:
        avg_plan_score = sum(resolution_plan_scores) / len(resolution_plan_scores)
        component_scores["resolution_plan_quality"] = avg_plan_score * 0.10
        explanations.append(f"Resolution plan quality: {avg_plan_score:.2f}.")
    else:
        component_scores["resolution_plan_quality"] = 0.0
        explanations.append("No resolution plans to evaluate.")

    # Penalties: resolved flagged as dropped
    penalties = resolved_flagged * 0.05
    if resolved_flagged > 0:
        explanations.append(f"Penalty: {resolved_flagged} resolved commitments incorrectly flagged (-{penalties:.2f}).")

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
