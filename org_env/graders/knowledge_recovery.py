"""
Grader 3: Knowledge Amnesia Recovery
Evaluates agent's ability to reconstruct institutional knowledge.

Scoring:
- System coverage: 0.25
- Decision ownership accuracy: 0.25 (F1)
- Relationship map completeness: 0.20
- Temporal accuracy: 0.15 (day ± 5 tolerance)
- Actionability score: 0.15 (LLM-as-judge with rule-based fallback)
- Penalty: confidently wrong facts: -0.10 each

Confidently wrong facts:
- System mentioned that departing employee has no involvement with
- Decision attributed incorrectly
- Collaborator named who has no messages with departing employee
"""

import json
import os
from typing import Dict, Any, List, Set, Optional, Tuple
from ..models import GraderResult

HAS_OPENAI = False

def _try_import_openai():
    """Lazy import OpenAI to avoid import-time errors."""
    global HAS_OPENAI
    try:
        from openai import OpenAI
        HAS_OPENAI = True
        return OpenAI
    except (ImportError, OSError):
        HAS_OPENAI = False
        return None


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


# LLM-as-judge prompt for actionability
ACTIONABILITY_PROMPT = """You are evaluating a knowledge transfer document written for an engineer
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
{document}"""


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


def _fuzzy_match(text1: str, text2: str, threshold: float = 0.5) -> bool:
    """Fuzzy matching based on word overlap or semantic similarity."""
    if not text1 or not text2:
        return False

    text1_lower = text1.lower()
    text2_lower = text2.lower()

    # Direct substring match
    if text1_lower in text2_lower or text2_lower in text1_lower:
        return True

    # Word overlap
    words1 = set(text1_lower.split())
    words2 = set(text2_lower.split())

    if not words1 or not words2:
        return False

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    jaccard = intersection / union if union > 0 else 0

    if jaccard >= threshold:
        return True

    # Try semantic similarity if available
    model = _get_embedding_model()
    if model is not None:
        try:
            _, util = _try_import_sentence_transformers()
            if util is not None:
                embeddings = model.encode([text1, text2], convert_to_tensor=True)
                similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
                return similarity >= 0.7
        except Exception:
            pass

    return False


def _llm_judge_actionability(artifact: Dict[str, Any]) -> Tuple[float, str]:
    """
    Use LLM to judge actionability of the knowledge transfer document.

    Returns:
        Tuple of (score, rationale)
    """
    OpenAI = _try_import_openai()
    if OpenAI is None:
        return None, "OpenAI not available"

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None, "No API key"

    # Format the artifact as a document
    doc_parts = []

    systems = artifact.get("systems", [])
    if systems:
        doc_parts.append("## Systems\n" + "\n".join(f"- {s}" for s in systems))

    decisions = artifact.get("decisions", [])
    if decisions:
        doc_parts.append("## Decisions\n" + "\n".join(
            f"- {d.get('decision_text', 'Unknown')}: owned by {d.get('owner', 'Unknown')}"
            for d in decisions
        ))

    collaborators = artifact.get("collaborators", [])
    if collaborators:
        doc_parts.append("## Key Collaborators\n" + "\n".join(f"- {c}" for c in collaborators))

    timeline = artifact.get("timeline", [])
    if timeline:
        doc_parts.append("## Timeline\n" + "\n".join(
            f"- Day {t.get('day', '?')}: {t.get('event', 'Unknown')}"
            for t in timeline
        ))

    open_items = artifact.get("open_items", [])
    if open_items:
        doc_parts.append("## Open Items\n" + "\n".join(f"- {item}" for item in open_items))

    notes = artifact.get("freeform_notes", "")
    if notes:
        doc_parts.append(f"## Notes\n{notes}")

    document = "\n\n".join(doc_parts) if doc_parts else "(Empty document)"

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": ACTIONABILITY_PROMPT.format(document=document)}
            ],
            max_tokens=150,
            temperature=0.0,
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON response
        # Handle potential markdown code block wrapping
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        result = json.loads(content)
        score = float(result.get("score", 0.5))
        rationale = result.get("rationale", "")

        return max(0.0, min(1.0, score)), rationale

    except Exception as e:
        return None, str(e)


def _rule_based_actionability(artifact: Dict[str, Any]) -> float:
    """
    Rule-based fallback for actionability scoring.

    Checks:
    - Systems described with enough detail (weight: 0.25)
    - Key contacts identified (weight: 0.25)
    - Ongoing tasks/open items stated (weight: 0.25)
    - Decisions explained (weight: 0.25)
    """
    score = 0.0

    # Systems with detail
    systems = artifact.get("systems", [])
    if len(systems) >= 2:
        score += 0.25
    elif len(systems) >= 1:
        score += 0.125

    # Key contacts
    collaborators = artifact.get("collaborators", [])
    if len(collaborators) >= 3:
        score += 0.25
    elif len(collaborators) >= 1:
        score += 0.125

    # Open items
    open_items = artifact.get("open_items", [])
    if len(open_items) >= 2:
        score += 0.25
    elif len(open_items) >= 1:
        score += 0.125

    # Decisions
    decisions = artifact.get("decisions", [])
    if len(decisions) >= 1:
        score += 0.25

    # Bonus for freeform notes with substance
    notes = artifact.get("freeform_notes", "")
    if len(notes) > 100:
        score = min(score + 0.1, 1.0)

    return score


def _build_sofia_contact_set(corpus: List[Dict[str, Any]], sofia_id: str) -> Set[str]:
    """Build set of all person IDs Sofia has exchanged messages with."""
    contacts = set()
    for msg in corpus:
        if msg.get("sender_id") == sofia_id:
            contacts.update(msg.get("recipient_ids", []))
        elif sofia_id in msg.get("recipient_ids", []):
            contacts.add(msg.get("sender_id"))
    return contacts


def _detect_wrong_facts(
    artifact: Dict[str, Any],
    ground_truth: Dict[str, Any],
    corpus: List[Dict[str, Any]],
    sofia_id: str,
) -> List[str]:
    """
    Detect confidently wrong facts in the submission.

    A fact is "confidently wrong" if:
    1. A system is mentioned that Sofia has no involvement with
    2. A decision is attributed to Sofia that GT assigns to different person
    3. A collaborator is named who has no messages with Sofia
    """
    wrong_facts = []

    sofia_profile = ground_truth.get("sofia_knowledge_profile", {})
    gt_systems = {s["name"].lower() for s in sofia_profile.get("systems_owned", [])}

    # Get Sofia's projects
    projects = ground_truth.get("projects", {})
    sofia_projects = set()
    for proj_id, proj in projects.items():
        if sofia_id in proj.get("team_ids", []):
            sofia_projects.add(proj_id.lower())

    # Build contact set
    sofia_contacts = _build_sofia_contact_set(corpus, sofia_id)

    # Check systems
    sub_systems = artifact.get("systems", [])
    for sys in sub_systems:
        sys_lower = sys.lower()

        # Check if it matches a known system or project
        is_valid = False
        for gt_sys in gt_systems:
            if _fuzzy_match(sys_lower, gt_sys, threshold=0.4):
                is_valid = True
                break

        for proj in sofia_projects:
            if proj in sys_lower or sys_lower in proj:
                is_valid = True
                break

        if not is_valid:
            # Check if Sofia has any messages mentioning this system
            system_mentioned = any(
                sys_lower in msg.get("body", "").lower()
                for msg in corpus
                if msg.get("sender_id") == sofia_id
            )
            if not system_mentioned:
                wrong_facts.append(f"System '{sys}' has no connection to Sofia")

    # Check collaborators
    sub_collaborators = artifact.get("collaborators", [])
    for collab in sub_collaborators:
        if collab != sofia_id and collab not in sofia_contacts:
            wrong_facts.append(f"Collaborator '{collab}' has no message history with Sofia")

    # Check decision attributions
    decisions_gt = ground_truth.get("decisions", {})
    sub_decisions = artifact.get("decisions", [])

    for sub_dec in sub_decisions:
        sub_owner = sub_dec.get("owner", "")
        sub_text = sub_dec.get("decision_text", "")

        for dec_id, gt_dec in decisions_gt.items():
            gt_title = gt_dec.get("title", "")
            gt_owner = gt_dec.get("accountable_person_id", "")

            if _fuzzy_match(sub_text, gt_title, threshold=0.4):
                if sub_owner and gt_owner and sub_owner != gt_owner:
                    wrong_facts.append(
                        f"Decision '{sub_text[:50]}...' attributed to {sub_owner}, "
                        f"but actually owned by {gt_owner}"
                    )
                break

    return wrong_facts


def grade_knowledge_recovery(
    submission: Dict[str, Any],
    ground_truth: Dict[str, Any],
    corpus: List[Dict[str, Any]],
    use_llm_judge: bool = True,
) -> GraderResult:
    """
    Grade the knowledge recovery task submission.

    Expected submission format:
    {
        "artifact": {
            "systems": [str, ...],
            "decisions": [{"decision_text": str, "owner": str}, ...],
            "collaborators": [str, ...],
            "timeline": [{"day": int, "event": str}, ...],
            "open_items": [str, ...],
            "freeform_notes": str
        }
    }
    """
    component_scores = {}
    penalties = 0.0
    explanations = []

    # Get Sofia's knowledge profile from ground truth
    sofia_profile = ground_truth.get("sofia_knowledge_profile", {})
    sofia_id = sofia_profile.get("person_id", "P06")

    # Ground truth data
    gt_systems = [s["name"].lower() for s in sofia_profile.get("systems_owned", [])]
    gt_decisions = sofia_profile.get("decisions_owned", [])
    gt_collaborators = [r["person_id"] for r in sofia_profile.get("key_relationships", [])]

    # Extract submission data
    artifact = submission.get("artifact", {})
    sub_systems = [s.lower() for s in artifact.get("systems", [])]
    sub_decisions = artifact.get("decisions", [])
    sub_collaborators = artifact.get("collaborators", [])
    sub_timeline = artifact.get("timeline", [])
    sub_open_items = artifact.get("open_items", [])

    # ═══════════════════════════════════════════════════════════════════════════
    # Score: System coverage (0.25)
    # ═══════════════════════════════════════════════════════════════════════════
    matched_systems = 0
    for gt_sys in gt_systems:
        for sub_sys in sub_systems:
            if _fuzzy_match(sub_sys, gt_sys, threshold=0.4):
                matched_systems += 1
                break

    system_coverage = matched_systems / len(gt_systems) if gt_systems else 0
    component_scores["system_coverage"] = system_coverage * 0.25
    explanations.append(f"System coverage: {matched_systems}/{len(gt_systems)}.")

    # ═══════════════════════════════════════════════════════════════════════════
    # Score: Decision ownership accuracy (0.25)
    # ═══════════════════════════════════════════════════════════════════════════
    decisions_gt = ground_truth.get("decisions", {})

    decision_tp = 0
    decision_fp = 0
    decisions_found = set()

    for sub_dec in sub_decisions:
        sub_text = sub_dec.get("decision_text", "")
        sub_owner = sub_dec.get("owner", "")

        found_match = False
        for dec_id in gt_decisions:
            if dec_id in decisions_found:
                continue

            gt_dec = decisions_gt.get(dec_id, {})
            gt_title = gt_dec.get("title", "")
            gt_owner = gt_dec.get("accountable_person_id", "")

            if _fuzzy_match(sub_text, gt_title, threshold=0.4) or dec_id.lower() in sub_text.lower():
                found_match = True
                decisions_found.add(dec_id)
                if sub_owner == gt_owner:
                    decision_tp += 1
                else:
                    decision_fp += 1
                break

        if not found_match and sub_text:
            decision_fp += 1

    # F1 score
    if decision_tp + decision_fp > 0:
        precision = decision_tp / (decision_tp + decision_fp)
        recall = decision_tp / len(gt_decisions) if gt_decisions else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        component_scores["decision_ownership_accuracy"] = f1 * 0.25
        explanations.append(f"Decision accuracy: P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}.")
    else:
        component_scores["decision_ownership_accuracy"] = 0.0
        explanations.append("No decisions submitted.")

    # ═══════════════════════════════════════════════════════════════════════════
    # Score: Relationship completeness (0.20)
    # ═══════════════════════════════════════════════════════════════════════════
    matched_collaborators = len(set(sub_collaborators) & set(gt_collaborators))
    relationship_score = matched_collaborators / len(gt_collaborators) if gt_collaborators else 0
    component_scores["relationship_completeness"] = relationship_score * 0.20
    explanations.append(f"Collaborators: {matched_collaborators}/{len(gt_collaborators)}.")

    # ═══════════════════════════════════════════════════════════════════════════
    # Score: Temporal accuracy (0.15) - day ± 5 tolerance
    # ═══════════════════════════════════════════════════════════════════════════
    if sub_timeline:
        # Get ground truth timeline events from decisions and commitments
        gt_timeline_days = set()

        # Add decision days
        for dec_id in gt_decisions:
            gt_dec = decisions_gt.get(dec_id, {})
            gt_timeline_days.add(gt_dec.get("root_message_day", 0))
            gt_timeline_days.add(gt_dec.get("final_decision_day", 0))

        # Add Sofia's departure day
        departure_day = sofia_profile.get("departure_day", 45)
        gt_timeline_days.add(departure_day)

        # Score timeline events with ± 5 day tolerance
        temporal_matches = 0
        for event in sub_timeline:
            event_day = event.get("day", 0)
            if not isinstance(event_day, int):
                continue

            # Check if within simulation bounds
            if not (1 <= event_day <= 60):
                continue

            # Check if close to any known event
            for gt_day in gt_timeline_days:
                if gt_day and abs(event_day - gt_day) <= 5:
                    temporal_matches += 1
                    break

        temporal_score = temporal_matches / len(sub_timeline) if sub_timeline else 0
        component_scores["temporal_accuracy"] = temporal_score * 0.15
        explanations.append(f"Timeline: {temporal_matches}/{len(sub_timeline)} events within tolerance.")
    else:
        component_scores["temporal_accuracy"] = 0.0
        explanations.append("No timeline submitted.")

    # ═══════════════════════════════════════════════════════════════════════════
    # Score: Actionability (0.15) - LLM-as-judge with fallback
    # ═══════════════════════════════════════════════════════════════════════════
    actionability_score = None
    actionability_rationale = ""

    if use_llm_judge:
        actionability_score, actionability_rationale = _llm_judge_actionability(artifact)

    if actionability_score is not None:
        component_scores["actionability_score"] = actionability_score * 0.15
        explanations.append(f"Actionability (LLM): {actionability_score:.2f}. {actionability_rationale}")
    else:
        # Fallback to rule-based
        actionability_score = _rule_based_actionability(artifact)
        component_scores["actionability_score"] = actionability_score * 0.15
        explanations.append(f"Actionability (rule-based): {actionability_score:.2f}.")

    # ═══════════════════════════════════════════════════════════════════════════
    # Penalties: Confidently wrong facts (-0.10 each)
    # ═══════════════════════════════════════════════════════════════════════════
    wrong_facts = _detect_wrong_facts(artifact, ground_truth, corpus, sofia_id)
    wrong_fact_penalty = len(wrong_facts) * 0.10
    penalties += wrong_fact_penalty

    if wrong_facts:
        explanations.append(f"Wrong facts detected ({len(wrong_facts)}): {'; '.join(wrong_facts[:3])}")
        if len(wrong_facts) > 3:
            explanations.append(f"...and {len(wrong_facts) - 3} more.")

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
