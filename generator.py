"""
OrgTrace — Synthetic Corpus Generator
Generates the full 60-day communication corpus for Meridian Labs.
Weaves all seeded decisions and commitments into realistic messages.
Output: data/generated/corpus.json + ground_truth.json
"""

import json
import random
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data.personas   import PERSONAS, PERSONA_BY_ID, DEPARTING_EMPLOYEE
from data.seeds      import (
    PROJECTS, PROJECT_BY_ID,
    DECISION_SEEDS, DECISION_BY_ID,
    COMMITMENT_SEEDS, COMMITMENT_BY_ID,
    DROPPED_COMMITMENT_IDS,
    SOFIA_KNOWLEDGE_PROFILE,
)
from data.templates  import OPENERS, CLOSERS, FILLER_TEMPLATES, SLACK_CHANNELS


# ── CONFIG ───────────────────────────────────────────────────────────────────

SEED       = 42
SIM_DAYS   = 60
BASE_DATE  = datetime(2024, 1, 8)          # Monday Jan 8, 2024
OUTPUT_DIR = Path(__file__).parent / "data" / "generated"

random.seed(SEED)


# ── HELPERS ──────────────────────────────────────────────────────────────────

def day_to_dt(day: int, hour: int = 10, minute: int = 0) -> str:
    """Convert simulation day (1-indexed) to ISO datetime string."""
    dt = BASE_DATE + timedelta(days=day - 1, hours=hour - 10, minutes=minute)
    # Skip weekends
    while dt.weekday() >= 5:
        dt += timedelta(days=1)
    return dt.isoformat()


def make_id(prefix: str, *parts) -> str:
    raw = prefix + "_" + "_".join(str(p) for p in parts)
    return prefix + "_" + hashlib.md5(raw.encode()).hexdigest()[:8]


def pick(lst: list):
    return random.choice(lst)


def opener(style: str, name: str = "") -> str:
    tmpl = pick(OPENERS.get(style, OPENERS["casual"]))
    return tmpl.format(name=name.split()[0] if name else "")


def closer(style: str, sender_name: str = "") -> str:
    tmpl = pick(CLOSERS.get(style, CLOSERS["casual"]))
    return tmpl.format(
        sender_name=sender_name,
        sender_first=sender_name.split()[0] if sender_name else ""
    )


def project_team_ids(project_id: str) -> List[str]:
    return PROJECT_BY_ID[project_id].team_ids


# ── SEEDED DECISION MESSAGE GENERATOR ────────────────────────────────────────

def generate_decision_chain_messages(decision_seed) -> List[Dict]:
    """Turn a DecisionSeed chain into concrete messages."""
    messages = []
    thread_id = make_id("thread", decision_seed.id)

    for i, hop in enumerate(decision_seed.chain):
        sender   = PERSONA_BY_ID[hop["sender_id"]]
        style    = sender.comm_style.value

        # Determine recipients: next person in chain OR relevant leads
        if i + 1 < len(decision_seed.chain):
            next_sender_id = decision_seed.chain[i + 1]["sender_id"]
            recipients = [next_sender_id]
        else:
            recipients = [decision_seed.root_sender_id]

        # Add CC for decision messages
        if hop.get("is_decision"):
            recipients += [decision_seed.accountable_person_id]
            recipients = list(set(recipients))

        recipient_names = [PERSONA_BY_ID[r].name for r in recipients if r in PERSONA_BY_ID]
        primary_name    = recipient_names[0] if recipient_names else ""

        subject = (
            f"Re: {decision_seed.title}"
            if i > 0
            else decision_seed.title
        )

        body = (
            f"{opener(style, primary_name)}\n\n"
            f"{hop['summary']}\n\n"
            f"{closer(style, sender.name)}"
        )

        # Slack vs email — first message email, replies can be slack
        channel = "email" if i == 0 else pick(["email", "slack"])
        if channel == "slack":
            # Pick relevant slack channel
            proj = PROJECT_BY_ID.get(decision_seed.project_id)
            if proj and sender.id in proj.team_ids:
                channel_name = f"#{decision_seed.project_id}-eng" if "eng" in sender.team.value.lower() else f"#{decision_seed.project_id}"
            else:
                channel_name = "#engineering"
            channel = f"slack:{channel_name}"

        msg = {
            "message_id":   make_id("msg", decision_seed.id, i),
            "thread_id":    thread_id,
            "day":          hop["day"],
            "timestamp":    day_to_dt(hop["day"], hour=random.randint(9, 17)),
            "sender_id":    sender.id,
            "sender_name":  sender.name,
            "sender_email": sender.email,
            "recipient_ids":    recipients,
            "recipient_names":  recipient_names,
            "channel":      channel,
            "subject":      subject if "email" in channel else "",
            "body":         body,
            "project_tag":  decision_seed.project_id,
            # Ground truth annotations (hidden from agent observations)
            "_ground_truth": {
                "decision_id":  decision_seed.id,
                "hop_index":    i,
                "is_root":      hop.get("is_root", False),
                "is_decision":  hop.get("is_decision", False),
            },
        }
        messages.append(msg)

    return messages


# ── SEEDED COMMITMENT MESSAGE GENERATOR ──────────────────────────────────────

def generate_commitment_message(commitment) -> Dict:
    """Generate a single message containing a seeded commitment phrase."""
    sender  = PERSONA_BY_ID[commitment.speaker_id]
    style   = sender.comm_style.value

    # Pick a recipient
    recip_id   = pick(commitment.recipient_ids)
    recip      = PERSONA_BY_ID.get(recip_id)
    recip_name = recip.name if recip else ""

    # The commitment phrase goes verbatim into the body
    commitment_phrase = commitment.raw_phrase

    # Wrap in a short surrounding message
    prefix_options = [
        f"Also —",
        f"One more thing:",
        f"Side note:",
        f"By the way,",
        f"Almost forgot —",
    ]
    suffix_options = [
        f"Let me know if that works.",
        f"Does that timeline work for you?",
        f"Any concerns on your end?",
        f"",
    ]

    body = (
        f"{opener(style, recip_name)}\n\n"
        f"{pick(prefix_options)} {commitment_phrase}. "
        f"{pick(suffix_options)}\n\n"
        f"{closer(style, sender.name)}"
    )

    # Channel
    channel = pick(["email", "slack", "slack"])

    msg = {
        "message_id":    make_id("msg", commitment.id),
        "thread_id":     make_id("thread", "commitment", commitment.id),
        "day":           commitment.day,
        "timestamp":     day_to_dt(commitment.day, hour=random.randint(9, 17)),
        "sender_id":     sender.id,
        "sender_name":   sender.name,
        "sender_email":  sender.email,
        "recipient_ids": commitment.recipient_ids,
        "recipient_names": [PERSONA_BY_ID[r].name for r in commitment.recipient_ids if r in PERSONA_BY_ID],
        "channel":       channel,
        "subject":       f"Re: {commitment.project_id.title() if commitment.project_id else 'General'} update" if channel == "email" else "",
        "body":          body,
        "project_tag":   commitment.project_id,
        "_ground_truth": {
            "commitment_id":  commitment.id,
            "is_commitment":  True,
            "resolved":       commitment.resolved,
            "risk_level":     commitment.risk_level,
        },
    }

    # If resolved: also generate a follow-up message
    followup = None
    if commitment.resolved and commitment.resolution_day:
        followup_body = (
            f"{opener(style, recip_name)}\n\n"
            f"Following up on my earlier note — {commitment.resolution_summary}. "
            f"Let me know if you need anything else.\n\n"
            f"{closer(style, sender.name)}"
        )
        followup = {
            "message_id":    make_id("msg", commitment.id, "resolved"),
            "thread_id":     msg["thread_id"],
            "day":           commitment.resolution_day,
            "timestamp":     day_to_dt(commitment.resolution_day, hour=random.randint(9, 17)),
            "sender_id":     sender.id,
            "sender_name":   sender.name,
            "sender_email":  sender.email,
            "recipient_ids": commitment.recipient_ids,
            "recipient_names": msg["recipient_names"],
            "channel":       channel,
            "subject":       msg["subject"],
            "body":          followup_body,
            "project_tag":   commitment.project_id,
            "_ground_truth": {
                "commitment_id":      commitment.id,
                "is_resolution":      True,
                "resolves_commitment": commitment.id,
            },
        }
    return msg, followup


# ── FILLER MESSAGE GENERATOR ─────────────────────────────────────────────────

FILLER_CONTEXTS = {
    "atlas": {
        "items":       ["auth service", "DB migration", "multi-tenant routing", "API rate limiting", "frontend redesign"],
        "milestones":  ["alpha", "beta", "v2.0 launch"],
        "reasons":     ["unexpected complexity", "dependency on Legal review", "scope change"],
    },
    "beacon": {
        "items":       ["onboarding UI", "push notifications", "user progress tracking", "A/B test setup"],
        "milestones":  ["design freeze", "mobile beta", "GA launch"],
        "reasons":     ["design iteration", "mobile platform issues", "data pipeline delays"],
    },
    "crest": {
        "items":       ["vendor DPA review", "pen test findings", "access log audit", "policy documentation"],
        "milestones":  ["gap analysis", "audit readiness", "certification"],
        "reasons":     ["vendor non-response", "Legal bandwidth", "scope expansion"],
    },
    "general": {
        "items":       ["weekly sync", "quarterly planning", "headcount review", "tooling update"],
        "milestones":  ["Q2 targets", "board meeting", "all-hands"],
        "reasons":     ["competing priorities", "holiday schedule", "stakeholder availability"],
    },
}


def generate_filler_messages(n: int = 400) -> List[Dict]:
    """Generate n filler messages to pad the corpus with realistic noise."""
    messages = []

    for i in range(n):
        day    = random.randint(1, SIM_DAYS)
        sender = pick(PERSONAS)

        # Skip messages after departure day for Sofia
        if sender.departure_day and day > sender.departure_day:
            sender = pick([p for p in PERSONAS if not p.departure_day])

        style = sender.comm_style.value

        # Pick project context
        proj_id   = pick(["atlas", "beacon", "crest", "general"])
        ctx       = FILLER_CONTEXTS[proj_id]
        template  = pick(FILLER_TEMPLATES)

        # Build recipients — same team or cross-team
        possible_recipients = [
            p for p in PERSONAS
            if p.id != sender.id and (p.team == sender.team or random.random() < 0.3)
        ]
        recipients  = random.sample(possible_recipients, k=min(random.randint(1, 3), len(possible_recipients)))
        recip_names = [r.name for r in recipients]
        primary     = recip_names[0] if recip_names else ""

        # Simple substitution
        sprint_n    = random.randint(1, 8)
        items       = random.sample(ctx["items"], k=min(3, len(ctx["items"])))
        milestone   = pick(ctx["milestones"])
        reason      = pick(ctx["reasons"])

        subject = template["subject_template"].format(
            project=proj_id.title(), n=sprint_n,
            topic=pick(ctx["items"]), artifact=pick(ctx["items"]),
        )

        body_raw = template["body_template"].format(
            opener=opener(style, primary),
            closer=closer(style, sender.name),
            project=proj_id.title(),
            n=sprint_n,
            item1=items[0], item2=items[1] if len(items) > 1 else items[0],
            item3=items[2] if len(items) > 2 else items[0],
            reason=reason,
            milestone=milestone,
            topic=pick(ctx["items"]),
            question=f"how we should handle {pick(ctx['items'])}",
            output=f"the {milestone} plan",
            reference=pick(ctx["items"]),
            interpretation_a=f"doing it before {milestone}",
            interpretation_b=f"deferring until next sprint",
            action=f"finalizing the {pick(ctx['items'])} approach",
            issue_description=f"We're seeing delays in {pick(ctx['items'])}",
            blocked_item=f"the {milestone} milestone",
            decision_needed=f"how to handle {pick(ctx['items'])}",
            deadline=f"day {day + random.randint(2, 7)}",
            option_a=f"Defer {pick(ctx['items'])} to next sprint",
            option_b=f"Pull in extra help from {pick(PERSONAS).name}",
            artifact=pick(ctx["items"]).replace(" ", "-"),
            feedback_focus=f"the approach to {pick(ctx['items'])}",
            artifact_slug=pick(ctx["items"]).replace(" ", "-"),
            information=f"the {pick(ctx['items'])} work is progressing well",
            context=f"the {milestone} planning",
            stakeholders=", ".join([pick(PERSONAS).name for _ in range(2)]),
            decision=f"go with option A on {pick(ctx['items'])}",
            rationale=f"it unblocks {milestone} faster",
            next_steps=f"{pick(PERSONAS).name} will drive this",
            owner=pick(PERSONAS).name,
            message=f"{pick(ctx['items'])} update — {reason}. Will share more soon.",
        )

        channel = pick(["email", "email", "slack", "slack", "slack"])

        messages.append({
            "message_id":    make_id("filler", i, day, sender.id),
            "thread_id":     make_id("thread", "filler", i),
            "day":           day,
            "timestamp":     day_to_dt(day, hour=random.randint(8, 18), minute=random.randint(0, 59)),
            "sender_id":     sender.id,
            "sender_name":   sender.name,
            "sender_email":  sender.email,
            "recipient_ids": [r.id for r in recipients],
            "recipient_names": recip_names,
            "channel":       channel,
            "subject":       subject if channel == "email" else "",
            "body":          body_raw,
            "project_tag":   proj_id if proj_id != "general" else None,
            "_ground_truth": {"is_filler": True},
        })

    return messages


# ── MAIN GENERATOR ───────────────────────────────────────────────────────────

def generate_corpus() -> Tuple[List[Dict], Dict]:
    print("🔧 OrgTrace Corpus Generator")
    print(f"   Seed: {SEED} | Days: {SIM_DAYS} | Personas: {len(PERSONAS)}")

    all_messages: List[Dict] = []

    # 1. Decision chain messages
    print("   Generating decision chains...")
    for ds in DECISION_SEEDS:
        msgs = generate_decision_chain_messages(ds)
        all_messages.extend(msgs)
        print(f"     D{ds.id}: {len(msgs)} messages (chain depth {len(ds.chain)})")

    # 2. Commitment messages
    print("   Generating commitment messages...")
    for cs in COMMITMENT_SEEDS:
        msg, followup = generate_commitment_message(cs)
        all_messages.append(msg)
        if followup:
            all_messages.append(followup)
    print(f"     {len(COMMITMENT_SEEDS)} commitments seeded ({len(DROPPED_COMMITMENT_IDS)} dropped)")

    # 3. Filler messages
    print("   Generating filler messages...")
    fillers = generate_filler_messages(n=450)
    all_messages.extend(fillers)
    print(f"     {len(fillers)} filler messages")

    # Sort by timestamp
    all_messages.sort(key=lambda m: m["timestamp"])

    # Assign sequential indices
    for i, m in enumerate(all_messages):
        m["index"] = i

    print(f"\n   ✅ Total messages: {len(all_messages)}")

    # ── GROUND TRUTH DOCUMENT ────────────────────────────────────────────────
    ground_truth = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "seed":         SEED,
            "sim_days":     SIM_DAYS,
            "total_messages": len(all_messages),
            "persona_count": len(PERSONAS),
        },

        # Task 1: Decision chains
        "decisions": {
            ds.id: {
                "title":                  ds.title,
                "root_message_day":       ds.root_day,
                "root_sender_id":         ds.root_sender_id,
                "chain_length":           len(ds.chain),
                "chain_sender_ids":       [h["sender_id"] for h in ds.chain],
                "final_decision_day":     ds.final_decision_day,
                "final_decision_maker_id":ds.final_decision_maker_id,
                "accountable_person_id":  ds.accountable_person_id,
                "project_id":             ds.project_id,
                "consequence":            ds.consequence,
                "message_ids": [
                    m["message_id"] for m in all_messages
                    if m.get("_ground_truth", {}).get("decision_id") == ds.id
                ],
            }
            for ds in DECISION_SEEDS
        },

        # Task 2: Commitments
        "commitments": {
            cs.id: {
                "speaker_id":       cs.speaker_id,
                "recipient_ids":    cs.recipient_ids,
                "day":              cs.day,
                "raw_phrase":       cs.raw_phrase,
                "type":             cs.commitment_type,
                "due_day":          cs.due_day,
                "resolved":         cs.resolved,
                "resolution_day":   cs.resolution_day,
                "risk_level":       cs.risk_level,
                "project_id":       cs.project_id,
                "message_id": next(
                    (m["message_id"] for m in all_messages
                     if m.get("_ground_truth", {}).get("commitment_id") == cs.id
                     and not m.get("_ground_truth", {}).get("is_resolution")),
                    None
                ),
            }
            for cs in COMMITMENT_SEEDS
        },

        "dropped_commitment_ids": DROPPED_COMMITMENT_IDS,

        # Task 3: Sofia's knowledge profile
        "sofia_knowledge_profile": SOFIA_KNOWLEDGE_PROFILE,

        # Project states
        "projects": {
            p.id: {
                "name":         p.name,
                "description":  p.description,
                "owner_id":     p.owner_id,
                "lead_eng_id":  p.lead_eng_id,
                "team_ids":     p.team_ids,
                "status_changes": p.status_changes,
            }
            for p in PROJECTS
        },
    }

    return all_messages, ground_truth


def save(corpus: List[Dict], ground_truth: Dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Strip _ground_truth from agent-facing corpus
    public_corpus = []
    for m in corpus:
        pub = {k: v for k, v in m.items() if k != "_ground_truth"}
        public_corpus.append(pub)

    corpus_path = OUTPUT_DIR / "corpus.json"
    gt_path     = OUTPUT_DIR / "ground_truth.json"
    full_path   = OUTPUT_DIR / "corpus_annotated.json"   # includes GT, for debugging

    with open(corpus_path, "w") as f:
        json.dump(public_corpus, f, indent=2)

    with open(gt_path, "w") as f:
        json.dump(ground_truth, f, indent=2)

    with open(full_path, "w") as f:
        json.dump(corpus, f, indent=2)

    print(f"\n   📁 Saved:")
    print(f"      {corpus_path}        ({len(public_corpus)} messages, no GT)")
    print(f"      {gt_path}   (ground truth for all 3 tasks)")
    print(f"      {full_path}  (annotated, for debugging)")


if __name__ == "__main__":
    corpus, ground_truth = generate_corpus()
    save(corpus, ground_truth)

    # Quick stats
    print("\n── CORPUS STATS ──────────────────────────────────────")
    channels = {}
    for m in corpus:
        c = m["channel"].split(":")[0]
        channels[c] = channels.get(c, 0) + 1
    for ch, cnt in sorted(channels.items()):
        print(f"   {ch:10s}: {cnt:4d} messages")

    days_with_msgs = len(set(m["day"] for m in corpus))
    print(f"   Active days : {days_with_msgs}/{SIM_DAYS}")
    print(f"   Avg/day     : {len(corpus)/days_with_msgs:.1f}")
    print("──────────────────────────────────────────────────────")
