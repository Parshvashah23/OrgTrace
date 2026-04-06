"""
OrgTrace — Message Templates
Style-aware templates for generating realistic messages.
Each template maps (comm_style, message_type) → list of format strings.
"""

from typing import Dict, List

# ── STYLE-AWARE OPENERS ──────────────────────────────────────────────────────

OPENERS: Dict[str, List[str]] = {
    "terse":    ["Hey,", "Hi,", "FYI —", "Quick note:", ""],
    "verbose":  ["Hi {name},", "Hello {name},", "Hope you're doing well, {name}.", "Hey {name}, hope the week is going well —"],
    "formal":   ["Dear {name},", "Hello {name},", "Good morning {name},", "Hi {name},"],
    "casual":   ["Hey {name}! 👋", "Yo {name} —", "Hey! Quick thing —", "Hey {name}, hope you're good!"],
    "technical":["Hi {name},", "Hey {name} —", "Quick technical note:"],
}

CLOSERS: Dict[str, List[str]] = {
    "terse":    ["", "Thanks.", "– {sender_first}"],
    "verbose":  ["Let me know if you have any questions or need more context. Happy to jump on a call.", "Looking forward to your thoughts!", "Thanks for your time on this."],
    "formal":   ["Best regards,\n{sender_name}", "Kind regards,\n{sender_name}", "Thank you,\n{sender_name}"],
    "casual":   ["lmk what you think! 🙏", "let me know!", "thanks!! 😊", "cheers ✌️"],
    "technical":["Let me know if the above is unclear or if you need the raw config.", "Happy to pair on this if needed.", "Ping me if you need clarification."],
}

# ── COMMITMENT PHRASES (by type) ─────────────────────────────────────────────
# These are the raw implied commitment phrases seeded into messages.
# The generator substitutes these verbatim to ensure grader can find them.

COMMITMENT_PHRASE_PATTERNS: Dict[str, List[str]] = {
    "follow_up":   [
        "I'll get back to you on {topic} by {timeframe}",
        "I'll follow up on {topic} {timeframe}",
        "Let me look into {topic} and come back to you",
        "I'll check on {topic} and ping you",
    ],
    "meeting":     [
        "We should sync on {topic} — {timeframe} work for you?",
        "Let's set up a quick call to discuss {topic}",
        "Can we do a quick {timeframe} sync on {topic}?",
        "Shall we block some time to go over {topic}?",
    ],
    "deliverable": [
        "I'll have {deliverable} ready by {timeframe}",
        "I'll send you {deliverable} {timeframe}",
        "I'll put together {deliverable} and share it",
        "I'll get {deliverable} over to you {timeframe}",
    ],
    "decision":    [
        "I'll make a call on {topic} by {timeframe}",
        "Let me decide on {topic} and let you know",
        "I'll get alignment on {topic} and come back with a decision",
    ],
    "intro":       [
        "I'll loop in {person} on this",
        "Let me intro you to {person}",
        "I'll cc {person} — they should be part of this conversation",
        "I'll bring {person} into the thread",
    ],
}

# ── CONTEXTUAL MESSAGE BODIES (for non-seeded filler messages) ───────────────
# Used by the generator to pad the corpus with realistic noise.

FILLER_TEMPLATES: List[Dict] = [
    # Status updates
    {
        "type": "status_update",
        "subject_template": "Re: {project} — Sprint {n} update",
        "body_template": (
            "{opener}\n\n"
            "Quick update on {project} — we wrapped up sprint {n} yesterday. "
            "Main completed items: {item1}, {item2}. "
            "We're carrying over {item3} to next sprint due to {reason}. "
            "Overall still on track for the {milestone} milestone.\n\n"
            "{closer}"
        ),
    },
    # Meeting request
    {
        "type": "meeting_request",
        "subject_template": "Quick sync on {topic}?",
        "body_template": (
            "{opener}\n\n"
            "Do you have 30 min this week to chat about {topic}? "
            "I want to get your input on {question} before we finalize {output}.\n\n"
            "{closer}"
        ),
    },
    # Question/clarification
    {
        "type": "question",
        "subject_template": "Question on {topic}",
        "body_template": (
            "{opener}\n\n"
            "Quick question — when you mentioned {reference} in the last meeting, "
            "did you mean {interpretation_a} or {interpretation_b}? "
            "Trying to make sure I'm aligned before I proceed with {action}.\n\n"
            "{closer}"
        ),
    },
    # Escalation
    {
        "type": "escalation",
        "subject_template": "⚠️ Blocking issue on {project}",
        "body_template": (
            "{opener}\n\n"
            "Flagging a potential blocker on {project}. "
            "{issue_description}. "
            "This is blocking {blocked_item} and we need a decision on {decision_needed} "
            "by {deadline} or we'll slip the {milestone} milestone.\n\n"
            "Options as I see them:\n"
            "1. {option_a}\n"
            "2. {option_b}\n\n"
            "Happy to discuss — what do you prefer?\n\n"
            "{closer}"
        ),
    },
    # Slack-style short message
    {
        "type": "slack_short",
        "subject_template": "",
        "body_template": "{message}",
    },
    # Review request
    {
        "type": "review_request",
        "subject_template": "Review request: {artifact}",
        "body_template": (
            "{opener}\n\n"
            "Can you take a look at {artifact}? "
            "Main thing I want feedback on: {feedback_focus}. "
            "Targeting to finalize by {deadline}.\n\n"
            "Link: [notion/{artifact_slug}]\n\n"
            "{closer}"
        ),
    },
    # FYI / info share
    {
        "type": "fyi",
        "subject_template": "FYI: {topic}",
        "body_template": (
            "{opener}\n\n"
            "Just wanted to flag — {information}. "
            "No action needed from you right now, but good to be aware for {context}.\n\n"
            "{closer}"
        ),
    },
    # Decision announcement
    {
        "type": "decision_announcement",
        "subject_template": "Decision: {topic}",
        "body_template": (
            "{opener}\n\n"
            "After discussion with {stakeholders}, we've decided to go with {decision}. "
            "Rationale: {rationale}. "
            "Next steps: {next_steps}. "
            "{owner} will be driving this forward.\n\n"
            "{closer}"
        ),
    },
]

# ── SLACK CHANNEL DEFINITIONS ────────────────────────────────────────────────

SLACK_CHANNELS = [
    {"id": "general",         "name": "#general",          "members": "all"},
    {"id": "engineering",     "name": "#engineering",      "members": "engineering+leadership"},
    {"id": "atlas-eng",       "name": "#atlas-eng",        "members": "atlas_team"},
    {"id": "beacon-team",     "name": "#beacon-team",      "members": "beacon_team"},
    {"id": "crest-compliance","name": "#crest-compliance", "members": "crest_team"},
    {"id": "product",         "name": "#product",          "members": "product+leadership"},
    {"id": "design",          "name": "#design",           "members": "design+product"},
    {"id": "sales",           "name": "#sales",            "members": "sales+leadership"},
    {"id": "legal-private",   "name": "#legal-private",    "members": "legal+leadership"},
    {"id": "hr-internal",     "name": "#hr-internal",      "members": "hr+leadership"},
    {"id": "random",          "name": "#random",           "members": "all"},
    {"id": "announcements",   "name": "#announcements",    "members": "all"},
]
