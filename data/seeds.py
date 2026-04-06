"""
OrgTrace — Project Definitions & Seeded Ground Truth
These seeds are the SOURCE OF TRUTH for all three graders.
The generator will weave these facts into the message corpus.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


# ── PROJECTS ────────────────────────────────────────────────────────────────

@dataclass
class Project:
    id: str
    name: str
    description: str
    owner_id: str           # PM who owns it
    lead_eng_id: str        # engineering lead
    team_ids: List[str]     # all involved persona ids
    start_day: int
    status_changes: List[Dict]   # [{day, status, reason}]


PROJECTS = [
    Project(
        id="atlas",
        name="Project Atlas",
        description="Core product v2 rewrite — new multi-tenant architecture",
        owner_id="P16",
        lead_eng_id="P06",   # Sofia — the departing employee
        team_ids=["P05","P06","P07","P08","P09","P10","P15","P16","P23","P24"],
        start_day=1,
        status_changes=[
            {"day": 12, "status": "on_track",  "reason": "Sprint 1 completed"},
            {"day": 28, "status": "at_risk",   "reason": "Auth service scope creep"},
            {"day": 35, "status": "blocked",   "reason": "DB migration decision unresolved"},
            {"day": 42, "status": "on_track",  "reason": "Migration path decided"},
            {"day": 55, "status": "at_risk",   "reason": "Sofia leaving, knowledge gap"},
        ]
    ),
    Project(
        id="beacon",
        name="Project Beacon",
        description="New AI-powered onboarding flow for SMB customers",
        owner_id="P17",
        lead_eng_id="P07",
        team_ids=["P07","P11","P13","P17","P22","P24","P25"],
        start_day=8,
        status_changes=[
            {"day": 20, "status": "on_track",   "reason": "Designs approved"},
            {"day": 38, "status": "at_risk",    "reason": "Mobile scope unclear"},
            {"day": 50, "status": "on_track",   "reason": "Scope locked"},
        ]
    ),
    Project(
        id="crest",
        name="Project Crest",
        description="SOC2 Type II compliance certification",
        owner_id="P18",
        lead_eng_id="P14",
        team_ids=["P14","P18","P34","P35","P36","P09","P04"],
        start_day=3,
        status_changes=[
            {"day": 15, "status": "on_track",  "reason": "Gap analysis complete"},
            {"day": 30, "status": "blocked",   "reason": "Legal review of vendor contracts pending"},
            {"day": 44, "status": "on_track",  "reason": "Vendor contracts signed"},
            {"day": 58, "status": "at_risk",   "reason": "Audit finding in auth logs"},
        ]
    ),
]

PROJECT_BY_ID = {p.id: p for p in PROJECTS}


# ── SEEDED DECISIONS (Ground Truth for Task 1) ───────────────────────────────
# Each decision is a causal chain of messages.
# The grader checks if the agent traced the chain correctly.

@dataclass
class DecisionSeed:
    id: str
    title: str
    root_day: int               # day the ROOT cause message was sent
    root_sender_id: str
    chain: List[Dict]           # ordered list of {day, sender_id, summary, is_root, is_decision}
    final_decision_day: int
    final_decision_maker_id: str
    accountable_person_id: str  # who is responsible for follow-through
    project_id: str
    consequence: str            # what happened BECAUSE of this decision


DECISION_SEEDS = [
    DecisionSeed(
        id="D01",
        title="PostgreSQL → CockroachDB migration decision",
        root_day=18,
        root_sender_id="P08",   # Amara raised scaling concern
        chain=[
            {"day": 18, "sender_id": "P08", "summary": "Amara raises concern: current Postgres setup won't scale beyond 10M rows for Atlas multi-tenant", "is_root": True,  "is_decision": False},
            {"day": 19, "sender_id": "P06", "summary": "Sofia acknowledges, says she'll investigate CockroachDB vs Citus", "is_root": False, "is_decision": False},
            {"day": 22, "sender_id": "P06", "summary": "Sofia shares analysis: CockroachDB preferred for geo-distribution but 3x cost", "is_root": False, "is_decision": False},
            {"day": 24, "sender_id": "P02", "summary": "Carlos asks for cost breakdown before deciding", "is_root": False, "is_decision": False},
            {"day": 27, "sender_id": "P04", "summary": "James (CFO) says budget not available for CockroachDB this quarter", "is_root": False, "is_decision": False},
            {"day": 29, "sender_id": "P06", "summary": "Sofia proposes Citus as compromise — same Postgres base, horizontal sharding", "is_root": False, "is_decision": False},
            {"day": 31, "sender_id": "P02", "summary": "Carlos approves Citus. Sofia to own migration plan.", "is_root": False, "is_decision": True},
        ],
        final_decision_day=31,
        final_decision_maker_id="P02",
        accountable_person_id="P06",
        project_id="atlas",
        consequence="Atlas was blocked on day 35 because migration plan was only in Sofia's head — never documented after the decision."
    ),
    DecisionSeed(
        id="D02",
        title="Beacon mobile scope: native vs React Native",
        root_day=14,
        root_sender_id="P17",
        chain=[
            {"day": 14, "sender_id": "P17", "summary": "Owen asks Tariq: can we build Beacon mobile in React Native or do we need native?", "is_root": True, "is_decision": False},
            {"day": 15, "sender_id": "P11", "summary": "Tariq says React Native fine for MVP but push notifications unreliable on Android", "is_root": False, "is_decision": False},
            {"day": 16, "sender_id": "P22", "summary": "Aisha shares customer research: 70% of SMB users on Android", "is_root": False, "is_decision": False},
            {"day": 17, "sender_id": "P03", "summary": "Priya (CPO) says push notifications are non-negotiable for onboarding completion rates", "is_root": False, "is_decision": False},
            {"day": 21, "sender_id": "P05", "summary": "Ravi says no native iOS engineer available until Q3", "is_root": False, "is_decision": False},
            {"day": 23, "sender_id": "P17", "summary": "Owen decides: ship React Native for now, revisit native in Q3. Tariq to implement custom push handler.", "is_root": False, "is_decision": True},
        ],
        final_decision_day=23,
        final_decision_maker_id="P17",
        accountable_person_id="P11",
        project_id="beacon",
        consequence="The custom push handler was never specced out. Beacon mobile had a silent 40% push delivery failure in production on day 52."
    ),
    DecisionSeed(
        id="D03",
        title="SOC2 vendor contract review process",
        root_day=8,
        root_sender_id="P18",
        chain=[
            {"day":  8, "sender_id": "P18", "summary": "Zara asks Victor (Legal): do all 3rd-party vendors need SOC2 contracts reviewed?", "is_root": True, "is_decision": False},
            {"day":  9, "sender_id": "P34", "summary": "Victor says yes, all vendors with data access need DPA review — estimated 3 weeks", "is_root": False, "is_decision": False},
            {"day": 11, "sender_id": "P18", "summary": "Zara asks: can we prioritize top 5 vendors to unblock the audit?", "is_root": False, "is_decision": False},
            {"day": 12, "sender_id": "P35", "summary": "Mia (Legal Counsel) agrees to prioritize but needs vendor list from Zara by Friday", "is_root": False, "is_decision": False},
            {"day": 14, "sender_id": "P18", "summary": "Zara sends vendor list. Asks Mia to confirm receipt.", "is_root": False, "is_decision": False},
            {"day": 20, "sender_id": "P36", "summary": "Tom (Paralegal) starts reviews. Notes Stripe and AWS DPAs already signed.", "is_root": False, "is_decision": False},
            {"day": 30, "sender_id": "P34", "summary": "Victor flags: Crest is blocked — DataDog and Segment DPAs still unsigned. Needs CFO approval on DataDog pricing addendum.", "is_root": False, "is_decision": True},
        ],
        final_decision_day=30,
        final_decision_maker_id="P34",
        accountable_person_id="P04",
        project_id="crest",
        consequence="Crest was blocked for 14 days (day 30–44) because CFO approval on DataDog addendum was never formally requested — it was buried in a thread."
    ),
]

DECISION_BY_ID = {d.id: d for d in DECISION_SEEDS}


# ── SEEDED COMMITMENTS (Ground Truth for Task 2) ─────────────────────────────
# 40 total: 25 followed-up, 15 dropped (the ones agents need to find)

@dataclass
class CommitmentSeed:
    id: str
    day: int
    speaker_id: str             # who made the commitment
    recipient_ids: List[str]
    raw_phrase: str             # exact implied commitment phrase
    commitment_type: str        # "follow_up" | "meeting" | "deliverable" | "decision" | "intro"
    due_day: Optional[int]      # explicit or implied deadline
    resolved: bool              # True = followed up, False = DROPPED
    resolution_day: Optional[int]
    resolution_summary: Optional[str]
    risk_level: str             # "critical" | "high" | "medium" | "low"
    project_id: Optional[str]


COMMITMENT_SEEDS = [

    # ── DROPPED COMMITMENTS (15) — agents must find these ──────────────────

    CommitmentSeed("C01", 6,  "P06", ["P05"],        "I'll write up the auth service design doc this week",              "deliverable", 11, False, None, None, "critical", "atlas"),
    CommitmentSeed("C02", 13, "P17", ["P15"],        "Let me loop in Priya on the Beacon scope before we finalize",      "intro",       16, False, None, None, "high",     "beacon"),
    CommitmentSeed("C03", 19, "P04", ["P34"],        "I'll get back to you on the DataDog addendum by end of week",      "follow_up",   23, False, None, None, "critical", "crest"),
    CommitmentSeed("C04", 21, "P08", ["P06"],        "I'll send you the updated data model for the migration",           "deliverable", 25, False, None, None, "high",     "atlas"),
    CommitmentSeed("C05", 25, "P05", ["P12"],        "We should set up a proper QA process for Atlas — sync Thursday?",  "meeting",     28, False, None, None, "medium",   "atlas"),
    CommitmentSeed("C06", 27, "P11", ["P17"],        "I'll have the push notification POC ready for review by Monday",   "deliverable", 31, False, None, None, "high",     "beacon"),
    CommitmentSeed("C07", 30, "P16", ["P21"],        "Can you compile the user feedback from last sprint? I need it for the roadmap review", "deliverable", 33, False, None, None, "medium", "atlas"),
    CommitmentSeed("C08", 33, "P14", ["P18"],        "I'll send over the pen test report findings relevant to Crest",    "deliverable", 37, False, None, None, "critical", "crest"),
    CommitmentSeed("C09", 36, "P28", ["P27"],        "I'll follow up with Acme Corp on the enterprise contract renewal", "follow_up",   40, False, None, None, "high",     None),
    CommitmentSeed("C10", 38, "P07", ["P13"],        "Let's pair on the design system tokens migration next week",       "meeting",     43, False, None, None, "low",      None),
    CommitmentSeed("C11", 41, "P03", ["P16"],        "I'll review the Atlas Q3 success metrics and send feedback",       "deliverable", 45, False, None, None, "high",     "atlas"),
    CommitmentSeed("C12", 43, "P09", ["P06"],        "I'll update the CI/CD pipeline config for the Citus migration",   "deliverable", 47, False, None, None, "critical", "atlas"),
    CommitmentSeed("C13", 47, "P37", ["P06"],        "I'll send you the offboarding checklist by tomorrow",              "deliverable", 48, False, None, None, "high",     None),
    CommitmentSeed("C14", 50, "P32", ["P28"],        "I'll put together the technical requirements doc for Acme",        "deliverable", 54, False, None, None, "high",     None),
    CommitmentSeed("C15", 53, "P02", ["P05"],        "We need to do a proper knowledge transfer from Sofia — I'll schedule it", "meeting", 56, False, None, None, "critical", "atlas"),

    # ── RESOLVED COMMITMENTS (25) — should NOT be flagged as dropped ───────

    CommitmentSeed("C16",  3, "P16", ["P05"],        "I'll share the Atlas PRD with engineering by end of day",         "deliverable",  4, True,  4,  "PRD shared via Notion link",                    "medium", "atlas"),
    CommitmentSeed("C17",  5, "P09", ["P05"],        "I'll set up the staging environment this week",                   "deliverable",  8, True,  8,  "Staging env live, link shared in #atlas-eng",   "medium", "atlas"),
    CommitmentSeed("C18",  7, "P23", ["P24"],        "Let's review the new design system components together on Friday", "meeting",      9, True,  9,  "Design review held, notes in Notion",           "low",    None),
    CommitmentSeed("C19", 10, "P34", ["P35"],        "I'll forward you the Segment DPA for review",                     "deliverable", 11, True, 11,  "DPA forwarded",                                 "medium", "crest"),
    CommitmentSeed("C20", 12, "P05", ["P06"],        "Can we do a quick sync on the auth service scope tomorrow?",      "meeting",     13, True, 13,  "Sync happened, notes shared",                   "medium", "atlas"),
    CommitmentSeed("C21", 15, "P27", ["P28"],        "I'll send you the Q2 forecast template by Monday",               "deliverable", 17, True, 17,  "Template sent via email",                       "low",    None),
    CommitmentSeed("C22", 17, "P38", ["P39"],        "Let's sync on the open engineering roles tomorrow morning",       "meeting",     18, True, 18,  "Recruiting sync held",                          "low",    None),
    CommitmentSeed("C23", 18, "P06", ["P08"],        "I'll review your data model PR by Thursday",                      "deliverable", 20, True, 20,  "PR reviewed, comments left",                    "medium", "atlas"),
    CommitmentSeed("C24", 20, "P35", ["P36"],        "Can you start the Stripe and AWS DPA review this week?",          "deliverable", 22, True, 22,  "Both DPAs reviewed and signed",                 "medium", "crest"),
    CommitmentSeed("C25", 22, "P17", ["P22"],        "Can you get me the Android user data from the survey by Friday?", "deliverable", 24, True, 24,  "Data sent, summarized in Beacon doc",           "medium", "beacon"),
    CommitmentSeed("C26", 24, "P07", ["P13"],        "I'll share the Figma component library access with you",          "deliverable", 25, True, 25,  "Access granted",                                "low",    None),
    CommitmentSeed("C27", 26, "P16", ["P15"],        "I'll update the Atlas milestone tracker after the sprint review", "deliverable", 28, True, 28,  "Tracker updated in Jira",                       "medium", "atlas"),
    CommitmentSeed("C28", 28, "P14", ["P02"],        "I'll have the initial security gap analysis ready by Monday",     "deliverable", 31, True, 31,  "Gap analysis delivered via email",              "high",   "crest"),
    CommitmentSeed("C29", 29, "P33", ["P27"],        "I'll prepare the churn risk report for the board meeting",        "deliverable", 32, True, 32,  "Report prepared and sent",                      "medium", None),
    CommitmentSeed("C30", 31, "P18", ["P34"],        "I'll send you the updated vendor list with data classification",  "deliverable", 33, True, 33,  "Vendor list sent with DPA status",              "high",   "crest"),
    CommitmentSeed("C31", 33, "P08", ["P09"],        "I'll document the ETL pipeline config changes",                   "deliverable", 35, True, 35,  "Documented in Confluence",                      "medium", "atlas"),
    CommitmentSeed("C32", 35, "P29", ["P33"],        "I'll send you the onboarding feedback from the TechCorp trial",  "deliverable", 37, True, 37,  "Feedback doc shared",                           "medium", None),
    CommitmentSeed("C33", 37, "P01", ["P04"],        "I'll review the board deck numbers before Thursday",              "deliverable", 39, True, 39,  "Numbers confirmed over email",                  "high",   None),
    CommitmentSeed("C34", 39, "P24", ["P23"],        "I'll have the updated onboarding screens ready for review Friday","deliverable", 41, True, 41,  "Screens uploaded to Figma",                     "medium", "beacon"),
    CommitmentSeed("C35", 40, "P36", ["P35"],        "I'll compile the compliance checklist for the DataDog review",   "deliverable", 42, True, 42,  "Checklist compiled and shared",                 "medium", "crest"),
    CommitmentSeed("C36", 42, "P10", ["P16"],        "I'll have the webhook retry logic spec ready by end of sprint",  "deliverable", 45, True, 45,  "Spec merged into Atlas tech doc",               "medium", "atlas"),
    CommitmentSeed("C37", 44, "P15", ["P03"],        "I'll schedule the Q3 roadmap review for next week",              "meeting",     46, True, 46,  "Meeting scheduled and held",                    "medium", None),
    CommitmentSeed("C38", 46, "P39", ["P38"],        "I'll send you the shortlist of backend candidates by Wednesday", "deliverable", 48, True, 48,  "Shortlist of 4 candidates sent",                "medium", None),
    CommitmentSeed("C39", 49, "P06", ["P05"],        "I'll document the Citus shard key decisions before I leave",     "deliverable", 52, True, 52,  "Doc created in Confluence — partial only",      "critical","atlas"),
    CommitmentSeed("C40", 51, "P20", ["P10"],        "I'll intro you to the Zapier partnership contact this week",     "intro",       53, True, 53,  "Intro email sent",                              "low",    None),
]

COMMITMENT_BY_ID = {c.id: c for c in COMMITMENT_SEEDS}
DROPPED_COMMITMENT_IDS = [c.id for c in COMMITMENT_SEEDS if not c.resolved]


# ── SOFIA'S KNOWLEDGE PROFILE (Ground Truth for Task 3) ─────────────────────

SOFIA_KNOWLEDGE_PROFILE = {
    "person_id": "P06",
    "name": "Sofia Reyes",
    "departure_day": 45,
    "systems_owned": [
        {
            "name": "Auth Service",
            "description": "JWT-based authentication with refresh token rotation. Sofia designed the token invalidation strategy.",
            "critical_knowledge": "The 15-minute token expiry was a deliberate security decision (day 6) after a pen test finding. Not documented anywhere.",
            "related_decision_ids": ["D01"],
        },
        {
            "name": "Citus Migration",
            "description": "Postgres → Citus sharding migration for Atlas multi-tenant. Sofia owns the shard key design.",
            "critical_knowledge": "Shard key is tenant_id + created_at composite. Single-tenant queries must always include tenant_id or they'll do full table scans.",
            "related_decision_ids": ["D01"],
        },
        {
            "name": "Database Migration Scripts",
            "description": "Custom migration tooling built on top of Flyway. Sofia wrote the rollback procedures.",
            "critical_knowledge": "Migration V018 has a known issue with NULL handling in legacy accounts — Sofia has a hotfix but never merged it.",
            "related_decision_ids": [],
        },
    ],
    "key_relationships": [
        {"person_id": "P08", "nature": "Sofia relied on Amara for data modeling input on all schema decisions"},
        {"person_id": "P02", "nature": "Sofia had direct escalation access to Carlos for architectural decisions, bypassing Ravi"},
        {"person_id": "P14", "nature": "Sofia and Ingrid had a standing weekly sync on security implications of auth changes"},
    ],
    "decisions_owned": ["D01"],
    "implied_commitments_dropped": ["C01", "C04", "C12"],
    "undocumented_facts": [
        "The 15-min JWT expiry was a security decision, not a default",
        "Migration V018 has an unmerged hotfix for NULL handling",
        "Shard key design requires tenant_id in all queries",
        "Sofia had a verbal agreement with Carlos to do a second Citus performance review in Q4",
    ],
}
