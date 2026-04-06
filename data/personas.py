"""
OrgTrace — Synthetic Persona Definitions
40 employees at Meridian Labs (B2B SaaS, ~Series B)
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class Team(str, Enum):
    ENGINEERING = "Engineering"
    PRODUCT = "Product"
    SALES = "Sales"
    LEGAL = "Legal"
    HR = "HR"
    LEADERSHIP = "Leadership"
    DESIGN = "Design"


class CommStyle(str, Enum):
    TERSE = "terse"           # short, direct, few words
    VERBOSE = "verbose"       # long explanations, context-heavy
    FORMAL = "formal"         # professional, structured
    CASUAL = "casual"         # informal, emoji-prone
    TECHNICAL = "technical"   # jargon-heavy, precise


@dataclass
class Persona:
    id: str
    name: str
    email: str
    team: Team
    role: str
    seniority: int            # 1=junior, 2=mid, 3=senior, 4=lead/manager, 5=exec
    comm_style: CommStyle
    expertise: List[str]      # domains they own
    reports_to: Optional[str] = None   # persona id
    direct_reports: List[str] = field(default_factory=list)
    slack_handle: str = ""
    departure_day: Optional[int] = None  # if set, they leave on this day (Task 3)

    def __post_init__(self):
        if not self.slack_handle:
            self.slack_handle = self.name.lower().replace(" ", ".")


PERSONAS: List[Persona] = [

    # ── LEADERSHIP (4) ──────────────────────────────────────────────────────
    Persona("P01", "Diana Marsh",    "diana@meridian.io",   Team.LEADERSHIP, "CEO",             5, CommStyle.FORMAL,    ["strategy","fundraising","company_vision"],       reports_to=None),
    Persona("P02", "Carlos Vega",    "carlos@meridian.io",  Team.LEADERSHIP, "CTO",             5, CommStyle.TECHNICAL, ["architecture","technical_strategy","security"],  reports_to="P01"),
    Persona("P03", "Priya Nair",     "priya@meridian.io",   Team.LEADERSHIP, "CPO",             5, CommStyle.VERBOSE,   ["product_strategy","roadmap","customer_insight"],  reports_to="P01"),
    Persona("P04", "James Okafor",   "james@meridian.io",   Team.LEADERSHIP, "CFO",             5, CommStyle.FORMAL,    ["finance","compliance","legal_oversight"],         reports_to="P01"),

    # ── ENGINEERING (10) ────────────────────────────────────────────────────
    Persona("P05", "Ravi Shankar",   "ravi@meridian.io",    Team.ENGINEERING, "Eng Manager",    4, CommStyle.TERSE,     ["team_management","sprint_planning","hiring"],     reports_to="P02"),
    Persona("P06", "Sofia Reyes",    "sofia@meridian.io",   Team.ENGINEERING, "Sr Engineer",    3, CommStyle.TECHNICAL, ["backend","auth_service","database_migrations"],   reports_to="P05",
            departure_day=45),   # ← DEPARTING EMPLOYEE for Task 3
    Persona("P07", "Liam Chen",      "liam@meridian.io",    Team.ENGINEERING, "Sr Engineer",    3, CommStyle.CASUAL,    ["frontend","design_system","performance"],         reports_to="P05"),
    Persona("P08", "Amara Diallo",   "amara@meridian.io",   Team.ENGINEERING, "Engineer",       2, CommStyle.VERBOSE,   ["data_pipeline","etl","reporting"],                reports_to="P05"),
    Persona("P09", "Noah Kim",       "noah@meridian.io",    Team.ENGINEERING, "Engineer",       2, CommStyle.TERSE,     ["devops","ci_cd","infrastructure"],                reports_to="P05"),
    Persona("P10", "Elena Popov",    "elena@meridian.io",   Team.ENGINEERING, "Engineer",       2, CommStyle.TECHNICAL, ["api_design","integrations","webhooks"],           reports_to="P05"),
    Persona("P11", "Tariq Hassan",   "tariq@meridian.io",   Team.ENGINEERING, "Engineer",       2, CommStyle.CASUAL,    ["mobile","ios","push_notifications"],              reports_to="P05"),
    Persona("P12", "Yuki Tanaka",    "yuki@meridian.io",    Team.ENGINEERING, "Jr Engineer",    1, CommStyle.VERBOSE,   ["testing","qa","test_automation"],                 reports_to="P05"),
    Persona("P13", "Ben Foster",     "ben@meridian.io",     Team.ENGINEERING, "Jr Engineer",    1, CommStyle.CASUAL,    ["frontend","accessibility"],                       reports_to="P07"),
    Persona("P14", "Ingrid Holm",    "ingrid@meridian.io",  Team.ENGINEERING, "Security Eng",   3, CommStyle.FORMAL,    ["security_audits","pen_testing","compliance_tech"],reports_to="P02"),

    # ── PRODUCT (8) ─────────────────────────────────────────────────────────
    Persona("P15", "Marcus Webb",    "marcus@meridian.io",  Team.PRODUCT, "Head of Product",   4, CommStyle.VERBOSE,   ["roadmap_execution","stakeholder_mgmt","metrics"], reports_to="P03"),
    Persona("P16", "Fatima Al-Zahra","fatima@meridian.io",  Team.PRODUCT, "Sr PM",              3, CommStyle.FORMAL,    ["project_atlas","user_research","kpis"],           reports_to="P15"),
    Persona("P17", "Owen Blake",     "owen@meridian.io",    Team.PRODUCT, "Sr PM",              3, CommStyle.CASUAL,    ["project_beacon","growth","onboarding"],           reports_to="P15"),
    Persona("P18", "Zara Singh",     "zara@meridian.io",    Team.PRODUCT, "PM",                 2, CommStyle.VERBOSE,   ["project_crest","compliance_features","legal_pm"], reports_to="P15"),
    Persona("P19", "Hugo Laurent",   "hugo@meridian.io",    Team.PRODUCT, "PM",                 2, CommStyle.TERSE,     ["analytics","dashboards","reporting_features"],    reports_to="P15"),
    Persona("P20", "Mei Ling",       "mei@meridian.io",     Team.PRODUCT, "PM",                 2, CommStyle.CASUAL,    ["integrations","marketplace","partners"],          reports_to="P15"),
    Persona("P21", "Derek Stone",    "derek@meridian.io",   Team.PRODUCT, "Jr PM",              1, CommStyle.VERBOSE,   ["user_feedback","support_insights"],               reports_to="P16"),
    Persona("P22", "Aisha Mensah",   "aisha@meridian.io",   Team.PRODUCT, "Jr PM",              1, CommStyle.CASUAL,    ["mobile_pm","app_store"],                          reports_to="P17"),

    # ── DESIGN (4) ──────────────────────────────────────────────────────────
    Persona("P23", "Leo Martinez",   "leo@meridian.io",     Team.DESIGN, "Design Lead",         4, CommStyle.CASUAL,    ["design_system","brand","ux_strategy"],            reports_to="P03"),
    Persona("P24", "Nadia Petrov",   "nadia@meridian.io",   Team.DESIGN, "Sr Designer",         3, CommStyle.VERBOSE,   ["product_design","prototyping","user_testing"],    reports_to="P23"),
    Persona("P25", "Finn O'Brien",   "finn@meridian.io",    Team.DESIGN, "Designer",            2, CommStyle.CASUAL,    ["marketing_design","landing_pages"],               reports_to="P23"),
    Persona("P26", "Layla Hassan",   "layla@meridian.io",   Team.DESIGN, "Designer",            2, CommStyle.TERSE,     ["iconography","illustration","motion"],            reports_to="P23"),

    # ── SALES (7) ───────────────────────────────────────────────────────────
    Persona("P27", "Rachel Kim",     "rachel@meridian.io",  Team.SALES, "VP Sales",             4, CommStyle.FORMAL,    ["enterprise_sales","revenue","forecasting"],       reports_to="P01"),
    Persona("P28", "Anton Volkov",   "anton@meridian.io",   Team.SALES, "Account Exec",         3, CommStyle.VERBOSE,   ["enterprise_accounts","demos","negotiations"],     reports_to="P27"),
    Persona("P29", "Chloe Martin",   "chloe@meridian.io",   Team.SALES, "Account Exec",         3, CommStyle.CASUAL,    ["smb_accounts","trials","churn_prevention"],       reports_to="P27"),
    Persona("P30", "David Osei",     "david@meridian.io",   Team.SALES, "SDR",                  2, CommStyle.FORMAL,    ["prospecting","outbound","lead_qualification"],    reports_to="P27"),
    Persona("P31", "Sara Johansson", "sara@meridian.io",    Team.SALES, "SDR",                  1, CommStyle.CASUAL,    ["inbound_leads","demos"],                          reports_to="P27"),
    Persona("P32", "Kai Nakamura",   "kai@meridian.io",     Team.SALES, "Sales Engineer",       3, CommStyle.TECHNICAL, ["technical_demos","poc","integrations_sales"],     reports_to="P27"),
    Persona("P33", "Bianca Russo",   "bianca@meridian.io",  Team.SALES, "Customer Success",     3, CommStyle.VERBOSE,   ["onboarding","retention","upsell"],                reports_to="P27"),

    # ── LEGAL (3) ───────────────────────────────────────────────────────────
    Persona("P34", "Victor Stein",   "victor@meridian.io",  Team.LEGAL, "General Counsel",      4, CommStyle.FORMAL,    ["contracts","gdpr","ip","litigation"],             reports_to="P04"),
    Persona("P35", "Mia Johansson",  "mia@meridian.io",     Team.LEGAL, "Legal Counsel",        3, CommStyle.FORMAL,    ["vendor_contracts","employment_law","privacy"],    reports_to="P34"),
    Persona("P36", "Tom Bradley",    "tom@meridian.io",     Team.LEGAL, "Paralegal",            2, CommStyle.TERSE,     ["contract_review","filing","compliance_tracking"], reports_to="P34"),

    # ── HR (4) ──────────────────────────────────────────────────────────────
    Persona("P37", "Grace Liu",      "grace@meridian.io",   Team.HR, "HR Director",             4, CommStyle.FORMAL,    ["hiring","culture","performance","offboarding"],   reports_to="P01"),
    Persona("P38", "Kwame Asante",   "kwame@meridian.io",   Team.HR, "HR Manager",              3, CommStyle.VERBOSE,   ["recruiting","onboarding","benefits"],             reports_to="P37"),
    Persona("P39", "Iris Novak",     "iris@meridian.io",    Team.HR, "Recruiter",               2, CommStyle.CASUAL,    ["technical_recruiting","sourcing"],                reports_to="P38"),
    Persona("P40", "Samuel Park",    "samuel@meridian.io",  Team.HR, "HR Coordinator",          1, CommStyle.FORMAL,    ["admin","scheduling","compliance_hr"],             reports_to="P38"),
]

# Quick lookup dicts
PERSONA_BY_ID   = {p.id: p for p in PERSONAS}
PERSONA_BY_EMAIL = {p.email: p for p in PERSONAS}

# The departing employee (Task 3 ground truth subject)
DEPARTING_EMPLOYEE = next(p for p in PERSONAS if p.departure_day is not None)
