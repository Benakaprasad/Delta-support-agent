"""
Central configuration: brand, intent taxonomy, routing policy, file paths.
Everything else in this repo imports from here rather than hardcoding these
choices, so the taxonomy/policy is defined in exactly one place.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Brand
# ---------------------------------------------------------------------------
# Exact author_id string used for Delta's replies in the raw twcs.csv dataset.
BRAND_AUTHOR_ID = "Delta"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
EVAL_DIR = ROOT_DIR / "eval"

RAW_TWCS_CSV = DATA_RAW_DIR / "twcs.csv"
CLEAN_THREADS_CSV = DATA_PROCESSED_DIR / "delta_threads_clean.csv"

GOLDEN_CANDIDATES_CSV = EVAL_DIR / "golden_candidates_draft.csv"
GOLDEN_SET_CSV = EVAL_DIR / "golden_set.csv"
JUDGE_OUTPUT_CSV = EVAL_DIR / "judge_scores.csv"
HUMAN_CALIBRATION_CSV = EVAL_DIR / "human_calibration.csv"

# ---------------------------------------------------------------------------
# Subsampling (data_prep.py)
# ---------------------------------------------------------------------------
SUBSAMPLE_SIZE = 8000
RANDOM_SEED = 42
MAX_PRIOR_CONTEXT_TURNS = 2

# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------
# Built bottom-up by reading ~120 real Delta customer messages, then
# stress-tested against a larger keyword-matched sample. See
# report/decision_log.md for how/why each category earned its place.
INTENTS = [
    "delay_cancellation_rebooking",
    "baggage_lost_item",
    "booking_change_refund",
    "checkin_seating_boarding",
    "loyalty_skymiles_upgrades",
    "service_complaint_general",
    "policy_info_question",
    "compliment_praise",
    "not_support_request",
]

# Rare-but-critical sub-intents. NOT part of the 9-way primary taxonomy --
# too low-frequency (~0.2-0.7% base rate) to earn a top-level bucket -- but
# tracked separately because they trigger a hard escalation override
# regardless of which of the 9 intents the message also matches.
CRITICAL_OVERRIDE_INTENTS = [
    "accessibility_special_assistance",
    "account_security_pii",
]

# When a message plausibly matches multiple of the 9 intents (~19% of real
# traffic), the classifier picks ONE primary intent using this priority
# order (earlier = higher priority). A stated, defensible design choice
# rather than an arbitrary tie-break -- concrete actionable issues outrank
# compliments, operational/time-sensitive issues outrank generic ones.
INTENT_PRIORITY_ORDER = [
    "delay_cancellation_rebooking",
    "baggage_lost_item",
    "booking_change_refund",
    "checkin_seating_boarding",
    "loyalty_skymiles_upgrades",
    "service_complaint_general",
    "policy_info_question",
    "compliment_praise",
    "not_support_request",
]

# ---------------------------------------------------------------------------
# Routing policy
# ---------------------------------------------------------------------------
ROUTE_AUTO_HANDLE = "AUTO_HANDLE"
ROUTE_AUTO_HANDLE_NOOP = "AUTO_HANDLE_NOOP"  # not_support_request: no reply needed
ROUTE_ESCALATE = "ESCALATE"

# Only these intents are EVER eligible for auto-handling. Everything else
# escalates by default. Deliberately conservative: Delta's own historical
# replies show Twitter is triage, not resolution -- anything account-
# specific gets pushed to DMs in real Delta replies, so an agent that
# confidently resolves those in-thread would be fabricating.
AUTO_HANDLE_ELIGIBLE_INTENTS = {
    "compliment_praise",
    "policy_info_question",
}

NOOP_INTENTS = {"not_support_request"}

# Classifier confidence below this always forces escalation, independent of
# intent eligibility (gate #2 of the 3-gate routing policy).
MIN_CONFIDENCE_FOR_AUTO_HANDLE = 0.75

# Keyword triggers for the cross-cutting hard override (gate #1, checked
# BEFORE intent/confidence gates). Deliberately broad/recall-oriented: a
# false positive here just costs an unnecessary escalation; a false
# negative means a credentialed/accessibility/safety case gets
# auto-handled, which is the costliest possible mistake this system can
# make.
HARD_OVERRIDE_KEYWORDS = {
    "accessibility_special_assistance": [
        "wheelchair", "special assist", "mobility", "disab",
    ],
    "account_security_pii": [
        "password", "user name", "username", "login", "log in",
        "hacked", "compromised", "credit card number", "verify my identity",
    ],
    "safety_incident": [
        "emergency", "medical emergency", "injured", "injury", "assault",
        "threat", "unsafe", "smoke in the cabin", "evacuat",
    ],
    "legal_regulatory": [
        "lawsuit", "lawyer", "attorney", "sue you", "dot complaint",
        "department of transportation", "legal action",
    ],
    "minor_traveling_alone": [
        "unaccompanied minor", "traveling alone", "travelling alone",
        "minor alone",
    ],
}


def get_intent_priority_rank(intent: str) -> int:
    """Lower = higher priority. Used to pick a primary intent among several
    matches. Unknown intents sort last."""
    try:
        return INTENT_PRIORITY_ORDER.index(intent)
    except ValueError:
        return len(INTENT_PRIORITY_ORDER)
