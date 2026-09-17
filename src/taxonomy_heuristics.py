"""
Regex heuristics for the 9-intent taxonomy + 2 rare-critical sub-intents.

IMPORTANT: these patterns are NOT the classifier. The real classifier
(src/classifier.py) is an LLM call. These regexes exist for exactly one
purpose: stratified sampling when building the golden set
(eval/build_golden_set.py) -- finding a pool of "probably ambiguous",
"probably rare-critical", and "probably hard-negative" candidates to
oversample from, since random sampling alone would barely surface the rare
cases (~0.2-0.7% base rate) or the ~19% multi-intent cases.

Using keyword matching AS the classifier was tried and rejected during
taxonomy design: ~35-40% of real messages don't hit any keyword even though
they clearly fit one of the 9 intents (e.g. "2 hours late" instead of
"delayed"), which just reflects that phrasing varies more than fixed
patterns capture -- exactly why the actual classifier is an LLM call, not
this.
"""
import re
from typing import Dict, List

# ---------------------------------------------------------------------------
# Primary 9-intent taxonomy patterns
# ---------------------------------------------------------------------------
INTENT_PATTERNS: Dict[str, str] = {
    "delay_cancellation_rebooking": (
        r"\bdelay|late\b|cancel|cancell|missed (my )?(flight|connection)|"
        r"reschedul|rebook|running late|held on the (runway|tarmac)|"
        r"sat on (the )?(runway|tarmac)"
    ),
    "baggage_lost_item": (
        r"\bbag(gage)?\b|luggage|suitcase|lost (my|it|her|his)|"
        r"left (my|it|her|his) on|checked bag|carry.?on|stroller"
    ),
    "booking_change_refund": (
        r"\brefund|credit|money back|change (my|the) (flight|ticket|reservation)|"
        r"reservation|book(ing)?|cancel my (flight|ticket)|\bfee\b|charged|compensat"
    ),
    "checkin_seating_boarding": (
        r"\bcheck.?in|seat assign|boarding|upgrade list|standby|gate\b|"
        r"middle seat|comfort\+|first class|economy|seat error|choose seats|"
        r"can't (choose|select) (my |a )?seat"
    ),
    "loyalty_skymiles_upgrades": (
        r"skymiles|sky miles|mqm|mqd|medallion|platinum|diamond|"
        r"gold status|silver status|elite|\bmiles\b|\bpoints\b"
    ),
    "service_complaint_general": (
        r"worst (airline|customer service|experience)|rude|hold time|on hold|"
        r"call.?back|customer service|disappoint|unacceptable|ridiculous|"
        r"horrible|terrible|wait time|understaff|call center|"
        r"two hours to|garbage|owe me"
    ),
    "policy_info_question": (
        r"\bhow do i|how can i|is there a|what is the|can i bring|policy|"
        r"sky ?club|weather|wifi|do you have|quick question|"
        r"international (or domestic)?terminal|which terminal"
    ),
    "compliment_praise": (
        r"\bthank(s| you)|great (service|flight|crew|staff)|awesome|amazing|"
        r"love (flying|delta)|kudos|shout ?out|appreciat|"
        r"you'?re the best|tysm|bearable|welcoming"
    ),
    # not_support_request deliberately has NO pattern: it's the "matched
    # nothing else" bucket -- ambient chatter, travel photos, hashtag noise.
    # See report/decision_log.md for why this is a real intent, not a gap.
}

# ---------------------------------------------------------------------------
# Rare-but-critical sub-intents (see config.CRITICAL_OVERRIDE_INTENTS)
# ---------------------------------------------------------------------------
RARE_CRITICAL_PATTERNS: Dict[str, str] = {
    "accessibility_special_assistance": r"wheelchair|special assist|mobility|disab",
    "account_security_pii": (
        r"password|user ?name|login|log in|"
        r"account.*(hack|compromis|access)|credit card number|verify my identity"
    ),
}

# ---------------------------------------------------------------------------
# Hard-negative signal: messages with sentiment-ish language that DON'T hit
# any primary pattern -- these look like they need a reply but often don't
# (rhetorical complaints, sarcasm, ambiguous fragments). Used to oversample
# the golden set's "hard_negative" bucket.
# ---------------------------------------------------------------------------
HARD_NEGATIVE_SIGNAL_PATTERN = (
    r"thank|great|love|awesome|worst|terrible|ridiculous|disappoint"
)


def match_all_intents(text: str) -> List[str]:
    """Return every primary intent whose pattern matches. Used to find
    multi-intent candidates for the golden set's ambiguous bucket, and to
    measure genuine multi-label overlap (~19% of traffic) vs. keyword-
    matching noise."""
    if not isinstance(text, str):
        return []
    t = text.lower()
    return [intent for intent, pat in INTENT_PATTERNS.items() if re.search(pat, t, re.I)]


def match_rare_critical(text: str) -> List[str]:
    """Return every rare-critical sub-intent whose pattern matches."""
    if not isinstance(text, str):
        return []
    t = text.lower()
    return [intent for intent, pat in RARE_CRITICAL_PATTERNS.items() if re.search(pat, t, re.I)]


def is_hard_negative_candidate(text: str) -> bool:
    """True if the message has sentiment-ish language but matches no
    primary intent pattern -- a candidate for the golden set's
    hard_negative stratum."""
    if not isinstance(text, str):
        return False
    if match_all_intents(text):
        return False
    return bool(re.search(HARD_NEGATIVE_SIGNAL_PATTERN, text, re.I))
