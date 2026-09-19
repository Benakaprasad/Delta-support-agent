"""
LLM intent classifier. Given a customer message (+ optional prior thread
context), returns a primary intent from the 9-way taxonomy (or a
rare-critical sub-intent), a confidence score, and which rare-critical
keywords/patterns the message plausibly touches.

This is deliberately an LLM call, not the regex heuristics in
taxonomy_heuristics.py -- those exist only for golden-set stratification.
Real messages vary in phrasing far more than fixed patterns can capture
(see decision log), which is exactly the gap an LLM classifier is meant to
close.
"""
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, llm_client

_SYSTEM_PROMPT = f"""You are an intent classifier for Delta Air Lines' customer support Twitter channel.

Classify the customer's message into EXACTLY ONE of these primary intents:
{chr(10).join(f"- {intent}" for intent in config.INTENTS)}

Definitions:
- delay_cancellation_rebooking: flight delayed, cancelled, missed connection, needs rebooking
- baggage_lost_item: lost, delayed, or damaged baggage; item left on plane
- booking_change_refund: refund request, credit, change/cancel a reservation, billing/fee dispute
- checkin_seating_boarding: check-in problems, seat assignment, boarding, gate, upgrade standby
- loyalty_skymiles_upgrades: SkyMiles, MQM/MQD, Medallion status, points/miles issues
- service_complaint_general: general dissatisfaction, rudeness, hold times -- NOT tied to one fixable transaction above
- policy_info_question: factual question about Delta policy, no problem to resolve
- compliment_praise: positive feedback, thanks, no resolution needed
- not_support_request: not actually a support request -- ambient chatter, travel photos, hashtag noise, rhetorical statements

ALSO check whether the message touches either of these rare-but-critical sub-intents
(these can co-occur with a primary intent above -- report them separately):
- accessibility_special_assistance: wheelchair, mobility, disability-related assistance
- account_security_pii: password/login/account-access/credential issues, requests to look up someone's account by email/password

If the message plausibly fits multiple primary intents, choose the ONE that represents
the most concrete, actionable core of the message (an operational issue like a delay or
lost bag outranks a general complaint about it; a general complaint outranks a policy
question; a policy question outranks a compliment).

Respond with ONLY this JSON structure, no other text:
{{
  "intent": "<one of the 9 primary intents above>",
  "confidence": <float 0.0-1.0, how confident you are in this primary intent>,
  "rare_critical_flags": [<zero or more of: "accessibility_special_assistance", "account_security_pii">],
  "reasoning": "<one short sentence>"
}}"""


def classify(message: str, prior_context: str = "") -> Dict:
    """Returns {"intent", "confidence", "rare_critical_flags", "reasoning"}.
    Falls back to a safe default (not_support_request, confidence 0.0) if
    the LLM response can't be parsed -- a parse failure should never crash
    the pipeline, and a 0.0 confidence guarantees the router escalates
    rather than silently auto-handling on bad data."""
    user_prompt = f"Customer message: \"{message}\""
    if prior_context:
        user_prompt = f"Prior context: {prior_context}\n\n{user_prompt}"

    try:
        result = llm_client.call_json(_SYSTEM_PROMPT, user_prompt, max_tokens=300)
    except Exception as e:
        return {
            "intent": "not_support_request",
            "confidence": 0.0,
            "rare_critical_flags": [],
            "reasoning": f"classifier_error: {e}",
        }

    intent = result.get("intent", "not_support_request")
    if intent not in config.INTENTS:
        intent = "not_support_request"

    confidence = result.get("confidence", 0.0)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0

    rare_flags = result.get("rare_critical_flags", []) or []
    rare_flags = [f for f in rare_flags if f in config.CRITICAL_OVERRIDE_INTENTS]

    return {
        "intent": intent,
        "confidence": confidence,
        "rare_critical_flags": rare_flags,
        "reasoning": result.get("reasoning", ""),
    }


if __name__ == "__main__":
    tests = [
        "my bag never showed up in ATL and nobody will help",
        "thanks for the great flight crew today!",
        "just landed in Cancun!! #vacation",
        "I need a wheelchair for my connecting flight, can someone confirm this is arranged?",
    ]
    for t in tests:
        print(t, "->", classify(t))
