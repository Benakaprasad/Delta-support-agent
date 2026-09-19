"""
Routing policy: auto-handle vs escalate, with a stated reason. Three
independent gates, checked in order -- ANY gate can force escalation, none
can override an escalation decided by an earlier gate:

  Gate 1 (cross-cutting hard override): keyword-based, deterministic, checks
      the raw message text against config.HARD_OVERRIDE_KEYWORDS regardless
      of what the classifier says. Accessibility, account-security, safety,
      legal, minors-alone -- always escalate. This is rule-based rather than
      LLM-based on purpose: the cost of a false negative here (missing a
      real accessibility/security case) is much higher than the cost of a
      false positive (an unnecessary escalation), so recall-oriented regex
      is the right tool even though it's cruder than the LLM classifier.

  Gate 2 (classifier confidence): if the LLM classifier wasn't confident
      about the intent, don't act on it -- escalate regardless of what the
      guessed intent was.

  Gate 3 (intent eligibility + grounding confidence): only compliment_praise
      and policy_info_question are ever auto-handle eligible, and even a
      policy_info_question needs a well-grounded retrieval match (real
      historical precedent) to be trusted for auto-handling. Anything else,
      or a policy question with weak retrieval grounding, escalates.
      not_support_request routes to a special no-op (no reply needed at all,
      not an escalation and not a handled reply).
"""
import re
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config


def _check_hard_override(message: str) -> Optional[str]:
    """Returns the override category name if the message trips any
    cross-cutting hard-override keyword pattern, else None."""
    if not isinstance(message, str):
        return None
    text = message.lower()
    for category, keywords in config.HARD_OVERRIDE_KEYWORDS.items():
        for kw in keywords:
            if re.search(kw, text, re.I):
                return category
    return None


def route(message: str, classifier_result: Dict, top_retrieval_similarity: float = 0.0) -> Dict:
    """Returns {"route", "reason", "gate_triggered"}.

    classifier_result: output of src.classifier.classify()
    top_retrieval_similarity: best TF-IDF cosine similarity found by
        src.retrieval.ReplyRetriever for this message (0.0 if none/unknown)
    """
    intent = classifier_result.get("intent", "not_support_request")
    confidence = classifier_result.get("confidence", 0.0)
    rare_flags = classifier_result.get("rare_critical_flags", [])

    # Gate 1: cross-cutting hard override. Two independent signals, checked
    # separately so the stated reason is honest about provenance:
    #   (a) deterministic keyword match on the raw text -- high confidence
    #   (b) classifier-reported rare_critical_flags -- an LLM guess, kept
    #       as a second, more cautious signal because a smaller local model
    #       can occasionally hallucinate these (observed in testing -- see
    #       decision log). We still escalate on it rather than ignore it,
    #       because a false positive here is cheap (unnecessary escalation)
    #       and a false negative is not (a real accessibility/security case
    #       silently auto-handled) -- but the reason string says "the model
    #       flagged this" rather than implying a literal keyword was found.
    override_category = _check_hard_override(message)
    if override_category:
        return {
            "route": config.ROUTE_ESCALATE,
            "reason": f"deterministic keyword override: \'{override_category}\' pattern "
                      f"matched in message text -- always escalates regardless of intent",
            "gate_triggered": "hard_override_keyword",
        }
    if rare_flags:
        return {
            "route": config.ROUTE_ESCALATE,
            "reason": f"classifier flagged rare-critical sub-intent(s) {rare_flags} "
                      f"(LLM-reported, not independently keyword-verified) -- escalating "
                      f"out of caution per policy: a false positive here is cheap, a false "
                      f"negative is not",
            "gate_triggered": "hard_override_classifier_flag",
        }

    # not_support_request is a no-op regardless of confidence -- there's
    # nothing to auto-handle OR escalate, the classifier is just noting
    # this isn't actionable at all.
    if intent in config.NOOP_INTENTS:
        return {
            "route": config.ROUTE_AUTO_HANDLE_NOOP,
            "reason": "not an actionable support request, no reply needed",
            "gate_triggered": "noop_intent",
        }

    # Gate 2: classifier confidence
    if confidence < config.MIN_CONFIDENCE_FOR_AUTO_HANDLE:
        return {
            "route": config.ROUTE_ESCALATE,
            "reason": f"classifier confidence {confidence:.2f} below threshold "
                      f"{config.MIN_CONFIDENCE_FOR_AUTO_HANDLE} -- too uncertain to auto-act",
            "gate_triggered": "low_confidence",
        }

    # Gate 3: intent eligibility + grounding confidence
    if intent not in config.AUTO_HANDLE_ELIGIBLE_INTENTS:
        return {
            "route": config.ROUTE_ESCALATE,
            "reason": f"intent '{intent}' is not auto-handle eligible -- "
                      f"likely needs account/PNR-specific lookup or operational action",
            "gate_triggered": "intent_ineligible",
        }

    if intent == "policy_info_question" and top_retrieval_similarity < config.MIN_SIMILARITY_FOR_GROUNDED_REPLY:
        return {
            "route": config.ROUTE_ESCALATE,
            "reason": f"policy question but no well-grounded historical precedent "
                      f"(similarity {top_retrieval_similarity:.2f} < {config.MIN_SIMILARITY_FOR_GROUNDED_REPLY}) "
                      f"-- risk of confidently answering with fabricated policy detail",
            "gate_triggered": "weak_grounding",
        }

    return {
        "route": config.ROUTE_AUTO_HANDLE,
        "reason": f"intent '{intent}' is auto-handle eligible, classifier confident, "
                  f"grounding adequate" if intent == "policy_info_question"
                  else f"intent '{intent}' is low-risk, template-appropriate",
        "gate_triggered": "auto_handle_eligible",
    }


if __name__ == "__main__":
    print(route("thanks for the great crew today!", {"intent": "compliment_praise", "confidence": 0.95, "rare_critical_flags": []}))
    print(route("I need a wheelchair confirmed for my flight", {"intent": "checkin_seating_boarding", "confidence": 0.9, "rare_critical_flags": ["accessibility_special_assistance"]}))
    print(route("my bag is lost", {"intent": "baggage_lost_item", "confidence": 0.9, "rare_critical_flags": []}))
