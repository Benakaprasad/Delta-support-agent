"""
LLM-as-judge: scores the agent's drafted reply on 4 axes against the
customer message, Delta's actual historical reply (if available), and the
classified intent. Used by eval/run_eval.py, and validated against a human
via eval/human_calibration.py -- the assignment explicitly requires
evidence the judge agrees with a human, not just a judge score on its own.
"""
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import llm_client

_SYSTEM_PROMPT = """You are evaluating a draft customer-support reply for Delta Air Lines'
Twitter support agent. Score the DRAFT REPLY on 4 axes, each 1-5 (5 = best):

- groundedness: Does the reply avoid inventing account-specific facts (refund amounts,
  confirmation numbers, dates, compensation figures) it couldn't actually know? A reply
  that asks the customer to DM/follow up rather than inventing specifics should score
  HIGH here, even if it "resolves" less. A reply that confidently states a fabricated
  detail should score LOW regardless of how fluent it sounds.
- helpfulness: Does it actually move the customer's situation forward (acknowledge the
  issue, give a clear next step)? A generic non-answer scores low even if harmless.
- tone: Is it professional, empathetic, and appropriately brief for a Twitter reply
  (not curt, not overwrought)?
- correctness: Does it correctly reflect the customer's actual issue and intent, with no
  factual errors about airline policy?

Then give an "overall" score 1-5 as your holistic judgment (not simply an average).

Respond with ONLY this JSON, no other text:
{
  "groundedness": <1-5>,
  "helpfulness": <1-5>,
  "tone": <1-5>,
  "correctness": <1-5>,
  "overall": <1-5>,
  "notes": "<one short sentence on the main strength or weakness>"
}"""


def judge_reply(customer_message: str, draft_reply: str, historical_reply: str, intent: str) -> Dict:
    """Returns axis scores 1-5, or a safe all-1 fallback with an error note
    if judging itself fails -- a judge failure should surface as an
    obviously-bad score, not silently vanish from the eval."""
    user_prompt = f"""Customer message (classified intent: {intent}): "{customer_message}"

Draft reply being evaluated: "{draft_reply}"

For reference, here is a REAL historical Delta reply to a similar situation (not
necessarily to THIS exact message -- use it only as a rough tone/pattern reference,
not as the "correct answer"): "{historical_reply}"
"""
    try:
        result = llm_client.call_json(_SYSTEM_PROMPT, user_prompt, max_tokens=300)
    except Exception as e:
        return {
            "groundedness": 1, "helpfulness": 1, "tone": 1, "correctness": 1, "overall": 1,
            "notes": f"judge_error: {e}",
        }

    scores = {}
    for axis in ["groundedness", "helpfulness", "tone", "correctness", "overall"]:
        val = result.get(axis, 1)
        try:
            val = int(round(float(val)))
        except (TypeError, ValueError):
            val = 1
        scores[axis] = max(1, min(5, val))
    scores["notes"] = result.get("notes", "")
    return scores


if __name__ == "__main__":
    result = judge_reply(
        customer_message="flight got cancelled and I need a refund",
        draft_reply="Hi, sorry to hear that. Please DM your confirmation number and we'll look into a refund.",
        historical_reply="Hi Cassandra. Pls follow/DM your confirmation code or tkt # to better assist. *TKR",
        intent="booking_change_refund",
    )
    print(result)
