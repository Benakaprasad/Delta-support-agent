"""
Drafts a reply grounded in retrieved historical Delta replies. The LLM is
explicitly told to imitate the PATTERN of the retrieved examples (tone,
what they ask for, whether they commit to specifics) rather than invent a
new resolution -- see src/retrieval.py's module docstring for why this
matters more than it might seem.

Deliberately never asked to invent account-specific facts (refund amounts,
confirmation numbers, rebooking details) -- Delta's own real replies almost
never do this on Twitter either; they ask the customer to DM. The prompt
makes this explicit rather than relying on the model to infer it.
"""
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config, llm_client
from src.retrieval import ReplyRetriever

_SYSTEM_PROMPT = """You are drafting a reply as Delta Air Lines' Twitter customer support team.

You will be shown real examples of how Delta has actually replied to similar customer
messages in the past. Study their pattern: tone, length, what they ask the customer for,
and crucially -- what they DON'T do.

STRICT RULES:
1. NEVER invent account-specific facts: no fabricated refund amounts, confirmation
   numbers, rebooking details, dates, or compensation figures. If the real examples ask
   the customer to DM their confirmation number, do the same rather than pretending to
   already have their account details.
2. Match Delta's real tone: brief, empathetic, professional. Twitter-length (under ~280
   characters), not a long email.
3. If the historical examples show Delta routing this type of issue to DMs, your reply
   should do the same -- don't try to resolve in-thread what Delta itself doesn't
   resolve in-thread.
4. Do not sign with an agent initials placeholder like "*XYZ" -- that's Delta's internal
   convention, not something to imitate literally.

Respond with ONLY the reply text, nothing else -- no preamble, no quotes around it."""


def generate_reply(
    message: str,
    prior_context: str = "",
    retriever: Optional[ReplyRetriever] = None,
    intent: str = "",
) -> dict:
    """Returns {"reply": str, "grounding_examples": str, "top_similarity": float}.

    Callers (src.pipeline.SupportAgent) are responsible for deciding
    WHETHER a reply should be generated at all (e.g. not_support_request
    gets no reply) -- this function always drafts something if called.
    """
    if retriever is None:
        retriever = ReplyRetriever.from_csv()

    retrieved = retriever.retrieve(message, k=5)
    top_similarity = retrieved[0]["similarity"] if retrieved else 0.0
    grounding_block = retriever.format_for_prompt(message, k=5)

    user_prompt = f"""Historical examples of how Delta replied to similar messages:
{grounding_block}

---
Customer's message now (intent: {intent or 'unknown'}): "{message}"
"""
    if prior_context:
        user_prompt += f"\nPrior thread context: {prior_context}\n"

    try:
        reply_text = llm_client.call(_SYSTEM_PROMPT, user_prompt, max_tokens=200, temperature=0.3)
        reply_text = reply_text.strip().strip('"')
    except Exception as e:
        reply_text = ""
        grounding_block = f"(reply generation failed: {e})"

    return {
        "reply": reply_text,
        "grounding_examples": grounding_block,
        "top_similarity": top_similarity,
    }


if __name__ == "__main__":
    retriever = ReplyRetriever.from_csv()
    result = generate_reply(
        "flight got cancelled and I need a refund",
        retriever=retriever,
        intent="booking_change_refund",
    )
    print("Reply:", result["reply"])
    print("Top similarity:", result["top_similarity"])
