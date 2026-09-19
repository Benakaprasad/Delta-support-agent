"""
Two baselines the LLM agent must beat to justify its cost/complexity (per
assignment requirements):

  TrivialBaseline: always predicts the single most common intent (measured
      via regex heuristics over the training pool, since we have no true
      labels for the full 8000-thread set) and always drafts the SAME
      canned reply regardless of what the customer said. Routing is always
      ESCALATE -- the safest possible default with zero understanding of
      the message.

  KeywordBaseline: uses the same regex heuristics as the golden-set
      stratification (src/taxonomy_heuristics.py) AS the actual classifier,
      applies the same rule-based routing policy as
      eval/build_golden_set.py's draft_route(), and for a reply, returns
      the single nearest historical Delta reply via TF-IDF retrieval
      verbatim (no generation at all -- pure nearest-neighbor lookup).

Neither baseline calls an LLM, so eval/run_eval.py --skip-llm can score
both without an Ollama server running.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.taxonomy_heuristics import match_all_intents, match_rare_critical
from src.retrieval import ReplyRetriever

_CANNED_REPLY = (
    "Hi there, thanks for reaching out. Please follow/DM us your confirmation "
    "or ticket number and we'll take a look. *BOT"
)


class TrivialBaseline:
    def __init__(self, train_pool: pd.DataFrame):
        hits = train_pool["customer_message"].apply(match_all_intents)
        primary = hits.apply(lambda h: h[0] if h else "not_support_request")
        self.majority_intent = primary.value_counts().idxmax()

    def predict(self, message: str) -> dict:
        return {
            "intent": self.majority_intent,
            "route": config.ROUTE_ESCALATE,
            "reply": _CANNED_REPLY,
        }


class KeywordBaseline:
    """Reimplements the same priority-order intent pick and rule-based
    routing used to draft-label the golden set (eval/build_golden_set.py),
    so it's directly comparable: this is literally "what if we shipped the
    stratification heuristic as the product." For replies, does pure
    nearest-neighbor retrieval (no generation) via the same TF-IDF index
    the LLM pipeline uses for grounding."""

    _NEGATIVE_SENTIMENT_WORDS = [
        "worst", "terrible", "unacceptable", "ridiculous", "horrible",
        "disgust", "furious", "never fly", "awful",
    ]

    def __init__(self, retriever: ReplyRetriever = None):
        self.retriever = retriever or ReplyRetriever.from_csv()

    def _draft_intent(self, message: str) -> str:
        rare = match_rare_critical(message)
        if rare:
            return rare[0]
        hits = match_all_intents(message)
        for intent in config.INTENT_PRIORITY_ORDER:
            if intent in hits:
                return intent
        return "not_support_request"

    def _draft_route(self, message: str, intent: str) -> str:
        text = message.lower()
        if intent in config.CRITICAL_OVERRIDE_INTENTS:
            return config.ROUTE_ESCALATE
        if intent in config.NOOP_INTENTS:
            return config.ROUTE_AUTO_HANDLE_NOOP
        if intent in config.AUTO_HANDLE_ELIGIBLE_INTENTS:
            return config.ROUTE_AUTO_HANDLE
        if intent == "service_complaint_general" and any(w in text for w in self._NEGATIVE_SENTIMENT_WORDS):
            return config.ROUTE_ESCALATE
        return config.ROUTE_ESCALATE

    def predict(self, message: str) -> dict:
        intent = self._draft_intent(message)
        route_val = self._draft_route(message, intent)

        reply = ""
        if route_val != config.ROUTE_AUTO_HANDLE_NOOP:
            retrieved = self.retriever.retrieve(message, k=1)
            reply = retrieved[0]["historical_delta_reply"] if retrieved else _CANNED_REPLY

        return {"intent": intent, "route": route_val, "reply": reply}


if __name__ == "__main__":
    train_pool = pd.read_csv(config.CLEAN_THREADS_CSV)
    trivial = TrivialBaseline(train_pool)
    keyword = KeywordBaseline()
    test_msg = "my bag never showed up in ATL and nobody will help"
    print("Trivial:", trivial.predict(test_msg))
    print("Keyword:", keyword.predict(test_msg))
