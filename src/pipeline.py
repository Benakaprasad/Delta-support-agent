"""
End-to-end agent: wires classifier -> router -> reply_generator into one
call. This is the "system" eval/run_eval.py scores against the two
baselines.
"""
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.classifier import classify
from src.router import route
from src.reply_generator import generate_reply
from src.retrieval import ReplyRetriever


class SupportAgent:
    def __init__(self, retriever: Optional[ReplyRetriever] = None):
        self.retriever = retriever or ReplyRetriever.from_csv()

    def handle(self, message: str, prior_context: str = "") -> dict:
        """Returns {"intent", "confidence", "route", "route_reason", "reply",
        "grounding_examples", "top_similarity"} -- the full trace of what
        the agent decided and why, not just the final answer. Keeping the
        intermediate reasoning visible is deliberate: a routing decision
        with no stated reason isn't auditable, and this assignment's whole
        point is proving the system is trustworthy, not just that it works.
        """
        classifier_result = classify(message, prior_context)

        # Look up retrieval similarity once, used by both the router's
        # weak-grounding gate and (if we proceed) the reply generator.
        retrieved = self.retriever.retrieve(message, k=5)
        top_similarity = retrieved[0]["similarity"] if retrieved else 0.0

        route_result = route(message, classifier_result, top_retrieval_similarity=top_similarity)

        reply = ""
        grounding_examples = ""
        if route_result["route"] != config.ROUTE_AUTO_HANDLE_NOOP:
            # Draft a reply even for ESCALATE cases -- a human agent can
            # use/edit the draft rather than starting from a blank page.
            # Only true no-ops (not_support_request) skip this entirely.
            reply_result = generate_reply(
                message, prior_context, retriever=self.retriever,
                intent=classifier_result["intent"],
            )
            reply = reply_result["reply"]
            grounding_examples = reply_result["grounding_examples"]

        return {
            "intent": classifier_result["intent"],
            "confidence": classifier_result["confidence"],
            "rare_critical_flags": classifier_result["rare_critical_flags"],
            "route": route_result["route"],
            "route_reason": route_result["reason"],
            "gate_triggered": route_result["gate_triggered"],
            "reply": reply,
            "grounding_examples": grounding_examples,
            "top_similarity": top_similarity,
        }


if __name__ == "__main__":
    agent = SupportAgent()
    result = agent.handle("my bag never showed up in ATL and nobody will help")
    for k, v in result.items():
        print(f"{k}: {v}")
