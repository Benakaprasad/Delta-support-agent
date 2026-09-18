"""
Retrieval-grounding layer: given a new customer message, finds the most
similar historical customer messages (via TF-IDF + cosine similarity) and
returns Delta's actual past replies to them.

This is what "grounded reply" means in this project: the reply generator
never freewheels an answer -- it's shown 3-5 real examples of how Delta
actually responded to similar situations and asked to draft something
consistent with that pattern, not to invent a new one. See
report/decision_log.md for why this matters more than it might seem: it's
the difference between "plausible-sounding" and "actually representative
of the brand's real behavior."
"""
import re
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config

# Strip tokens that are lexically "rare" (and so get high IDF weight) but
# carry no semantic content about WHAT the issue is -- flight numbers,
# confirmation codes, airport codes, raw dates. Left in, these dominate
# cosine similarity (e.g. two totally unrelated complaints that happen to
# both mention "ATL" score as more similar than two baggage complaints
# using different phrasing). This was a real bug caught by manually
# inspecting retrieval output, not a hypothetical -- see decision log.
_FLIGHT_NUM_RE = re.compile(r"\b(DL|dl)\s?\d{2,4}\b")
_AIRPORT_CODE_RE = re.compile(r"\b[A-Z]{3}\b")  # must run before lowercasing
_NUMERIC_TOKEN_RE = re.compile(r"\b\d+\b")


def preprocess_for_retrieval(text: str) -> str:
    """Strips flight numbers / airport codes / raw numbers so TF-IDF
    similarity is driven by the actual content of the complaint, not by
    incidental route/flight-number overlap."""
    if not isinstance(text, str):
        return ""
    text = _FLIGHT_NUM_RE.sub(" ", text)
    text = _AIRPORT_CODE_RE.sub(" ", text)
    text = _NUMERIC_TOKEN_RE.sub(" ", text)
    return text


class ReplyRetriever:
    """Wraps a fitted TF-IDF index over historical customer messages so
    reply generation can look up 'what did Delta actually say to people in
    situations like this one' at inference time."""

    def __init__(self, threads_df: pd.DataFrame):
        self.threads_df = threads_df.reset_index(drop=True)
        self._retrieval_texts = self.threads_df["customer_message"].apply(preprocess_for_retrieval)
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        self._matrix = self.vectorizer.fit_transform(self._retrieval_texts)

    @classmethod
    def from_csv(cls, path: Path = None) -> "ReplyRetriever":
        path = path or config.CLEAN_THREADS_CSV
        df = pd.read_csv(path)
        return cls(df)

    def retrieve(self, query_message: str, k: int = 5) -> List[Dict]:
        """Returns the top-k most similar historical threads, each as a
        dict with the historical customer_message, Delta's actual reply,
        and the cosine similarity score. Ordered by descending similarity.
        """
        if not isinstance(query_message, str) or not query_message.strip():
            return []
        query_clean = preprocess_for_retrieval(query_message)
        query_vec = self.vectorizer.transform([query_clean])
        sims = cosine_similarity(query_vec, self._matrix).flatten()
        top_idx = sims.argsort()[::-1][:k]

        results = []
        for idx in top_idx:
            score = float(sims[idx])
            if score <= 0.0:
                continue  # no lexical overlap at all -- not a useful match
            row = self.threads_df.iloc[idx]
            results.append({
                "similarity": score,
                "historical_customer_message": row["customer_message"],
                "historical_delta_reply": row["delta_reply"],
                "thread_id": row.get("thread_id", None),
            })
        return results

    def format_for_prompt(self, query_message: str, k: int = 5) -> str:
        """Formats retrieved examples as a numbered block ready to drop
        into an LLM prompt. Returns a note if nothing similar was found --
        this matters for the reply generator, which should NOT fabricate
        confidence when there's genuinely no grounding available."""
        examples = self.retrieve(query_message, k=k)
        if not examples:
            return "(No sufficiently similar historical examples found.)"
        lines = []
        for i, ex in enumerate(examples, 1):
            lines.append(
                f"{i}. [similarity={ex['similarity']:.2f}] "
                f"Customer said: \"{ex['historical_customer_message']}\"\n"
                f"   Delta replied: \"{ex['historical_delta_reply']}\""
            )
        return "\n".join(lines)


if __name__ == "__main__":
    # Quick manual smoke test.
    retriever = ReplyRetriever.from_csv()
    test_query = "my bag never showed up in ATL and nobody will help"
    print(f"Query: {test_query}\n")
    print(retriever.format_for_prompt(test_query, k=3))
