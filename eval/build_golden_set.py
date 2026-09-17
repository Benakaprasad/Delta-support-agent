"""
Builds the golden-set CANDIDATE pool: 200 stratified examples with draft
labels from regex heuristics, flagged for human review.

Stratification (see report/decision_log.md for why):
    130 random              -- natural class balance, including noise
    35  ambiguous_multi_intent -- messages matching 2+ intent patterns
    20  rare_critical        -- accessibility / account-security signal
    15  hard_negative        -- sentiment language, no clean intent match

Draft labels are a STARTING POINT, not ground truth. Every row must be
reviewed and corrected by hand before eval/golden_set.csv is considered
"hand-labelled" -- see decision log for a concrete example of what goes
wrong if that step is skipped.

Usage:
    python -m eval.build_golden_set
Writes: eval/golden_candidates_draft.csv
"""
import ast
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.taxonomy_heuristics import (
    match_all_intents,
    match_rare_critical,
    is_hard_negative_candidate,
)

RANDOM_SEED = config.RANDOM_SEED

N_RANDOM = 130
N_AMBIGUOUS = 35
N_RARE_CRITICAL = 20
N_HARD_NEGATIVE = 15

NEGATIVE_SENTIMENT_WORDS = [
    "worst", "terrible", "unacceptable", "ridiculous", "horrible",
    "disgust", "furious", "never fly", "awful",
]


def sample_pool(pool: pd.DataFrame, n: int, seed: int, used_ids: set) -> pd.DataFrame:
    pool = pool[~pool["thread_id"].isin(used_ids)]
    n = min(n, len(pool))
    sampled = pool.sample(n, random_state=seed)
    used_ids.update(sampled["thread_id"].tolist())
    return sampled


def draft_intent(row) -> str:
    """Priority: rare-critical > primary intents in INTENT_PRIORITY_ORDER >
    not_support_request fallback."""
    if row["rare_hits"]:
        return row["rare_hits"][0]
    for intent in config.INTENT_PRIORITY_ORDER:
        if intent in row["hits"]:
            return intent
    return "not_support_request"


def draft_route(row, intent: str) -> tuple:
    text = str(row["customer_message"]).lower()
    if row["rare_hits"]:
        return config.ROUTE_ESCALATE, "cross-cutting override: accessibility/account-security always escalates"
    if intent in config.NOOP_INTENTS:
        return config.ROUTE_AUTO_HANDLE_NOOP, "not an actionable support request, no reply needed"
    if intent in config.AUTO_HANDLE_ELIGIBLE_INTENTS:
        if intent == "compliment_praise":
            return config.ROUTE_AUTO_HANDLE, "positive feedback, low risk, template ack sufficient"
        return config.ROUTE_AUTO_HANDLE, "factual/policy question, answerable without account access"
    if intent == "service_complaint_general" and any(w in text for w in NEGATIVE_SENTIMENT_WORDS):
        return config.ROUTE_ESCALATE, "strong negative sentiment, human judgment on compensation likely needed"
    return config.ROUTE_ESCALATE, "intent requires account/PNR-specific lookup or operational action beyond generic reply"


def main():
    df = pd.read_csv(config.CLEAN_THREADS_CSV)

    df["hits"] = df["customer_message"].apply(match_all_intents)
    df["n_hits"] = df["hits"].apply(len)
    df["rare_hits"] = df["customer_message"].apply(match_rare_critical)
    df["is_hard_neg"] = df["customer_message"].apply(is_hard_negative_candidate)

    used_ids = set()

    random_pool = sample_pool(df, N_RANDOM, seed=1, used_ids=used_ids)

    multi_pool = df[df["n_hits"] >= 2]
    ambiguous = sample_pool(multi_pool, N_AMBIGUOUS, seed=2, used_ids=used_ids)

    rare_pool = df[df["rare_hits"].apply(len) > 0]
    rare = sample_pool(rare_pool, N_RARE_CRITICAL, seed=3, used_ids=used_ids)
    print(f"Rare-critical pool available: {len(rare_pool)} -> sampled {len(rare)}")

    hardneg_pool = df[df["is_hard_neg"]]
    hardneg = sample_pool(hardneg_pool, N_HARD_NEGATIVE, seed=4, used_ids=used_ids)
    print(f"Hard-negative pool available: {len(hardneg_pool)} -> sampled {len(hardneg)}")

    golden = pd.concat([
        random_pool.assign(sample_bucket="random"),
        ambiguous.assign(sample_bucket="ambiguous_multi_intent"),
        rare.assign(sample_bucket="rare_critical"),
        hardneg.assign(sample_bucket="hard_negative"),
    ], ignore_index=True)

    golden = golden.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)  # shuffle
    golden.insert(0, "golden_id", [f"G{i+1:03d}" for i in range(len(golden))])

    print(f"\nTotal golden candidates: {len(golden)}")
    print(golden["sample_bucket"].value_counts())

    rows = []
    for _, row in golden.iterrows():
        intent = draft_intent(row)
        route, reason = draft_route(row, intent)
        needs_review = (
            row["sample_bucket"] in ("ambiguous_multi_intent", "rare_critical", "hard_negative")
            or row["n_hits"] == 0
        )
        rows.append({
            "golden_id": row["golden_id"],
            "sample_bucket": row["sample_bucket"],
            "customer_message": row["customer_message"],
            "prior_context": row.get("prior_context", ""),
            "historical_delta_reply": row["delta_reply"],
            "draft_primary_intent": intent,
            "draft_all_signal_hits": row["hits"],
            "draft_route": route,
            "draft_route_reason": reason,
            "needs_human_review": needs_review,
            # empty columns for you to fill by hand -- these become the
            # ground truth eval/run_eval.py actually scores against
            "final_intent": "",
            "final_route": "",
            "your_notes": "",
        })

    out = pd.DataFrame(rows)
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.GOLDEN_CANDIDATES_CSV, index=False)
    print(f"\nSaved {config.GOLDEN_CANDIDATES_CSV}, {len(out)} rows")
    print(f"Rows flagged needs_human_review: {out['needs_human_review'].sum()}")
    print("\nDraft intent distribution:")
    print(out["draft_primary_intent"].value_counts())
    print("\nDraft route distribution:")
    print(out["draft_route"].value_counts())
    print(f"\nNext step: open {config.GOLDEN_CANDIDATES_CSV.name}, review EVERY row, "
          f"fill in final_intent/final_route, save as {config.GOLDEN_SET_CSV.name}")


if __name__ == "__main__":
    main()
