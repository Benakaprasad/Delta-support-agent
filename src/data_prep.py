"""
Pairs each Delta reply in the raw twcs.csv with the customer tweet it
answered, attaches up to MAX_PRIOR_CONTEXT_TURNS turns of prior context,
filters to English, dedupes, and takes a seeded subsample.

Usage:
    python -m src.data_prep

Writes: data/processed/delta_threads_clean.csv
Columns: thread_id, customer_tweet_id, delta_tweet_id, customer_message,
         delta_reply, prior_context, created_at
"""
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config

# langdetect raises on empty/garbage input and is non-deterministic across
# runs unless seeded -- seed it for reproducibility.
from langdetect import detect, DetectorFactory, LangDetectException
DetectorFactory.seed = 0

MENTION_RE = re.compile(r"^(@\S+\s*)+")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip leading @mentions (noise for grounding/classification, not
    signal) and normalize whitespace. Keeps URLs and body text as-is."""
    if not isinstance(text, str):
        return ""
    text = MENTION_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def is_english(text: str) -> bool:
    """Very short strings make langdetect unreliable and are usually noise
    anyway (emoji-only, single word) -- treat as non-English rather than
    risk a bad detection."""
    if not text or len(text) < 8:
        return False
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def build_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """Index by tweet_id for fast O(1)-ish lookups when walking reply
    chains backwards."""
    return df.set_index("tweet_id", drop=False)


def get_prior_context(lookup: pd.DataFrame, start_tweet_id, max_turns: int) -> str:
    """Walk backwards via in_response_to_tweet_id from the customer's tweet,
    collecting up to max_turns prior turns (oldest first). Returns them
    joined as a single string, or "" if there's no prior context."""
    turns = []
    current_id = start_tweet_id
    for _ in range(max_turns):
        if pd.isna(current_id) or current_id not in lookup.index:
            break
        row = lookup.loc[current_id]
        # set_index with drop=False + non-unique safety: .loc can return a
        # DataFrame if tweet_id somehow duplicated -- guard against that.
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        text = clean_text(row["text"])
        if text:
            speaker = "brand" if row["author_id"] == config.BRAND_AUTHOR_ID else "customer"
            turns.append(f"[{speaker}] {text}")
        current_id = row["in_response_to_tweet_id"]
    turns.reverse()  # oldest first
    return " | ".join(turns)


def main():
    print(f"Loading {config.RAW_TWCS_CSV} ...")
    df = pd.read_csv(
        config.RAW_TWCS_CSV,
        dtype={
            "tweet_id": "int64",
            "author_id": "str",
            "text": "str",
        },
    )
    df["inbound"] = df["inbound"].astype(str) == "True"
    print(f"Loaded {len(df):,} total rows")

    lookup = build_lookup(df)

    delta_replies = df[(df["inbound"] == False) & (df["author_id"] == config.BRAND_AUTHOR_ID)]
    print(f"Found {len(delta_replies):,} Delta reply tweets")

    rows = []
    skipped_no_customer_tweet = 0
    skipped_not_inbound = 0

    for _, reply in tqdm(delta_replies.iterrows(), total=len(delta_replies), desc="Pairing threads"):
        customer_tweet_id = reply["in_response_to_tweet_id"]
        if pd.isna(customer_tweet_id):
            skipped_no_customer_tweet += 1
            continue
        customer_tweet_id = int(customer_tweet_id)
        if customer_tweet_id not in lookup.index:
            skipped_no_customer_tweet += 1
            continue

        customer_row = lookup.loc[customer_tweet_id]
        if isinstance(customer_row, pd.DataFrame):
            customer_row = customer_row.iloc[0]

        # Only pair with genuine inbound customer messages -- Delta
        # sometimes replies to its own prior tweet in a thread.
        if not customer_row["inbound"]:
            skipped_not_inbound += 1
            continue

        customer_message = clean_text(customer_row["text"])
        delta_reply_text = clean_text(reply["text"])
        if not customer_message or not delta_reply_text:
            continue

        prior_context = get_prior_context(
            lookup,
            customer_row["in_response_to_tweet_id"],
            config.MAX_PRIOR_CONTEXT_TURNS,
        )

        rows.append({
            "customer_tweet_id": customer_tweet_id,
            "delta_tweet_id": reply["tweet_id"],
            "customer_message": customer_message,
            "delta_reply": delta_reply_text,
            "prior_context": prior_context,
            "created_at": reply["created_at"],
        })

    print(f"Paired {len(rows):,} threads "
          f"(skipped {skipped_no_customer_tweet:,} with no resolvable customer tweet, "
          f"{skipped_not_inbound:,} where 'customer' tweet was brand-authored)")

    out = pd.DataFrame(rows)

    # Dedupe on exact customer_message text (retweet-style spam / repeated
    # canned complaints inflate volume without adding signal).
    before = len(out)
    out = out.drop_duplicates(subset=["customer_message"]).reset_index(drop=True)
    print(f"Deduped: {before:,} -> {len(out):,}")

    # English filter (this is the slow step -- langdetect per row).
    print("Filtering to English (this takes a few minutes)...")
    tqdm.pandas(desc="Language detection")
    out = out[out["customer_message"].progress_apply(is_english)].reset_index(drop=True)
    print(f"After English filter: {len(out):,} threads")

    # Seeded subsample.
    if len(out) > config.SUBSAMPLE_SIZE:
        out = out.sample(n=config.SUBSAMPLE_SIZE, random_state=config.RANDOM_SEED).reset_index(drop=True)
    print(f"Final subsample: {len(out):,} threads")

    out.insert(0, "thread_id", [f"T{i+1:05d}" for i in range(len(out))])

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.CLEAN_THREADS_CSV, index=False)
    print(f"Saved {config.CLEAN_THREADS_CSV}")


if __name__ == "__main__":
    main()
