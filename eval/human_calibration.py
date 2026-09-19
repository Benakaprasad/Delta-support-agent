"""
Validates the LLM judge against your own human judgment (required by the
assignment: "evidence of how well your judge agrees with a human").

Usage:
    python eval/human_calibration.py --sample 30
      -> writes eval/human_calibration_TEMPLATE.csv with 30 (message, draft
         reply) pairs and the judge's "overall" score already filled in.
         Open it and fill your_overall_score (1-5) for each row yourself,
         WITHOUT looking at the judge's score first if you want an unbiased read.
         Save as eval/human_calibration.csv.

    python eval/human_calibration.py --score
      -> reads eval/human_calibration.csv and prints Cohen's kappa + raw
         agreement between your_overall_score and the judge's overall score
         (both bucketed as <=3 vs >=4, since exact 5-point agreement is a
         much stricter and less meaningful bar for this kind of rubric).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
import metrics


def build_template(sample_n: int):
    judge_scores = pd.read_csv(config.JUDGE_OUTPUT_CSV)
    preds = pd.read_csv(config.EVAL_DIR / "predictions_agent.csv")
    merged = preds.merge(judge_scores, on="golden_id", how="inner")
    sample = merged.sample(min(sample_n, len(merged)), random_state=1)
    out = sample[["golden_id", "customer_message", "agent_reply", "overall"]].rename(
        columns={"overall": "judge_overall_score"}
    )
    out["your_overall_score"] = ""
    out_path = config.EVAL_DIR / "human_calibration_TEMPLATE.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path} — fill your_overall_score for each row, save as "
          f"{config.HUMAN_CALIBRATION_CSV.name}, then rerun with --score")


def score():
    df = pd.read_csv(config.HUMAN_CALIBRATION_CSV)
    df = df.dropna(subset=["your_overall_score"])
    judge_bucket = (df["judge_overall_score"] >= 4).astype(int)
    human_bucket = (df["your_overall_score"] >= 4).astype(int)
    result = metrics.agreement(human_bucket, judge_bucket)
    print(f"n={len(df)}  cohen_kappa={result['cohen_kappa']:.3f}  "
          f"raw_agreement={result['raw_agreement']:.3f}")
    print("(kappa > 0.4 = moderate, > 0.6 = substantial agreement — report "
          "whatever you actually get, don't round up)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=30)
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.score:
        score()
    else:
        build_template(args.sample)
