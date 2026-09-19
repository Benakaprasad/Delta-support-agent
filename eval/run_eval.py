"""
Runs the LLM pipeline AND both baselines on the golden set, scores everything,
and writes eval/pipeline_predictions.csv + eval/eval_summary.md.

Usage:
    python eval/run_eval.py                 # full run: pipeline + baselines + judge
    python eval/run_eval.py --skip-llm       # baselines only, no Ollama needed
    python eval/run_eval.py --n 40           # smoke-test on first 40 golden rows
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.baselines import KeywordBaseline, TrivialBaseline
from src.pipeline import SupportAgent
from src.retrieval import ReplyRetriever
import metrics
import judge as judge_mod


def load_golden():
    path = config.GOLDEN_SET_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m eval.build_golden_set`, hand-label "
            f"eval/golden_candidates_draft.csv (fill in final_intent / final_route "
            f"for every row), then save it as eval/golden_set.csv."
        )
    df = pd.read_csv(path)
    required = {"customer_message", "final_intent", "final_route"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"golden_set.csv is missing columns: {missing}")
    df = df.dropna(subset=["final_intent", "final_route"])
    df = df[(df["final_intent"] != "") & (df["final_route"] != "")]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-llm", action="store_true", help="only run baselines, skip Ollama calls")
    ap.add_argument("--n", type=int, default=None, help="limit to first N golden rows (smoke test)")
    args = ap.parse_args()

    golden = load_golden()
    if args.n:
        golden = golden.head(args.n)
    print(f"Evaluating on {len(golden)} golden-set rows"
          + (" (LLM pipeline skipped)" if args.skip_llm else ""))

    labels = config.INTENTS
    all_results = {}

    # ---- Baseline 1: trivial ----
    train_pool = pd.read_csv(config.CLEAN_THREADS_CSV)
    trivial = TrivialBaseline(train_pool)
    trivial_preds = [trivial.predict(m) for m in golden["customer_message"]]
    all_results["trivial"] = trivial_preds

    # ---- Baseline 2: keyword rules ----
    keyword_retriever = ReplyRetriever.from_csv()
    keyword = KeywordBaseline(retriever=keyword_retriever)
    keyword_preds = [keyword.predict(m) for m in golden["customer_message"]]
    all_results["keyword"] = keyword_preds

    # ---- System: LLM pipeline ----
    if not args.skip_llm:
        agent = SupportAgent(retriever=keyword_retriever)  # reuse the same TF-IDF index
        agent_preds = []
        for i, (_, row) in enumerate(golden.iterrows()):
            print(f"  agent {i+1}/{len(golden)}...", end="\r")
            agent_preds.append(agent.handle(row["customer_message"], row.get("prior_context", "") or ""))
        print()
        all_results["agent"] = agent_preds

    # ---- Score each system ----
    summary_lines = ["# Eval summary\n"]
    for system_name, preds in all_results.items():
        y_pred_intent = [p["intent"] for p in preds]
        y_pred_route = [p["route"] for p in preds]
        y_true_intent = golden["final_intent"].tolist()
        y_true_route = golden["final_route"].tolist()

        clf_metrics = metrics.classification_metrics(y_true_intent, y_pred_intent, labels=labels)
        route_metrics = metrics.routing_metrics(y_true_route, y_pred_route)

        summary_lines.append(f"## {system_name}")
        summary_lines.append(f"- accuracy: {clf_metrics['accuracy']:.3f}")
        summary_lines.append(f"- macro_f1: {clf_metrics['macro_f1']:.3f}")
        summary_lines.append(f"- escalate_precision: {route_metrics['escalate_precision']:.3f}")
        summary_lines.append(f"- escalate_recall: {route_metrics['escalate_recall']:.3f}")
        summary_lines.append(f"- auto_handle_rate: {route_metrics['auto_handle_rate']:.3f}\n")

        out_df = golden.copy()
        out_df[f"{system_name}_intent"] = y_pred_intent
        out_df[f"{system_name}_route"] = y_pred_route
        out_df[f"{system_name}_reply"] = [p["reply"] for p in preds]
        out_path = config.EVAL_DIR / f"predictions_{system_name}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved {out_path}")

    # ---- LLM-as-judge on the agent's replies (if run) ----
    if not args.skip_llm:
        judge_rows = []
        for i, (row, pred) in enumerate(zip(golden.to_dict("records"), all_results["agent"])):
            if not pred["reply"]:
                continue
            print(f"  judging {i+1}/{len(golden)}...", end="\r")
            score = judge_mod.judge_reply(
                row["customer_message"], pred["reply"], row.get("historical_delta_reply", "") or "", pred["intent"]
            )
            score["golden_id"] = row.get("golden_id")
            judge_rows.append(score)
        print()
        judge_df = pd.DataFrame(judge_rows)
        judge_df.to_csv(config.JUDGE_OUTPUT_CSV, index=False)
        summary_lines.append("## LLM-judge (agent replies)")
        for axis in ["groundedness", "helpfulness", "tone", "correctness", "overall"]:
            if axis in judge_df.columns:
                summary_lines.append(f"- mean {axis}: {judge_df[axis].mean():.2f}")

    summary_path = config.EVAL_DIR / "eval_summary.md"
    summary_path.write_text("\n".join(summary_lines))
    print(f"\nSaved {summary_path}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
