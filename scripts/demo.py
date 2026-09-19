"""
Try the agent on a single message from the command line.

Usage:
    python scripts/demo.py --message "my bag never showed up in ATL and nobody will help"
    python scripts/demo.py --message "..." --baseline keyword    # compare to keyword baseline
    python scripts/demo.py --baseline trivial --message "..."    # compare to trivial baseline
    python scripts/demo.py --interactive                          # loop, one message per line
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import SupportAgent
from src.baselines import TrivialBaseline, KeywordBaseline
from src.retrieval import ReplyRetriever
from src import config

import pandas as pd


def print_result(message: str, result: dict, system_label: str):
    print(f"\n{'='*70}")
    print(f"[{system_label}] {message}")
    print(f"{'='*70}")
    print(f"  intent:  {result.get('intent')}")
    if "confidence" in result:
        print(f"  confidence: {result['confidence']:.2f}")
    print(f"  route:   {result.get('route')}")
    if "route_reason" in result:
        print(f"  reason:  {result['route_reason']}")
    print(f"  reply:   {result.get('reply') or '(no reply -- not_support_request / noop)'}")


def build_system(name: str, retriever: ReplyRetriever):
    if name == "agent":
        return SupportAgent(retriever=retriever)
    if name == "keyword":
        return KeywordBaseline(retriever=retriever)
    if name == "trivial":
        train_pool = pd.read_csv(config.CLEAN_THREADS_CSV)
        return TrivialBaseline(train_pool)
    raise ValueError(f"unknown system: {name}")


def run_one(system, name: str, message: str):
    if name == "agent":
        result = system.handle(message)
    else:
        result = system.predict(message)
    print_result(message, result, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--message", type=str, help="a single customer message to run through the agent")
    ap.add_argument("--baseline", choices=["keyword", "trivial"], default=None,
                     help="run a baseline instead of the full LLM agent")
    ap.add_argument("--interactive", action="store_true", help="loop, prompting for messages")
    args = ap.parse_args()

    retriever = ReplyRetriever.from_csv()
    system_name = args.baseline or "agent"
    system = build_system(system_name, retriever)

    if args.interactive:
        print("Interactive mode. Type a customer message, or 'quit' to exit.")
        while True:
            try:
                message = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if message.lower() in ("quit", "exit", ""):
                break
            run_one(system, system_name, message)
        return

    if not args.message:
        ap.error("--message is required unless --interactive is set")
    run_one(system, system_name, args.message)


if __name__ == "__main__":
    main()
