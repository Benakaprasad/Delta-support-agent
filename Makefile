.PHONY: data golden eval eval-fast demo clean

data:
python -m src.data_prep

golden:
python -m eval.build_golden_set

eval:
python eval/run_eval.py

eval-fast:
python eval/run_eval.py --skip-llm

demo:
python scripts/demo.py --interactive

clean:
rm -f eval/predictions_*.csv eval/judge_scores.csv eval/eval_summary.md
