# Delta Twitter Support Agent

An AI customer-support agent for **Delta Air Lines**, built on the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset. It classifies an incoming customer message into a data-derived intent
taxonomy, drafts a reply grounded in how Delta has historically resolved similar
issues, and decides whether to auto-handle or escalate — with a stated reason.

Built for the Hiver SDE Intern take-home. **The proof matters more than the
system**, so start with [`report/REPORT.md`](report/REPORT.md) — especially
§6, "What is misleading about my headline number?"

- [Report](report/REPORT.md) — framing, results vs. baselines, failure analysis
- [Decision log](report/decision_log.md) — 16 non-obvious decisions and why

---

## Reproduce the headline results in under 15 minutes

### 1. Setup (~2 min)

```bash
git clone <this-repo>
cd delta-support-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env     # then put your real key in .env
export ANTHROPIC_API_KEY=sk-ant-...
```

### 2. Get the data (~3 min)

Download `twcs.csv` from the Kaggle dataset above and place it at:

```
data/raw/twcs.csv
```

(~500 MB; gitignored, so it isn't in this repo.)

### 3. Build the cleaned subsample (~2 min)

```bash
make data          # or: python -m src.data_prep
```

Pairs each Delta reply with the customer tweet it answered, attaches up to 2
prior turns of context, filters to English, dedupes, and takes a seeded
8,000-thread subsample → `data/processed/delta_threads_clean.csv`.

### 4. Run the evaluation (~6 min)

```bash
make eval-fast     # baselines only — no API key needed, ~10 seconds
make eval          # full: LLM agent + baselines + LLM-as-judge
```

Writes `eval/predictions_*.csv`, `eval/judge_scores.csv`, and
`eval/eval_summary.md`.

Smoke-test a subset first if you want:

```bash
python eval/run_eval.py --n 25
```

### 5. Try it on a single message

```bash
python scripts/demo.py --message "my bag never showed up in ATL and nobody will help"
python scripts/demo.py --message "..." --baseline keyword    # compare to baseline
python scripts/demo.py --interactive
```

---

## Rebuilding the golden set from scratch

The hand-labelled golden set (`eval/golden_set.csv`) ships in this repo. To
regenerate the *candidate pool* it was built from:

```bash
make golden        # → eval/golden_candidates_draft.csv
```

This produces 200 stratified candidates with **draft** labels from regex
heuristics and flags the rows most likely to be wrong
(`needs_human_review`). Those drafts are a starting point, not ground truth —
the shipped `golden_set.csv` was reviewed and corrected row by row. See
decision log #13 for a concrete demonstration of what happens when you skip
that step.

## Validating the LLM judge against a human

```bash
python eval/human_calibration.py --sample 30   # → fill in your own scores
python eval/human_calibration.py --score       # → Cohen's kappa + raw agreement
```

---

## Repo layout

```
delta-support-agent/
├── README.md
├── Makefile                      # make data / golden / eval / eval-fast / demo
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/twcs.csv              # you provide (gitignored)
│   └── processed/
│       └── delta_threads_clean.csv
├── src/
│   ├── config.py                 # brand, taxonomy, priority order, override rules
│   ├── data_prep.py              # thread pairing, cleaning, English filter, subsample
│   ├── taxonomy_heuristics.py    # regex patterns (stratification only, NOT the classifier)
│   ├── retrieval.py              # TF-IDF grounding over historical replies
│   ├── classifier.py             # LLM intent classifier (+ confidence)
│   ├── reply_generator.py        # retrieval-grounded reply drafting
│   ├── router.py                 # auto-handle vs escalate + cross-cutting overrides
│   ├── baselines.py              # trivial + keyword baselines
│   ├── pipeline.py               # end-to-end agent
│   └── llm_client.py             # Anthropic wrapper w/ retries + JSON parsing
├── eval/
│   ├── build_golden_set.py       # stratified sampling + draft labels
│   ├── golden_set.csv            # 200 hand-labelled examples
│   ├── metrics.py                # accuracy, macro-F1, escalation P/R, Cohen's kappa
│   ├── judge.py                  # LLM-as-judge rubric (4 axes)
│   ├── human_calibration.py      # judge-vs-human agreement
│   └── run_eval.py               # main harness
├── scripts/demo.py
└── report/
    ├── REPORT.md
    └── decision_log.md
```

## Design in one paragraph

Delta's Twitter channel is **triage, not resolution** — their real replies
almost always route account-specific issues to DMs rather than resolving
in-thread. So the agent optimises for never fabricating a resolution and for
reliably catching the expensive cases, not for maximising auto-handle rate.
Routing sits in its own layer with three independent escalation gates
(cross-cutting overrides for accessibility / account-security / safety /
legal / minors → a classifier-confidence gate → intent eligibility), because
a wrong auto-handle on those is far costlier than an unnecessary escalation.
Only compliments, generic policy questions, and non-support chatter are ever
auto-handle eligible.

## Credits / borrowed

- Dataset: [thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (Kaggle).
- `scikit-learn` for TF-IDF, cosine similarity, and metrics (accuracy, F1,
  confusion matrix, Cohen's kappa).
- LLM calls via the `anthropic` Python SDK.
- Taxonomy, prompts, routing policy, sampling strategy, and evaluation design
  are my own, derived from reading the data.
