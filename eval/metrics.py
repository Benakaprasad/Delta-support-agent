"""
Scoring functions used by eval/run_eval.py and eval/human_calibration.py.
Kept dependency-light (sklearn only) and pure functions -- no file I/O --
so they're easy to unit-test or reuse from a notebook.
"""
from typing import Dict, List

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    confusion_matrix,
)


def classification_metrics(y_true: List[str], y_pred: List[str], labels: List[str] = None) -> Dict:
    """Accuracy + macro-F1 for intent classification. macro-F1 matters more
    than accuracy here because the intent distribution is heavily skewed
    (not_support_request alone is ~38% of traffic) -- a classifier that
    just always guesses the majority class would score well on accuracy
    but terribly on macro-F1, which is exactly what the trivial baseline
    is meant to expose."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
    }


def routing_metrics(y_true_route: List[str], y_pred_route: List[str]) -> Dict:
    """Precision/recall on ESCALATE specifically (not just overall
    accuracy), because the two error directions have very different costs:
    a missed escalation (false negative on ESCALATE) can mean a real
    problem got auto-handled wrong; an unnecessary escalation (false
    positive) just costs a human a few extra seconds. Recall matters more
    than precision for this system by design."""
    y_true_bin = [1 if r == "ESCALATE" else 0 for r in y_true_route]
    y_pred_bin = [1 if r == "ESCALATE" else 0 for r in y_pred_route]
    return {
        "escalate_precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "escalate_recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "auto_handle_rate": sum(1 for r in y_pred_route if r == "AUTO_HANDLE") / len(y_pred_route) if y_pred_route else 0.0,
    }


def confusion(y_true: List[str], y_pred: List[str], labels: List[str]) -> List[List[int]]:
    """Full confusion matrix for failure-mode analysis in the report --
    which specific intents get confused with which."""
    return confusion_matrix(y_true, y_pred, labels=labels).tolist()


def agreement(a: List[int], b: List[int]) -> Dict:
    """Cohen's kappa + raw agreement between two binary raters (used by
    human_calibration.py to check judge-vs-human agreement). Kappa
    corrects for chance agreement, which matters when one class dominates
    -- raw agreement alone can look deceptively high."""
    return {
        "cohen_kappa": cohen_kappa_score(a, b),
        "raw_agreement": sum(1 for x, y in zip(a, b) if x == y) / len(a) if a else 0.0,
    }
