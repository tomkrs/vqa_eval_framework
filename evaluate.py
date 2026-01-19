from typing import Dict, List, Any, Optional
import re
import random
import math

def normalize(text: str) -> str:
    if text is None:
        return ""
    t = str(text).strip().lower()
    # drop common prefixes
    t = re.sub(r"^\s*answer\s*:\s*", "", t)
    # collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

def exact_match(rows: List[Dict[str, Any]]) -> float:
    """Accuracy as exact string match after normalization."""
    if not rows:
        return 0.0
    correct = 0
    for r in rows:
        gt = normalize(r.get("answer", ""))
        pred = normalize(r.get("prediction", ""))
        if gt == pred:
            correct += 1
    return correct / len(rows)

def bootstrap_std(
    rows: List[Dict[str, Any]],
    n_boot: int = 300,
    seed: int = 123,
) -> float:
    """
    Bootstrap standard deviation for exact_match accuracy.
    Returns 0.0 if there are too few rows.
    """
    n = len(rows)
    if n <= 1:
        return 0.0

    rng = random.Random(seed)
    accs = []
    for _ in range(n_boot):
        sample = [rows[rng.randrange(n)] for _ in range(n)]
        accs.append(exact_match(sample))

    mean = sum(accs) / len(accs)
    var = sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)
    return math.sqrt(var)

def compute_metrics(
    rows: List[Dict[str, Any]],
    n_boot: int = 300,
    seed: int = 123,
) -> Dict[str, float]:
    """
    Base metrics for a set of rows: accuracy + bootstrap std.
    """
    acc = exact_match(rows)
    std = bootstrap_std(rows, n_boot=n_boot, seed=seed) if len(rows) > 1 else 0.0
    return {
        "n": float(len(rows)),
        "accuracy": acc,
        "accuracy_std": std,
    }

def infer_answer_type(row: Dict[str, Any]) -> str:
    """
    Fallback if dataset loader doesn't provide answer_type.
    """
    gt = normalize(row.get("answer", ""))
    pred = normalize(row.get("prediction", ""))

    # if either looks like yes/no, treat as closed
    if gt in ("yes", "no") or pred in ("yes", "no"):
        return "closed"
    return "open"

def compute_metrics_with_splits(
    rows: List[Dict[str, Any]],
    n_boot: int = 300,
    seed: int = 123,
) -> Dict[str, Any]:
    """
    Returns overall + per answer_type metrics (accuracy + std).
    """
    open_rows, closed_rows, other_rows = [], [], []
    for r in rows:
        at = r.get("answer_type")
        if not at:
            at = infer_answer_type(r)

        at = str(at).lower().strip()
        if at in ("open", "o"):
            open_rows.append(r)
        elif at in ("closed", "c", "yes/no", "yesno"):
            closed_rows.append(r)
        else:
            other_rows.append(r)

    return {
        "overall": compute_metrics(rows, n_boot=n_boot, seed=seed),
        "open": compute_metrics(open_rows, n_boot=n_boot, seed=seed),
        "closed": compute_metrics(closed_rows, n_boot=n_boot, seed=seed),
        "other": compute_metrics(other_rows, n_boot=n_boot, seed=seed),
    }
