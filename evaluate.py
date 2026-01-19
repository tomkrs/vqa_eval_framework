from typing import Dict, List, Any
import re
from collections import Counter

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
    if not rows:
        return 0.0
    correct = 0
    for r in rows:
        gt = normalize(r.get("answer", ""))
        pred = normalize(r.get("prediction", ""))
        if gt == pred:
            correct += 1
    return correct / len(rows)

def token_f1(rows: List[Dict[str, Any]]) -> float:
    """
    Simple token-level F1 (helpful for open-ended answers).
    """
    if not rows:
        return 0.0

    def f1(a: str, b: str) -> float:
        a_toks = normalize(a).split()
        b_toks = normalize(b).split()
        if len(a_toks) == 0 and len(b_toks) == 0:
            return 1.0
        if len(a_toks) == 0 or len(b_toks) == 0:
            return 0.0
        common = Counter(a_toks) & Counter(b_toks)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(b_toks)
        recall = num_same / len(a_toks)
        return 2 * precision * recall / (precision + recall)

    scores = [f1(r.get("answer", ""), r.get("prediction", "")) for r in rows]
    return sum(scores) / len(scores)

def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Base metrics for a set of rows.
    """
    return {
        "n": float(len(rows)),
        "exact_match": exact_match(rows),
        "token_f1": token_f1(rows),
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

def compute_metrics_with_splits(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns overall + per answer_type metrics.
    """
    overall = compute_metrics(rows)

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
        "overall": overall,
        "open": compute_metrics(open_rows),
        "closed": compute_metrics(closed_rows),
        "other": compute_metrics(other_rows),
    }
