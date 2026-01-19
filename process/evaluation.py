from __future__ import annotations

import json
import re
import string
from collections import Counter
from typing import Any, Dict, Iterable, Sequence, Union

from ..datasets.base_dataset import AnswerType


# -----------------------------
# Answer extraction / post-processing
# -----------------------------

_ANSWER_PREFIX_RE = re.compile(
    r"^\s*(answer|final answer|prediction|output)\s*[:\-]\s*", re.IGNORECASE
)

_YES_PAT = re.compile(r"\b(yes|yeah|yep|true|positive|present)\b", re.IGNORECASE)
_NO_PAT = re.compile(r"\b(no|nope|false|negative|absent)\b", re.IGNORECASE)


def postprocess_prediction(raw: Any, answer_type: AnswerType = AnswerType.UNKNOWN) -> str:
    """Convert a raw model output into a short answer string.

    This is intentionally conservative and dataset-agnostic:
    - Strips common "Answer:" prefixes
    - Uses first line
    - For CLOSED questions: extracts yes/no where possible
    - For OPEN questions: takes first sentence-like chunk
    """

    if raw is None:
        return ""

    s = str(raw).replace("\x00", "").strip()

    # First line (many models add extra lines / explanations)
    if "\n" in s:
        s = s.splitlines()[0].strip()

    # Remove common prefixes repeatedly (handles "Answer: Answer: ...")
    for _ in range(3):
        new = _ANSWER_PREFIX_RE.sub("", s).strip()
        if new == s:
            break
        s = new

    # If the model output is JSON-like, try to extract a field
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                for k in ("answer", "prediction", "output", "text"):
                    if k in obj:
                        s = str(obj[k]).strip()
                        break
        except Exception:
            pass

    if answer_type == AnswerType.CLOSED:
        # Choose yes/no if we can detect it
        yes = _YES_PAT.search(s)
        no = _NO_PAT.search(s)
        if yes and not no:
            return "yes"
        if no and not yes:
            return "no"
        if yes and no:
            # If both appear, take the last one mentioned
            return "yes" if yes.start() > no.start() else "no"

        # Fallback: first token
        return s.split()[0].strip().lower() if s.split() else s.lower()

    # OPEN / UNKNOWN: take first sentence-like chunk
    first = re.split(r"[.!?]\s+", s)[0].strip()
    return first.strip(string.punctuation + " ")


# -----------------------------
# Normalization (simple, dataset-agnostic)
# -----------------------------

_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans({p: " " for p in string.punctuation})


def normalize_answer(ans: Any) -> str:
    """Normalize answers for exact-match style evaluation."""

    if ans is None:
        return ""

    s = str(ans).strip().lower()
    s = s.replace("\u00a0", " ")

    # Basic yes/no synonym mapping
    if s in {"y", "yes", "yeah", "yep", "true"}:
        s = "yes"
    elif s in {"n", "no", "nope", "false"}:
        s = "no"

    # Replace punctuation with spaces
    s = s.translate(_PUNCT_TABLE)

    tokens = [t for t in s.split() if t and t not in _ARTICLES]
    return " ".join(tokens)


def exact_match(pred: str, gt: str) -> float:
    """Exact match accuracy after normalization."""

    return 1.0 if normalize_answer(pred) == normalize_answer(gt) else 0.0


def vqa_consensus_accuracy(pred: str, gts: Sequence[str]) -> float:
    """VQA consensus accuracy: min(1, count/3).

    This is the standard used in the original VQA benchmark and works for any number
    of reference answers (though it is typically used with 10 human answers).
    """

    npred = normalize_answer(pred)
    norm_gts = [normalize_answer(a) for a in gts]
    counts = Counter(norm_gts)
    return float(min(1.0, counts.get(npred, 0) / 3.0))


def score_prediction(prediction: str, reference: Union[str, Sequence[str]]) -> float:
    """Score a prediction against a single or multiple reference answers."""

    if isinstance(reference, (list, tuple)):
        if len(reference) == 0:
            return 0.0
        if len(reference) == 1:
            return exact_match(prediction, reference[0])
        return vqa_consensus_accuracy(prediction, reference)

    return exact_match(prediction, reference)


def score_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Compute score/is_correct fields in-place for one record."""

    ref = record.get("reference_answer")
    if ref is None:
        ref = record.get("reference_answers")

    pred = record.get("prediction", "")
    score = score_prediction(pred, ref if ref is not None else "")

    record["score"] = float(score)
    record["is_correct"] = bool(score >= 1.0 - 1e-9)
    return record


def compute_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics over a set of scored records."""

    total = 0
    sum_score = 0.0

    by_type = {
        AnswerType.CLOSED.value: {"n": 0, "sum": 0.0},
        AnswerType.OPEN.value: {"n": 0, "sum": 0.0},
        AnswerType.UNKNOWN.value: {"n": 0, "sum": 0.0},
    }

    for r in records:
        total += 1
        score = float(r.get("score", 0.0))
        sum_score += score

        at = str(r.get("answer_type", AnswerType.UNKNOWN.value)).strip().lower()
        if at not in by_type:
            at = AnswerType.UNKNOWN.value

        by_type[at]["n"] += 1
        by_type[at]["sum"] += score

    metrics: Dict[str, Any] = {
        "n": total,
        "mean_score": (sum_score / total) if total else 0.0,
    }

    for at, v in by_type.items():
        n = v["n"]
        metrics[f"{at}_n"] = n
        metrics[f"{at}_mean_score"] = (v["sum"] / n) if n else 0.0

    return metrics
