from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

from ..datasets.base_dataset import BaseVQADataset, VQASample
from ..models.base_model import BaseVQAModel
from .evaluation import normalize_answer, postprocess_prediction, score_record, compute_metrics


DEFAULT_PROMPT_PREFIX = (
    "This is a visual question answering task. "
    "If a yes/no answer is applicable, answer with only 'yes' or 'no'. "
    "Otherwise answer with a single short word or phrase.\n"
)


@dataclass
class RunConfig:
    """Configuration for inference runs."""

    output_dir: str
    prompt_prefix: str = DEFAULT_PROMPT_PREFIX

    # Generation parameters passed to the model adapter
    gen_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "max_new_tokens": 16,
        "do_sample": False,
        "temperature": 1.0,
    })

    seed: int = 0
    limit: Optional[int] = None

    write_jsonl: bool = True
    write_csv: bool = True


def _set_seed(seed: int) -> None:
    """Best-effort deterministic seeding."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import random

        random.seed(seed)
    except Exception:
        pass
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def run_inference(
    model: BaseVQAModel,
    dataset: BaseVQADataset,
    config: RunConfig,
) -> Dict[str, Any]:
    """Run inference and save artifacts.

    Saved files in output_dir:
      - predictions.jsonl (one record per sample)
      - predictions.csv (same data; easier to view)
      - metrics.json (aggregate metrics)
      - run_metadata.json (model/dataset/config, timestamp)

    Returns
    -------
    dict
        metrics
    """

    _set_seed(config.seed)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "predictions.jsonl"
    csv_path = out_dir / "predictions.csv"
    metrics_path = out_dir / "metrics.json"
    meta_path = out_dir / "run_metadata.json"

    run_meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {"name": getattr(dataset, "name", dataset.__class__.__name__)},
        "model": {
            "model_id": getattr(model, "info", None).model_id if getattr(model, "info", None) else model.__class__.__name__,
            "revision": getattr(model, "info", None).revision if getattr(model, "info", None) else None,
            "extra": getattr(model, "info", None).extra if getattr(model, "info", None) else {},
        },
        "prompt_prefix": config.prompt_prefix,
        "gen_kwargs": config.gen_kwargs,
        "seed": config.seed,
        "limit": config.limit,
    }

    (meta_path).write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    records: List[Dict[str, Any]] = []

    n_total = len(dataset)
    n_run = min(n_total, config.limit) if config.limit is not None else n_total

    iterator: Iterable[VQASample] = (dataset.get_sample(i) for i in range(n_run))
    if tqdm is not None:
        iterator = tqdm(iterator, total=n_run, desc="Inference")

    jsonl_f = open(jsonl_path, "w", encoding="utf-8") if config.write_jsonl else None

    try:
        for sample in iterator:
            raw = model.predict(
                image=sample.image,
                question=sample.question,
                answer_type=sample.answer_type,
                prompt_prefix=config.prompt_prefix,
                **config.gen_kwargs,
            )

            pred = postprocess_prediction(raw, sample.answer_type)

            record: Dict[str, Any] = {
                "sample_id": sample.sample_id,
                "question": sample.question,
                "answer_type": sample.answer_type.value,
                "reference_answer": sample.answers,
                "prediction_raw": raw,
                "prediction": pred,
                "prediction_normalized": normalize_answer(pred),
                "meta": sample.meta,
            }

            score_record(record)

            if jsonl_f is not None:
                jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            records.append(record)

    finally:
        if jsonl_f is not None:
            jsonl_f.close()

    # Aggregate metrics
    metrics = compute_metrics(records)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Optional CSV for quick inspection
    if config.write_csv:
        _write_csv(csv_path, records)

    return metrics


def _write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    # Flatten meta for CSV (keep it as JSON string)
    flat = []
    for r in records:
        rr = dict(r)
        rr["meta"] = json.dumps(rr.get("meta", {}), ensure_ascii=False)
        rr["reference_answer"] = json.dumps(rr.get("reference_answer", ""), ensure_ascii=False)
        flat.append(rr)

    # Stable header order
    preferred = [
        "sample_id",
        "question",
        "answer_type",
        "reference_answer",
        "prediction",
        "prediction_raw",
        "prediction_normalized",
        "score",
        "is_correct",
        "meta",
    ]
    keys = list({k for r in flat for k in r.keys()})
    header = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in flat:
            w.writerow(r)
