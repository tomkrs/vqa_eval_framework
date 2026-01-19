import math
from typing import Dict, Any

def fmt_acc(acc: float, std: float) -> str:
    return f"{acc*100:.2f}% (+/- {std*100:.2f}%)"

def _acc(split: Dict[str, Any]) -> float:
    return float(split.get("accuracy", split.get("exact_match", 0.0)))

def _std(split: Dict[str, Any]) -> float:
    if "accuracy_std" in split:
        return float(split["accuracy_std"])
    n = int(split.get("n", 0))
    p = _acc(split)
    return math.sqrt(p * (1 - p) / n) if n > 0 else 0.0

def print_metrics(model_name: str, dataset_name: str, metrics: Dict[str, Any]) -> None:
    o  = metrics.get("overall", {})
    op = metrics.get("open", {})
    cl = metrics.get("closed", {})

    print(f"model: {model_name}")
    print(f"dataset: {dataset_name}")
    print(f"accuracy overall (+/- std): {fmt_acc(_acc(o),  _std(o))}   (n={int(o.get('n',0))})")
    print(f"accuracy open (+/- std):    {fmt_acc(_acc(op), _std(op))}   (n={int(op.get('n',0))})")
    print(f"accuracy closed (+/- std):  {fmt_acc(_acc(cl), _std(cl))}   (n={int(cl.get('n',0))})")
    print("-" * 60)
