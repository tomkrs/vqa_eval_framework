
import json, os
from typing import Any, Dict, List

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[IO] Saved {len(rows)} rows -> {path}")

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    print(f"[IO] Loaded {len(rows)} rows <- {path}")
    return rows
