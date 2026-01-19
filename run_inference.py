from typing import Dict, Iterable, List, Optional
from tqdm import tqdm

# Optional: set a global/default instruction prefix here
# (or pass it in from your notebook/script).
DEFAULT_PROMPT_PREFIX = (
    "This is a visual question answering task. "
    "If yes/no is applicable, answer only yes or no. "
    "Otherwise answer with as few words as possible."
)

def postprocess_prediction(text: str) -> str:
    """Light cleanup to improve evaluation consistency."""
    if text is None:
        return ""
    t = str(text).strip()

    # common patterns from some VQA models
    for prefix in ("answer:", "Answer:", "ANSWER:"):
        if t.startswith(prefix):
            t = t[len(prefix):].strip()

    # keep only first line (many models add explanations)
    t = t.splitlines()[0].strip()

    # normalize yes/no variants a bit
    low = t.lower().strip(" .,!?:;")
    if low in ("yes", "y"):
        return "yes"
    if low in ("no", "n"):
        return "no"

    return t

def run_inference(
    model,
    dataset_iter: Iterable[Dict],
    limit: Optional[int] = None,
    prompt_prefix: Optional[str] = DEFAULT_PROMPT_PREFIX,
) -> List[Dict]:
    rows = []
    print(f"[Run] Model: {model.name}")

    for i, ex in enumerate(tqdm(dataset_iter, desc=f"Inference::{model.name}")):
        if limit is not None and i >= limit:
            break

        q = ex["question"]
        q_for_model = f"{prompt_prefix}\n{q}" if prompt_prefix else q

        raw_pred = model.answer(ex["image"], q_for_model)
        pred = postprocess_prediction(raw_pred)

        rows.append({
            "id": ex["id"],
            "dataset": ex["dataset"],
            "question": q,
            "answer": ex["answer"],
            "answer_type": ex.get("answer_type"),   # add in dataset loaders
            "raw_prediction": raw_pred,
            "prediction": pred,
        })

        if (i + 1) % 25 == 0:
            print(f"[Run] {i+1} examples done...")

    print(f"[Run] Completed: {len(rows)} examples")
    return rows
