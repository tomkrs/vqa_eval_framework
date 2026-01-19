from __future__ import annotations

from typing import Dict, Iterable, Optional
from datasets import load_dataset

from .base import BaseVQADataset, VQASample, make_sample


class VQARADDataset(BaseVQADataset):
    name = "VQA-RAD"

    def __init__(self, dataset_id: str = "flaviagiammarino/vqa-rad", split: str = "test"):
        super().__init__(dataset_id=dataset_id, split=split)

    def __iter__(self) -> Iterable[VQASample]:
        print(f"[Dataset] Loading VQA-RAD: {self.dataset_id} split={self.split}")
        ds = load_dataset(self.dataset_id, split=self.split)

        for i, ex in enumerate(ds):
            # HF schema: image, question, answer (+ sometimes id)
            sample_id = ex.get("id", str(i))
            image = ex["image"]
            question = ex["question"]
            answer = ex.get("answer", "")

            # If the dataset ever provides an explicit type, you can wire it here:
            # answer_type = ex.get("answer_type") or ex.get("question_type")
            answer_type = None

            yield make_sample(
                sample_id=str(sample_id),
                dataset_name=self.name,
                image=image,
                question=question,
                answer=answer,
                answer_type=answer_type,
            )


def load_vqarad(dataset_id: str = "flaviagiammarino/vqa-rad", split: str = "test") -> Iterable[Dict]:
    """
    Backwards-compatible functional API.
    Returns dicts with keys: id, image, question, answer, dataset, answer_type
    """
    return iter(VQARADDataset(dataset_id=dataset_id, split=split))
