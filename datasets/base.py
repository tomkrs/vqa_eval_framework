from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional, TypedDict, Any


class VQASample(TypedDict, total=False):
    id: str
    dataset: str
    image: Any              # usually PIL.Image.Image or HF "Image" type
    question: str
    answer: str
    answer_type: str        # "open" or "closed"


def infer_answer_type(answer: str) -> str:
    """
    Minimal heuristic:
    - yes/no => closed
    - else => open
    """
    a = (answer or "").strip().lower()
    if a in {"yes", "no"}:
        return "closed"
    return "open"


def make_sample(
    *,
    sample_id: str,
    dataset_name: str,
    image: Any,
    question: str,
    answer: str,
    answer_type: Optional[str] = None,
) -> VQASample:
    """
    Standardize the sample format used by the whole framework.
    """
    at = answer_type or infer_answer_type(answer)
    return {
        "id": str(sample_id),
        "dataset": dataset_name,
        "image": image,
        "question": question,
        "answer": answer or "",
        "answer_type": at,
    }


class BaseVQADataset(ABC):
    """
    Base interface for all VQA datasets in this framework.

    Subclasses should implement __iter__ and yield standardized samples
    (use make_sample()).
    """

    name: str

    def __init__(self, dataset_id: str, split: str):
        self.dataset_id = dataset_id
        self.split = split

    @abstractmethod
    def __iter__(self) -> Iterable[VQASample]:
        raise NotImplementedError
