from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Union


class AnswerType(str, Enum):
    """Answer format expectation.

    - CLOSED: fixed vocabulary / categorical (commonly yes/no)
    - OPEN: free-form short text
    - UNKNOWN: dataset does not provide this information
    """

    CLOSED = "closed"
    OPEN = "open"
    UNKNOWN = "unknown"


@dataclass
class VQASample:
    """Dataset-agnostic VQA sample representation."""

    sample_id: str
    image: Any  # typically PIL.Image.Image or a path; model adapters decide what to accept
    question: str
    answers: Union[str, List[str]]  # single reference or multiple references
    answer_type: AnswerType = AnswerType.UNKNOWN
    meta: Dict[str, Any] = field(default_factory=dict)


class BaseVQADataset:
    """Minimal dataset interface.

    Implementations must provide:
      - __len__
      - get_sample(i) -> VQASample

    The goal is that *every* benchmark can be wrapped behind this interface.
    """

    name: str = "base"

    def __len__(self) -> int:  # pragma: no cover
        raise NotImplementedError

    def get_sample(self, index: int) -> VQASample:  # pragma: no cover
        raise NotImplementedError

    def __iter__(self) -> Iterator[VQASample]:
        for i in range(len(self)):
            yield self.get_sample(i)

    def iter_samples(self) -> Iterator[VQASample]:
        """Alias for __iter__ (explicit name reads nicer in notebooks)."""
        return iter(self)


@dataclass
class FieldMap:
    """Maps columns from a tabular dataset row into VQASample fields."""

    image: str
    question: str
    answer: str
    answer_type: Optional[str] = None
    sample_id: Optional[str] = None
    # Any extra columns you want preserved under sample.meta
    keep_meta: Sequence[str] = ()


class HuggingFaceVQADataset(BaseVQADataset):
    """Wraps a Hugging Face datasets.Dataset split into BaseVQADataset.

    This is intentionally simple: it assumes each row corresponds to *one* question-image pair.

    Parameters
    ----------
    ds:
        A datasets.Dataset object.
    field_map:
        Explicit mapping. If not provided, we try to infer common column names.
    image_loader:
        Optional function to turn the row's image field into the desired format.
        By default we return row[field_map.image] as-is.
    name:
        Optional dataset name for metadata.
    """

    def __init__(
        self,
        ds: Any,
        field_map: Optional[FieldMap] = None,
        image_loader=None,
        name: str = "hf_dataset",
    ):
        self.ds = ds
        self.name = name
        self.field_map = field_map or self._infer_field_map(ds)
        self.image_loader = image_loader

    def __len__(self) -> int:
        return len(self.ds)

    def get_sample(self, index: int) -> VQASample:
        row = self.ds[index]
        fm = self.field_map

        img_val = row[fm.image]
        image = self.image_loader(img_val) if self.image_loader else img_val

        answers = row[fm.answer]
        # Some datasets store answers as dict/list structures.
        # Keep it flexible: if it's a dict with 'answer' field, extract.
        if isinstance(answers, dict):
            if "answer" in answers:
                answers = answers["answer"]

        at = AnswerType.UNKNOWN
        if fm.answer_type and fm.answer_type in row and row[fm.answer_type] is not None:
            at = self._parse_answer_type(row[fm.answer_type])

        sid = str(row[fm.sample_id]) if (fm.sample_id and fm.sample_id in row) else str(index)

        meta: Dict[str, Any] = {}
        for k in fm.keep_meta:
            if k in row:
                meta[k] = row[k]

        return VQASample(
            sample_id=sid,
            image=image,
            question=str(row[fm.question]),
            answers=answers,
            answer_type=at,
            meta=meta,
        )

    @staticmethod
    def _parse_answer_type(value: Any) -> AnswerType:
        s = str(value).strip().lower()
        if s in {"closed", "close", "yesno", "yes/no", "binary"}:
            return AnswerType.CLOSED
        if s in {"open", "free", "open-ended", "openended"}:
            return AnswerType.OPEN
        return AnswerType.UNKNOWN

    @staticmethod
    def _infer_field_map(ds: Any) -> FieldMap:
        cols = set(getattr(ds, "column_names", []) or [])

        def pick(*candidates: str) -> Optional[str]:
            for c in candidates:
                if c in cols:
                    return c
            return None

        image = pick("image", "img", "image_path", "path")
        question = pick("question", "query", "prompt")
        answer = pick("answer", "answers", "label", "gt_answer")
        answer_type = pick("answer_type", "answerType", "type", "question_type")
        sample_id = pick("id", "qid", "question_id", "sample_id")

        if not image or not question or not answer:
            raise ValueError(
                "Could not infer required columns (image/question/answer). "
                f"Found columns: {sorted(cols)}. Provide an explicit FieldMap."
            )

        # Keep some common useful fields if present
        keep = [c for c in ("modality", "organ", "anatomy", "split", "lang") if c in cols]

        return FieldMap(
            image=image,
            question=question,
            answer=answer,
            answer_type=answer_type,
            sample_id=sample_id,
            keep_meta=keep,
        )
