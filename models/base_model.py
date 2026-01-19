from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..datasets.base_dataset import AnswerType


@dataclass
class ModelInfo:
    """Metadata to store alongside inference outputs for reproducibility."""

    model_id: str
    revision: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class BaseVQAModel:
    """Common model interaction contract.

    Given an image and a question, return a textual answer.

    - `predict(...)` should return the *raw* model output (may be verbose).
    - Post-processing / normalization is handled in process/evaluation.py.

    Implementations should set `info` for logging and reproducibility.
    """

    info: ModelInfo

    def __init__(self, model_id: str, revision: Optional[str] = None, **extra: Any):
        self.info = ModelInfo(model_id=model_id, revision=revision, extra=dict(extra))

    def predict(
        self,
        image: Any,
        question: str,
        answer_type: AnswerType = AnswerType.UNKNOWN,
        prompt_prefix: str = "",
        **gen_kwargs: Any,
    ) -> str:  # pragma: no cover
        """Return a raw string prediction.

        Parameters
        ----------
        image:
            PIL image, numpy array, path, or any representation your adapter supports.
        question:
            The question text.
        answer_type:
            CLOSED/OPEN/UNKNOWN. Can be used by the model adapter (optional).
        prompt_prefix:
            Shared instruction prefix applied by the framework.
        gen_kwargs:
            Generation params (max_new_tokens, do_sample, temperature, num_beams, ...)

        Returns
        -------
        str
            Raw model output.
        """
        raise NotImplementedError
