
from abc import ABC, abstractmethod
from typing import Any, Dict
from PIL import Image

class VQAModel(ABC):
    name: str

    @abstractmethod
    def answer(self, image: Image.Image, question: str, **kwargs) -> str:
        raise NotImplementedError

    def __call__(self, image: Image.Image, question: str, **kwargs) -> str:
        return self.answer(image=image, question=question, **kwargs)
