
from typing import Optional
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from .base import VQAModel

class BLIP2VQA(VQAModel):
    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "float16", max_new_tokens: int = 32):
        self.name = f"blip2::{model_id}"
        print(f"[Model] Loading {self.name}")

        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        self.processor = Blip2Processor.from_pretrained(model_id)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            self.model.to(device)

        self.device = device
        self.max_new_tokens = max_new_tokens

    @torch.inference_mode()
    def answer(self, image, question: str, **kwargs) -> str:
        prompt = f"Question: {question} Answer:"
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text = self.processor.decode(out[0], skip_special_tokens=True)

        # Often returns the full prompt + answer; strip if needed
        if "Answer:" in text:
            text = text.split("Answer:", 1)[-1].strip()
        return text.strip()
