
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from .base import VQAModel

class GITVQAV2(VQAModel):
    def __init__(self, model_id: str = "microsoft/git-base-vqav2", device: str = "cuda", dtype: str = "float16", max_new_tokens: int = 16):
        self.name = f"git::{model_id}"
        print(f"[Model] Loading {self.name}")

        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype if device == "cuda" else None,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            self.model.to(device)

        self.device = device
        self.max_new_tokens = max_new_tokens

        # convenient handles
        self.tokenizer = self.processor.tokenizer
        self.cls_id = self.tokenizer.cls_token_id
        self.eos_id = self.tokenizer.eos_token_id

    @torch.inference_mode()
    def answer(self, image, question: str, **kwargs) -> str:
        # 1) image -> pixel_values
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.model.device)

        # 2) question -> input_ids (GIT VQA pattern: CLS + question tokens)
        q_ids = self.processor(text=question, add_special_tokens=False).input_ids
        input_ids = torch.tensor([([self.cls_id] + q_ids)], device=self.model.device)

        # 3) generate
        gen_ids = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            eos_token_id=self.eos_id,
            pad_token_id=self.eos_id,
        )

        text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        # Postprocess: GIT often returns "question answer" — remove the question prefix if present
        q_norm = question.strip().lower()
        t_norm = text.lower()
        if t_norm.startswith(q_norm):
            text = text[len(question):].strip()

        # If it's still empty, fall back to original decoded text
        return text if text else self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
