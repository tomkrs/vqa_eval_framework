
from dataclasses import dataclass

@dataclass
class RunConfig:
    # Models
    blip2_model_id: str = "Salesforce/blip2-opt-2.7b"
    llava_med_model_id: str = "Veda0718/llava-med-v1.5-mistral-7b-finetuned"

    # Datasets
    slake_dataset_id: str = "BoKelvin/SLAKE"
    vqarad_dataset_id: str = "flaviagiammarino/vqa-rad"

    # Runtime
    device: str = "cpu"  # "cpu" or "cuda"
    dtype: str = "float16"   # "float16" or "bfloat16" on supported GPUs
    max_new_tokens: int = 32

    # Output
    outputs_dir: str = "/content/drive/MyDrive/medvqa_project/outputs"
