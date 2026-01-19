from __future__ import annotations

from typing import Dict, Iterable, Optional
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
import os
import zipfile

from .base import BaseVQADataset, VQASample, make_sample

_SLAKE_EXTRACT_DIR = None


def _prepare_slake_images(repo_id: str) -> str:
    global _SLAKE_EXTRACT_DIR
    if _SLAKE_EXTRACT_DIR is not None and os.path.isdir(_SLAKE_EXTRACT_DIR):
        return _SLAKE_EXTRACT_DIR

    print("[Dataset] Downloading imgs.zip from HF hub (SLAKE)")
    zip_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",  # important for datasets on HF hub
        filename="imgs.zip"
    )

    extract_dir = "/content/slake_imgs_extracted"
    if not os.path.isdir(extract_dir) or len(os.listdir(extract_dir)) == 0:
        print(f"[Dataset] Extracting imgs.zip -> {extract_dir}")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
    else:
        print(f"[Dataset] Using existing extracted images at {extract_dir}")

    _SLAKE_EXTRACT_DIR = extract_dir
    return extract_dir


class SlakeDataset(BaseVQADataset):
    name = "SLAKE"

    def __init__(
        self,
        dataset_id: str = "BoKelvin/SLAKE",
        split: str = "test",
        lang: Optional[str] = "en",
    ):
        super().__init__(dataset_id=dataset_id, split=split)
        self.lang = lang

    def __iter__(self) -> Iterable[VQASample]:
        print(f"[Dataset] Loading SLAKE: {self.dataset_id} split={self.split}")
        ds = load_dataset(self.dataset_id, split=self.split)
        print("[Dataset] SLAKE columns:", ds.column_names)

        img_root = _prepare_slake_images(self.dataset_id)

        kept = 0
        skipped = 0

        for i, ex in enumerate(ds):
            if self.lang is not None and ex.get("q_lang", None) != self.lang:
                continue

            img_rel = ex.get("img_name", None)
            if not img_rel:
                skipped += 1
                continue

            candidates = [
                os.path.join(img_root, img_rel),
                os.path.join(img_root, "imgs", img_rel),
                os.path.join(img_root, "images", img_rel),
            ]
            img_path = next((p for p in candidates if os.path.isfile(p)), None)
            if img_path is None:
                skipped += 1
                continue

            image = Image.open(img_path).convert("RGB")

            sample_id = str(ex.get("qid", i))
            question = ex.get("question", "")
            answer = ex.get("answer", "")

            # If SLAKE provides a type field in your version, wire it here:
            # answer_type = ex.get("answer_type") or ex.get("q_type")
            answer_type = None

            kept += 1
            yield make_sample(
                sample_id=sample_id,
                dataset_name=self.name,
                image=image,
                question=question,
                answer=answer,
                answer_type=answer_type,
            )

        print(f"[Dataset] SLAKE generator done. kept={kept}, skipped={skipped}")


def load_slake(
    dataset_id: str = "BoKelvin/SLAKE",
    split: str = "test",
    lang: str = "en",
) -> Iterable[Dict]:
    """
    Backwards-compatible functional API.
    Returns dicts with keys: id, image, question, answer, dataset, answer_type
    """
    return iter(SlakeDataset(dataset_id=dataset_id, split=split, lang=lang))
