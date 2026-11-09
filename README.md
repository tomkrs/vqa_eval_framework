# Medical VQA Evaluation Framework

A small, modular Python framework for evaluating **vision-language models on medical Visual Question Answering (VQA)** in radiology.

Built as part of my Master's research project *"Medical Visual Question Answering in Radiology: A Literature Landscape Analysis and Evaluation Framework"* (M.Sc. Medical Informatics, Heilbronn University & Heidelberg University).

## What it does

- Loads radiology VQA benchmarks from the Hugging Face Hub and converts them into **one common sample format** (`id, image, question, answer, answer_type`).
- Wraps different VQA models behind **one common interface** (`VQAModel.answer(image, question)`).
- Runs batched inference with consistent prompting and light post-processing of model outputs (prefix stripping, yes/no normalisation).
- Computes **exact-match accuracy with bootstrap standard deviation**, reported **overall and disaggregated by question type** (closed yes/no vs. open-ended).
- Saves raw and post-processed predictions as JSONL for later error analysis.

## Supported datasets and models

| Datasets | Hugging Face ID |
|---|---|
| VQA-RAD | `flaviagiammarino/vqa-rad` |
| SLAKE (English subset) | `BoKelvin/SLAKE` |

| Models | Hugging Face ID |
|---|---|
| BLIP-2 (OPT-2.7B) | `Salesforce/blip2-opt-2.7b` |
| GIT (fine-tuned on VQAv2) | `microsoft/git-base-vqav2` |

New datasets subclass `BaseVQADataset`, and new models subclass `VQAModel`. The rest of the pipeline stays the same.

## Project structure

```
vqa_eval_framework/
├── config.py               # RunConfig dataclass: model/dataset IDs, device, dtype, output dir
├── dataset_loaders/
│   ├── base.py             # BaseVQADataset + standard sample format
│   ├── vqarad.py           # VQA-RAD loader
│   └── slake.py            # SLAKE loader (downloads and extracts image archive)
├── models/
│   ├── base.py             # VQAModel interface
│   ├── blip2.py            # BLIP-2 wrapper
│   └── git_vqa.py          # GIT-VQAv2 wrapper
├── run_inference.py        # prompting, inference loop, prediction post-processing
├── evaluate.py             # normalisation, exact match, bootstrap std, open/closed splits
├── utils_reporting.py      # formatted metric output
├── utils_io.py             # JSONL export
└── utils_runtime.py        # GPU memory cleanup between models
```

## Quick start

The framework was developed for **Google Colab with a GPU**. For example, the SLAKE images are extracted to `/content/`. To run locally, change the paths in `config.py` and `dataset_loaders/slake.py`.

```bash
pip install -r requirements.txt
```

```python
from config import RunConfig
from dataset_loaders.vqarad import VQARADDataset
from models.blip2 import BLIP2VQA
from run_inference import run_inference
from evaluate import compute_metrics_with_splits
from utils_reporting import print_metrics
from utils_io import save_jsonl
from utils_runtime import free_gpu

cfg = RunConfig(device="cuda", outputs_dir="outputs")

model = BLIP2VQA(cfg.blip2_model_id, device=cfg.device, dtype=cfg.dtype,
                 max_new_tokens=cfg.max_new_tokens)
dataset = VQARADDataset(cfg.vqarad_dataset_id, split="test")

rows = run_inference(model, dataset, limit=None)
metrics = compute_metrics_with_splits(rows, n_boot=300)

print_metrics(model.name, dataset.name, metrics)
save_jsonl(f"{cfg.outputs_dir}/blip2_vqarad.jsonl", rows)
free_gpu()
```

Example output format:

```
model: blip2::Salesforce/blip2-opt-2.7b
dataset: VQA-RAD
accuracy overall: xx.xx% (+/- x.xx%)   (n=...)
accuracy open:    xx.xx% (+/- x.xx%)   (n=...)
accuracy closed:  xx.xx% (+/- x.xx%)   (n=...)
```

## Evaluation details

- **Normalisation:** lower-casing, whitespace collapsing, and removal of `answer:` prefixes before comparison.
- **Exact match:** a prediction counts as correct only if it equals the reference answer after normalisation. This is a strict metric, so open-ended answers with synonyms count as wrong.
- **Uncertainty:** bootstrap resampling (default 300 resamples, fixed seed) gives a standard deviation for every accuracy value.
- **Disaggregation:** results are reported separately for closed (yes/no) and open questions. These behave very differently: general-domain models often reach reasonable yes/no accuracy but struggle with open medical answers.

## Roadmap

- Wrapper for LLaVA-Med (model ID is already part of `RunConfig`)
- Softer metrics for open questions (token F1, BERTScore)
- CLI entry point for running a full model × dataset grid

## Tech stack

Python · PyTorch · Hugging Face Transformers · Hugging Face Datasets · Pillow
