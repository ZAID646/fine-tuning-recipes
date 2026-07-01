# Fine-Tuning Recipes

Production-ready fine-tuning pipelines: LoRA, QLoRA, DPO with config-driven YAML recipes.

## Features

- **LoRA / QLoRA** — Parameter-efficient fine-tuning of 1B-70B models on consumer GPUs
- **DPO** — Preference optimization with configurable datasets
- **Config-Driven** — YAML recipes, swap configs instead of editing code
- **Inference** — Load and compare base vs fine-tuned models
- **HuggingFace Hub** — Push adapters directly to hub

## Quick Start

```bash
pip install -e .
```

### Train with LoRA
```bash
python -m src.cli train --config recipes/lora.yaml
```

### Train with QLoRA (4-bit)
```bash
python -m src.cli train --config recipes/qlora.yaml
```

### Train with DPO
```bash
python -m src.cli train --config recipes/dpo.yaml
```

### Inference
```bash
python -m src.cli infer --model microsoft/phi-2 --adapter ./outputs/lora --prompt "Write a poem about AI"
```

### Push to Hub
```bash
python -m src.cli push --adapter-path ./outputs/lora --repo-id your-username/model-name
```

## Recipe Structure

```yaml
model:
  name: "microsoft/phi-2"
  torch_dtype: "bfloat16"

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  learning_rate: 2e-4

dataset:
  path: "yahma/alpaca-cleaned"
  split: "train[:1000]"
  template: "alpaca"

quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "bfloat16"
```

## GPU Requirements

| Method | Min VRAM | Recommended |
|---|---|---|
| LoRA (1B) | 8GB | RTX 3090 |
| QLoRA (7B) | 12GB | RTX 3090 |
| QLoRA (13B) | 20GB | RTX 3090/4090 |
| DPO (7B QLoRA) | 16GB | RTX 3090 |
