import yaml
from pathlib import Path
from pydantic import BaseModel


class QuantizationConfig(BaseModel):
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"


class LoRAConfig(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = ["q_proj", "v_proj"]


class TrainingConfig(BaseModel):
    output_dir: str = "./outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 200
    max_seq_length: int = 512
    packing: bool = True
    max_length: int = 512
    max_prompt_length: int = 256


class DatasetConfig(BaseModel):
    path: str = "yahma/alpaca-cleaned"
    split: str = "train"
    template: str = "alpaca"


class ModelConfig(BaseModel):
    name: str = "microsoft/phi-2"
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = None


class Recipe(BaseModel):
    model: ModelConfig
    lora: LoRAConfig
    training: TrainingConfig
    dataset: DatasetConfig
    quantization: QuantizationConfig | None = None


def load_recipe(path: str | Path) -> Recipe:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Recipe(**data)
