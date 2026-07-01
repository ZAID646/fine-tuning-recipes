from datasets import load_dataset
from transformers import AutoTokenizer


def format_alpaca(example):
    prompt = f"### Instruction:\n{example['instruction']}\n"
    if example.get("input"):
        prompt += f"### Input:\n{example['input']}\n"
    prompt += "### Response:\n"
    return {"text": prompt + example["output"], "prompt": prompt, "chosen": prompt + example["output"], "rejected": prompt + "I don't know."}


def format_dpo(example):
    prompt = example.get("prompt", "")
    chosen = example.get("chosen", "")
    rejected = example.get("rejected", "")
    if not chosen and "messages" in example:
        messages = example["messages"]
        chosen = messages[-1]["content"] if messages else ""
        rejected = messages[-2]["content"] if len(messages) > 1 else ""
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected, "text": prompt + chosen}


FORMATTERS = {
    "alpaca": format_alpaca,
    "dpo": format_dpo,
}


def load_and_format(cfg) -> tuple:
    dataset = load_dataset(cfg.path, split=cfg.split)
    formatter = FORMATTERS.get(cfg.template, format_alpaca)
    dataset = dataset.map(formatter, remove_columns=dataset.column_names)
    return dataset


def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer
