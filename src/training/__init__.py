import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from trl import SFTTrainer
from src.configs import Recipe
from src.data import load_and_format, get_tokenizer


def train_sft(recipe: Recipe):
    tokenizer = get_tokenizer(recipe.model.name)

    quantization = None
    if recipe.quantization and recipe.quantization.load_in_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, recipe.quantization.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=recipe.quantization.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=recipe.quantization.bnb_4bit_quant_type,
        )

    model = AutoModelForCausalLM.from_pretrained(
        recipe.model.name,
        torch_dtype=getattr(torch, recipe.model.torch_dtype),
        quantization_config=quantization,
        device_map="auto",
        attn_implementation=recipe.model.attn_implementation,
        trust_remote_code=True,
    )

    if quantization:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=recipe.lora.r,
        lora_alpha=recipe.lora.alpha,
        lora_dropout=recipe.lora.dropout,
        target_modules=recipe.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = load_and_format(recipe.dataset)

    training_args = TrainingArguments(
        output_dir=recipe.training.output_dir,
        num_train_epochs=recipe.training.num_train_epochs,
        per_device_train_batch_size=recipe.training.per_device_train_batch_size,
        gradient_accumulation_steps=recipe.training.gradient_accumulation_steps,
        learning_rate=recipe.training.learning_rate,
        warmup_steps=recipe.training.warmup_steps,
        logging_steps=recipe.training.logging_steps,
        save_steps=recipe.training.save_steps,
        bf16=True,
        logging_dir=f"{recipe.training.output_dir}/logs",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
        max_seq_length=recipe.training.max_seq_length,
        packing=recipe.training.packing,
        dataset_text_field="text",
    )

    trainer.train()
    trainer.save_model(recipe.training.output_dir)
    tokenizer.save_pretrained(recipe.training.output_dir)

    return recipe.training.output_dir
