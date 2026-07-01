import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class InferenceEngine:
    def __init__(self, base_model: str, adapter_path: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model = self.model.merge_and_unload()

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def compare(self, prompt: str, base_model: str = "") -> dict:
        base_result = self.generate(prompt)
        return {
            "prompt": prompt,
            "base_model" if base_model else "fine_tuned": base_result,
        }
