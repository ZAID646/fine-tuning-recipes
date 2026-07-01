import fire
from src.configs import load_recipe
from src.training import train_sft


def train(config: str):
    recipe = load_recipe(config)
    output_dir = train_sft(recipe)
    print(f"Training complete. Model saved to {output_dir}")


def push(adapter_path: str, repo_id: str = ""):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if repo_id:
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        print(f"Pushed to https://huggingface.co/{repo_id}")


def infer(model: str, adapter: str = "", prompt: str = ""):
    from src.inference import InferenceEngine

    engine = InferenceEngine(base_model=model, adapter_path=adapter or None)
    if prompt:
        result = engine.generate(prompt)
        print(f"Prompt: {prompt}\n\nResponse: {result}")
    else:
        print("Interactive mode. Type 'exit' to quit.")
        while True:
            p = input("\n> ")
            if p.lower() == "exit":
                break
            print(engine.generate(p))


if __name__ == "__main__":
    fire.Fire()
