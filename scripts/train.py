"""Configurable extraction of the original Qwen3 8B/14B QLoRA scripts."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from common import read_jsonl, validate_sample


def load_config(path: Path) -> dict:
    import yaml
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["model_name"] not in ("Qwen/Qwen3-8B", "Qwen/Qwen3-14B"):
        raise ValueError("Expected a Qwen3 8B or 14B model")
    if not 0 < config["data"]["sample_ratio"] <= 1:
        raise ValueError("sample_ratio must be in (0, 1]")
    return config


def check_data(config: dict) -> dict:
    counts, keys = {}, {}
    for split in ("train", "validation"):
        count, split_keys = 0, set()
        for sample in read_jsonl(Path(config["data"][f"{split}_file"])):
            direction, source, _ = validate_sample(sample)
            split_keys.add((direction, source))
            count += 1
        selected = int(count * config["data"]["sample_ratio"])
        if selected < 1:
            raise ValueError(f"{split}: sampling selects zero rows; increase sample_ratio or provide more data")
        counts[split] = {"total": count, "selected": selected}
        keys[split] = split_keys
    if keys["train"] & keys["validation"]:
        raise ValueError("Source text overlaps between train and validation")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--sample-ratio", type=float, help="Override the historical 10%% sampling, e.g. 1 for a small fixture")
    parser.add_argument("--check-only", action="store_true", help="Validate settings/data without ML libraries or a model download")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.sample_ratio is not None:
        if not 0 < args.sample_ratio <= 1:
            parser.error("sample-ratio must be in (0, 1]")
        config["data"]["sample_ratio"] = args.sample_ratio
    counts = check_data(config)
    print(json.dumps({"model": config["model_name"], "data": counts}, indent=2))
    if args.check_only:
        return
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()):
        parser.error("output_dir is not empty; choose a new run directory in the config")

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("This 4-bit training path requires an NVIDIA CUDA GPU")
    set_seed(config["training"]["seed"])
    bf16 = torch.cuda.is_bf16_supported()
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"], device_map={"": torch.cuda.current_device()},
        quantization_config=BitsAndBytesConfig(
            **config["quantization"],
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        ),
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(**config["lora"]))
    model.config.use_cache = False
    model.print_trainable_parameters()
    dataset = load_dataset("json", data_files={
        split: config["data"][f"{split}_file"] for split in ("train", "validation")
    }, keep_in_memory=False)

    def formatting_prompts_func(examples):
        return {"text": [tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
        ) for messages in examples["messages"]]}

    for split in dataset:
        dataset[split] = dataset[split].shuffle(seed=config["training"]["seed"]).select(range(counts[split]["selected"]))
        dataset[split] = dataset[split].map(
            formatting_prompts_func, batched=True, batch_size=500,
            remove_columns=dataset[split].column_names,
        )
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": config, "data_counts": counts,
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "trl", "peft", "datasets", "bitsandbytes", "accelerate")},
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    training_args = SFTConfig(
        **config["training"], output_dir=str(output), run_name=output.name,
        logging_dir=str(output / "logs"), bf16=bf16, fp16=not bf16,
    )
    trainer = SFTTrainer(
        model=model, args=training_args, train_dataset=dataset["train"],
        eval_dataset=dataset["validation"], processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(str(output / "lora_adapters"))
    tokenizer.save_pretrained(output / "lora_adapters")
    trainer.save_metrics("train", result.metrics)
    trainer.save_metrics("eval", trainer.evaluate())
    # Unlike the originals, do not automatically merge into quantized weights.
    # Inference loads the base plus this adapter explicitly.


if __name__ == "__main__":
    main()
