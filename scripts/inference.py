"""Compare an untuned Qwen3 model and its LoRA adapter with identical prompts."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from common import DIRECTIONS, make_messages, read_jsonl, write_jsonl


def load_cases(path: Path) -> list[dict]:
    cases = list(read_jsonl(path))
    if not cases:
        raise ValueError("No evaluation cases")
    identifiers = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"]:
            raise ValueError("Every evaluation case needs a non-empty string id")
        if case["id"] in identifiers:
            raise ValueError("Duplicate evaluation case id")
        identifiers.add(case["id"])
        make_messages(case.get("source"), case.get("direction"))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("Qwen/Qwen3-8B", "Qwen/Qwen3-14B"))
    parser.add_argument("--adapter", type=Path, help="Omit to run only the untuned baseline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="JSONL with id, direction, source, optional reference")
    group.add_argument("--text")
    parser.add_argument("--direction", choices=tuple(DIRECTIONS), default="ko2ja")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    cases = load_cases(args.input) if args.input else [{"id": "manual-1", "direction": args.direction, "source": args.text}]
    for case in cases:
        make_messages(case["source"], case["direction"])
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    if args.adapter:
        adapter_config = json.loads((args.adapter / "adapter_config.json").read_text(encoding="utf-8"))
        if adapter_config.get("base_model_name_or_path") != args.model:
            parser.error("Adapter base_model_name_or_path does not match --model")
    if args.output.exists():
        parser.error("Output already exists; choose a new filename")
    if args.check_only:
        print(json.dumps({"model": args.model, "cases": len(cases), "adapter": str(args.adapter)}, indent=2))
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("This 4-bit inference path requires an NVIDIA CUDA GPU")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.eval()
    # Preserve the original sampling settings; explicitly fix top_k and seed.
    generation = dict(max_new_tokens=args.max_new_tokens, do_sample=True,
                      temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.1)

    def generate(case):
        set_seed(args.seed)
        prompt = tokenizer.apply_chat_template(
            make_messages(case["source"], case["direction"]),
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.get_input_embeddings().weight.device)
        with torch.inference_mode():
            output = model.generate(**inputs, **generation, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    metadata = {"model": args.model, "model_revision": getattr(model.config, "_commit_hash", None),
                "adapter": str(args.adapter) if args.adapter else None, "seed": args.seed,
                "generation": generation, "enable_thinking": False,
                "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "bitsandbytes")}}
    results = [{**case, "base": generate(case), "run": metadata} for case in cases]
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter))
        model.eval()
        for result, case in zip(results, cases):
            result["lora"] = generate(case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, results)
    print(f"Saved {len(results)} cases to {args.output}")


if __name__ == "__main__":
    main()
