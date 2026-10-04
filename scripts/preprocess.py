"""Adapt prepare_dataset_flexible.py + transJson.py + valida.py into one CLI.

New behavior: retain the source, validate records, deduplicate by source and
direction, and split deterministically without overwriting the original data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

from common import make_messages, read_jsonl, validate_sample, write_jsonl


def convert_record(record: dict) -> dict:
    if not isinstance(record, dict):
        raise ValueError("Record must be a JSON object")
    if "messages" in record:
        direction, source, target = validate_sample(record)
    elif "tc_text" in record:
        languages = (record.get("origin_lang"), record.get("tl_trans_lang"))
        if languages == ("한국어", "일본어"):
            direction = "ko2ja"
        elif languages == ("일본어", "한국어"):
            direction = "ja2ko"
        else:
            raise ValueError("Unsupported language pair")
        source, target = record.get("tc_text"), record.get("tl_trans_text")
    else:
        instruction = record.get("instruction", "")
        if not isinstance(instruction, str):
            raise ValueError("Instruction must be text")
        if "Korean to Japanese" in instruction or "Korean->Japanese" in instruction:
            direction, label = "ko2ja", "Korean"
        elif "Japanese to Korean" in instruction or "Japanese->Korean" in instruction:
            direction, label = "ja2ko", "Japanese"
        else:
            raise ValueError("Cannot determine translation direction")
        source = record.get("input", "")
        # The original preparation script stored the source in instruction,
        # while transJson.py read only input (which was empty).
        if not isinstance(source, str):
            raise ValueError("Input must be text")
        if not source.strip():
            match = re.search(rf"(?:^|\n){label}:\s*(.+)\Z", instruction, re.DOTALL)
            source = match.group(1) if match else ""
        target = record.get("output")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("Missing target text")
    return {"messages": make_messages(source, direction, target)}


def load_records(input_path: Path):
    paths = sorted(input_path.rglob("*.json")) + sorted(input_path.rglob("*.jsonl")) if input_path.is_dir() else [input_path]
    if not paths:
        raise ValueError("No JSON/JSONL files found")
    for path in paths:
        if path.suffix == ".jsonl":
            yield from read_jsonl(path)
        elif path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            yield from data if isinstance(data, list) else [data]
        else:
            raise ValueError(f"Unsupported input: {path}")


def prepare(records, validation_ratio: float, seed: int):
    if not 0 < validation_ratio < 1:
        raise ValueError("validation-ratio must be between 0 and 1")
    groups = {"ko2ja": [], "ja2ko": []}
    seen = set()
    stats = Counter()
    for record in records:
        stats["input_records"] += 1
        try:
            sample = convert_record(record)
            direction, source, _ = validate_sample(sample)
        except ValueError:
            stats["invalid_records"] += 1
            continue
        key = (direction, source)
        if key in seen:
            stats["duplicate_sources"] += 1
            continue
        seen.add(key)
        groups[direction].append(sample)
    train, validation = [], []
    rng = random.Random(seed)
    for direction, rows in groups.items():
        rng.shuffle(rows)
        count = min(len(rows) - 1, max(1, int(len(rows) * validation_ratio))) if len(rows) >= 2 else 0
        validation.extend(rows[:count])
        train.extend(rows[count:])
        stats[f"{direction}_train"] = len(rows) - count
        stats[f"{direction}_validation"] = count
    if not train or not validation:
        raise ValueError("Need at least two valid, unique sources in one direction")
    rng.shuffle(train)
    rng.shuffle(validation)
    stats.update(train_records=len(train), validation_records=len(validation))
    return train, validation, dict(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output_dir.resolve()
    if source.is_dir() and (source == output or source in output.parents):
        parser.error("Output directory must be outside the input directory")
    paths = [output / name for name in ("train.jsonl", "validation.jsonl", "preprocess_stats.json")]
    if any(path.exists() for path in paths):
        parser.error("Output files already exist; choose a new output directory")
    train, validation, stats = prepare(load_records(source), args.validation_ratio, args.seed)
    stats.update(seed=args.seed, requested_validation_ratio=args.validation_ratio)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths[0], train)
    write_jsonl(paths[1], validation)
    paths[2].write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
