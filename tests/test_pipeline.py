"""CPU-only regression tests for the extracted data pipeline."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import make_messages, validate_sample, write_jsonl
from preprocess import convert_record, prepare
from train import check_data, load_config
from inference import load_cases

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_legacy_empty_input_keeps_source_from_instruction(self):
        sample = convert_record({
            "instruction": "Translate the following Korean to Japanese naturally.\nSpeaker: unknown, unknown\n\nKorean: 원문입니다.\n둘째 줄입니다.",
            "input": "", "output": "原文です。\n二行目です。",
        })
        self.assertEqual(validate_sample(sample), ("ko2ja", "원문입니다.\n둘째 줄입니다.", "原文です。\n二行目です。"))

    def test_legacy_explicit_input_is_preserved(self):
        sample = convert_record({"instruction": "Japanese->Korean", "input": "こんにちは", "output": "안녕하세요"})
        self.assertEqual(validate_sample(sample), ("ja2ko", "こんにちは", "안녕하세요"))

    def test_invalid_source_target_and_direction_are_rejected(self):
        for source, target, direction in [("", "x", "ko2ja"), ("x", "", "ko2ja"), ("x", "y", "en2ko")]:
            with self.subTest(source=source, target=target, direction=direction), self.assertRaises(ValueError):
                make_messages(source, direction, target)
        with self.assertRaises(ValueError):
            convert_record({"tc_text": "x", "tl_trans_text": None, "origin_lang": "한국어", "tl_trans_lang": "일본어"})

    def test_direction_stratified_split_has_no_duplicate_source_leakage(self):
        records = [{"messages": make_messages(f"source {i}", direction, f"target {i}")}
                   for direction in ("ko2ja", "ja2ko") for i in range(20)]
        records += [records[0], {"messages": make_messages("source 0", "ko2ja", "alternate target")}, {}]
        first = prepare(records, 0.1, 42)
        self.assertEqual(first, prepare(records, 0.1, 42))
        train, validation, stats = first
        keys = lambda rows: {validate_sample(row)[:2] for row in rows}
        self.assertFalse(keys(train) & keys(validation))
        self.assertEqual((len(train), len(validation)), (36, 4))
        self.assertEqual(stats["duplicate_sources"], 2)
        self.assertEqual(stats["invalid_records"], 1)
        self.assertEqual(stats["ko2ja_validation"], 2)
        self.assertEqual(stats["ja2ko_validation"], 2)

    def test_corrupted_messages_are_rejected(self):
        for change in ("empty_source", "wrong_role"):
            sample = {"messages": make_messages("x", "ko2ja", "y")}
            if change == "empty_source":
                sample["messages"][1]["content"] = "[Korean to Japanese]\n"
            else:
                sample["messages"][2]["role"] = "user"
            with self.assertRaises(ValueError):
                validate_sample(sample)

    def test_training_preflight_detects_empty_sampling_and_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            train, validation = Path(directory) / "train.jsonl", Path(directory) / "validation.jsonl"
            write_jsonl(train, [{"messages": make_messages("train", "ko2ja", "a")}])
            write_jsonl(validation, [{"messages": make_messages("val", "ko2ja", "b")}])
            config = {"data": {"train_file": str(train), "validation_file": str(validation), "sample_ratio": 0.1}}
            with self.assertRaisesRegex(ValueError, "zero rows"):
                check_data(config)
            config["data"]["sample_ratio"] = 1
            self.assertEqual(check_data(config)["train"]["selected"], 1)
            write_jsonl(validation, [{"messages": make_messages("train", "ko2ja", "different target")}])
            with self.assertRaisesRegex(ValueError, "overlaps"):
                check_data(config)

    def test_config_values_match_source_variants(self):
        for size, batch, accum in ((8, 6, 3), (14, 1, 16)):
            config = load_config(ROOT / f"configs/qwen3_{size}b_lora.yaml")
            self.assertEqual(config["training"]["per_device_train_batch_size"], batch)
            self.assertEqual(config["training"]["gradient_accumulation_steps"], accum)
            self.assertEqual(config["data"]["sample_ratio"], 0.1)

    def test_evaluation_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.jsonl"
            case = {"id": "one", "direction": "ko2ja", "source": "x"}
            write_jsonl(path, [case, case])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_cases(path)


if __name__ == "__main__":
    unittest.main()
