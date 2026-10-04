"""Shared translation format, derived from Translate-Project/transJson.py."""
from __future__ import annotations

import json
from pathlib import Path

SYSTEM_PROMPT = "You are a professional Korean-Japanese bilingual translator."
DIRECTIONS = {"ko2ja": "[Korean to Japanese]", "ja2ko": "[Japanese to Korean]"}


def make_messages(text: str, direction: str, target: str | None = None) -> list[dict]:
    if not isinstance(direction, str) or direction not in DIRECTIONS or not isinstance(text, str) or not text.strip():
        raise ValueError("A supported direction and non-empty source text are required")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{DIRECTIONS[direction]}\n{text.strip()}"},
    ]
    if target is not None:
        if not isinstance(target, str) or not target.strip():
            raise ValueError("Target text must be non-empty")
        messages.append({"role": "assistant", "content": target.strip()})
    return messages


def validate_sample(sample: dict) -> tuple[str, str, str]:
    """Validate the repository's exact three-message training schema."""
    messages = sample.get("messages") if isinstance(sample, dict) else None
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("Expected system/user/assistant messages")
    for message, role in zip(messages, ("system", "user", "assistant")):
        if not isinstance(message, dict) or message.get("role") != role:
            raise ValueError(f"Expected {role} message")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"Empty or invalid {role} content")
    if messages[0]["content"] != SYSTEM_PROMPT:
        raise ValueError("Unexpected system prompt; normalize before training")
    tag, separator, source = messages[1]["content"].partition("\n")
    direction = next((key for key, value in DIRECTIONS.items() if value == tag), None)
    if direction is None or not separator or not source.strip():
        raise ValueError("Missing direction tag or source text")
    return direction, source.strip(), messages[2]["content"].strip()


def read_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
