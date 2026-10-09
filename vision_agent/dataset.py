from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

LABELS = {"type", "wait", "open", "press", "click"}


@dataclass(frozen=True)
class Example:
    text: str
    label: str

    def as_dict(self) -> dict[str, str]:
        return {"text": self.text, "label": self.label}


def load_dataset(path: str | Path) -> list[Example]:
    examples: list[Example] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            text, label = str(row["text"]).strip(), str(row["label"]).strip().lower()
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"Invalid dataset row {line_number}: {exc}") from exc
        if not text or label not in LABELS:
            raise ValueError(f"Invalid dataset row {line_number}: text is required and label must be one of {sorted(LABELS)}")
        examples.append(Example(text, label))
    if len(examples) < 2:
        raise ValueError("Dataset needs at least two examples")
    return examples


def clean_dataset(examples: Iterable[Example]) -> list[Example]:
    seen: set[tuple[str, str]] = set()
    clean: list[Example] = []
    for item in examples:
        text = re.sub(r"\s+", " ", item.text.strip())
        key = (text.casefold(), item.label)
        if text and key not in seen:
            seen.add(key)
            clean.append(Example(text, item.label))
    return clean


def save_dataset(path: str | Path, examples: Iterable[Example]) -> None:
    rows = [e.as_dict() for e in clean_dataset(examples)]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
