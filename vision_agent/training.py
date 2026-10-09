from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .dataset import Example, clean_dataset, load_dataset

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class IntentClassifier:
    def __init__(self, labels: list[str], priors: dict[str, float], word_counts: dict[str, dict[str, int]], totals: dict[str, int], vocabulary: list[str]):
        self.labels, self.priors, self.word_counts, self.totals, self.vocabulary = labels, priors, word_counts, totals, vocabulary

    @classmethod
    def train(cls, examples: Iterable[Example]) -> "IntentClassifier":
        items = clean_dataset(examples)
        if not items:
            raise ValueError("Cannot train on an empty dataset")
        labels = sorted({e.label for e in items})
        counts = Counter(e.label for e in items)
        word_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for e in items:
            word_counts[e.label].update(tokens(e.text))
        vocab = sorted({word for counts_for_label in word_counts.values() for word in counts_for_label})
        return cls(labels, {label: counts[label] / len(items) for label in labels}, {k: dict(v) for k, v in word_counts.items()}, {k: sum(v.values()) for k, v in word_counts.items()}, vocab)

    def predict(self, text: str) -> tuple[str, float]:
        words = tokens(text)
        scores: dict[str, float] = {}
        for label in self.labels:
            score = math.log(self.priors[label])
            denom = self.totals[label] + len(self.vocabulary)
            score += sum(math.log((self.word_counts[label].get(word, 0) + 1) / denom) for word in words)
            scores[label] = score
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if len(ranked) == 1:
            return ranked[0][0], 1.0
        gap = ranked[0][1] - ranked[1][1]
        return ranked[0][0], 1 - math.exp(-max(0.0, gap))

    def save(self, path: str | Path) -> None:
        payload = {"labels": self.labels, "priors": self.priors, "word_counts": self.word_counts, "totals": self.totals, "vocabulary": self.vocabulary}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "IntentClassifier":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def train_file(dataset_path: str | Path, model_path: str | Path) -> IntentClassifier:
    model = IntentClassifier.train(load_dataset(dataset_path))
    model.save(model_path)
    return model
