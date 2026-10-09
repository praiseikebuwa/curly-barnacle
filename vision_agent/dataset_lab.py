from __future__ import annotations
import json, random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
from .schemas import BoundingBox

@dataclass
class Annotation:
    annotation_id: str; image_id: str; label: str; bbox: BoundingBox; reviewed: bool = False; approved: bool = False; source: str = "human"; license: str | None = None
    def validate(self) -> None:
        if not self.label.strip(): raise ValueError("Annotation label cannot be empty")
        if self.bbox.width <= 0 or self.bbox.height <= 0: raise ValueError("Annotation bounding box must have positive dimensions")

def split_examples(items: list[dict], seed: int = 7, train_ratio: float = .7, val_ratio: float = .15) -> dict[str, list[dict]]:
    if not items: return {"train": [], "validation": [], "test": []}
    rows = list(items); random.Random(seed).shuffle(rows); n = len(rows); train = int(n * train_ratio); val = train + int(n * val_ratio)
    return {"train": rows[:train], "validation": rows[train:val], "test": rows[val:]}

def export_coco(images: Iterable[dict], annotations: Iterable[Annotation], output: str | Path) -> None:
    labels = sorted({a.label for a in annotations}); categories = [{"id": i + 1, "name": name} for i, name in enumerate(labels)]; ids = {c["name"]: c["id"] for c in categories}
    anns = list(annotations)
    for a in anns: a.validate()
    payload = {"info": {"description": "VisionOS AI dataset", "version": "1.0"}, "images": list(images), "categories": categories, "annotations": [{"id": a.annotation_id, "image_id": a.image_id, "category_id": ids[a.label], "bbox": [a.bbox.x, a.bbox.y, a.bbox.width, a.bbox.height], "area": a.bbox.width * a.bbox.height, "iscrowd": 0} for a in anns]}
    Path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
