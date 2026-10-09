from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any
import json


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Element:
    text: str
    kind: str = "text"
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
    confidence: float = 0.0

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass(frozen=True)
class Observation:
    timestamp: str
    width: int
    height: int
    image_path: str | None = None
    elements: tuple[Element, ...] = ()
    fingerprint: str | None = None
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=list)


@dataclass(frozen=True)
class Action:
    kind: str
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    risk: str = "low"


@dataclass(frozen=True)
class Plan:
    task: str
    actions: tuple[Action, ...]
    notes: str = ""


@dataclass(frozen=True)
class RunResult:
    success: bool
    task: str
    executed: tuple[Action, ...]
    message: str
    verification_score: float | None = None
