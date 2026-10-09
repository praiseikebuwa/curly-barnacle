from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class AgentEvent:
    type: str
    message: str
    data: dict[str, Any]
    timestamp: str


class EventBus:
    def __init__(self, max_events: int = 500):
        self._events: deque[AgentEvent] = deque(maxlen=max_events)
        self._lock = Lock()

    def publish(self, event_type: str, message: str, **data: Any) -> AgentEvent:
        event = AgentEvent(event_type, message, data, datetime.now(timezone.utc).isoformat())
        with self._lock:
            self._events.append(event)
        return event

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return [asdict(e) for e in list(self._events)[-limit:]]
