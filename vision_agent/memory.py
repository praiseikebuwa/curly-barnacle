from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .models import Action, Observation, now_iso


class MemoryStore:
    """Small durable store for observations and verified interaction outcomes."""

    def __init__(self, path: str | Path = "vision_agent.db") -> None:
        self.path = str(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY, created_at TEXT NOT NULL,
            payload TEXT NOT NULL, fingerprint TEXT
        );
        CREATE TABLE IF NOT EXISTS actions (
            id INTEGER PRIMARY KEY, created_at TEXT NOT NULL,
            kind TEXT NOT NULL, args TEXT NOT NULL, reason TEXT, risk TEXT,
            verified INTEGER NOT NULL DEFAULT 0, outcome TEXT
        );
        CREATE TABLE IF NOT EXISTS workflows (
            id INTEGER PRIMARY KEY, created_at TEXT NOT NULL,
            task TEXT NOT NULL, steps TEXT NOT NULL, verified INTEGER NOT NULL DEFAULT 0
        );
        """)
        self.connection.commit()

    def save_observation(self, observation: Observation) -> int:
        cur = self.connection.execute(
            "INSERT INTO observations(created_at,payload,fingerprint) VALUES(?,?,?)",
            (observation.timestamp, observation.to_json(), observation.fingerprint),
        )
        self.connection.commit()
        return int(cur.lastrowid)

    def save_action(self, action: Action, verified: bool, outcome: str) -> int:
        cur = self.connection.execute(
            "INSERT INTO actions(created_at,kind,args,reason,risk,verified,outcome) VALUES(?,?,?,?,?,?,?)",
            (now_iso(), action.kind, json.dumps(action.args), action.reason, action.risk, int(verified), outcome),
        )
        self.connection.commit()
        return int(cur.lastrowid)

    def save_workflow(self, task: str, actions: list[Action], verified: bool = False) -> int:
        cur = self.connection.execute(
            "INSERT INTO workflows(created_at,task,steps,verified) VALUES(?,?,?,?)",
            (now_iso(), task, json.dumps([a.__dict__ for a in actions]), int(verified)),
        )
        self.connection.commit()
        return int(cur.lastrowid)

    def recent_workflows(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM workflows ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self.connection.close()
