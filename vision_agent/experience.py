from __future__ import annotations
import sqlite3
from datetime import datetime, timezone
from typing import Any

class ExperienceStore:
    def __init__(self, path: str = "vision_agent.db"):
        self.db = sqlite3.connect(path); self.db.row_factory = sqlite3.Row
        self.db.execute("CREATE TABLE IF NOT EXISTS experiences (id INTEGER PRIMARY KEY, application TEXT, task_context TEXT, action TEXT, expected_outcome TEXT, observed_outcome TEXT, verification TEXT, confidence REAL, source TEXT, created_at TEXT, reviewed INTEGER DEFAULT 0, stale INTEGER DEFAULT 0)"); self.db.commit()
    def add(self, application: str, task_context: str, action: str, expected_outcome: str, observed_outcome: str, verification: str, confidence: float | None, source: str = "verified", reviewed: bool = False) -> int:
        if source not in {"verified", "human_review"}: raise ValueError("Only verified or human_review experiences can enter trusted memory")
        cur = self.db.execute("INSERT INTO experiences(application,task_context,action,expected_outcome,observed_outcome,verification,confidence,source,created_at,reviewed) VALUES(?,?,?,?,?,?,?,?,?,?)", (application,task_context,action,expected_outcome,observed_outcome,verification,confidence,source,datetime.now(timezone.utc).isoformat(),int(reviewed))); self.db.commit(); return int(cur.lastrowid)
    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM experiences WHERE stale=0 AND (task_context LIKE ? OR observed_outcome LIKE ?) ORDER BY id DESC LIMIT ?", (f"%{query}%", f"%{query}%", limit)).fetchall(); return [dict(r) for r in rows]
    def mark_stale(self, experience_id: int) -> None: self.db.execute("UPDATE experiences SET stale=1 WHERE id=?", (experience_id,)); self.db.commit()
