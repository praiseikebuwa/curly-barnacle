from __future__ import annotations

import re

from .models import Action, Observation, Plan


class Planner:
    """Deterministic starter planner; a VLM can replace this class later."""

    def plan(self, task: str, observation: Observation | None = None) -> Plan:
        text = task.strip()
        lower = text.lower()
        actions: list[Action] = []
        if "wait" in lower:
            match = re.search(r"wait\s+(\d+(?:\.\d+)?)", lower)
            actions.append(Action("wait", {"seconds": float(match.group(1)) if match else 1}, "Allow the UI to settle"))
        if "type " in lower or "search for " in lower:
            value = re.search(r"(?:type|search for)\s+(.+?)(?:\s+and\s+|$)", text, re.I)
            if value:
                actions.append(Action("type", {"text": value.group(1).strip()}, "Enter the requested text"))
                actions.append(Action("press", {"key": "enter"}, "Submit the text"))
        if "open" in lower and "browser" in lower:
            actions.insert(0, Action("wait", {"seconds": 2}, "Wait for the browser to open"))
        if not actions:
            raise ValueError("I cannot safely plan that task yet; provide an explicit text entry or wait action.")
        return Plan(text, tuple(actions), "Conservative MVP plan; coordinates and risky actions require a dedicated adapter.")
