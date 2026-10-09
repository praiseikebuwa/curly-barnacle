from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from .models import Action, Observation, Plan


PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {"type": "array", "items": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["click", "type", "press", "wait"]},
            "args": {"type": "object"}, "reason": {"type": "string"}, "risk": {"type": "string"}
        }, "required": ["kind", "args", "reason", "risk"], "additionalProperties": False}},
        "notes": {"type": "string"}
    },
    "required": ["actions", "notes"], "additionalProperties": False,
}


class MultimodalReasoner:
    """Optional OpenAI-compatible vision planner. It is never required for local tests."""

    def __init__(self, model: str | None = None, client: Any | None = None):
        self.model = model or os.getenv("VISION_AGENT_MODEL", "gemini-3-flash-preview")
        self.client = client

    def _client(self) -> Any:
        if self.client is None:
            try:
                from openai import OpenAI
                self.client = OpenAI()
            except Exception as exc:
                raise RuntimeError("Install the llm extra and configure OPENAI_API_KEY/OPENAI_API_BASE") from exc
        return self.client

    @staticmethod
    def _image_data(path: str | Path) -> str:
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def plan(self, task: str, observation: Observation) -> Plan:
        if not observation.image_path:
            raise ValueError("A screenshot is required for multimodal planning")
        messages = [{"role": "system", "content": "You are a cautious desktop automation planner. Use only safe, reversible actions. Never invent coordinates; click only visible UI targets. Return JSON matching the schema."}, {"role": "user", "content": [{"type": "text", "text": f"Task: {task}\nVisible OCR: {[e.text for e in observation.elements]}"}, {"type": "image_url", "image_url": {"url": self._image_data(observation.image_path), "detail": "auto"}}]}]
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "response_format": {"type": "json_schema", "json_schema": {"name": "desktop_plan", "strict": True, "schema": PLAN_SCHEMA}}}
        if self.model.startswith("gpt-"):
            kwargs["max_completion_tokens"] = 1200
        else:
            kwargs["max_tokens"] = 1200
        response = self._client().chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        payload = json.loads(content)
        actions = tuple(Action(a["kind"], a["args"], a["reason"], a["risk"]) for a in payload["actions"])
        return Plan(task, actions, payload.get("notes", ""))
