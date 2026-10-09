from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from .models import Action
from .safety import SafetyPolicy


class ActionExecutionError(RuntimeError):
    pass


@dataclass
class ActionExecutor:
    dry_run: bool = True
    confirm: Callable[[Action], bool] | None = None
    policy: SafetyPolicy | None = None
    action_count: int = 0

    ALLOWED = {"click", "type", "press", "wait"}

    def execute(self, action: Action) -> str:
        if action.kind not in self.ALLOWED:
            raise ActionExecutionError(f"Unsupported action: {action.kind}")
        if not self.dry_run:
            (self.policy or SafetyPolicy()).check(action.kind, action.args, self.action_count)
        if self.confirm and not self.confirm(action):
            return "rejected by confirmation policy"
        if self.dry_run:
            return f"dry-run: {action.kind} {action.args}"
        try:
            import pyautogui
        except ImportError as exc:
            raise ActionExecutionError("Install the desktop extra: pip install -e '.[desktop]'") from exc
        try:
            if action.kind == "click":
                pyautogui.click(int(action.args["x"]), int(action.args["y"]))
            elif action.kind == "type":
                pyautogui.write(str(action.args["text"]), interval=0.01)
            elif action.kind == "press":
                pyautogui.press(str(action.args["key"]))
            elif action.kind == "wait":
                time.sleep(float(action.args.get("seconds", 1)))
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ActionExecutionError(str(exc)) from exc
        self.action_count += 1
        return "executed"
