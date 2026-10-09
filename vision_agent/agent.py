from __future__ import annotations

from .controls import ActionExecutor
from .memory import MemoryStore
from .models import RunResult
from .perception import ScreenObserver
from .planner import Planner


class VisionAgent:
    def __init__(self, memory: MemoryStore | None = None, observer: ScreenObserver | None = None, planner: Planner | None = None, executor: ActionExecutor | None = None) -> None:
        self.memory = memory or MemoryStore()
        self.observer = observer or ScreenObserver()
        self.planner = planner or Planner()
        self.executor = executor or ActionExecutor()

    def run(self, task: str, observe: bool = False) -> RunResult:
        before = self.observer.capture() if observe else None
        plan = self.planner.plan(task, before)
        executed = []
        for action in plan.actions:
            outcome = self.executor.execute(action)
            self.memory.save_action(action, outcome == "executed" or outcome.startswith("dry-run"), outcome)
            executed.append(action)
        self.memory.save_workflow(task, executed, verified=False)
        return RunResult(True, task, tuple(executed), "Plan completed" + (" in dry-run mode" if self.executor.dry_run else ""))
