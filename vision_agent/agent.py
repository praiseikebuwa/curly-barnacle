from __future__ import annotations

from .controls import ActionExecutor
from .events import EventBus
from .llm import MultimodalReasoner
from .memory import MemoryStore
from .models import Plan, RunResult
from .perception import ScreenObserver
from .planner import Planner
from .recovery import RetryError, RetryPolicy, RetryRunner


class VisionAgent:
    def __init__(self, memory: MemoryStore | None = None, observer: ScreenObserver | None = None, planner: Planner | None = None, executor: ActionExecutor | None = None, reasoner: MultimodalReasoner | None = None, events: EventBus | None = None, retry_policy: RetryPolicy | None = None) -> None:
        self.memory = memory or MemoryStore()
        self.observer = observer or ScreenObserver()
        self.planner = planner or Planner()
        self.executor = executor or ActionExecutor()
        self.reasoner = reasoner
        self.events = events or EventBus()
        self.retry = RetryRunner(retry_policy, on_retry=lambda op, attempt, exc: self.events.publish("retry", f"Retrying {op}", attempt=attempt, error=str(exc)))

    def _make_plan(self, task: str, before) -> Plan:
        if self.reasoner and before:
            self.events.publish("reasoning", "Asking multimodal model for a plan", model=self.reasoner.model)
            return self.retry.run("multimodal planning", lambda: self.reasoner.plan(task, before))
        return self.planner.plan(task, before)

    def run(self, task: str, observe: bool = False) -> RunResult:
        self.events.publish("task_started", "Task started", task=task)
        before = self.observer.capture() if observe else None
        try:
            plan = self._make_plan(task, before)
            self.events.publish("plan_created", "Plan created", action_count=len(plan.actions), notes=plan.notes)
            executed = []
            for action in plan.actions:
                self.events.publish("action_started", f"Executing {action.kind}", action=action.__dict__)
                try:
                    outcome = self.retry.run(f"action:{action.kind}", lambda a=action: self.executor.execute(a))
                except RetryError as exc:
                    self.events.publish("action_failed", str(exc), action=action.__dict__)
                    self.memory.save_action(action, False, str(exc))
                    return RunResult(False, task, tuple(executed), str(exc))
                verified = outcome == "executed" or outcome.startswith("dry-run")
                self.memory.save_action(action, verified, outcome)
                self.events.publish("action_completed", outcome, action=action.__dict__)
                executed.append(action)
            self.memory.save_workflow(task, executed, verified=not self.executor.dry_run)
            self.events.publish("task_completed", "Task completed", task=task)
            return RunResult(True, task, tuple(executed), "Plan completed" + (" in dry-run mode" if self.executor.dry_run else ""))
        except Exception as exc:
            self.events.publish("task_failed", str(exc), task=task)
            return RunResult(False, task, (), str(exc))
