from __future__ import annotations
from .schemas import TaskState

LEGAL: dict[TaskState, set[TaskState]] = {
    TaskState.created: {TaskState.validating, TaskState.cancelled},
    TaskState.validating: {TaskState.planning, TaskState.failed, TaskState.cancelled},
    TaskState.planning: {TaskState.waiting_for_approval, TaskState.ready, TaskState.failed, TaskState.cancelled},
    TaskState.waiting_for_approval: {TaskState.ready, TaskState.cancelled, TaskState.needs_user_input},
    TaskState.ready: {TaskState.running, TaskState.cancelled},
    TaskState.running: {TaskState.verifying, TaskState.recovering, TaskState.failed, TaskState.cancelled},
    TaskState.verifying: {TaskState.completed, TaskState.recovering, TaskState.failed, TaskState.cancelled},
    TaskState.recovering: {TaskState.planning, TaskState.running, TaskState.failed, TaskState.needs_user_input, TaskState.cancelled},
    TaskState.needs_user_input: {TaskState.ready, TaskState.cancelled},
    TaskState.completed: set(), TaskState.failed: set(), TaskState.cancelled: set(),
}

def can_transition(current: TaskState, target: TaskState) -> bool:
    return target in LEGAL[current]

def transition(current: TaskState, target: TaskState) -> TaskState:
    if not can_transition(current, target):
        raise ValueError(f"Illegal task transition: {current.value} -> {target.value}")
    return target
