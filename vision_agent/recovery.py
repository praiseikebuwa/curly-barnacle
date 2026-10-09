from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_delay: float = 0.25
    backoff: float = 2.0
    max_delay: float = 4.0

    def delay_for(self, attempt: int) -> float:
        return min(self.max_delay, self.initial_delay * (self.backoff ** max(0, attempt - 1)))


class RetryError(RuntimeError):
    def __init__(self, operation: str, attempts: int, last_error: Exception):
        super().__init__(f"{operation} failed after {attempts} attempt(s): {last_error}")
        self.operation, self.attempts, self.last_error = operation, attempts, last_error


class RetryRunner:
    def __init__(self, policy: RetryPolicy | None = None, sleep: Callable[[float], None] = time.sleep, on_retry: Callable[[str, int, Exception], None] | None = None):
        self.policy = policy or RetryPolicy()
        self.sleep = sleep
        self.on_retry = on_retry

    def run(self, operation: str, fn: Callable[[], T], is_retryable: Callable[[Exception], bool] | None = None) -> T:
        last: Exception | None = None
        for attempt in range(1, self.policy.max_attempts + 1):
            try:
                return fn()
            except Exception as exc:  # recovery boundary intentionally catches adapter failures
                last = exc
                retryable = is_retryable(exc) if is_retryable else True
                if not retryable or attempt == self.policy.max_attempts:
                    raise RetryError(operation, attempt, exc) from exc
                if self.on_retry:
                    self.on_retry(operation, attempt, exc)
                self.sleep(self.policy.delay_for(attempt))
        raise RetryError(operation, self.policy.max_attempts, last or RuntimeError("unknown failure"))
