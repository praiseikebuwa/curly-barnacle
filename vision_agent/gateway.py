from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Any, Protocol

class TextModelProvider(Protocol):
    def complete(self, prompt: str) -> str: ...
class VisionLanguageProvider(Protocol):
    def analyze(self, prompt: str, image_path: str) -> dict[str, Any]: ...
@dataclass
class ProviderStatus:
    provider_id: str; model_id: str; capabilities: set[str]; available: bool = False; latency_ms: float | None = None; error: str | None = None
class ModelGateway:
    def __init__(self): self.providers: dict[str, Any] = {}; self.status: dict[str, ProviderStatus] = {}
    def register(self, provider_id: str, provider: Any, model_id: str, capabilities: set[str]) -> None:
        self.providers[provider_id] = provider; self.status[provider_id] = ProviderStatus(provider_id, model_id, capabilities)
    def health_check(self, provider_id: str) -> ProviderStatus:
        if provider_id not in self.providers: raise KeyError(provider_id)
        start = time.perf_counter(); provider = self.providers[provider_id]; status = self.status[provider_id]
        try:
            if hasattr(provider, "health_check"): provider.health_check()
            status.available = True; status.error = None; status.latency_ms = (time.perf_counter() - start) * 1000
        except Exception as exc: status.available = False; status.error = str(exc)
        return status
    def available(self, capability: str) -> list[ProviderStatus]: return [s for s in self.status.values() if s.available and capability in s.capabilities]
