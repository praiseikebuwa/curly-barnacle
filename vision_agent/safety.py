from __future__ import annotations
from dataclasses import dataclass, field
from threading import Event
from urllib.parse import urlparse
import ipaddress

@dataclass
class SafetyPolicy:
    desktop_control_enabled: bool = False
    approved_domains: set[str] = field(default_factory=set)
    max_actions: int = 20
    require_confirmation_for: set[str] = field(default_factory=lambda: {"open_authorized_url", "type_text"})
    _stopped: Event = field(default_factory=Event, repr=False)
    def stop(self) -> None: self._stopped.set()
    def reset_stop(self) -> None: self._stopped.clear()
    def check(self, action_type: str, parameters: dict, action_count: int) -> None:
        if self._stopped.is_set(): raise PermissionError("Emergency stop is active")
        if action_count >= self.max_actions: raise PermissionError("Per-task action limit exceeded")
        if action_type not in {"wait", "capture_screen", "finish_task"} and not self.desktop_control_enabled:
            raise PermissionError("Desktop control is disabled by safety policy")
        if action_type == "open_authorized_url":
            host = (urlparse(str(parameters.get("url", ""))).hostname or "").lower()
            if urlparse(str(parameters.get("url", ""))).scheme != "https" or host not in self.approved_domains:
                raise PermissionError("URL is not in the approved HTTPS domain allowlist")

def is_public_host(host: str) -> bool:
    try: return ipaddress.ip_address(host).is_global
    except ValueError: return host not in {"localhost", "metadata.google.internal"} and not host.endswith(".local")
