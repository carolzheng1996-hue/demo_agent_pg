from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from config import STATE_FILE


class GlobalState:
    """Shared global state bus for all agents."""

    def __init__(self, persist_path: Path = STATE_FILE, initial: Optional[Dict[str, Any]] = None):
        self._persist_path = persist_path
        self._lock = threading.RLock()
        self._state: Dict[str, Any] = {}
        self._runtime: Dict[str, Any] = {}
        self.load()
        if initial:
            self.update(initial)

    def load(self) -> None:
        with self._lock:
            if self._persist_path.exists():
                self._state = json.loads(self._persist_path.read_text(encoding="utf-8"))
            else:
                self._state = {}

    def persist(self) -> None:
        with self._lock:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(
                json.dumps(self._state, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

    def read(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._state.get(key, default)

    def write(self, key: str, value: Any, persist: bool = True) -> None:
        with self._lock:
            self._state[key] = value
            if persist:
                self.persist()

    def update(self, payload: Dict[str, Any], persist: bool = True) -> None:
        with self._lock:
            self._state.update(payload)
            if persist:
                self.persist()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def write_runtime(self, key: str, value: Any) -> None:
        with self._lock:
            self._runtime[key] = value

    def read_runtime(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._runtime.get(key, default)
