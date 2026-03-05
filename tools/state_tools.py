from __future__ import annotations

from typing import Any, Optional

from global_state import GlobalState


def read_state(state: GlobalState, key: Optional[str] = None) -> Any:
    if key is None:
        return state.snapshot()
    return state.read(key)


def write_state(state: GlobalState, key: str, value: Any) -> None:
    state.write(key, value)
