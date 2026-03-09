from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, List
from uuid import uuid4

from ..config import OUTPUT_DIR
from ..state import DGGlobalState


def _dump(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def ensure_task_context(state: DGGlobalState) -> Path:
    task_id = state.read("task_id")
    if not task_id:
        task_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
        state.write("task_id", task_id)
    task_dir = OUTPUT_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    state.write("task_dir", str(task_dir))
    return task_dir


def prepare_iteration_artifacts(state: DGGlobalState, plan: List[str], iteration_index: int) -> Path:
    task_dir = ensure_task_context(state)
    iteration_id = f"iteration_{iteration_index:03d}"
    iteration_dir = task_dir / iteration_id
    iteration_dir.mkdir(parents=True, exist_ok=True)
    for step in plan:
        (iteration_dir / step).mkdir(parents=True, exist_ok=True)
    (iteration_dir / "plan.json").write_text(
        _dump(
            {
                "task_id": state.read("task_id"),
                "iteration_id": iteration_id,
                "iteration_index": iteration_index,
                "plan": plan,
                "user_query": state.read("user_query", ""),
                "task_type": state.read("plan_meta", {}).get("task_type"),
            }
        ),
        encoding="utf-8",
    )
    state.update(
        {
            "current_iteration_index": iteration_index,
            "current_iteration_id": iteration_id,
            "current_iteration_dir": str(iteration_dir),
        }
    )
    return iteration_dir


def write_step_artifact(state: DGGlobalState, step_name: str, payload: Any, filename: str = "result.json") -> Path:
    iteration_dir = Path(state.read("current_iteration_dir"))
    step_dir = iteration_dir / step_name
    step_dir.mkdir(parents=True, exist_ok=True)
    target = step_dir / filename
    if filename.endswith(".json"):
        target.write_text(_dump(payload), encoding="utf-8")
    else:
        target.write_text(str(payload), encoding="utf-8")
    return target


def write_task_text_artifact(state: DGGlobalState, relative_name: str, content: str) -> Path:
    task_dir = ensure_task_context(state)
    target = task_dir / relative_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target
