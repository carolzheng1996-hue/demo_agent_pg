from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional
from uuid import uuid4

import pandas as pd

try:
    from ..config import OUTPUT_DIR
    from ..state import DGGlobalState
except ImportError:
    from config import OUTPUT_DIR
    from state import DGGlobalState


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
    state.update(
        {
            "current_iteration_index": iteration_index,
            "current_iteration_id": iteration_id,
            "current_iteration_dir": str(iteration_dir),
            "current_iteration_plan": plan,
        }
    )
    state.write_runtime("iteration_step_artifacts", {})
    return iteration_dir


def write_step_artifact(state: DGGlobalState, step_name: str, payload: Any, filename: str = "result.json") -> Path:
    iteration_dir = Path(state.read("current_iteration_dir"))
    artifacts = dict(state.read_runtime("iteration_step_artifacts", {}) or {})
    artifacts[step_name] = payload
    state.write_runtime("iteration_step_artifacts", artifacts)
    return iteration_dir / "summary.md"


def write_step_dataframe_artifact(
    state: DGGlobalState,
    step_name: str,
    dataframe: pd.DataFrame,
    filename: str = "dataset.parquet",
) -> Path:
    iteration_dir = Path(state.read("current_iteration_dir"))
    target = iteration_dir / f"{step_name}_{filename}"
    suffix = target.suffix.lower()
    if suffix == ".csv":
        dataframe.to_csv(target, index=False)
    else:
        dataframe.to_parquet(target, index=False)
    return target


def write_task_dataframe_artifact(
    state: DGGlobalState,
    relative_name: str,
    dataframe: pd.DataFrame,
) -> Path:
    task_dir = ensure_task_context(state)
    target = task_dir / relative_name
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower()
    if suffix == ".csv":
        dataframe.to_csv(target, index=False)
    else:
        dataframe.to_parquet(target, index=False)
    return target


def write_step_code_artifact(
    state: DGGlobalState,
    step_name: str,
    code: str,
    filename: str = "executed_code.py",
) -> Path:
    iteration_dir = Path(state.read("current_iteration_dir"))
    target = iteration_dir / f"{step_name}_{filename}"
    target.write_text(str(code), encoding="utf-8")
    return target


def read_step_code_artifact(
    state: DGGlobalState,
    step_name: str,
    filename: str = "executed_code.py",
) -> Optional[str]:
    iteration_dir_value = state.read("current_iteration_dir")
    if not iteration_dir_value:
        return None
    target = Path(str(iteration_dir_value)) / f"{step_name}_{filename}"
    if not target.exists():
        return None
    return target.read_text(encoding="utf-8")


def write_task_text_artifact(state: DGGlobalState, relative_name: str, content: str) -> Path:
    task_dir = ensure_task_context(state)
    target = task_dir / relative_name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target
