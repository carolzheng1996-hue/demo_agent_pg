from __future__ import annotations

from datetime import datetime
from typing import Callable, Dict, List

try:
    from .state import DGGlobalState
    from .task_manager import DGTaskManager
except ImportError:
    from state import DGGlobalState
    from task_manager import DGTaskManager

SubagentFunc = Callable[[DGGlobalState], Dict]


def run_subagent(name: str, fn: SubagentFunc, state: DGGlobalState, task_manager: DGTaskManager) -> Dict:
    task_manager.start(name)
    state.write("tasks", task_manager.list_tasks())
    iteration_index = int(state.read("current_iteration_index", 1) or 1)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] iteration={iteration_index} step={name} start", flush=True)
    try:
        result = fn(state)
        detail = result.get("message", "ok") if isinstance(result, dict) else "ok"
        task_manager.complete(name, detail=detail)
        state.write("tasks", task_manager.list_tasks())
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] iteration={iteration_index} step={name} done: {detail}",
            flush=True,
        )
        return result if isinstance(result, dict) else {}
    except Exception as exc:
        task_manager.fail(name, detail=str(exc))
        state.write("tasks", task_manager.list_tasks())
        state.write("error", {"subagent": name, "message": str(exc)})
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] iteration={iteration_index} step={name} failed: {exc}",
            flush=True,
        )
        raise


def agent_loop(plan: List[str], registry: Dict[str, SubagentFunc], state: DGGlobalState, task_manager: DGTaskManager) -> Dict:
    outputs: Dict[str, Dict] = {}
    for step in plan:
        if step not in registry:
            raise KeyError(f"Unknown subagent: {step}")
        outputs[step] = run_subagent(step, registry[step], state, task_manager)
        if outputs[step].get("pause_execution"):
            break
    return outputs
