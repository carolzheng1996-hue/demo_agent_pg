from __future__ import annotations

from typing import Callable, Dict, List

from global_state import GlobalState
from task_manager import TaskManager

SubagentFunc = Callable[[GlobalState], Dict]


def run_subagent(name: str, fn: SubagentFunc, state: GlobalState, task_manager: TaskManager) -> Dict:
    task_manager.start(name)
    state.write("tasks", task_manager.list_tasks())
    try:
        result = fn(state)
        detail = result.get("message", "ok") if isinstance(result, dict) else "ok"
        task_manager.complete(name, detail=detail)
        state.write("tasks", task_manager.list_tasks())
        return result if isinstance(result, dict) else {}
    except Exception as exc:
        task_manager.fail(name, detail=str(exc))
        state.write("tasks", task_manager.list_tasks())
        state.write("error", {"subagent": name, "message": str(exc)})
        raise


def agent_loop(plan: List[str], registry: Dict[str, SubagentFunc], state: GlobalState, task_manager: TaskManager) -> Dict:
    outputs: Dict[str, Dict] = {}
    for step in plan:
        if step not in registry:
            task_manager.fail(step, detail="subagent not found")
            state.write("tasks", task_manager.list_tasks())
            raise KeyError(f"Unknown subagent: {step}")
        outputs[step] = run_subagent(step, registry[step], state, task_manager)
    return outputs
