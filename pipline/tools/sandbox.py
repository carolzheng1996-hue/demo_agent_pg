from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from .codegen import regenerate_code_with_feedback
except ImportError:
    from codegen import regenerate_code_with_feedback


SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

ALLOWED_IMPORTS = {
    "numpy": np,
    "pandas": pd,
    "json": importlib.import_module("json"),
    "math": importlib.import_module("math"),
    "statistics": importlib.import_module("statistics"),
    "collections": importlib.import_module("collections"),
}


def _safe_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    root_name = str(name or "").split(".", 1)[0]
    if level != 0:
        raise ImportError("Relative imports are not allowed in generated code.")
    if root_name not in ALLOWED_IMPORTS:
        raise ImportError(f"Import of module '{root_name}' is not allowed in generated code.")
    module = ALLOWED_IMPORTS[root_name]
    if fromlist and root_name in {"collections", "math", "statistics", "json"}:
        return module
    return module


SAFE_BUILTINS["__import__"] = _safe_import


def execute_user_code_safely(code: str, context: Dict[str, Any]) -> Dict[str, Any]:
    globals_dict = {
        "__builtins__": SAFE_BUILTINS,
        "np": np,
        "pd": pd,
    }
    locals_dict = dict(context)
    compiled = compile(code, "<generated_data_reader>", "exec")
    exec(compiled, globals_dict, locals_dict)
    return locals_dict


def execute_user_code_with_repair(
    *,
    code: str,
    context: Dict[str, Any],
    subagent: str,
    repair_context: Dict[str, Any],
    validator: Optional[Callable[[Dict[str, Any]], None]] = None,
    max_attempts: int = 2,
) -> Dict[str, Any]:
    current_code = code
    attempts: List[Dict[str, str]] = []
    last_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            result = execute_user_code_safely(current_code, context)
            if validator is not None:
                validator(result)
            return {
                "result": result,
                "executed_code": current_code,
                "repair_attempts": attempts,
                "used_repair": bool(attempts),
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            attempts.append(
                {
                    "attempt": str(attempt),
                    "error_type": exc.__class__.__name__,
                    "error_message": str(exc),
                }
            )
            if attempt >= max_attempts:
                break
            feedback = (
                f"The generated code failed at runtime with {exc.__class__.__name__}: {exc}. "
                "Rewrite the code to keep the same contract, avoid unavailable packages, and use only pandas/numpy/basic stdlib imports."
            )
            rewritten = regenerate_code_with_feedback(
                subagent=subagent,
                original_code=current_code,
                feedback=feedback,
                context=repair_context,
            )
            if not rewritten or rewritten.strip() == current_code.strip():
                break
            current_code = rewritten

    raise RuntimeError(
        f"Generated code execution failed after {len(attempts)} attempt(s): "
        f"{last_error.__class__.__name__ if last_error else 'UnknownError'}: {last_error}"
    ) from last_error
