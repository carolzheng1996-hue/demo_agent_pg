from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


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
