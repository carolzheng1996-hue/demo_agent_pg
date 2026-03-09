from __future__ import annotations

from typing import Dict

from ..state import DGGlobalState
from ..tools import write_step_artifact


def run(state: DGGlobalState) -> Dict:
    profile = state.read("dataset_profile", {})
    rows = int(profile.get("shape", [0, 0])[0])
    ratios = profile.get("split_ratios") or {
        "train_ratio": state.read("train_ratio"),
        "val_ratio": state.read("val_ratio"),
        "test_ratio": state.read("test_ratio"),
    }
    if not all(ratios.get(k) is not None for k in ["train_ratio", "val_ratio", "test_ratio"]):
        raise ValueError("split_strategy requires train/val/test ratios.")

    train_end = int(rows * float(ratios["train_ratio"]))
    val_end = train_end + int(rows * float(ratios["val_ratio"]))
    payload = {
        "strategy": "time_order_split",
        "rows": rows,
        "ratios": ratios,
        "indices": {
            "train": [0, train_end],
            "val": [train_end, val_end],
            "test": [val_end, rows],
        },
    }
    state.write("split_strategy_result", payload)
    state.write_runtime("split_indices", payload["indices"])
    write_step_artifact(state, "split_strategy", payload)
    return {"message": "split strategy prepared", "indices": payload["indices"]}
