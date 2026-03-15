from __future__ import annotations

from typing import Dict

try:
    from ..state import DGGlobalState
    from ..tools import write_step_artifact
except ImportError:
    from state import DGGlobalState
    from tools import write_step_artifact


def _default_window_config(state: DGGlobalState) -> Dict:
    requires_modeling = bool(state.read("plan_meta", {}).get("requires_modeling", False))
    query = str(state.read("user_query", "")).lower()
    output_len = 24 if any(keyword in query for keyword in ["24", "day ahead", "one day"]) else 1
    return {
        "input_length": 96 if requires_modeling else None,
        "output_length": output_len if requires_modeling else None,
        "time_increment": 1 if requires_modeling else None,
    }


def _normalized_ratios(profile: Dict) -> Dict[str, float]:
    ratios = profile.get("split_ratios") or {"train_ratio": 0.7, "val_ratio": 0.1, "test_ratio": 0.2}
    normalized = {key: float(ratios[key]) for key in ["train_ratio", "val_ratio", "test_ratio"]}
    if any(value < 0 for value in normalized.values()):
        raise ValueError(f"split ratios must be non-negative, got {normalized}")
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("split ratios must contain at least one positive value")
    return {key: value / total for key, value in normalized.items()}


def run(state: DGGlobalState) -> Dict:
    profile = state.read("dataset_profile", {})
    rows = int(profile.get("shape", [0, 0])[0])
    if rows <= 0:
        raise ValueError("split_strategy requires dataset_profile.shape.")

    ratios = _normalized_ratios(profile)
    user_window = {
        "input_length": state.read("input_length"),
        "output_length": state.read("output_length"),
        "time_increment": state.read("time_increment"),
    }
    window_payload = dict(user_window) if any(value is not None for value in user_window.values()) else _default_window_config(state)
    input_length = int(window_payload.get("input_length") or 1)
    output_length = int(window_payload.get("output_length") or 1)
    minimum_required_rows = max(8, input_length + output_length + 2)
    if rows < minimum_required_rows:
        raise ValueError(f"Dataset is too short for forecasting window configuration. Need at least {minimum_required_rows} rows, got {rows}.")

    train_end = int(rows * ratios["train_ratio"])
    val_end = train_end + int(rows * ratios["val_ratio"])
    train_end = max(train_end, input_length + output_length)
    val_end = max(val_end, train_end + output_length)
    val_end = min(val_end, rows - output_length)
    if val_end <= train_end or rows <= val_end:
        raise ValueError("Split ratios produced an empty validation or test segment. Adjust ratios or window sizes.")

    payload = {
        "strategy": "time_order_split",
        "rows": rows,
        "ratios": ratios,
        "window_config": window_payload,
        "indices": {
            "train": [0, train_end],
            "val": [train_end, val_end],
            "test": [val_end, rows],
        },
        "pipeline_source": "deterministic_splitter",
    }
    state.write_runtime("split_indices", payload["indices"])
    state.write("split_strategy_result", payload)
    write_step_artifact(state, "split_strategy", payload)
    return {"message": "split strategy created", **payload}
