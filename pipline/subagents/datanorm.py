from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

try:
    from ..state import DGGlobalState
    from ..tools import write_step_artifact
except ImportError:
    from state import DGGlobalState
    from tools import write_step_artifact


def _auto_decision(state: DGGlobalState) -> Dict:
    profile = state.read("dataset_profile", {})
    requires_modeling = bool(state.read("plan_meta", {}).get("requires_modeling", False))
    target_col = profile.get("target_column")
    station_paths = profile.get("formatted_dataset_paths") or {}
    formatted_path = profile.get("formatted_dataset_path")
    if not requires_modeling or (not station_paths and not formatted_path) or not target_col:
        return {"should_normalize": False, "recommended_mode": "skip", "reason": "analysis_only"}

    if station_paths:
        raw_df = pd.concat(
            [pd.read_parquet(Path(str(path))) for path in station_paths.values()],
            axis=0,
            ignore_index=True,
        )
    else:
        raw_df = pd.read_parquet(Path(str(formatted_path)))
    if target_col not in raw_df.columns:
        return {"should_normalize": False, "recommended_mode": "skip", "reason": "analysis_only"}

    series = raw_df[target_col]
    dynamic_range = float(series.max() - series.min()) if len(series) else 0.0
    should_normalize = dynamic_range > 1.0
    return {
        "should_normalize": should_normalize,
        "recommended_mode": "zscore" if should_normalize else "skip",
        "reason": "modeling_numeric_target" if should_normalize else "low_dynamic_range",
    }


def run(state: DGGlobalState) -> Dict:
    policy_override = str(state.read("normalization_policy") or "").lower()
    if not policy_override:
        raise ValueError("normalization_policy is missing in state. Please provide it in config_all.json or runtime args.")
    if policy_override in {"off", "skip", "none"}:
        result = {"should_normalize": False, "recommended_mode": "skip", "reason": "user_skip"}
    elif policy_override == "zscore":
        result = {"should_normalize": True, "recommended_mode": "zscore", "reason": "user_selected_zscore"}
    elif policy_override == "minmax":
        result = {"should_normalize": True, "recommended_mode": "minmax", "reason": "user_selected_minmax"}
    else:
        result = _auto_decision(state)

    state.write("datanorm_result", result)
    write_step_artifact(state, "datanorm", result)
    return {"message": "normalization policy decided", **result}
