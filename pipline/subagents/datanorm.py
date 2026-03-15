from __future__ import annotations

from typing import Dict

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
    raw_df = state.read_runtime("raw_df")
    if not requires_modeling or raw_df is None or not target_col or target_col not in raw_df.columns:
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
    policy_override = str(state.read("normalization_policy", "auto") or "auto").lower()
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
