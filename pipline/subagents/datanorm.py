from __future__ import annotations

from typing import Dict

try:
    from ..state import DGGlobalState
    from ..tools import write_step_artifact
    from .ds_pipeline_utils import load_ds_dataframe, sequence_to_scalar
except ImportError:
    from state import DGGlobalState
    from tools import write_step_artifact
    from subagents.ds_pipeline_utils import load_ds_dataframe, sequence_to_scalar


def _auto_decision(state: DGGlobalState) -> Dict:
    profile = state.read("dataset_profile", {})
    requires_modeling = bool(state.read("plan_meta", {}).get("requires_modeling", False))
    target_col = profile.get("target_column")
    if not requires_modeling or not target_col:
        return {"should_normalize": False, "recommended_mode": "skip", "reason": "analysis_only"}

    raw_df = load_ds_dataframe(state)
    if target_col not in raw_df.columns:
        return {"should_normalize": False, "recommended_mode": "skip", "reason": "analysis_only"}

    series = raw_df[target_col].apply(sequence_to_scalar)
    dynamic_range = float(series.max() - series.min()) if len(series) else 0.0
    should_normalize = dynamic_range > 1.0
    return {
        "should_normalize": should_normalize,
        "recommended_mode": "zscore" if should_normalize else "skip",
        "reason": "modeling_numeric_target" if should_normalize else "low_dynamic_range",
    }


def run(state: DGGlobalState) -> Dict:
    enable_normalization = state.read("enable_normalization")
    if enable_normalization is None:
        raise ValueError("enable_normalization is missing in state. Please provide it in config_all.json or runtime args.")
    if not bool(enable_normalization):
        result = {"should_normalize": False, "recommended_mode": "skip", "reason": "user_disabled_normalization"}
        state.write("datanorm_result", result)
        write_step_artifact(state, "datanorm", result)
        return {"message": "normalization policy decided", **result}

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
