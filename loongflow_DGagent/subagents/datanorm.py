from __future__ import annotations

import json
from typing import Dict, List

try:
    from ..llm_utils import invoke_json
    from ..state import DGGlobalState
    from ..tools import write_step_artifact
except ImportError:
    from llm_utils import invoke_json
    from state import DGGlobalState
    from tools import write_step_artifact


FOUNDATION_MODEL_KEYWORDS: List[str] = [
    "foundation model",
    "timesfm",
    "chronos",
    "moirai",
    "timer",
    "timellm",
    "zero-shot",
    "few-shot",
]


def _fallback_decision(query: str, plan_meta: Dict) -> Dict:
    q = str(query).lower()
    skip_norm = any(keyword in q for keyword in FOUNDATION_MODEL_KEYWORDS)
    requires_modeling = bool(plan_meta.get("requires_modeling", False))
    should_normalize = requires_modeling and not skip_norm
    return {
        "should_normalize": should_normalize,
        "reason": "foundation_model_detected" if skip_norm else "default_modeling_policy",
        "recommended_mode": "skip" if not should_normalize else "zscore",
    }


def run(state: DGGlobalState) -> Dict:
    query = state.read("user_query", "")
    plan_meta = state.read("plan_meta", {})
    policy_override = str(state.read("normalization_policy", "auto") or "auto")
    if policy_override == "force_on":
        result = {"should_normalize": True, "recommended_mode": "zscore", "reason": "user_force_on"}
        state.write("datanorm_result", result)
        write_step_artifact(state, "datanorm", result)
        return {"message": "normalization policy decided", **result}
    if policy_override == "force_off":
        result = {"should_normalize": False, "recommended_mode": "skip", "reason": "user_force_off"}
        state.write("datanorm_result", result)
        write_step_artifact(state, "datanorm", result)
        return {"message": "normalization policy decided", **result}

    payload = invoke_json(
        system_prompt=(
            "You decide whether time series data should be normalized before downstream modeling. "
            "Return JSON with keys: should_normalize (bool), recommended_mode (string), reason (string). "
            "For foundation models or zero-shot/few-shot large pretrained forecasting models, prefer should_normalize=false."
        ),
        user_prompt=json.dumps(
            {
                "task_description": query,
                "plan_meta": plan_meta,
                "dataset_profile": state.read("dataset_profile", {}),
                "available_models": ["arima", "xgboost", "lstm", "foundation_model_like"],
            },
            ensure_ascii=False,
        ),
        max_tokens=180,
        temperature=0.0,
    ) or _fallback_decision(query, plan_meta)

    result = {
        "should_normalize": bool(payload.get("should_normalize", False)),
        "recommended_mode": payload.get("recommended_mode", "skip"),
        "reason": payload.get("reason", "fallback"),
    }
    state.write("datanorm_result", result)
    write_step_artifact(state, "datanorm", result)
    return {
        "message": "normalization policy decided",
        "should_normalize": result["should_normalize"],
        "recommended_mode": result["recommended_mode"],
    }
