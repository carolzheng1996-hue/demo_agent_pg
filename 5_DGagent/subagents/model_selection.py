from __future__ import annotations

import json
from typing import Dict, List

try:
    from ..config import DEFAULT_ALLOWED_MODELS, DEFAULT_TRAINING_PARAMS
    from ..llm_utils import invoke_json
    from ..state import DGGlobalState
except ImportError:
    from config import DEFAULT_ALLOWED_MODELS, DEFAULT_TRAINING_PARAMS
    from llm_utils import invoke_json
    from state import DGGlobalState


FALLBACK_MODELS = [
    {"name": "arima", "params": DEFAULT_TRAINING_PARAMS["arima"], "reason": "适合提供统计基线。"},
    {"name": "xgboost", "params": DEFAULT_TRAINING_PARAMS["xgboost"], "reason": "适合表格化滑窗特征。"},
    {"name": "lstm", "params": DEFAULT_TRAINING_PARAMS["lstm"], "reason": "适合学习较长序列依赖。"},
]


def _normalize_models(raw_models: List[Dict]) -> List[Dict]:
    normalized: List[Dict] = []
    for item in raw_models:
        name = str(item.get("name", "")).strip().lower()
        if name not in DEFAULT_ALLOWED_MODELS:
            continue
        normalized.append(
            {
                "name": name,
                "params": item.get("params") or DEFAULT_TRAINING_PARAMS[name],
                "reason": item.get("reason", ""),
            }
        )
    uniq = []
    seen = set()
    for item in normalized:
        if item["name"] not in seen:
            uniq.append(item)
            seen.add(item["name"])
    if len(uniq) < 3:
        for item in FALLBACK_MODELS:
            if item["name"] not in seen:
                uniq.append(item)
                seen.add(item["name"])
            if len(uniq) == 3:
                break
    return uniq[:3]


def run(state: DGGlobalState) -> Dict:
    analysis = state.read("data_analysis_result", {})
    payload = invoke_json(
        system_prompt=(
            "You are a time-series model selector. Return JSON with key models, an array of exactly three items. "
            "Each item must contain name, params, reason. Allowed names: arima, xgboost, lstm."
        ),
        user_prompt=json.dumps(
            {
                "task_description": state.read("user_query", ""),
                "dataset_profile": state.read("dataset_profile", {}),
                "analysis_result": analysis,
                "allowed_models": DEFAULT_ALLOWED_MODELS,
                "default_params": DEFAULT_TRAINING_PARAMS,
            },
            ensure_ascii=False,
        ),
        max_tokens=500,
        temperature=0.0,
    ) or {}

    models = _normalize_models(llm_payload.get("models", []) if (llm_payload := payload) else [])
    selection_payload = {
        "models": models,
        "selection_source": "llm" if payload else "fallback",
    }
    state.write("model_selection_result", selection_payload)
    return {
        "message": "selected three models",
        "models": [item["name"] for item in models],
    }
