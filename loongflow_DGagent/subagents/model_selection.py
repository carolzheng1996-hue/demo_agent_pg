from __future__ import annotations

import json
from typing import Dict, List

from ..config import DEFAULT_ALLOWED_MODELS, DEFAULT_TRAINING_PARAMS
from ..llm_utils import invoke_json
from ..state import DGGlobalState
from ..tools import write_step_artifact


FALLBACK_MODELS = [
    {"name": "arima", "params": DEFAULT_TRAINING_PARAMS["arima"], "reason": "适合提供统计基线。"},
    {"name": "xgboost", "params": DEFAULT_TRAINING_PARAMS["xgboost"], "reason": "适合表格化滑窗特征。"},
    {"name": "lstm", "params": DEFAULT_TRAINING_PARAMS["lstm"], "reason": "适合学习较长序列依赖。"},
]


def _mutate_fallback_models(iteration_index: int) -> List[Dict]:
    factor = max(1, iteration_index)
    return [
        {
            "name": "arima",
            "params": {"order": [min(4, 1 + factor % 3), 1, min(4, 1 + (factor + 1) % 3)]},
            "reason": f"第{iteration_index}轮统计基线扰动",
        },
        {
            "name": "xgboost",
            "params": {
                "window": 24 if factor % 2 else 48,
                "n_estimators": 150 + factor * 30,
                "max_depth": 4 + factor % 3,
                "learning_rate": max(0.02, 0.08 - factor * 0.01),
            },
            "reason": f"第{iteration_index}轮树模型参数搜索",
        },
        {
            "name": "lstm",
            "params": {
                "seq_len": 48 if factor % 2 else 96,
                "hidden_size": 32 + factor * 16,
                "num_layers": 1 + factor % 2,
                "dropout": 0.1,
                "epochs": 6 + factor,
                "lr": max(0.0005, 0.001 - factor * 0.0001),
                "batch_size": 64,
            },
            "reason": f"第{iteration_index}轮深度模型参数搜索",
        },
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
    iteration_index = int(state.read("current_iteration_index", 1))
    iteration_history = state.read("iteration_history", [])
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
                "feature_engineering_result": state.read("feature_engineering_result", {}),
                "preprocess_result": state.read("preprocess_result", {}),
                "iteration_index": iteration_index,
                "iteration_history": iteration_history,
                "allowed_models": DEFAULT_ALLOWED_MODELS,
                "default_params": DEFAULT_TRAINING_PARAMS,
            },
            ensure_ascii=False,
        ),
        max_tokens=500,
        temperature=0.0,
    ) or {}

    models = _normalize_models(llm_payload.get("models", []) if (llm_payload := payload) else _mutate_fallback_models(iteration_index))
    selection_payload = {
        "models": models,
        "selection_source": "llm" if payload else "fallback",
        "iteration_index": iteration_index,
    }
    state.write("model_selection_result", selection_payload)
    write_step_artifact(state, "model_selection", selection_payload)
    return {
        "message": "selected three models",
        "models": [item["name"] for item in models],
    }
