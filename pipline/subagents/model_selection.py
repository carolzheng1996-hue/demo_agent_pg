from __future__ import annotations

import hashlib
import json
from typing import Dict, List

try:
    from ..config import DEFAULT_ALLOWED_MODELS, DEFAULT_TRAINING_PARAMS
    from ..state import DGGlobalState
    from ..tools import write_step_artifact
except ImportError:
    from config import DEFAULT_ALLOWED_MODELS, DEFAULT_TRAINING_PARAMS
    from state import DGGlobalState
    from tools import write_step_artifact


FALLBACK_MODELS = [
    {"name": "arima", "params": DEFAULT_TRAINING_PARAMS["arima"], "reason": "提供稳定的统计基线。"},
    {"name": "xgboost", "params": DEFAULT_TRAINING_PARAMS["xgboost"], "reason": "适合滑窗和统计特征。"},
    {"name": "linear", "params": DEFAULT_TRAINING_PARAMS["linear"], "reason": "提供快速线性基线，便于调试与快速测试。"},
]


def _iteration_seed(iteration_index: int, model_name: str, feature_methods: List[str]) -> int:
    feature_factor = sum(sum(ord(ch) for ch in name) for name in feature_methods) % 997
    model_factor = sum(ord(ch) for ch in model_name)
    return iteration_index * 1009 + feature_factor + model_factor


def _history_bias(history: List[Dict]) -> int:
    if not history:
        return 0
    material = "|".join(f"{item.get('selected_strategy','')}:{item.get('best_score','')}" for item in history[-3:])
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return int(digest[:6], 16) % 11


def _param_signature(params: Dict) -> str:
    normalized: Dict[str, object] = {}
    for key, value in params.items():
        if isinstance(value, float):
            normalized[key] = round(value, 8)
        else:
            normalized[key] = value
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False)


def _bump_params(name: str, params: Dict, attempt: int, seed: int) -> Dict:
    updated = dict(params)
    if name == "arima":
        order = list(updated.get("order", [2, 1, 2]))
        order[0] = min(5, max(0, int(order[0]) + (attempt % 2)))
        order[1] = 1 if (attempt + seed) % 3 else 0
        order[2] = min(5, max(0, int(order[2]) + ((attempt + 1) % 2)))
        updated["order"] = order
    elif name == "xgboost":
        updated["window"] = int(updated.get("window", 48)) + attempt * 6
        updated["n_estimators"] = int(updated.get("n_estimators", 200)) + attempt * 25
        updated["max_depth"] = min(12, int(updated.get("max_depth", 6)) + (attempt % 3))
        updated["learning_rate"] = max(0.01, float(updated.get("learning_rate", 0.05)) * (0.92 ** attempt))
        updated["subsample"] = min(0.98, max(0.55, float(updated.get("subsample", 0.8)) - 0.03 * attempt))
        updated["colsample_bytree"] = min(0.98, max(0.55, float(updated.get("colsample_bytree", 0.8)) - 0.02 * attempt))
        updated["random_seed"] = int(updated.get("random_seed", seed)) + attempt * 97
    elif name == "linear":
        updated["window"] = int(updated.get("window", 96)) + attempt * 12
        updated["fit_intercept"] = bool((attempt + seed) % 2)
    return updated


def _ensure_unique_params(name: str, params: Dict, seen_signatures: set[str], seed: int) -> Dict:
    candidate = dict(params)
    for attempt in range(0, 12):
        signature = f"{name}:{_param_signature(candidate)}"
        if signature not in seen_signatures:
            seen_signatures.add(signature)
            return candidate
        candidate = _bump_params(name, candidate, attempt + 1, seed)
    seen_signatures.add(f"{name}:{_param_signature(candidate)}")
    return candidate


def _select_models(iteration_index: int, feature_methods: List[str], history: List[Dict], param_history: List[Dict]) -> List[Dict]:
    has_diff = "difference_signature" in feature_methods
    has_ewm = "ewm_signature" in feature_methods
    has_peak = "peak_signature" in feature_methods
    has_calendar = "calendar_signature" in feature_methods
    has_lag = "lag_signature" in feature_methods
    bias = _history_bias(history)
    seen_signatures = {
        f"{item.get('name')}:{_param_signature(item.get('params', {}))}"
        for item in param_history
        if item.get("name") and item.get("params")
    }

    models: List[Dict] = []
    for item in FALLBACK_MODELS:
        name = item["name"]
        params = dict(item["params"])
        seed = _iteration_seed(iteration_index, name, feature_methods)
        if name == "arima":
            params["order"] = [
                min(5, 1 + ((iteration_index + bias) % 4)),
                0 if has_diff and iteration_index % 2 else 1,
                min(5, 1 + ((iteration_index + 1 + bias) % 4)),
            ]
        elif name == "xgboost":
            params["window"] = int(params.get("window", 48)) + (24 if has_lag else 0) + 12 * ((iteration_index + bias) % 3)
            params["n_estimators"] = int(params.get("n_estimators", 200)) + iteration_index * 40 + bias * 5
            params["max_depth"] = min(10, int(params.get("max_depth", 6)) + (1 if has_calendar else 0) + ((iteration_index + bias) % 2))
            params["learning_rate"] = max(0.015, float(params.get("learning_rate", 0.05)) - iteration_index * 0.006 + bias * 0.001)
            params["subsample"] = 0.7 if has_peak else (0.8 if iteration_index % 2 else 0.92)
            params["colsample_bytree"] = 0.68 if has_diff else (0.78 if bias % 2 else 0.9)
            params["random_seed"] = seed
        elif name == "linear":
            params["window"] = int(params.get("window", 96)) + (24 if has_lag else 0) + 12 * ((iteration_index + bias) % 3)
            params["fit_intercept"] = bool((iteration_index + bias + (1 if has_calendar else 0)) % 2)
        params = _ensure_unique_params(name, params, seen_signatures, seed)
        models.append(
            {
                "name": name,
                "params": params,
                "reason": f"{item['reason']} iteration={iteration_index}, features={','.join(feature_methods) or 'default'}",
            }
        )
    return [item for item in models if item["name"] in DEFAULT_ALLOWED_MODELS]


def run(state: DGGlobalState) -> Dict:
    iteration_index = int(state.read("current_iteration_index", 1))
    feature_methods = list(state.read("feature_engineering_result", {}).get("selected_methods", []) or [])
    history = list(state.read("iteration_history", []) or [])
    param_history = list(state.read("model_selection_history", []) or [])
    models = _select_models(iteration_index, feature_methods, history, param_history)
    selection_payload = {
        "models": models,
        "selection_source": "deterministic_rules",
        "iteration_index": iteration_index,
        "feature_methods": feature_methods,
        "history_length": len(history),
    }
    param_history.extend(
        [{"iteration_index": iteration_index, "name": item["name"], "params": item["params"]} for item in models]
    )
    state.write("model_selection_history", param_history)
    state.write("model_selection_result", selection_payload)
    write_step_artifact(state, "model_selection", selection_payload)
    return {
        "message": "selected three models",
        "models": [item["name"] for item in models],
    }
