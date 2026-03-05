from __future__ import annotations

import json
from typing import Dict, List

from config import MODEL_CONFIG
from global_state import GlobalState
from llm_client import LLMClient


def _fallback_model_names(rows: int) -> List[str]:
    names = ["arima", "xgboost", "lstm"]
    if rows > 30000:
        names = [n for n in names if n != "arima"]
    return names


def run(state: GlobalState) -> Dict:
    rows = int(state.read("data_shape", [0, 0])[0])
    llm = LLMClient(state.read("api_config", {}))
    selected_names = _fallback_model_names(rows)
    selection_reason = "fallback_rules"

    analysis = state.read("analysis", {})
    payload = llm.complete_json(
        system_prompt=(
            "Pick model candidates for time-series forecast from fixed names: arima, xgboost, lstm. "
            "Return JSON with keys: models (array), reason (string). "
            "Rules: keep 2-3 models, only choose from allowed names."
        ),
        user_prompt=json.dumps(
            {
                "rows": rows,
                "intent": state.read("intent"),
                "analysis_statistics": analysis.get("statistics", {}),
                "analysis_stationarity": analysis.get("stationarity", {}),
                "analysis_seasonality": analysis.get("seasonality", {}),
            },
            ensure_ascii=False,
        ),
        max_tokens=180,
        temperature=0.0,
    )
    if payload:
        raw_models = payload.get("models", [])
        if isinstance(raw_models, list):
            clean = []
            for name in raw_models:
                n = str(name).strip().lower()
                if n in MODEL_CONFIG and n not in clean:
                    clean.append(n)
            if clean:
                selected_names = clean
                selection_reason = str(payload.get("reason", "llm_selected"))

    candidates: List[Dict] = [{"name": name, **MODEL_CONFIG[name]} for name in selected_names]

    state.update(
        {
            "model_candidates": candidates,
            "selected_model": candidates[0] if candidates else None,
            "model_selection_reason": selection_reason,
        }
    )

    return {
        "message": f"selected {len(candidates)} model candidates ({selection_reason})",
        "candidates": [c["name"] for c in candidates],
    }
