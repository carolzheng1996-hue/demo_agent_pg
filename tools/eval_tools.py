from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    mape = float(np.mean(np.abs((y_pred - y_true) / denom)) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def compare_models(model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ranked = sorted(model_results, key=lambda x: x.get("metrics", {}).get("mae", float("inf")))
    best = ranked[0] if ranked else None
    return {
        "ranking": [
            {
                "name": x.get("name"),
                "mae": x.get("metrics", {}).get("mae"),
                "rmse": x.get("metrics", {}).get("rmse"),
                "mape": x.get("metrics", {}).get("mape"),
            }
            for x in ranked
        ],
        "best_model": best,
    }


def create_ensemble(
    model_results: List[Dict[str, Any]],
    actual: Optional[np.ndarray] = None,
    top_k: int = 2,
) -> Dict[str, Any]:
    ranked = sorted(model_results, key=lambda x: x.get("metrics", {}).get("mae", float("inf")))
    members = ranked[: max(1, top_k)]
    preds = np.mean([np.asarray(m["predictions"], dtype=np.float64) for m in members], axis=0)

    output: Dict[str, Any] = {
        "member_models": [m["name"] for m in members],
        "predictions": preds.tolist(),
    }
    if actual is not None:
        output["metrics"] = compute_metrics(np.asarray(actual, dtype=np.float64), preds)
    return output
