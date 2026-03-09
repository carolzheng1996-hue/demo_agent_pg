from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def compute_metrics_extended(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = float(np.mean((y_pred - y_true) ** 2))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(mse))
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    mape = float(np.mean(np.abs((y_pred - y_true) / denom)) * 100.0)
    return {"mse": mse, "mae": mae, "rmse": rmse, "mape": mape}


def compare_models_with_metrics(model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ranked = sorted(model_results, key=lambda item: item.get("metrics", {}).get("mae", float("inf")))
    return {
        "ranking": [
            {
                "name": item.get("name"),
                "backend": item.get("backend"),
                "metrics": item.get("metrics", {}),
                "params": item.get("params", {}),
            }
            for item in ranked
        ],
        "best_model": ranked[0] if ranked else None,
    }


def mean_ensemble(model_results: List[Dict[str, Any]], actual: np.ndarray) -> Dict[str, Any]:
    if not model_results:
        raise ValueError("model_results is empty")
    preds = np.mean([np.asarray(row["predictions"], dtype=np.float64) for row in model_results], axis=0)
    metrics = compute_metrics_extended(actual, preds)
    return {
        "strategy": "simple_average_all_models",
        "member_models": [row.get("name") for row in model_results],
        "member_count": len(model_results),
        "predictions": preds.tolist(),
        "metrics": metrics,
    }
