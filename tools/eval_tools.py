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
                "backend": x.get("backend"),
                "mae": x.get("metrics", {}).get("mae"),
                "rmse": x.get("metrics", {}).get("rmse"),
                "mape": x.get("metrics", {}).get("mape"),
                "params": x.get("params", {}),
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
    weight = 1.0 / float(len(members))
    member_details = [
        {
            "name": m.get("name"),
            "backend": m.get("backend"),
            "metrics": m.get("metrics", {}),
            "params": m.get("params", {}),
            "weight": weight,
        }
        for m in members
    ]

    output: Dict[str, Any] = {
        "strategy": "simple_average",
        "selection_rule": "top_k_by_lowest_mae",
        "top_k_requested": int(top_k),
        "top_k_used": int(len(members)),
        "member_models": [m["name"] for m in members],
        "member_details": member_details,
        "predictions": preds.tolist(),
    }
    if actual is not None:
        ens_metrics = compute_metrics(np.asarray(actual, dtype=np.float64), preds)
        output["metrics"] = ens_metrics
        best_metrics = members[0].get("metrics", {}) if members else {}
        if best_metrics:
            output["delta_vs_best_single"] = {
                "mae": float(ens_metrics.get("mae", 0.0) - float(best_metrics.get("mae", 0.0))),
                "rmse": float(ens_metrics.get("rmse", 0.0) - float(best_metrics.get("rmse", 0.0))),
                "mape": float(ens_metrics.get("mape", 0.0) - float(best_metrics.get("mape", 0.0))),
            }
    return output
