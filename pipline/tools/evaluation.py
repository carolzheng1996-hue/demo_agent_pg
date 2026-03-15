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


def evaluate_iteration(
    current_result: Dict[str, Any],
    history: List[Dict[str, Any]],
    iteration_index: int,
    max_iterations: int,
    min_iterations: int = 2,
    patience: int = 2,
) -> Dict[str, Any]:
    current_score = float(current_result.get("best_score", float("inf")))
    best_previous = min((float(item.get("best_score", float("inf"))) for item in history), default=float("inf"))
    improved = current_score < best_previous

    stagnant_rounds = 0
    for item in reversed(history):
        prev = float(item.get("best_score", float("inf")))
        if current_score >= prev:
            stagnant_rounds += 1
        else:
            break

    should_continue = iteration_index < max_iterations and (
        iteration_index < min_iterations or stagnant_rounds < patience
    )
    return {
        "iteration_index": iteration_index,
        "best_score": current_score,
        "best_previous_score": None if best_previous == float("inf") else best_previous,
        "improved": improved,
        "stagnant_rounds": stagnant_rounds,
        "should_continue": should_continue,
        "stop_reason": "max_iterations_reached"
        if iteration_index >= max_iterations
        else "stagnation_limit_reached"
        if not should_continue and iteration_index >= min_iterations
        else "continue_search",
    }
