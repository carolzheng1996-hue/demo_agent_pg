from __future__ import annotations

from typing import Dict

import numpy as np

try:
    from ..config import DEFAULT_MAX_ITERATIONS
    from ..state import DGGlobalState
    from ..tools import evaluate_iteration, write_step_artifact
except ImportError:
    from config import DEFAULT_MAX_ITERATIONS
    from state import DGGlobalState
    from tools import evaluate_iteration, write_step_artifact


def run(state: DGGlobalState) -> Dict:
    training = state.read("model_training_result", {})
    integration = state.read("model_integration_result", {})
    ranking = training.get("ranking", {})
    best_model = ranking.get("best_model") or {}
    best_single_score = float(best_model.get("metrics", {}).get("mae", float("inf")))
    ensemble_score = float(integration.get("metrics", {}).get("mae", float("inf")))
    chosen = "ensemble" if ensemble_score <= best_single_score else "best_single"
    best_score = min(best_single_score, ensemble_score)
    iteration_index = int(state.read("current_iteration_index", 1))

    history = state.read("iteration_history", [])
    evaluation = evaluate_iteration(
        current_result={
            "best_score": best_score,
            "selected_strategy": chosen,
            "best_single_score": best_single_score,
            "ensemble_score": ensemble_score,
        },
        history=history,
        iteration_index=iteration_index,
        max_iterations=int(state.read("max_iterations", DEFAULT_MAX_ITERATIONS)),
    )
    chosen_result = integration if chosen == "ensemble" else best_model
    predictions = np.asarray(chosen_result.get("predictions", []), dtype=float)
    best_iteration_artifact = {
        "iteration_index": iteration_index,
        "strategy": chosen,
        "name": "iteration_ensemble" if chosen == "ensemble" else chosen_result.get("name"),
        "backend": "ensemble" if chosen == "ensemble" else chosen_result.get("backend"),
        "member_models": integration.get("member_models", []) if chosen == "ensemble" else [chosen_result.get("name")],
        "params": chosen_result.get("params", {}),
        "metrics": chosen_result.get("metrics", {}),
        "predictions": predictions.tolist(),
    }
    payload = {
        "selected_strategy": chosen,
        "best_single_score": best_single_score,
        "ensemble_score": ensemble_score,
        "iteration_best_result": {
            "iteration_index": iteration_index,
            "strategy": best_iteration_artifact["strategy"],
            "name": best_iteration_artifact["name"],
            "backend": best_iteration_artifact["backend"],
            "member_models": best_iteration_artifact["member_models"],
            "metrics": best_iteration_artifact["metrics"],
            "prediction_points": int(predictions.size),
        },
        **evaluation,
    }
    state.write_runtime("iteration_best_artifact", best_iteration_artifact)
    state.write("iteration_best_result", payload["iteration_best_result"])
    state.write("evaluator_result", payload)
    write_step_artifact(state, "evaluator", payload)
    return {"message": "iteration evaluated", **payload}
