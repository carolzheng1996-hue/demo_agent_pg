from __future__ import annotations

from typing import Dict

from ..config import DEFAULT_MAX_ITERATIONS
from ..state import DGGlobalState
from ..tools import evaluate_iteration, write_step_artifact


def run(state: DGGlobalState) -> Dict:
    training = state.read("model_training_result", {})
    integration = state.read("model_integration_result", {})
    ranking = training.get("ranking", {})
    best_model = ranking.get("best_model") or {}
    best_single_score = float(best_model.get("metrics", {}).get("mae", float("inf")))
    ensemble_score = float(integration.get("metrics", {}).get("mae", float("inf")))
    chosen = "ensemble" if ensemble_score <= best_single_score else "best_single"
    best_score = min(best_single_score, ensemble_score)

    history = state.read("iteration_history", [])
    evaluation = evaluate_iteration(
        current_result={
            "best_score": best_score,
            "selected_strategy": chosen,
            "best_single_score": best_single_score,
            "ensemble_score": ensemble_score,
        },
        history=history,
        iteration_index=int(state.read("current_iteration_index", 1)),
        max_iterations=int(state.read("max_iterations", DEFAULT_MAX_ITERATIONS)),
    )
    payload = {
        "selected_strategy": chosen,
        "best_single_score": best_single_score,
        "ensemble_score": ensemble_score,
        **evaluation,
    }
    state.write("evaluator_result", payload)
    write_step_artifact(state, "evaluator", payload)
    return {"message": "iteration evaluated", **payload}
