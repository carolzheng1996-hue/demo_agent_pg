from __future__ import annotations

from typing import Dict

import numpy as np

from global_state import GlobalState
from tools.eval_tools import compare_models, create_ensemble


def run(state: GlobalState) -> Dict:
    model_results = state.read("trained_models", [])
    if not model_results:
        raise RuntimeError("No trained models found. Run model_training first.")

    test_target = state.read_runtime("test_target")
    if test_target is None:
        raise RuntimeError("Missing test target in runtime state.")

    ranking = compare_models(model_results)
    ensemble = create_ensemble(model_results, actual=np.asarray(test_target), top_k=2)

    ranking_rows = ranking.get("ranking", [])
    ensemble_members = ensemble.get("member_models", [])
    payload = {
        "ranking": ranking,
        "all_models": ranking_rows,
        "ensemble": ensemble,
        "ensemble_selection": {
            "strategy": ensemble.get("strategy"),
            "selection_rule": ensemble.get("selection_rule"),
            "top_k_requested": ensemble.get("top_k_requested"),
            "top_k_used": ensemble.get("top_k_used"),
            "member_models": ensemble_members,
        },
        "selected_model": ranking.get("best_model", {}).get("name") if ranking.get("best_model") else None,
    }
    state.write("integration", payload)

    return {
        "message": "integration completed",
        "selected_model": payload["selected_model"],
    }
