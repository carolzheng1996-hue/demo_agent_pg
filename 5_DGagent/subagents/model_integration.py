from __future__ import annotations

from typing import Dict

import numpy as np

try:
    from ..state import DGGlobalState
    from ..tools import mean_ensemble
except ImportError:
    from state import DGGlobalState
    from tools import mean_ensemble


def run(state: DGGlobalState) -> Dict:
    training_result = state.read("model_training_result", {})
    model_results = training_result.get("results", [])
    if not model_results:
        raise RuntimeError("No model results available. Run model_training first.")

    test_target = state.read_runtime("test_target")
    if test_target is None:
        raise RuntimeError("Missing test target in runtime state.")

    ensemble_result = mean_ensemble(model_results, np.asarray(test_target, dtype=float))
    state.write("model_integration_result", ensemble_result)
    return {
        "message": "integrated three model outputs by averaging",
        "member_models": ensemble_result.get("member_models", []),
    }
