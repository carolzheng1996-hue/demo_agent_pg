from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from ..tools.model_tools import train_arima, train_lstm, train_xgboost
    from ..state import DGGlobalState
    from ..tools import compare_models_with_metrics, compute_metrics_extended, write_step_artifact
except ImportError:
    from tools.model_tools import train_arima, train_lstm, train_xgboost
    from state import DGGlobalState
    from tools import compare_models_with_metrics, compute_metrics_extended, write_step_artifact


def run(state: DGGlobalState) -> Dict:
    df = state.read_runtime("raw_df")
    if df is None:
        raise RuntimeError("Missing raw dataframe in runtime state. Run approved data_reading first.")

    profile = state.read("dataset_profile", {})
    target_col = profile.get("target_column")
    split_indices = state.read_runtime("split_indices")
    preprocessed_series = state.read_runtime("preprocessed_target_series")
    if split_indices is None or preprocessed_series is None:
        raise RuntimeError("Missing split strategy or preprocessed series.")

    series = np.asarray(preprocessed_series, dtype=float)
    train = series[split_indices["train"][0] : split_indices["train"][1]]
    val = series[split_indices["val"][0] : split_indices["val"][1]]
    test = series[split_indices["test"][0] : split_indices["test"][1]]
    train_plus_val = np.concatenate([train, val]) if len(val) else train
    if len(test) == 0:
        raise ValueError("Test split is empty. Please adjust split ratios.")

    results: List[Dict] = []
    for item in state.read("model_selection_result", {}).get("models", []):
        name = item["name"]
        params = item.get("params", {})
        if name == "arima":
            output = train_arima(train_plus_val, test, order=tuple(params.get("order", [2, 1, 2])))
        elif name == "xgboost":
            output = train_xgboost(
                train_plus_val,
                test,
                window=int(params.get("window", 48)),
                n_estimators=int(params.get("n_estimators", 200)),
                max_depth=int(params.get("max_depth", 6)),
                learning_rate=float(params.get("learning_rate", 0.05)),
            )
        elif name == "lstm":
            output = train_lstm(
                train_plus_val,
                test,
                seq_len=int(params.get("seq_len", 96)),
                hidden_size=int(params.get("hidden_size", 64)),
                num_layers=int(params.get("num_layers", 2)),
                dropout=float(params.get("dropout", 0.1)),
                epochs=int(params.get("epochs", 8)),
                lr=float(params.get("lr", 0.001)),
                batch_size=int(params.get("batch_size", 64)),
            )
        else:
            continue
        preds = np.asarray(output["predictions"], dtype=float)
        output["metrics"] = compute_metrics_extended(test, preds)
        results.append(output)

    ranking = compare_models_with_metrics(results)
    state.write_runtime("test_target", test)
    payload = {"results": results, "ranking": ranking, "target_column": target_col}
    state.write("model_training_result", payload)
    write_step_artifact(state, "model_training", payload)
    return {
        "message": "trained selected models",
        "trained_models": [row["name"] for row in results],
    }
