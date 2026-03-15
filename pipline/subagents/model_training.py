from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from ..tools.model_tools import train_arima, train_linear, train_xgboost
    from ..state import DGGlobalState
    from ..tools import compare_models_with_metrics, compute_metrics_extended, write_step_artifact
except ImportError:
    from tools.model_tools import train_arima, train_linear, train_xgboost
    from state import DGGlobalState
    from tools import compare_models_with_metrics, compute_metrics_extended, write_step_artifact


def run(state: DGGlobalState) -> Dict:
    profile = state.read("dataset_profile", {})
    target_col = profile.get("target_column")
    split_indices = state.read_runtime("split_indices")
    preprocessed_series = state.read_runtime("preprocessed_target_series")
    preprocessed_df = state.read_runtime("preprocessed_df")
    if split_indices is None or preprocessed_series is None or preprocessed_df is None:
        raise RuntimeError("Missing split strategy or preprocessed series.")

    series = np.asarray(preprocessed_series, dtype=float)
    train = series[split_indices["train"][0] : split_indices["train"][1]]
    val = series[split_indices["val"][0] : split_indices["val"][1]]
    test = series[split_indices["test"][0] : split_indices["test"][1]]
    train_plus_val = np.concatenate([train, val]) if len(val) else train
    if len(test) == 0:
        raise ValueError("Test split is empty. Please adjust split ratios.")

    modeling_df = preprocessed_df.select_dtypes(include=["number"]).copy()
    train_frame = modeling_df.iloc[split_indices["train"][0] : split_indices["val"][1]].reset_index(drop=True)
    test_frame = modeling_df.iloc[split_indices["test"][0] : split_indices["test"][1]].reset_index(drop=True)

    results: List[Dict] = []
    feature_methods = list(state.read("feature_engineering_result", {}).get("selected_methods", []) or [])
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
                subsample=float(params.get("subsample", 0.8)),
                colsample_bytree=float(params.get("colsample_bytree", 0.8)),
                random_seed=int(params.get("random_seed", 42)),
                feature_methods=feature_methods,
                train_frame=train_frame,
                test_frame=test_frame,
                target_col=target_col,
            )
        elif name == "linear":
            output = train_linear(
                train_plus_val,
                test,
                window=int(params.get("window", 96)),
                fit_intercept=bool(params.get("fit_intercept", True)),
                train_frame=train_frame,
                test_frame=test_frame,
                target_col=target_col,
            )
        else:
            continue
        preds = np.asarray(output["predictions"], dtype=float)
        output["metrics"] = compute_metrics_extended(test, preds)
        results.append(output)

    ranking = compare_models_with_metrics(results)
    state.write_runtime("test_target", test)
    payload = {
        "results": results,
        "ranking": ranking,
        "target_column": target_col,
        "feature_methods": feature_methods,
        "modeling_feature_count": int(modeling_df.shape[1]),
        "modeling_features": modeling_df.columns.tolist(),
    }
    state.write("model_training_result", payload)
    write_step_artifact(state, "model_training", payload)
    return {
        "message": "trained selected models",
        "trained_models": [row["name"] for row in results],
    }
