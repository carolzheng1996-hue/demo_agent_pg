from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from ..config import DEFAULT_TRAINING_PARAMS
    from ..tools.model_tools import train_arima, train_linear, train_xgboost
    from ..state import DGGlobalState
    from ..tools import compare_models_with_metrics, compute_metrics_extended, write_step_artifact
except ImportError:
    from config import DEFAULT_TRAINING_PARAMS
    from tools.model_tools import train_arima, train_linear, train_xgboost
    from state import DGGlobalState
    from tools import compare_models_with_metrics, compute_metrics_extended, write_step_artifact


def _read_frame_parts(path_value: str | List[str], columns: List[str]) -> pd.DataFrame:
    paths = path_value if isinstance(path_value, list) else [path_value]
    parts = [pd.read_parquet(Path(str(path)), columns=columns) for path in paths if path]
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, axis=0, ignore_index=True)


def _load_split_frames(state: DGGlobalState, columns: List[str]) -> Dict[str, pd.DataFrame]:
    dataset_paths = state.read("preprocess_result", {}).get("dataset_paths", {})
    if not dataset_paths.get("train"):
        raise RuntimeError("Missing preprocessed train dataset path.")
    return {
        name: _read_frame_parts(path_value, columns)
        for name, path_value in dataset_paths.items()
        if path_value
    }


def run(state: DGGlobalState) -> Dict:
    profile = state.read("dataset_profile", {})
    target_col = profile.get("target_column")
    if not target_col:
        raise RuntimeError("Missing target column in dataset profile.")

    preprocess_result = state.read("preprocess_result", {}) or {}
    candidate_columns = list(dict.fromkeys([target_col] + list(preprocess_result.get("numeric_columns", []) or [])))
    frames = _load_split_frames(state, candidate_columns)
    train_df = frames["train"].reset_index(drop=True)
    val_df = frames.get("val", train_df.iloc[0:0].copy()).reset_index(drop=True)
    if any(target_col not in frame.columns for frame in frames.values()):
        raise ValueError(f"Target column {target_col} not found in preprocessed splits.")

    train = pd.to_numeric(train_df[target_col], errors="coerce").to_numpy(dtype=float)
    val = pd.to_numeric(val_df[target_col], errors="coerce").to_numpy(dtype=float)
    split_applied = len(val) > 0

    train_frame = train_df.select_dtypes(include=["number"]).astype("float32", copy=False)
    val_frame = val_df.select_dtypes(include=["number"]).astype("float32", copy=False)
    for frame in (train_frame, val_frame):
        if "__row_id__" in frame.columns:
            frame.drop(columns=["__row_id__"], inplace=True)

    results: List[Dict] = []
    feature_methods = list(state.read("feature_engineering_result", {}).get("selected_methods", []) or [])
    selected_models = state.read("model_selection_result", {}).get("models", []) or [
        {"name": "arima", "params": DEFAULT_TRAINING_PARAMS["arima"]},
        {"name": "xgboost", "params": DEFAULT_TRAINING_PARAMS["xgboost"]},
        {"name": "linear", "params": DEFAULT_TRAINING_PARAMS["linear"]},
    ]
    for item in selected_models:
        name = item["name"]
        params = item.get("params", {})
        if not split_applied:
            output = {
                "name": name,
                "backend": "fit_only",
                "params": params,
                "predictions": [],
                "metrics": {},
                "training_mode": "fit_only_without_split",
            }
            results.append(output)
            continue
        if name == "arima":
            output = train_arima(train, val, order=tuple(params.get("order", [2, 1, 2])))
        elif name == "xgboost":
            output = train_xgboost(
                train,
                val,
                window=int(params.get("window", 48)),
                n_estimators=int(params.get("n_estimators", 200)),
                max_depth=int(params.get("max_depth", 6)),
                learning_rate=float(params.get("learning_rate", 0.05)),
                subsample=float(params.get("subsample", 0.8)),
                colsample_bytree=float(params.get("colsample_bytree", 0.8)),
                random_seed=int(params.get("random_seed", 42)),
                feature_methods=feature_methods,
                train_frame=train_frame,
                test_frame=val_frame,
                target_col=target_col,
            )
        elif name == "linear":
            output = train_linear(
                train,
                val,
                window=int(params.get("window", 96)),
                fit_intercept=bool(params.get("fit_intercept", True)),
                train_frame=train_frame,
                test_frame=val_frame,
                target_col=target_col,
            )
        else:
            continue
        preds = np.asarray(output["predictions"], dtype=float)
        output["metrics"] = compute_metrics_extended(val, preds)
        results.append(output)

    ranking = compare_models_with_metrics(results)
    payload = {
        "results": results,
        "ranking": ranking,
        "target_column": target_col,
        "feature_methods": feature_methods,
        "modeling_feature_count": int(train_frame.shape[1]),
        "modeling_features": train_frame.columns.tolist(),
        "split_applied": split_applied,
        "split_counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
        },
    }
    state.write_runtime("validation_target", val)
    state.write("model_training_result", payload)
    write_step_artifact(state, "model_training", payload)
    return {
        "message": "trained selected models",
        "trained_models": [row["name"] for row in results],
    }
