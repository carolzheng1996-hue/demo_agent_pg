from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from tools.model_tools import train_arima, train_lstm, train_xgboost

from ..state import DGGlobalState
from ..tools import compare_models_with_metrics, compute_metrics_extended


def _slice_series(series: np.ndarray, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, np.ndarray]:
    n = len(series)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return {
        "train": series[:train_end],
        "val": series[train_end:val_end],
        "test": series[val_end:],
    }


def run(state: DGGlobalState) -> Dict:
    df = state.read_runtime("raw_df")
    if df is None:
        raise RuntimeError("Missing raw dataframe in runtime state. Run approved data_reading first.")

    profile = state.read("dataset_profile", {})
    target_col = profile.get("target_column")
    ratios = profile.get("split_ratios") or {}
    train_ratio = float(ratios.get("train_ratio"))
    val_ratio = float(ratios.get("val_ratio"))
    test_ratio = float(ratios.get("test_ratio"))

    series = pd.to_numeric(df[target_col], errors="coerce").ffill().bfill().to_numpy(dtype=float)
    splits = _slice_series(series, train_ratio, val_ratio, test_ratio)
    train_plus_val = np.concatenate([splits["train"], splits["val"]]) if len(splits["val"]) else splits["train"]
    test = splits["test"]
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
    state.write("model_training_result", {"results": results, "ranking": ranking})
    return {
        "message": "trained selected models",
        "trained_models": [row["name"] for row in results],
    }
