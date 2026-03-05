from __future__ import annotations

from typing import Dict, List

import numpy as np

from global_state import GlobalState
from tools.eval_tools import compute_metrics
from tools.model_tools import train_arima, train_lstm, train_xgboost


def run(state: GlobalState) -> Dict:
    df = state.read_runtime("df")
    if df is None:
        raise RuntimeError("Missing dataframe in runtime state. Run data_reading first.")

    target_col = state.read("target_column")
    split_idx = int(state.read("train_size", 0))
    if split_idx <= 0:
        raise ValueError("Invalid train/test split index.")

    series = df[target_col].astype(float).to_numpy()
    train = series[:split_idx]
    test = series[split_idx:]
    state.write_runtime("test_target", test)

    candidates = state.read("model_candidates", [])
    results: List[Dict] = []

    for cfg in candidates:
        name = cfg.get("name")
        if name == "arima":
            out = train_arima(train, test, order=tuple(cfg.get("order", (2, 1, 2))))
        elif name == "xgboost":
            out = train_xgboost(
                train,
                test,
                window=int(cfg.get("window", 48)),
                n_estimators=int(cfg.get("n_estimators", 200)),
                max_depth=int(cfg.get("max_depth", 6)),
                learning_rate=float(cfg.get("learning_rate", 0.05)),
            )
        elif name == "lstm":
            out = train_lstm(
                train,
                test,
                seq_len=int(cfg.get("seq_len", 96)),
                hidden_size=int(cfg.get("hidden_size", 64)),
                num_layers=int(cfg.get("num_layers", 2)),
                dropout=float(cfg.get("dropout", 0.1)),
                epochs=int(cfg.get("epochs", 8)),
                lr=float(cfg.get("lr", 1e-3)),
                batch_size=int(cfg.get("batch_size", 64)),
            )
        else:
            continue

        pred = np.asarray(out["predictions"], dtype=np.float64)
        out["metrics"] = compute_metrics(test, pred)
        results.append(out)

    state.write("trained_models", results)

    return {
        "message": f"trained {len(results)} models",
        "models": [m["name"] for m in results],
    }
