from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from ..state import DGGlobalState
    from ..tools import write_step_artifact
except ImportError:
    from state import DGGlobalState
    from tools import write_step_artifact


def _normalize_numeric_frame(df: pd.DataFrame, numeric_cols: List[str], mode: str, use_partial_fit: bool) -> tuple[pd.DataFrame, Dict]:
    scaled = df.copy()
    scaling: Dict = {}
    if not numeric_cols:
        return scaled, scaling

    # For directory inputs we prefer incremental fitting when sklearn is available,
    # which avoids assuming a single-file load path for large multi-file datasets.
    if use_partial_fit and mode == "zscore":
        try:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            chunk_size = max(256, min(len(scaled), 4096))
            values = scaled[numeric_cols].to_numpy(dtype=float)
            for start in range(0, len(values), chunk_size):
                scaler.partial_fit(values[start : start + chunk_size])
            transformed = scaler.transform(values)
            scaled.loc[:, numeric_cols] = transformed
            for index, col in enumerate(numeric_cols):
                scaling[col] = {
                    "mean": float(scaler.mean_[index]),
                    "scale": float(scaler.scale_[index] or 1.0),
                    "mode": "zscore_partial_fit",
                }
            return scaled, scaling
        except Exception:
            pass

    if use_partial_fit and mode == "minmax":
        try:
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()
            scaler.fit(scaled[numeric_cols].to_numpy(dtype=float))
            transformed = scaler.transform(scaled[numeric_cols].to_numpy(dtype=float))
            scaled.loc[:, numeric_cols] = transformed
            for index, col in enumerate(numeric_cols):
                scaling[col] = {
                    "data_min": float(scaler.data_min_[index]),
                    "data_max": float(scaler.data_max_[index]),
                    "mode": "minmax_sklearn",
                }
            return scaled, scaling
        except Exception:
            pass

    for col in numeric_cols:
        values = pd.to_numeric(scaled[col], errors="coerce").to_numpy(dtype=float)
        if mode == "minmax":
            min_value = float(np.nanmin(values))
            max_value = float(np.nanmax(values))
            scale = max(max_value - min_value, 1e-8)
            scaled[col] = (values - min_value) / scale
            scaling[col] = {"data_min": min_value, "data_max": max_value, "scale": scale, "mode": "minmax"}
        else:
            mean = float(np.nanmean(values))
            std = float(np.nanstd(values)) or 1.0
            scaled[col] = (values - mean) / std
            scaling[col] = {"mean": mean, "std": std, "mode": "zscore"}
    return scaled, scaling


def run(state: DGGlobalState) -> Dict:
    engineered_df = state.read_runtime("engineered_df")
    if engineered_df is None:
        raise RuntimeError("Missing engineered dataframe in runtime state.")

    df = engineered_df.copy()
    for col in df.columns:
        if str(df[col].dtype) != "datetime64[ns]":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.ffill().bfill().fillna(0.0)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    scaling_policy = state.read("datanorm_result", {})
    should_normalize = bool(scaling_policy.get("should_normalize", False))
    normalization_mode = str(scaling_policy.get("recommended_mode", "skip") or "skip").lower()
    scaled = df.copy()
    scaling: Dict = {}
    used_partial_fit = False
    if should_normalize and normalization_mode in {"zscore", "minmax"}:
        use_partial_fit = bool(state.read("dataset_loading_result", {}).get("is_directory", False))
        scaled, scaling = _normalize_numeric_frame(df, numeric_cols, normalization_mode, use_partial_fit)
        used_partial_fit = use_partial_fit and any(str(item.get("mode", "")).endswith(("partial_fit", "sklearn")) for item in scaling.values())

    target_col = state.read("dataset_profile", {}).get("target_column")
    target_series = pd.to_numeric(scaled[target_col], errors="coerce").ffill().bfill().to_numpy(dtype=float)
    state.write_runtime("preprocessed_df", scaled)
    state.write_runtime("preprocessed_target_series", np.asarray(target_series, dtype=float))
    payload = {
        "numeric_columns": numeric_cols,
        "scaling": scaling,
        "normalization_applied": should_normalize,
        "normalization_mode": normalization_mode,
        "preprocessed_shape": [int(scaled.shape[0]), int(scaled.shape[1])],
        "used_partial_fit": used_partial_fit,
    }
    state.write("preprocess_result", payload)
    write_step_artifact(state, "preprocess", payload)
    return {
        "message": "preprocess completed",
        "numeric_columns": len(numeric_cols),
        "normalization_applied": should_normalize,
        "normalization_mode": normalization_mode,
    }
