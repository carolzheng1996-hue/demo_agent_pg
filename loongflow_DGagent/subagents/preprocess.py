from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

try:
    from ..state import DGGlobalState
    from ..tools import write_step_artifact
except ImportError:
    from state import DGGlobalState
    from tools import write_step_artifact


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
    scaled = df.copy()
    scaling = {}
    if should_normalize:
        for col in numeric_cols:
            mean = float(scaled[col].mean())
            std = float(scaled[col].std()) or 1.0
            scaled[col] = (scaled[col] - mean) / std
            scaling[col] = {"mean": mean, "std": std}

    target_col = state.read("dataset_profile", {}).get("target_column")
    target_series = pd.to_numeric(df[target_col], errors="coerce").ffill().bfill().to_numpy(dtype=float)
    state.write_runtime("preprocessed_df", scaled)
    state.write_runtime("preprocessed_target_series", np.asarray(target_series, dtype=float))
    payload = {
        "numeric_columns": numeric_cols,
        "scaling": scaling,
        "normalization_applied": should_normalize,
        "normalization_mode": scaling_policy.get("recommended_mode", "skip"),
    }
    state.write("preprocess_result", payload)
    write_step_artifact(state, "preprocess", payload)
    return {"message": "preprocess completed", "numeric_columns": len(numeric_cols), "normalization_applied": should_normalize}
