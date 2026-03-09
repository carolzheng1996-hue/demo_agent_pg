from __future__ import annotations

from typing import Dict

import pandas as pd

from ..state import DGGlobalState
from ..tools import write_step_artifact


def run(state: DGGlobalState) -> Dict:
    df = state.read_runtime("raw_df")
    if df is None:
        raise RuntimeError("Missing raw dataframe in runtime state. Run approved data_reading first.")

    target_col = state.read("dataset_profile", {}).get("target_column")
    engineered = df.copy()
    target = pd.to_numeric(engineered[target_col], errors="coerce").ffill().bfill()
    engineered[f"{target_col}_lag_1"] = target.shift(1)
    engineered[f"{target_col}_lag_24"] = target.shift(24)
    engineered[f"{target_col}_rolling_mean_24"] = target.rolling(24, min_periods=1).mean()
    engineered[f"{target_col}_rolling_std_24"] = target.rolling(24, min_periods=1).std().fillna(0.0)

    date_col = state.read("dataset_profile", {}).get("date_column")
    if date_col and date_col in engineered.columns:
        dt = pd.to_datetime(engineered[date_col], errors="coerce")
        engineered["hour"] = dt.dt.hour.fillna(0).astype(int)
        engineered["dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)

    feature_payload = {
        "engineered_columns": [c for c in engineered.columns if c not in df.columns] + [],
        "shape": [int(engineered.shape[0]), int(engineered.shape[1])],
    }
    state.write_runtime("engineered_df", engineered)
    state.write("feature_engineering_result", feature_payload)
    write_step_artifact(state, "feature_engineering", feature_payload)
    return {"message": "feature engineering completed", **feature_payload}
