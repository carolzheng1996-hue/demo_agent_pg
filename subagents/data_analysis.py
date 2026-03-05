from __future__ import annotations

from typing import Dict

from global_state import GlobalState
from tools.analysis_tools import ACF, distribution, seasonality, stationarity, statistics, trend


def run(state: GlobalState) -> Dict:
    df = state.read_runtime("df")
    if df is None:
        raise RuntimeError("Missing dataframe in runtime state. Run data_reading first.")

    target_col = state.read("target_column")
    if target_col not in df.columns:
        raise ValueError(f"target column not found: {target_col}")

    s = df[target_col]
    analysis = {
        "statistics": statistics(s),
        "stationarity": stationarity(s),
        "acf": ACF(s),
        "seasonality": seasonality(s),
        "trend": trend(s),
        "distribution": distribution(s),
    }
    state.write("analysis", analysis)

    return {
        "message": "analysis completed",
        "rows": int(len(s)),
    }
