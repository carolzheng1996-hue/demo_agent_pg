from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from tools.analysis_tools import ACF, distribution, seasonality, stationarity, statistics, trend


def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    preview = df.head(5).to_dict(orient="records")
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": df.columns.tolist(),
        "dtypes": {k: str(v) for k, v in df.dtypes.astype(str).to_dict().items()},
        "numeric_columns": numeric_cols,
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "preview": preview,
    }


def compute_full_analysis(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    if target_column not in df.columns:
        raise ValueError(f"target column not found: {target_column}")
    series = df[target_column]
    return {
        "dataset_summary": summarize_dataframe(df),
        "target_column": target_column,
        "statistics": statistics(series),
        "stationarity": stationarity(series),
        "acf": ACF(series),
        "seasonality": seasonality(series),
        "trend": trend(series),
        "distribution": distribution(series),
    }
