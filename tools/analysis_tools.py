from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def _series(s: pd.Series) -> np.ndarray:
    return s.astype(float).to_numpy()


def statistics(series: pd.Series) -> Dict[str, Any]:
    x = _series(series)
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "missing_ratio": float(series.isna().mean()),
    }


def stationarity(series: pd.Series) -> Dict[str, Any]:
    x = _series(series)
    result = {"method": "heuristic", "is_stationary": False}
    try:
        from statsmodels.tsa.stattools import adfuller

        adf = adfuller(x)
        p_value = float(adf[1])
        result = {
            "method": "adfuller",
            "adf_stat": float(adf[0]),
            "p_value": p_value,
            "is_stationary": p_value < 0.05,
        }
    except Exception:
        half = len(x) // 2
        if half > 10:
            left_var = float(np.var(x[:half]))
            right_var = float(np.var(x[half:]))
            ratio = right_var / (left_var + 1e-8)
            result["variance_ratio"] = ratio
            result["is_stationary"] = 0.5 <= ratio <= 1.5
    return result


def ACF(series: pd.Series, max_lag: int = 48) -> Dict[str, Any]:
    x = _series(series)
    lags = {}
    for lag in range(1, min(max_lag, len(x) - 1) + 1):
        lags[str(lag)] = float(pd.Series(x).autocorr(lag=lag))
    return {"max_lag": max_lag, "acf": lags}


def seasonality(series: pd.Series, period_hint: int = 24) -> Dict[str, Any]:
    acf_map = ACF(series, max_lag=max(2 * period_hint, 48))["acf"]
    if not acf_map:
        return {"dominant_period": None, "seasonal_strength": 0.0}
    dominant_lag = max(acf_map, key=lambda k: abs(acf_map[k]))
    strength = float(abs(acf_map[dominant_lag]))
    return {"dominant_period": int(dominant_lag), "seasonal_strength": strength}


def trend(series: pd.Series) -> Dict[str, Any]:
    x = _series(series)
    t = np.arange(len(x), dtype=np.float64)
    slope, intercept = np.polyfit(t, x, deg=1)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "direction": "up" if slope > 0 else "down" if slope < 0 else "flat",
    }


def distribution(series: pd.Series) -> Dict[str, Any]:
    x = _series(series)
    mean = np.mean(x)
    std = np.std(x) + 1e-8
    skew = np.mean(((x - mean) / std) ** 3)
    kurt = np.mean(((x - mean) / std) ** 4) - 3.0
    q = np.quantile(x, [0.01, 0.25, 0.5, 0.75, 0.99])
    return {
        "skewness": float(skew),
        "kurtosis": float(kurt),
        "q01": float(q[0]),
        "q25": float(q[1]),
        "q50": float(q[2]),
        "q75": float(q[3]),
        "q99": float(q[4]),
    }
