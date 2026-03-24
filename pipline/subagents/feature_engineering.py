from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from ..state import DGGlobalState
    from ..tools import write_step_artifact, write_step_dataframe_artifact
    from .ds_pipeline_utils import load_ds_dataframe, load_ds_station_frames, normalize_sequence_value
except ImportError:
    from state import DGGlobalState
    from tools import write_step_artifact, write_step_dataframe_artifact
    from subagents.ds_pipeline_utils import load_ds_dataframe, load_ds_station_frames, normalize_sequence_value


FeatureMethod = Callable[[pd.DataFrame, Sequence[str]], Tuple[pd.DataFrame, List[str]]]


def _safe_sequence(series: pd.Series) -> List[np.ndarray]:
    values: List[np.ndarray] = []
    for value in series:
        arr = normalize_sequence_value(value)
        values.append(arr if arr is not None else np.zeros(0, dtype=float))
    return values


def _concat_feature_block(df: pd.DataFrame, block: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, List[str]]:
    filtered = {name: values for name, values in block.items() if name not in df.columns}
    if not filtered:
        return df, []
    return pd.concat([df, pd.DataFrame(filtered, index=df.index)], axis=1).copy(), list(filtered.keys())


def _safe_skew(arr: np.ndarray) -> float:
    if len(arr) < 3:
        return 0.0
    mean = float(arr.mean())
    std = float(arr.std())
    if std <= 1e-12:
        return 0.0
    centered = arr - mean
    return float(np.mean(centered ** 3) / (std ** 3))


def _safe_kurtosis(arr: np.ndarray) -> float:
    if len(arr) < 4:
        return 0.0
    mean = float(arr.mean())
    std = float(arr.std())
    if std <= 1e-12:
        return 0.0
    centered = arr - mean
    return float(np.mean(centered ** 4) / (std ** 4))


def _safe_mean_n_absolute_max(arr: np.ndarray, n: int = 3) -> float:
    if not len(arr):
        return 0.0
    ranked = np.sort(np.abs(arr))[-min(n, len(arr)) :]
    return float(ranked.mean()) if len(ranked) else 0.0


def _safe_absolute_sum_of_changes(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    return float(np.abs(np.diff(arr)).sum())


def _safe_mean_change(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    return float(np.diff(arr).mean())


def _safe_mean_abs_change(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    return float(np.abs(np.diff(arr)).mean())


def _safe_mean_second_derivative_central(arr: np.ndarray) -> float:
    if len(arr) < 3:
        return 0.0
    second = arr[2:] - 2 * arr[1:-1] + arr[:-2]
    return float(second.mean()) if len(second) else 0.0


def _safe_slope(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=float)
    x_mean = float(x.mean())
    y_mean = float(arr.mean())
    denominator = float(np.square(x - x_mean).sum())
    if denominator <= 1e-12:
        return 0.0
    numerator = float(((x - x_mean) * (arr - y_mean)).sum())
    return float(numerator / denominator)


def _add_distribution_features(df: pd.DataFrame, columns: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    block: Dict[str, pd.Series] = {}
    for column in columns:
        sequences = _safe_sequence(df[column])
        block[f"{column}__mean"] = pd.Series([float(arr.mean()) if len(arr) else 0.0 for arr in sequences], index=df.index)
        block[f"{column}__median"] = pd.Series([float(np.median(arr)) if len(arr) else 0.0 for arr in sequences], index=df.index)
        block[f"{column}__minimum"] = pd.Series([float(arr.min()) if len(arr) else 0.0 for arr in sequences], index=df.index)
        block[f"{column}__maximum"] = pd.Series([float(arr.max()) if len(arr) else 0.0 for arr in sequences], index=df.index)
        block[f"{column}__standard_deviation"] = pd.Series([float(arr.std()) if len(arr) else 0.0 for arr in sequences], index=df.index)
        block[f"{column}__variance"] = pd.Series([float(arr.var()) if len(arr) else 0.0 for arr in sequences], index=df.index)
        block[f"{column}__skewness"] = pd.Series([_safe_skew(arr) for arr in sequences], index=df.index)
        block[f"{column}__kurtosis"] = pd.Series([_safe_kurtosis(arr) for arr in sequences], index=df.index)
        block[f"{column}__absolute_maximum"] = pd.Series([float(np.abs(arr).max()) if len(arr) else 0.0 for arr in sequences], index=df.index)
        block[f"{column}__root_mean_square"] = pd.Series([float(np.sqrt(np.mean(np.square(arr)))) if len(arr) else 0.0 for arr in sequences], index=df.index)
        block[f"{column}__quantile_25"] = pd.Series([float(np.quantile(arr, 0.25)) if len(arr) else 0.0 for arr in sequences], index=df.index)
        block[f"{column}__quantile_50"] = pd.Series([float(np.quantile(arr, 0.50)) if len(arr) else 0.0 for arr in sequences], index=df.index)
        block[f"{column}__quantile_75"] = pd.Series([float(np.quantile(arr, 0.75)) if len(arr) else 0.0 for arr in sequences], index=df.index)
        block[f"{column}__mean_n_absolute_max"] = pd.Series([_safe_mean_n_absolute_max(arr) for arr in sequences], index=df.index)
    return _concat_feature_block(df, block)


def _add_lag_features(df: pd.DataFrame, columns: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    block: Dict[str, pd.Series] = {}
    for column in columns:
        sequences = _safe_sequence(df[column])
        block[f"{column}__lag_1"] = pd.Series([float(arr[-1]) if len(arr) >= 1 else 0.0 for arr in sequences], index=df.index)
        block[f"{column}__lag_6"] = pd.Series([float(arr[-6]) if len(arr) >= 6 else 0.0 for arr in sequences], index=df.index)
    return _concat_feature_block(df, block)


def _add_rolling_features(df: pd.DataFrame, columns: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    block: Dict[str, pd.Series] = {}
    for column in columns:
        sequences = _safe_sequence(df[column])
        block[f"{column}__rolling_mean_6"] = pd.Series([float(arr[-min(6, len(arr)) :].mean()) if len(arr) else 0.0 for arr in sequences], index=df.index)
        block[f"{column}__rolling_mean_full"] = pd.Series([float(arr.mean()) if len(arr) else 0.0 for arr in sequences], index=df.index)
        block[f"{column}__rolling_std_full"] = pd.Series([float(arr.std()) if len(arr) else 0.0 for arr in sequences], index=df.index)
        block[f"{column}__rolling_min_full"] = pd.Series([float(arr.min()) if len(arr) else 0.0 for arr in sequences], index=df.index)
        block[f"{column}__rolling_max_full"] = pd.Series([float(arr.max()) if len(arr) else 0.0 for arr in sequences], index=df.index)
    return _concat_feature_block(df, block)


def _add_difference_features(df: pd.DataFrame, columns: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    block: Dict[str, pd.Series] = {}
    for column in columns:
        sequences = _safe_sequence(df[column])
        block[f"{column}__diff_1"] = pd.Series([float(arr[-1] - arr[-2]) if len(arr) >= 2 else 0.0 for arr in sequences], index=df.index)
    return _concat_feature_block(df, block)


def _add_ewm_features(df: pd.DataFrame, columns: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    block: Dict[str, pd.Series] = {}
    for column in columns:
        sequences = _safe_sequence(df[column])
        ewm_mean: List[float] = []
        ewm_std: List[float] = []
        for arr in sequences:
            if not len(arr):
                ewm_mean.append(0.0)
                ewm_std.append(0.0)
                continue
            series = pd.Series(arr, dtype=float)
            ewm = series.ewm(span=min(12, max(2, len(arr))), adjust=False).mean()
            ewm_mean.append(float(ewm.iloc[-1]))
            ewm_std.append(float(series.ewm(span=min(24, max(2, len(arr))), adjust=False).std().fillna(0.0).iloc[-1]))
        block[f"{column}__ewm_mean"] = pd.Series(ewm_mean, index=df.index)
        block[f"{column}__ewm_std"] = pd.Series(ewm_std, index=df.index)
    return _concat_feature_block(df, block)


def _add_peak_features(df: pd.DataFrame, columns: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    block: Dict[str, pd.Series] = {}
    for column in columns:
        sequences = _safe_sequence(df[column])
        block[f"{column}__abs_energy"] = pd.Series([float(np.square(arr).sum()) if len(arr) else 0.0 for arr in sequences], index=df.index)
        block[f"{column}__value_range"] = pd.Series([float(arr.max() - arr.min()) if len(arr) else 0.0 for arr in sequences], index=df.index)
        block[f"{column}__absolute_sum_of_changes"] = pd.Series([_safe_absolute_sum_of_changes(arr) for arr in sequences], index=df.index)
        block[f"{column}__mean_change"] = pd.Series([_safe_mean_change(arr) for arr in sequences], index=df.index)
        block[f"{column}__mean_abs_change"] = pd.Series([_safe_mean_abs_change(arr) for arr in sequences], index=df.index)
        block[f"{column}__mean_second_derivative_central"] = pd.Series([_safe_mean_second_derivative_central(arr) for arr in sequences], index=df.index)
        block[f"{column}__slope"] = pd.Series([_safe_slope(arr) for arr in sequences], index=df.index)
        block[f"{column}__turning_points"] = pd.Series(
            [float(np.sum(np.diff(np.sign(np.diff(arr))) != 0)) if len(arr) > 2 else 0.0 for arr in sequences],
            index=df.index,
        )
    return _concat_feature_block(df, block)


def _add_calendar_features(df: pd.DataFrame, _: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    if "timestamp_win" not in df.columns:
        return df, []
    dt = pd.to_datetime(df["timestamp_win"], errors="coerce")
    block = {
        "hour": dt.dt.hour.fillna(0).astype(int),
        "dayofweek": dt.dt.dayofweek.fillna(0).astype(int),
        "day": dt.dt.day.fillna(0).astype(int),
        "month": dt.dt.month.fillna(0).astype(int),
        "is_weekend": dt.dt.dayofweek.isin([5, 6]).fillna(False).astype(int),
    }
    return _concat_feature_block(df, block)


FEATURE_METHOD_POOL: Dict[str, FeatureMethod] = {
    "lag_signature": _add_lag_features,
    "distribution_signature": _add_distribution_features,
    "rolling_signature": _add_rolling_features,
    "difference_signature": _add_difference_features,
    "ewm_signature": _add_ewm_features,
    "peak_signature": _add_peak_features,
    "calendar_signature": _add_calendar_features,
}


def _pick_methods(state: DGGlobalState, ds_df: pd.DataFrame, target_col: str) -> List[str]:
    analysis_result = state.read("data_analysis_result", {}) or {}
    quality_checks = analysis_result.get("quality_checks", {}) or {}
    base_analysis = analysis_result.get("base_analysis", {}) or {}
    stationarity = base_analysis.get("stationarity", {}) or {}
    seasonality = base_analysis.get("seasonality", {}) or {}
    target_quality = quality_checks.get("target_sequence_summary", {}) or {}

    selected = ["lag_signature", "distribution_signature", "rolling_signature", "difference_signature"]
    if not stationarity.get("is_stationary"):
        selected.append("ewm_signature")
    if seasonality.get("dominant_period") not in (None, "", 0) and "timestamp_win" in ds_df.columns:
        selected.append("calendar_signature")
    if int(target_quality.get("max_length", 0)) >= 24 or float(base_analysis.get("statistics", {}).get("std", 0.0) or 0.0) > 0:
        selected.append("peak_signature")
    return [name for idx, name in enumerate(selected) if name in FEATURE_METHOD_POOL and name not in selected[:idx]]


def _pick_source_columns(profile: Dict[str, object], analysis_result: Dict[str, object], target_col: str) -> List[str]:
    sequence_columns = list(dict.fromkeys(profile.get("feature_columns", []) or []))
    if sequence_columns:
        return sequence_columns
    quality_checks = analysis_result.get("quality_checks", {}) or {}
    sequence_summary = quality_checks.get("sequence_summary", {}) or {}
    inferred = [
        str(item.get("column"))
        for item in sequence_summary.get("sequence_columns", [])
        if str(item.get("column") or "") != target_col and int(item.get("max_length", 0)) > 0
    ]
    return inferred or [target_col]


def run(state: DGGlobalState) -> Dict:
    profile = state.read("dataset_profile", {})
    ds_df = load_ds_dataframe(state, columns=["timestamp_win", profile.get("target_column")] if profile.get("target_column") else None)
    target_col = profile.get("target_column")
    if not target_col:
        raise RuntimeError("Missing target column in dataset profile. Run data_reading first.")

    iteration_index = int(state.read("current_iteration_index", 1))
    analysis_result = state.read("data_analysis_result", {}) or {}
    source_columns = _pick_source_columns(profile, analysis_result, target_col)
    selected_methods = _pick_methods(state, ds_df, target_col)
    base_columns = ["station", "timestamp_win", "__row_id__", target_col] + source_columns
    station_frames = load_ds_station_frames(state, columns=list(dict.fromkeys(base_columns)))
    engineered_columns: List[str] = []
    engineered_dataset_paths: Dict[str, str] = {}
    total_rows = 0
    total_columns = 0
    for station_id, station_df in station_frames.items():
        engineered = station_df.copy()
        station_created: List[str] = []
        for method_name in selected_methods:
            engineered, created = FEATURE_METHOD_POOL[method_name](engineered, source_columns)
            station_created.extend(created)
        for column in station_created:
            engineered[column] = pd.to_numeric(engineered[column], errors="coerce").astype("float32")
        engineered = engineered.ffill().bfill().copy()
        station_path = write_step_dataframe_artifact(
            state,
            "feature_engineering",
            engineered,
            filename=f"engineered_dataset_{station_id}.parquet",
        )
        engineered_dataset_paths[str(station_id)] = str(station_path)
        engineered_columns.extend(station_created)
        total_rows += int(engineered.shape[0])
        total_columns = max(total_columns, int(engineered.shape[1]))
    engineered_columns = list(dict.fromkeys(engineered_columns))
    payload = {
        "selected_methods": selected_methods,
        "engineered_columns": engineered_columns,
        "source_columns": source_columns,
        "engineered_feature_count": len(engineered_columns),
        "shape": [int(total_rows), int(total_columns)],
        "iteration_index": iteration_index,
        "engineered_dataset_path": next(iter(engineered_dataset_paths.values())) if engineered_dataset_paths else "",
        "engineered_dataset_paths": engineered_dataset_paths,
        "selection_basis": "analysis_driven_deterministic",
        "dataset_kind": "ds",
    }
    state.write("feature_engineering_result", payload)
    write_step_artifact(state, "feature_engineering", payload)
    return {
        "message": "feature engineering completed",
        "selected_methods": selected_methods,
        "engineered_feature_count": len(engineered_columns),
    }
