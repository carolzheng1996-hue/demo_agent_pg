from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from ..state import DGGlobalState
    from ..tools import write_step_artifact, write_step_dataframe_artifact
except ImportError:
    from state import DGGlobalState
    from tools import write_step_artifact, write_step_dataframe_artifact


FeatureMethod = Callable[[pd.DataFrame, List[str]], Tuple[pd.DataFrame, List[str]]]


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce").ffill().bfill()


def _safe_feature_columns(df: pd.DataFrame, columns: List[str]) -> List[str]:
    return [column for column in columns if column in df.columns]


def _concat_feature_block(engineered: pd.DataFrame, feature_block: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, List[str]]:
    if not feature_block:
        return engineered, []
    filtered_block = {
        name: values for name, values in feature_block.items() if name not in engineered.columns
    }
    if not filtered_block:
        return engineered, []
    created = list(filtered_block.keys())
    block_df = pd.DataFrame(filtered_block, index=engineered.index)
    return pd.concat([engineered, block_df], axis=1).copy(), created


def _add_lag_features(engineered: pd.DataFrame, source_columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    feature_block: Dict[str, pd.Series] = {}
    for column in _safe_feature_columns(engineered, source_columns):
        series = _numeric_series(engineered, column)
        for lag in (1, 6):
            name = f"{column}_lag_{lag}"
            feature_block[name] = series.shift(lag)
    return _concat_feature_block(engineered, feature_block)


def _add_rolling_features(engineered: pd.DataFrame, source_columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    feature_block: Dict[str, pd.Series] = {}
    for column in _safe_feature_columns(engineered, source_columns):
        series = _numeric_series(engineered, column)
        specs = [
            (6, "mean"),
            (24, "mean"),
            (24, "std"),
            (24, "min"),
            (24, "max"),
        ]
        for window, kind in specs:
            name = f"{column}_rolling_{kind}_{window}"
            if kind == "mean":
                feature_block[name] = series.rolling(window, min_periods=1).mean()
            elif kind == "std":
                feature_block[name] = series.rolling(window, min_periods=1).std().fillna(0.0)
            elif kind == "min":
                feature_block[name] = series.rolling(window, min_periods=1).min()
            else:
                feature_block[name] = series.rolling(window, min_periods=1).max()
    return _concat_feature_block(engineered, feature_block)


def _add_difference_features(engineered: pd.DataFrame, source_columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    feature_block: Dict[str, pd.Series] = {}
    for column in _safe_feature_columns(engineered, source_columns):
        series = _numeric_series(engineered, column)
        mapping = {
            f"{column}_diff_1": series.diff(1),
        }
        feature_block.update(mapping)
    return _concat_feature_block(engineered, feature_block)


def _add_ewm_features(engineered: pd.DataFrame, source_columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    feature_block: Dict[str, pd.Series] = {}
    for column in _safe_feature_columns(engineered, source_columns):
        series = _numeric_series(engineered, column)
        mapping = {
            f"{column}_ewm_mean_12": series.ewm(span=12, adjust=False).mean(),
            f"{column}_ewm_mean_24": series.ewm(span=24, adjust=False).mean(),
            f"{column}_ewm_std_24": series.ewm(span=24, adjust=False).std().fillna(0.0),
        }
        feature_block.update(mapping)
    return _concat_feature_block(engineered, feature_block)


def _add_peak_features(engineered: pd.DataFrame, source_columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    feature_block: Dict[str, pd.Series] = {}
    for column in _safe_feature_columns(engineered, source_columns):
        series = _numeric_series(engineered, column)
        rolling_mean = series.rolling(24, min_periods=1).mean()
        rolling_std = series.rolling(24, min_periods=1).std().fillna(0.0)
        diff_sign = series.diff().fillna(0.0).apply(lambda value: 1 if value > 0 else (-1 if value < 0 else 0))
        mapping = {
            f"{column}_abs_energy_24": series.pow(2).rolling(24, min_periods=1).sum(),
            f"{column}_peak_count_24": (series > (rolling_mean + rolling_std)).astype(int).rolling(24, min_periods=1).sum(),
            f"{column}_turning_points_24": diff_sign.diff().abs().fillna(0.0).rolling(24, min_periods=1).sum(),
        }
        feature_block.update(mapping)
    return _concat_feature_block(engineered, feature_block)


def _add_calendar_features(engineered: pd.DataFrame, source_columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    _ = source_columns
    date_col = engineered.attrs.get("date_column")
    if not date_col or date_col not in engineered.columns:
        return engineered, []
    dt = pd.to_datetime(engineered[date_col], errors="coerce")
    feature_block = {
        "hour": dt.dt.hour.fillna(0).astype(int),
        "dayofweek": dt.dt.dayofweek.fillna(0).astype(int),
        "day": dt.dt.day.fillna(0).astype(int),
        "month": dt.dt.month.fillna(0).astype(int),
        "is_weekend": dt.dt.dayofweek.isin([5, 6]).fillna(False).astype(int),
    }
    return _concat_feature_block(engineered, feature_block)


FEATURE_METHOD_POOL: Dict[str, FeatureMethod] = {
    "lag_signature": _add_lag_features,
    "rolling_signature": _add_rolling_features,
    "difference_signature": _add_difference_features,
    "ewm_signature": _add_ewm_features,
    "peak_signature": _add_peak_features,
    "calendar_signature": _add_calendar_features,
}


def _feature_rng(state: DGGlobalState, iteration_index: int) -> random.Random:
    use_system_random = bool(state.read("use_system_random"))
    if use_system_random:
        return random.SystemRandom()

    task_id = str(state.read("task_id", "default_task"))
    history = state.read("iteration_history", []) or []
    history_basis = "|".join(
        ",".join(item.get("feature_methods", []) or []) for item in history[-2:]
    )
    seed_material = f"{task_id}:{iteration_index}:{history_basis}:{len(history)}:feature_engineering"
    digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _pick_methods(state: DGGlobalState, iteration_index: int, has_date_column: bool) -> List[str]:
    candidates = [name for name in FEATURE_METHOD_POOL if has_date_column or name != "calendar_signature"]
    history = state.read("iteration_history", []) or []
    recently_used = [item.get("feature_methods", []) or [] for item in history[-2:]]
    previous_signature = tuple(recently_used[-1]) if recently_used else ()
    rng = _feature_rng(state, iteration_index)
    sample_size = min(len(candidates), max(2, rng.randint(2, min(3, len(candidates)))))

    novelty_pool = [
        name
        for name in candidates
        if not recently_used or any(name not in used for used in recently_used)
    ] or candidates
    selected = rng.sample(novelty_pool, k=min(sample_size, len(novelty_pool)))
    if len(selected) < sample_size:
        remainder = [name for name in candidates if name not in selected]
        selected.extend(rng.sample(remainder, k=min(sample_size - len(selected), len(remainder))))
    if "lag_signature" not in selected:
        if len(selected) >= min(len(candidates), 5):
            selected[-1] = "lag_signature"
        else:
            selected.insert(0, "lag_signature")

    if tuple(selected) == previous_signature and len(candidates) > 1:
        remainder = [name for name in candidates if name not in selected]
        if remainder:
            swap_index = rng.randrange(len(selected))
            selected[swap_index] = rng.choice(remainder)
            if "lag_signature" not in selected:
                selected[0] = "lag_signature"
    return selected[: min(len(candidates), 3)]


def _load_station_frames(state: DGGlobalState) -> Dict[str, pd.DataFrame]:
    profile = state.read("dataset_profile", {})
    station_paths = profile.get("formatted_dataset_paths") or {}
    if station_paths:
        return {
            str(station_id): pd.read_parquet(Path(str(formatted_path)))
            for station_id, formatted_path in station_paths.items()
        }

    formatted_path = profile.get("formatted_dataset_path")
    if not formatted_path:
        raise RuntimeError("Missing formatted dataset path. Run data_formatter first.")
    return {"default": pd.read_parquet(Path(str(formatted_path)))}


def run(state: DGGlobalState) -> Dict:
    dataset_profile = state.read("dataset_profile", {})
    target_col = dataset_profile.get("target_column")
    if not target_col:
        raise RuntimeError("Missing target column in dataset profile. Run data_formatter first.")

    iteration_index = int(state.read("current_iteration_index", 1))
    date_col = dataset_profile.get("date_column")
    source_columns = list(dict.fromkeys((dataset_profile.get("feature_columns", []) or []) + [target_col]))
    selected_methods = _pick_methods(state, iteration_index, bool(date_col))
    station_frames = _load_station_frames(state)
    engineered_paths: Dict[str, str] = {}
    engineered_columns: List[str] = []
    total_rows = 0
    total_cols = 0
    for station_id, frame in station_frames.items():
        engineered = frame.copy()
        engineered.attrs["date_column"] = date_col
        station_created: List[str] = []
        for method_name in selected_methods:
            engineered, created = FEATURE_METHOD_POOL[method_name](engineered, source_columns)
            station_created.extend(created)

        for column in station_created:
            engineered[column] = pd.to_numeric(engineered[column], errors="coerce").astype("float32")
        engineered = engineered.ffill().bfill().fillna(0.0).copy()
        engineered_path = write_step_dataframe_artifact(
            state,
            "feature_engineering",
            engineered,
            filename=f"engineered_dataset_{station_id}.parquet",
        )
        engineered_paths[station_id] = str(engineered_path)
        engineered_columns.extend(station_created)
        total_rows += int(engineered.shape[0])
        total_cols = max(total_cols, int(engineered.shape[1]))

    engineered_columns = list(dict.fromkeys(engineered_columns))

    feature_payload = {
        "selected_methods": selected_methods,
        "engineered_columns": engineered_columns,
        "source_columns": source_columns,
        "engineered_feature_count": len(engineered_columns),
        "shape": [total_rows, total_cols],
        "station_count": len(engineered_paths),
        "iteration_index": iteration_index,
        "random_mode": "system_random" if bool(state.read("use_system_random")) else "deterministic_seeded",
        "strategy_seed_basis": (
            f"system_random:{iteration_index}:{int(time.time() * 1000)}"
            if bool(state.read("use_system_random"))
            else f"deterministic:{state.read('task_id', 'default_task')}:{iteration_index}:{len(state.read('iteration_history', []) or [])}"
        ),
        "tsfresh_inspired_notes": {
            "lag_signature": "多阶滞后特征",
            "rolling_signature": "滚动统计量特征",
            "difference_signature": "差分与变化率特征",
            "ewm_signature": "指数加权统计特征",
            "peak_signature": "局部峰值/能量/转折点特征",
            "calendar_signature": "时间日历特征",
        },
    }
    feature_payload["engineered_dataset_paths"] = engineered_paths
    feature_payload["engineered_dataset_path"] = next(iter(engineered_paths.values()))
    state.write("feature_engineering_result", feature_payload)
    write_step_artifact(state, "feature_engineering", feature_payload)
    return {"message": "feature engineering completed", **feature_payload}
