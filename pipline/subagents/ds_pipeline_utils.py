from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from ..state import DGGlobalState
    from ..tools import infer_target_column_from_query
except ImportError:
    from state import DGGlobalState
    from tools import infer_target_column_from_query


IDENTIFIER_COLUMNS = {"station", "timestamp_win"}


def _parse_name_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    return [str(item).strip() for item in raw if str(item).strip()]


def normalize_sequence_value(value: object) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(float, copy=True).reshape(-1)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float).reshape(-1)
    return None


def sequence_length(value: object) -> int:
    arr = normalize_sequence_value(value)
    return int(arr.shape[0]) if arr is not None else 0


def extract_sequence_width(series: pd.Series) -> int:
    max_width = 0
    for value in series:
        max_width = max(max_width, sequence_length(value))
    return max_width


def is_sequence_column(series: pd.Series) -> bool:
    return extract_sequence_width(series) > 0


def _selected_columns(df: pd.DataFrame, columns: Sequence[str] | None) -> pd.DataFrame:
    if not columns:
        return df
    existing = [column for column in columns if column in df.columns]
    return df[existing].copy()


def load_ds_station_frames(
    state: DGGlobalState,
    key: str = "dataset_profile",
    columns: Sequence[str] | None = None,
) -> Dict[str, pd.DataFrame]:
    payload = state.read(key, {}) or {}
    path_mapping = payload.get("ds_dataset_paths") or {}
    if not path_mapping:
        path_mapping = state.read("dataset_loading_result", {}).get("ds_dataset_paths") or {}
    frames: Dict[str, pd.DataFrame] = {}
    if path_mapping:
        for station_id, path in path_mapping.items():
            frame = pd.read_parquet(Path(str(path)), columns=list(columns) if columns else None)
            if "timestamp_win" in frame.columns:
                frame["timestamp_win"] = pd.to_datetime(frame["timestamp_win"], errors="coerce")
            frames[str(station_id)] = frame
        return frames

    df = load_ds_dataframe(state, key=key, columns=columns)
    if "station" in df.columns:
        for station_id, station_df in df.groupby(df["station"].astype(str), sort=True):
            frames[str(station_id)] = station_df.reset_index(drop=True)
    else:
        frames["default"] = df
    return frames


def load_ds_dataframe(state: DGGlobalState, key: str = "dataset_profile", columns: Sequence[str] | None = None) -> pd.DataFrame:
    payload = state.read(key, {}) or {}
    ds_path = payload.get("ds_dataset_path")
    if not ds_path:
        ds_path = state.read("dataset_loading_result", {}).get("ds_dataset_path")
    if not ds_path:
        raise RuntimeError("Missing ds dataset path in state.")
    df = pd.read_parquet(Path(str(ds_path)), columns=list(columns) if columns else None)
    if "timestamp_win" in df.columns:
        df["timestamp_win"] = pd.to_datetime(df["timestamp_win"], errors="coerce")
    return df


def resolve_target_sequence_columns(df: pd.DataFrame, state: DGGlobalState) -> List[str]:
    columns = df.columns.tolist()
    explicit_names = _parse_name_list(state.read("target_col"))
    resolved: List[str] = []
    for name in explicit_names:
        future = [column for column in columns if column == f"{name}_future"]
        exact = [column for column in columns if column == name]
        resolved.extend(future or exact)

    if resolved:
        return [column for idx, column in enumerate(resolved) if column not in resolved[:idx]]

    future_candidates = [column for column in columns if column.endswith("_future")]
    if future_candidates:
        preferred = infer_target_column_from_query(state.read("user_query", ""), future_candidates)
        if preferred and preferred in future_candidates:
            return [preferred]
        return [future_candidates[0]]

    raise ValueError("Unable to infer target sequence column from DS dataset.")


def resolve_feature_sequence_columns(df: pd.DataFrame, state: DGGlobalState, target_columns: Sequence[str]) -> List[str]:
    sequence_columns = [
        column
        for column in df.columns
        if column not in IDENTIFIER_COLUMNS and is_sequence_column(df[column])
    ]
    target_set = {str(column) for column in target_columns}
    explicit_names = _parse_name_list(state.read("input_feature_cols"))
    if explicit_names:
        resolved: List[str] = []
        for name in explicit_names:
            exact = [column for column in sequence_columns if column == name]
            historical = [column for column in sequence_columns if column == name.replace("_future", "")]
            resolved.extend(exact or historical)
        return [column for column in resolved if column not in target_set]
    return [column for column in sequence_columns if column not in target_set]


def ds_scalar_columns(df: pd.DataFrame) -> List[str]:
    return [
        column
        for column in df.columns
        if column not in IDENTIFIER_COLUMNS and not is_sequence_column(df[column])
    ]


def count_missing_values(value: object) -> int:
    arr = normalize_sequence_value(value)
    if arr is not None:
        return int(np.isnan(arr).sum())
    return int(pd.isna(value))


def ds_missing_summary(df: pd.DataFrame) -> Dict[str, Any]:
    missing_by_column = {str(column): int(df[column].apply(count_missing_values).sum()) for column in df.columns}
    total_missing = int(sum(missing_by_column.values()))
    total_cells = 0
    for column in df.columns:
        series = df[column]
        if is_sequence_column(series):
            total_cells += int(series.apply(sequence_length).sum())
        else:
            total_cells += int(len(series))
    return {
        "total_missing": total_missing,
        "total_cells": int(total_cells),
        "missing_ratio": float(total_missing / total_cells) if total_cells else 0.0,
        "missing_by_column": missing_by_column,
    }


def ds_sequence_summary(df: pd.DataFrame) -> Dict[str, Any]:
    sequence_columns = [
        column for column in df.columns if column not in IDENTIFIER_COLUMNS and is_sequence_column(df[column])
    ]
    details: List[Dict[str, Any]] = []
    for column in sequence_columns:
        lengths = df[column].apply(sequence_length)
        missing_counts = df[column].apply(count_missing_values)
        details.append(
            {
                "column": str(column),
                "min_length": int(lengths.min()) if len(lengths) else 0,
                "max_length": int(lengths.max()) if len(lengths) else 0,
                "mean_length": float(lengths.mean()) if len(lengths) else 0.0,
                "zero_length_rows": int((lengths == 0).sum()),
                "missing_values": int(missing_counts.sum()),
            }
        )
    return {
        "sequence_column_count": len(sequence_columns),
        "sequence_columns": details,
    }


def ds_dimension_summary(df: pd.DataFrame) -> Dict[str, Any]:
    station_count = int(df["station"].astype(str).nunique()) if "station" in df.columns else 0
    timestamp_min = None
    timestamp_max = None
    if "timestamp_win" in df.columns:
        timestamp_series = pd.to_datetime(df["timestamp_win"], errors="coerce")
        if not timestamp_series.dropna().empty:
            timestamp_min = str(timestamp_series.min())
            timestamp_max = str(timestamp_series.max())
    return {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "station_count": station_count,
        "columns": [str(column) for column in df.columns.tolist()],
        "timestamp_min": timestamp_min,
        "timestamp_max": timestamp_max,
    }


def build_ds_profile(
    df: pd.DataFrame,
    state: DGGlobalState,
    *,
    ds_dataset_path: str,
) -> Dict[str, Any]:
    target_columns = resolve_target_sequence_columns(df, state)
    feature_columns = resolve_feature_sequence_columns(df, state, target_columns)
    scalar_columns = ds_scalar_columns(df)
    station_ids = [str(value) for value in df.get("station", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()]
    sequence_columns = [
        column for column in df.columns if column not in IDENTIFIER_COLUMNS and is_sequence_column(df[column])
    ]
    train_ratio = state.read("train_ratio")
    val_ratio = state.read("val_ratio")
    return {
        "dataset_path": state.read("dataset_path"),
        "ds_dataset_path": ds_dataset_path,
        "ds_dataset_paths": state.read("dataset_loading_result", {}).get("ds_dataset_paths", {}),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": [str(column) for column in df.columns.tolist()],
        "date_column": "timestamp_win" if "timestamp_win" in df.columns else None,
        "target_column": target_columns[0],
        "target_columns": list(target_columns),
        "feature_columns": list(feature_columns),
        "input_feature_columns": list(feature_columns),
        "sequence_columns": sequence_columns,
        "scalar_columns": scalar_columns,
        "selected_units": station_ids,
        "sorted_by_time": "timestamp_win" in df.columns,
        "dataset_kind": "ds",
        "enable_split": bool(state.read("enable_split")),
        "enable_feature_engineering": bool(state.read("enable_feature_engineering")),
        "enable_normalization": bool(state.read("enable_normalization")),
        "need_split": bool(state.read("plan_meta", {}).get("requires_split", False)),
        "dimensions": ds_dimension_summary(df),
        "sequence_summary": ds_sequence_summary(df),
        "missing_summary": ds_missing_summary(df),
        "split_ratios": (
            {"train_ratio": float(train_ratio), "val_ratio": float(val_ratio)}
            if train_ratio is not None and val_ratio is not None
            else None
        ),
    }


def sequence_to_scalar(value: object, index: int = 0, default: float = 0.0) -> float:
    arr = normalize_sequence_value(value)
    if arr is None or arr.shape[0] <= index or np.isnan(arr[index]):
        return float(default)
    return float(arr[index])


def flatten_sequence_column(series: pd.Series, prefix: str, width: int | None = None) -> pd.DataFrame:
    width = int(width if width is not None else extract_sequence_width(series))
    if width <= 0:
        return pd.DataFrame(index=series.index)
    rows = []
    for value in series:
        arr = normalize_sequence_value(value)
        if arr is None:
            arr = np.full(width, np.nan, dtype=float)
        elif arr.shape[0] < width:
            arr = np.pad(arr, (0, width - arr.shape[0]), mode="constant", constant_values=np.nan)
        rows.append(arr[:width])
    flattened = pd.DataFrame(rows, index=series.index)
    flattened.columns = [f"{prefix}_step_{idx}" for idx in range(flattened.shape[1])]
    return flattened


def flatten_ds_for_model(
    df: pd.DataFrame,
    *,
    feature_sequence_columns: Sequence[str],
    target_column: str,
    scalar_feature_columns: Sequence[str],
    sequence_widths: Dict[str, int] | None = None,
) -> pd.DataFrame:
    base = pd.DataFrame(index=df.index)
    if "timestamp_win" in df.columns:
        base["timestamp_win"] = pd.to_datetime(df["timestamp_win"], errors="coerce")
    if "station" in df.columns:
        station_codes, uniques = pd.factorize(df["station"].astype(str), sort=True)
        base["station_code"] = station_codes.astype(int)
        base.attrs["station_labels"] = [str(item) for item in uniques.tolist()]
    for column in scalar_feature_columns:
        if column in df.columns:
            base[column] = pd.to_numeric(df[column], errors="coerce")

    for column in feature_sequence_columns:
        if column in df.columns:
            base = pd.concat([base, flatten_sequence_column(df[column], column, width=(sequence_widths or {}).get(column))], axis=1)

    if target_column not in df.columns:
        raise ValueError(f"Target sequence column not found in dataset: {target_column}")
    base[target_column] = df[target_column].apply(sequence_to_scalar)

    if "timestamp_win" in base.columns:
        dt = pd.to_datetime(base["timestamp_win"], errors="coerce")
        base["hour"] = dt.dt.hour.fillna(0).astype(int)
        base["dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)
        base["day"] = dt.dt.day.fillna(0).astype(int)
        base["month"] = dt.dt.month.fillna(0).astype(int)

    return base
