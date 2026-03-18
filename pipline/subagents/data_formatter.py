from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from ..state import DGGlobalState
    from ..tools import (
        infer_target_column_from_query,
        write_step_artifact,
        write_task_dataframe_artifact,
    )
    from ..tools.file_tools import detect_date, set_features_multi, set_target
except ImportError:
    from state import DGGlobalState
    from tools import (
        infer_target_column_from_query,
        write_step_artifact,
        write_task_dataframe_artifact,
    )
    from tools.file_tools import detect_date, set_features_multi, set_target


TIME_DERIVED_COLUMNS = ["date", "hour", "minute"]
IDENTIFIER_COLUMNS = {"station", "timestamp_win", "__row_id__"}


def _sequence_cleanup_config(state: DGGlobalState) -> Dict[str, int]:
    input_length = state.read("input_length")
    output_length = state.read("output_length")
    points_per_day = state.read("points_per_day")
    if input_length is None or output_length is None:
        raise ValueError("input_length/output_length is missing in state. Please provide them in config_all.json or runtime args.")
    if points_per_day is None:
        raise ValueError("points_per_day is missing in state. Please provide it in config_all.json or runtime args.")
    return {
        "input_length": int(input_length),
        "output_length": int(output_length),
        "points_per_day": int(points_per_day),
    }


def _ratio_payload(state: DGGlobalState) -> Dict[str, Optional[float]]:
    return {
        "train_ratio": state.read("train_ratio"),
        "val_ratio": state.read("val_ratio"),
    }


def _need_split(state: DGGlobalState) -> bool:
    return bool(state.read("plan_meta", {}).get("requires_split", False))


def _validate_ratios(ratios: Dict[str, Optional[float]]) -> Dict[str, float]:
    if any(ratios.get(name) is None for name in ["train_ratio", "val_ratio"]):
        raise ValueError("train_ratio/val_ratio is missing in state. Please provide them in config_all.json or runtime args.")
    normalized = {
        "train_ratio": float(ratios["train_ratio"]),
        "val_ratio": float(ratios["val_ratio"]),
    }
    if any(value < 0 for value in normalized.values()):
        raise ValueError(f"Split ratios must be non-negative, got {normalized}")
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("Split ratios must contain at least one positive value.")
    return {key: value / total for key, value in normalized.items()}


def _load_ds_dataframe(state: DGGlobalState) -> pd.DataFrame:
    ds_path = state.read("dataset_loading_result", {}).get("ds_dataset_path")
    if not ds_path:
        raise RuntimeError("Missing ds_dataset_path in state. Run data_reading first.")
    dataframe = pd.read_parquet(Path(str(ds_path)))
    if "timestamp_win" in dataframe.columns:
        dataframe["timestamp_win"] = pd.to_datetime(dataframe["timestamp_win"], errors="coerce")
    return dataframe


def _extract_sequence_width(series: pd.Series) -> int:
    max_width = 0
    for value in series:
        if isinstance(value, (list, tuple, np.ndarray)):
            max_width = max(max_width, len(value))
    return max_width


def _normalize_sequence_value(value: object) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.astype(float, copy=True).reshape(-1)
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float).reshape(-1)
    return None


def _sequence_length(value: object) -> int:
    arr = _normalize_sequence_value(value)
    return int(arr.shape[0]) if arr is not None else 0


def _pad_array_head(value: object, target_len: int) -> np.ndarray:
    arr = _normalize_sequence_value(value)
    if arr is None:
        return np.zeros(target_len, dtype=float)
    if arr.shape[0] >= target_len:
        return arr[:target_len]
    return np.pad(arr, (target_len - arr.shape[0], 0), mode="constant")


def _pad_array_tail(value: object, target_len: int) -> np.ndarray:
    arr = _normalize_sequence_value(value)
    if arr is None:
        return np.zeros(target_len, dtype=float)
    if arr.shape[0] >= target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - arr.shape[0]), mode="constant")


def _fill_nan_sequence(value: object, points_per_day: int) -> np.ndarray:
    arr = _normalize_sequence_value(value)
    if arr is None:
        return np.zeros(0, dtype=float)
    if arr.size == 0:
        return arr
    if np.all(np.isnan(arr)):
        return np.zeros_like(arr)

    filled = arr.copy()
    mean_val = float(np.nanmean(filled))
    for idx in range(filled.shape[0]):
        if not np.isnan(filled[idx]):
            continue
        previous_day_idx = idx - points_per_day
        if previous_day_idx >= 0 and not np.isnan(filled[previous_day_idx]):
            filled[idx] = filled[previous_day_idx]
        else:
            filled[idx] = mean_val
    return filled


def _sequence_target_length(column: str, config: Dict[str, int]) -> int:
    if "_predict" in column or "_future" in column:
        return int(config["output_length"])
    return int(config["input_length"])


def _clean_ds_sequences(df: pd.DataFrame, state: DGGlobalState) -> Tuple[pd.DataFrame, Dict]:
    config = _sequence_cleanup_config(state)
    cleaned = df.copy()
    repaired_columns: List[Dict[str, int | str]] = []

    for column in cleaned.columns:
        if column in {"station", "timestamp_win"}:
            continue
        series = cleaned[column]
        if _extract_sequence_width(series) <= 0:
            continue

        target_len = _sequence_target_length(column, config)
        pad_fn = _pad_array_tail if ("_predict" in column or "_future" in column) else _pad_array_head
        original_lengths = series.apply(_sequence_length)
        cleaned[column] = series.apply(lambda value: _fill_nan_sequence(pad_fn(value, target_len), config["points_per_day"]))
        repaired_columns.append(
            {
                "column": column,
                "target_length": target_len,
                "min_original_length": int(original_lengths.min()) if len(original_lengths) else 0,
                "max_original_length": int(original_lengths.max()) if len(original_lengths) else 0,
            }
        )

    return cleaned, {
        "input_length": config["input_length"],
        "output_length": config["output_length"],
        "points_per_day": config["points_per_day"],
        "sequence_columns_checked": len(repaired_columns),
        "repaired_columns": repaired_columns,
    }


def _flatten_sequence_columns(df: pd.DataFrame) -> pd.DataFrame:
    flattened = df.copy()
    for column in list(df.columns):
        width = _extract_sequence_width(df[column])
        if width <= 0:
            continue
        normalized_values = df[column].apply(
            lambda value: (
                list(value) + [np.nan] * max(0, width - len(value))
                if isinstance(value, (list, tuple, np.ndarray))
                else [np.nan] * width
            )
        ).tolist()
        expanded = pd.DataFrame(normalized_values, index=df.index)
        expanded = expanded.iloc[:, :width]
        expanded.columns = [f"{column}_step_{idx}" for idx in range(expanded.shape[1])]
        flattened = flattened.drop(columns=[column]).join(expanded)
    return flattened


def _expand_requested_columns(requested: List[str], available_columns: List[str]) -> List[str]:
    resolved: List[str] = []
    for name in requested:
        if name in available_columns:
            resolved.append(name)
            continue
        prefixed = [
            column
            for column in available_columns
            if column.startswith(f"{name}_step_") or column.startswith(f"{name}_future_step_")
        ]
        resolved.extend(prefixed)
    return [column for idx, column in enumerate(resolved) if column not in resolved[:idx]]


def _resolve_targets(df: pd.DataFrame, state: DGGlobalState) -> List[str]:
    explicit_target_names = [item.strip() for item in str(state.read("target_col", "")).split(",") if item.strip()]
    explicit_targets = _expand_requested_columns(explicit_target_names, df.columns.tolist())
    explicit_targets = [column for column in explicit_targets if column.endswith("_future_step_0")] or explicit_targets
    if explicit_targets:
        return explicit_targets

    future_candidates = [column for column in df.columns if str(column).endswith("_future_step_0")]
    if future_candidates:
        preferred = infer_target_column_from_query(state.read("user_query", ""), future_candidates)
        if preferred and preferred in future_candidates:
            return [preferred]
        return [future_candidates[0]]

    target_preference = infer_target_column_from_query(state.read("user_query", ""), df.columns.tolist())
    return [set_target(df, preferred=target_preference or "")]


def _resolve_feature_columns(df: pd.DataFrame, state: DGGlobalState, target_cols: List[str]) -> List[str]:
    target_related: set[str] = set(target_cols)
    for target in target_cols:
        if "_future_step_" in target:
            prefix = target.split("_future_step_", 1)[0]
            target_related.update(
                column for column in df.columns if column.startswith(f"{prefix}_future_step_")
            )

    explicit_feature_names = [item.strip() for item in str(state.read("input_feature_cols", "")).split(",") if item.strip()]
    explicit_features = _expand_requested_columns(explicit_feature_names, df.columns.tolist())
    if explicit_features:
        return [column for column in explicit_features if column in df.columns and column not in target_related]

    numeric_features = set_features_multi(df, target_cols)
    return [column for column in numeric_features if column not in IDENTIFIER_COLUMNS and column not in target_related]


def _prepare_station_dataframe(
    df: pd.DataFrame,
    date_col: Optional[str],
    feature_cols: List[str],
    target_cols: List[str],
    row_id_start: int,
) -> pd.DataFrame:
    working_df = df.copy()
    if date_col and date_col in working_df.columns:
        parsed_datetime = pd.to_datetime(working_df[date_col], errors="coerce")
        working_df[date_col] = parsed_datetime
        working_df = working_df.sort_values(date_col).reset_index(drop=True)
        if date_col != "date":
            working_df["date"] = parsed_datetime.dt.day.fillna(0).astype(int)
        if date_col != "hour":
            working_df["hour"] = parsed_datetime.dt.hour.fillna(0).astype(int)
        if date_col != "minute":
            working_df["minute"] = parsed_datetime.dt.minute.fillna(0).astype(int)

    numeric_candidates = [col for col in feature_cols + target_cols if col in working_df.columns]
    for col in numeric_candidates:
        working_df[col] = pd.to_numeric(working_df[col], errors="coerce")
    working_df = working_df.ffill().bfill().fillna(0.0).copy()
    working_df = working_df.assign(__row_id__=np.arange(row_id_start, row_id_start + len(working_df), dtype=int))
    return working_df


def _build_station_payload(
    station_df: pd.DataFrame,
    state: DGGlobalState,
    station_id: str,
    row_id_start: int,
) -> Tuple[pd.DataFrame, Dict]:
    date_col = "timestamp_win" if "timestamp_win" in station_df.columns else detect_date(station_df)
    target_cols = _resolve_targets(station_df, state)
    primary_target = target_cols[0]
    feature_cols = _resolve_feature_columns(station_df, state, target_cols)
    prepared_df = _prepare_station_dataframe(station_df, date_col, feature_cols, target_cols, row_id_start)
    derived_time_columns = [column for column in TIME_DERIVED_COLUMNS if column in prepared_df.columns]
    selected_columns = (
        ([date_col] if date_col and date_col in prepared_df.columns else [])
        + ["__row_id__"]
        + feature_cols
        + derived_time_columns
        + target_cols
    )
    selected_columns = [
        column
        for idx, column in enumerate(selected_columns)
        if column in prepared_df.columns and column not in selected_columns[:idx]
    ]
    formatted_df = prepared_df[selected_columns].copy()
    payload = {
        "station_id": station_id,
        "date_column": date_col,
        "target_column": primary_target,
        "target_columns": target_cols,
        "feature_columns": feature_cols,
        "derived_time_columns": derived_time_columns,
        "selected_columns": selected_columns,
        "shape": [int(formatted_df.shape[0]), int(formatted_df.shape[1])],
        "columns": formatted_df.columns.tolist(),
    }
    return formatted_df, payload


def run(state: DGGlobalState) -> Dict:
    ds_df = _load_ds_dataframe(state)
    ds_df, cleanup_payload = _clean_ds_sequences(ds_df, state)
    flat_df = _flatten_sequence_columns(ds_df)
    if "station" not in flat_df.columns:
        raise ValueError("Formatted station-wise output requires a station column in DS data.")

    need_split = _need_split(state)
    ratios = _validate_ratios(_ratio_payload(state)) if need_split else _ratio_payload(state)
    requested_units = [item.strip() for item in str(state.read("formatter_unit", "") or "").split(",") if item.strip()]
    if requested_units:
        flat_df = flat_df[flat_df["station"].astype(str).isin(requested_units)].copy()
    station_ids = [str(value) for value in flat_df["station"].dropna().astype(str).unique().tolist()]
    if not station_ids:
        raise ValueError("No station values found in DS dataset.")

    station_paths: Dict[str, str] = {}
    station_profiles: Dict[str, Dict] = {}
    first_profile: Optional[Dict] = None
    first_target: Optional[str] = None
    first_features: List[str] = []
    next_row_id = 0
    for station_id in station_ids:
        station_df = flat_df[flat_df["station"].astype(str) == station_id].copy()
        station_df = station_df.drop(columns=["station"])
        formatted_df, station_profile = _build_station_payload(
            station_df,
            state,
            station_id,
            row_id_start=next_row_id,
        )
        next_row_id += len(formatted_df)
        station_path = write_task_dataframe_artifact(
            state,
            f"data/formatted/formatted_dataset_{station_id}.parquet",
            formatted_df,
        )
        station_paths[station_id] = str(station_path)
        station_profiles[station_id] = station_profile
        if first_profile is None:
            first_profile = station_profile
            first_target = station_profile["target_column"]
            first_features = list(station_profile["feature_columns"])

    dataset_loading_result = state.read("dataset_loading_result", {})
    dataset_profile = {
        "dataset_path": state.read("dataset_path"),
        "raw_dataset_path": dataset_loading_result.get("raw_dataset_path"),
        "ds_dataset_path": dataset_loading_result.get("ds_dataset_path"),
        "formatted_dataset_paths": station_paths,
        "formatted_dataset_path": next(iter(station_paths.values())),
        "unit": dataset_loading_result.get("unit"),
        "selected_units": dataset_loading_result.get("selected_units", []),
        "enable_split": bool(state.read("enable_split")),
        "enable_feature_engineering": bool(state.read("enable_feature_engineering")),
        "split_method": str(state.read("split_method") or "").strip(),
        "enable_normalization": bool(state.read("enable_normalization")),
        "is_directory": True,
        "available_file_count": int(dataset_loading_result.get("available_file_count") or 0),
        "shape": [sum(item["shape"][0] for item in station_profiles.values()), first_profile["shape"][1] if first_profile else 0],
        "columns": first_profile["columns"] if first_profile else [],
        "date_column": first_profile["date_column"] if first_profile else None,
        "target_column": first_target,
        "target_columns": first_profile["target_columns"] if first_profile else [],
        "feature_columns": first_features,
        "derived_time_columns": first_profile["derived_time_columns"] if first_profile else [],
        "input_feature_columns": first_features,
        "need_split": need_split,
        "split_ratios": ratios if need_split else None,
        "sorted_by_time": bool(first_profile and first_profile["date_column"]),
        "station_column_removed": True,
        "station_profiles": station_profiles,
        "sequence_cleanup": cleanup_payload,
    }
    payload = {
        "dataset_profile": dataset_profile,
        "formatted_dataset_paths": station_paths,
        "station_profiles": station_profiles,
        "sequence_cleanup": cleanup_payload,
        "pipeline_source": "stationwise_formatter",
    }

    state.update({"dataset_profile": dataset_profile, "data_formatter_result": payload})
    write_step_artifact(state, "data_formatter", payload)
    return {
        "message": "data formatted successfully",
        "target_column": first_target,
        "feature_count": len(first_features),
        "formatted_station_count": len(station_paths),
    }
