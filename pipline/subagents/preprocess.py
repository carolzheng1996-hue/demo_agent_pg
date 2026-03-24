from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from ..state import DGGlobalState
    from ..tools import write_step_artifact, write_step_dataframe_artifact
    from .ds_pipeline_utils import extract_sequence_width, flatten_ds_for_model, load_ds_station_frames
except ImportError:
    from state import DGGlobalState
    from tools import write_step_artifact, write_step_dataframe_artifact
    from subagents.ds_pipeline_utils import extract_sequence_width, flatten_ds_for_model, load_ds_station_frames


def _load_source_station_frames(state: DGGlobalState) -> Dict[str, pd.DataFrame]:
    profile = state.read("dataset_profile", {}) or {}
    target_col = str(profile.get("target_column") or "")
    feature_columns = list(profile.get("feature_columns", []) or [])
    engineered_columns = list(state.read("feature_engineering_result", {}).get("engineered_columns", []) or [])
    base_columns = ["station", "timestamp_win", "__row_id__", target_col] + feature_columns + engineered_columns
    if bool(state.read("enable_feature_engineering")):
        engineered_paths = state.read("feature_engineering_result", {}).get("engineered_dataset_paths", {}) or {}
        if not engineered_paths:
            engineered_path = state.read("feature_engineering_result", {}).get("engineered_dataset_path")
            if engineered_path:
                engineered_paths = {"default": str(engineered_path)}
        if not engineered_paths:
            raise RuntimeError("Missing engineered DS dataset path. Run feature_engineering first.")
        frames: Dict[str, pd.DataFrame] = {}
        for station_id, path in engineered_paths.items():
            frame = pd.read_parquet(Path(str(path)), columns=list(dict.fromkeys(base_columns)))
            if "timestamp_win" in frame.columns:
                frame["timestamp_win"] = pd.to_datetime(frame["timestamp_win"], errors="coerce")
            frames[str(station_id)] = frame
        return frames
    return load_ds_station_frames(state, columns=list(dict.fromkeys(base_columns)))


def _load_split_row_ids(state: DGGlobalState) -> Dict[str, set[int]]:
    row_index_paths = state.read("split_strategy_result", {}).get("row_index_paths", {}) or {}
    if not row_index_paths:
        return {}
    result: Dict[str, set[int]] = {}
    for split_name, path in row_index_paths.items():
        if not path:
            continue
        frame = pd.read_parquet(Path(str(path)), columns=["__row_id__"])
        result[split_name] = {int(value) for value in frame["__row_id__"].tolist()}
    return result


def _sequence_width_map(frames: Dict[str, pd.DataFrame], columns: List[str]) -> Dict[str, int]:
    widths: Dict[str, int] = {}
    for column in columns:
        max_width = 0
        for frame in frames.values():
            if column not in frame.columns:
                continue
            max_width = max(max_width, int(extract_sequence_width(frame[column])))
        widths[column] = max_width
    return widths


def _prepare_model_frame(df: pd.DataFrame, state: DGGlobalState, sequence_widths: Dict[str, int]) -> pd.DataFrame:
    profile = state.read("dataset_profile", {})
    target_col = str(profile.get("target_column") or "")
    feature_sequence_columns = list(profile.get("feature_columns", []) or [])
    scalar_feature_columns = [
        column
        for column in df.columns
        if column not in set(feature_sequence_columns + [target_col, "station", "timestamp_win"])
    ]
    frame = flatten_ds_for_model(
        df,
        feature_sequence_columns=feature_sequence_columns,
        target_column=target_col,
        scalar_feature_columns=scalar_feature_columns,
        sequence_widths=sequence_widths,
    )
    for column in frame.columns:
        if str(frame[column].dtype) != "datetime64[ns]":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.ffill().bfill().fillna(0.0).copy()


def _normalize_with_train_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    numeric_cols: List[str],
    mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    scaling: Dict = {}
    if not numeric_cols or mode not in {"zscore", "minmax"}:
        return train_df, val_df, scaling

    scaled_train = train_df.copy()
    scaled_val = val_df.copy()
    for col in numeric_cols:
        train_values = pd.to_numeric(scaled_train[col], errors="coerce").to_numpy(dtype=float)
        val_values = pd.to_numeric(scaled_val[col], errors="coerce").to_numpy(dtype=float)
        if mode == "minmax":
            min_value = float(np.nanmin(train_values))
            max_value = float(np.nanmax(train_values))
            scale = max(max_value - min_value, 1e-8)
            scaled_train[col] = (train_values - min_value) / scale
            scaled_val[col] = (val_values - min_value) / scale
            scaling[col] = {"data_min": min_value, "data_max": max_value, "scale": scale, "mode": "minmax"}
        else:
            mean = float(np.nanmean(train_values))
            std = float(np.nanstd(train_values)) or 1.0
            scaled_train[col] = (train_values - mean) / std
            scaled_val[col] = (val_values - mean) / std
            scaling[col] = {"mean": mean, "std": std, "mode": "zscore"}
    return scaled_train, scaled_val, scaling


def _empty_like(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.iloc[0:0].copy().reset_index(drop=True)


def run(state: DGGlobalState) -> Dict:
    source_frames = _load_source_station_frames(state)
    profile = state.read("dataset_profile", {}) or {}
    target_col = str(profile.get("target_column") or "")
    if not target_col:
        raise RuntimeError("Missing target column in dataset profile.")

    feature_columns = list(profile.get("feature_columns", []) or [])
    sequence_widths = _sequence_width_map(source_frames, feature_columns)
    split_row_ids = _load_split_row_ids(state)
    split_applied = bool(split_row_ids.get("train")) and bool(split_row_ids.get("val"))

    train_outputs: List[str] = []
    val_outputs: List[str] = []
    scaling: Dict = {}
    train_shapes: List[List[int]] = []
    val_shapes: List[List[int]] = []
    numeric_columns_union: List[str] = []
    scaling_policy = state.read("datanorm_result", {})
    should_normalize = bool(scaling_policy.get("should_normalize", False))
    normalization_mode = str(scaling_policy.get("recommended_mode", "skip") or "skip").lower()

    for station_id, source_df in source_frames.items():
        if split_applied:
            train_source = source_df[source_df["__row_id__"].isin(split_row_ids.get("train", set()))].reset_index(drop=True)
            val_source = source_df[source_df["__row_id__"].isin(split_row_ids.get("val", set()))].reset_index(drop=True)
        else:
            train_source = source_df.reset_index(drop=True)
            val_source = _empty_like(source_df)
        if train_source.empty and val_source.empty:
            continue

        train_df = _prepare_model_frame(train_source, state, sequence_widths) if not train_source.empty else pd.DataFrame()
        val_df = _prepare_model_frame(val_source, state, sequence_widths) if not val_source.empty else train_df.iloc[0:0].copy()
        numeric_cols = train_df.select_dtypes(include=["number"]).columns.tolist() if not train_df.empty else []
        numeric_columns_union.extend([column for column in numeric_cols if column not in numeric_columns_union])
        scaled_train, scaled_val, station_scaling = _normalize_with_train_statistics(
            train_df,
            val_df,
            numeric_cols,
            normalization_mode if should_normalize else "skip",
        )
        if station_scaling:
            scaling[str(station_id)] = station_scaling

        if not scaled_train.empty:
            train_path = write_step_dataframe_artifact(
                state,
                "preprocess",
                scaled_train,
                filename=f"train_preprocessed_{station_id}.parquet",
            )
            train_outputs.append(str(train_path))
            train_shapes.append([int(scaled_train.shape[0]), int(scaled_train.shape[1])])
        if isinstance(scaled_val, pd.DataFrame) and not scaled_val.empty:
            val_path = write_step_dataframe_artifact(
                state,
                "preprocess",
                scaled_val,
                filename=f"val_preprocessed_{station_id}.parquet",
            )
            val_outputs.append(str(val_path))
            val_shapes.append([int(scaled_val.shape[0]), int(scaled_val.shape[1])])

    payload = {
        "numeric_columns": numeric_columns_union,
        "scaling": scaling,
        "normalization_applied": should_normalize,
        "normalization_mode": normalization_mode,
        "preprocessed_shapes": {
            "train": [
                int(sum(shape[0] for shape in train_shapes)),
                int(max((shape[1] for shape in train_shapes), default=0)),
            ],
            "val": [
                int(sum(shape[0] for shape in val_shapes)),
                int(max((shape[1] for shape in val_shapes), default=0)),
            ],
        },
        "dataset_paths": {
            "train": train_outputs,
            "val": val_outputs,
        },
        "target_column": target_col,
        "split_applied": split_applied,
        "pipeline_source": "station_chunked_preprocess",
    }
    state.write("preprocess_result", payload)
    write_step_artifact(state, "preprocess", payload)
    return {
        "message": "preprocess completed",
        "numeric_columns": len(numeric_columns_union),
        "normalization_applied": should_normalize,
        "normalization_mode": normalization_mode,
    }
