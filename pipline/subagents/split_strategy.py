from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    from ..data_loading_pg.data_split import (
        split_by_fixed_date,
        split_global_last_k,
        split_leave_stations_out,
        split_station_last_k,
        split_station_month_last_k,
    )
    from ..state import DGGlobalState
    from ..tools import write_step_artifact, write_task_dataframe_artifact
except ImportError:
    from data_loading_pg.data_split import (
        split_by_fixed_date,
        split_global_last_k,
        split_leave_stations_out,
        split_station_last_k,
        split_station_month_last_k,
    )
    from state import DGGlobalState
    from tools import write_step_artifact, write_task_dataframe_artifact


def _default_window_config(state: DGGlobalState) -> Dict:
    requires_modeling = bool(state.read("plan_meta", {}).get("requires_modeling", False))
    query = str(state.read("user_query", "")).lower()
    output_len = 24 if any(keyword in query for keyword in ["24", "day ahead", "one day"]) else 1
    return {
        "input_length": 96 if requires_modeling else None,
        "output_length": output_len if requires_modeling else None,
        "time_increment": 1 if requires_modeling else None,
    }


def _normalized_ratios(profile: Dict) -> Dict[str, float]:
    ratios = profile.get("split_ratios") or {"train_ratio": 0.7, "val_ratio": 0.1, "test_ratio": 0.2}
    normalized = {key: float(ratios[key]) for key in ["train_ratio", "val_ratio", "test_ratio"]}
    if any(value < 0 for value in normalized.values()):
        raise ValueError(f"split ratios must be non-negative, got {normalized}")
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("split ratios must contain at least one positive value")
    return {key: value / total for key, value in normalized.items()}


def _load_formatted_dataframe(state: DGGlobalState) -> pd.DataFrame:
    profile = state.read("dataset_profile", {})
    station_paths = profile.get("formatted_dataset_paths") or {}
    frames: List[pd.DataFrame] = []
    if station_paths:
        for station_id, formatted_path in station_paths.items():
            dataframe = pd.read_parquet(Path(str(formatted_path)))
            if "timestamp_win" in dataframe.columns:
                dataframe["timestamp_win"] = pd.to_datetime(dataframe["timestamp_win"], errors="coerce")
            dataframe["station"] = str(station_id)
            frames.append(dataframe)
        return pd.concat(frames, axis=0, ignore_index=True)

    formatted_path = profile.get("formatted_dataset_path")
    if not formatted_path:
        raise RuntimeError("split_strategy requires formatted_dataset_path.")
    dataframe = pd.read_parquet(Path(str(formatted_path)))
    if "timestamp_win" in dataframe.columns:
        dataframe["timestamp_win"] = pd.to_datetime(dataframe["timestamp_win"], errors="coerce")
    return dataframe


def _apply_primary_split(
    df: pd.DataFrame,
    method: str,
    test_ratio: float,
    cutoff_date: str,
    test_units: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    requires_station_column = method in {"station_last_k", "station_month_last_k", "leave_stations_out"}
    if requires_station_column and "station" not in df.columns:
        raise ValueError(
            f"split_method={method} requires a station column, but the formatted dataset has already been merged by station."
        )

    if method == "station_last_k":
        return split_station_last_k(df, k=test_ratio)
    if method == "station_month_last_k":
        return split_station_month_last_k(df, k=test_ratio)
    if method == "global_last_k":
        return split_global_last_k(df, k=test_ratio)
    if method == "fixed_date":
        if not cutoff_date:
            raise ValueError("split_cutoff_date is required when split_method=fixed_date")
        return split_by_fixed_date(df, cutoff_date=cutoff_date)
    if method == "leave_stations_out":
        if not test_units:
            raise ValueError("split_test_units is required when split_method=leave_stations_out")
        return split_leave_stations_out(df, test_station_ids=test_units)
    raise ValueError(f"Unsupported split_method: {method}")


def _apply_validation_split(df: pd.DataFrame, method: str, val_ratio_within_train: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) <= 1:
        raise ValueError("Training partition is too small to create a validation split.")

    if method == "station_last_k":
        return split_station_last_k(df, k=val_ratio_within_train)
    if method == "station_month_last_k":
        return split_station_month_last_k(df, k=val_ratio_within_train)
    return split_global_last_k(df, k=val_ratio_within_train)


def run(state: DGGlobalState) -> Dict:
    profile = state.read("dataset_profile", {})
    dataframe = _load_formatted_dataframe(state)
    rows = int(len(dataframe))
    if rows <= 0:
        raise ValueError("split_strategy requires a non-empty formatted dataset.")

    ratios = _normalized_ratios(profile)
    split_method = str(state.read("split_method", "global_last_k") or "global_last_k")
    split_cutoff_date = str(state.read("split_cutoff_date", "") or "")
    split_test_units = [item.strip() for item in str(state.read("split_test_units", "") or "").split(",") if item.strip()]
    user_window = {
        "input_length": state.read("input_length"),
        "output_length": state.read("output_length"),
        "time_increment": state.read("time_increment"),
    }
    window_payload = dict(user_window) if any(value is not None for value in user_window.values()) else _default_window_config(state)

    train_val_df, test_df = _apply_primary_split(
        dataframe,
        method=split_method,
        test_ratio=ratios["test_ratio"],
        cutoff_date=split_cutoff_date,
        test_units=split_test_units,
    )
    if train_val_df.empty or test_df.empty:
        raise ValueError("Split strategy produced an empty train/val or test dataset.")

    val_ratio_within_train = ratios["val_ratio"] / max(ratios["train_ratio"] + ratios["val_ratio"], 1e-8)
    train_df, val_df = _apply_validation_split(train_val_df, split_method, val_ratio_within_train)
    if train_df.empty or val_df.empty:
        raise ValueError("Split strategy produced an empty training or validation dataset.")

    train_path = write_task_dataframe_artifact(state, "data/split/train_dataset.parquet", train_df)
    val_path = write_task_dataframe_artifact(state, "data/split/val_dataset.parquet", val_df)
    test_path = write_task_dataframe_artifact(state, "data/split/test_dataset.parquet", test_df)

    payload = {
        "strategy": split_method,
        "rows": rows,
        "ratios": ratios,
        "window_config": window_payload,
        "split_cutoff_date": split_cutoff_date or None,
        "split_test_units": split_test_units,
        "counts": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "dataset_paths": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
        "row_ids": {
            "train": train_df["__row_id__"].astype(int).tolist() if "__row_id__" in train_df.columns else [],
            "val": val_df["__row_id__"].astype(int).tolist() if "__row_id__" in val_df.columns else [],
            "test": test_df["__row_id__"].astype(int).tolist() if "__row_id__" in test_df.columns else [],
        },
        "pipeline_source": "data_loading_pg.data_split",
    }
    state.write("split_strategy_result", payload)
    write_step_artifact(state, "split_strategy", payload)
    return {"message": "split strategy created", **payload}
