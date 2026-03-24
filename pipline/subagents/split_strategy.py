from __future__ import annotations

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
    from .ds_pipeline_utils import load_ds_station_frames
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
    from subagents.ds_pipeline_utils import load_ds_station_frames


def _window_config(state: DGGlobalState) -> Dict:
    input_length = state.read("input_length")
    output_length = state.read("output_length")
    if input_length is None or output_length is None:
        raise ValueError("input_length/output_length is missing in state. Please provide them in config_all.json or runtime args.")
    return {
        "input_length": int(input_length),
        "output_length": int(output_length),
    }


def _normalized_ratios(profile: Dict, state: DGGlobalState) -> Dict[str, float]:
    ratios = profile.get("split_ratios") or {
        "train_ratio": state.read("train_ratio"),
        "val_ratio": state.read("val_ratio"),
    }
    if any(ratios.get(key) is None for key in ["train_ratio", "val_ratio"]):
        raise ValueError("train_ratio/val_ratio is missing in state. Please provide them in config_all.json or runtime args.")
    normalized = {key: float(ratios[key]) for key in ["train_ratio", "val_ratio"]}
    if any(value < 0 for value in normalized.values()):
        raise ValueError(f"split ratios must be non-negative, got {normalized}")
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("split ratios must contain at least one positive value")
    return {key: value / total for key, value in normalized.items()}


def _load_ds_split_dataframe(state: DGGlobalState) -> pd.DataFrame:
    station_frames = load_ds_station_frames(state, columns=["__row_id__", "station", "timestamp_win"])
    dataframe = pd.concat(station_frames.values(), axis=0, ignore_index=True) if station_frames else pd.DataFrame()
    if "timestamp_win" in dataframe.columns:
        dataframe["timestamp_win"] = pd.to_datetime(dataframe["timestamp_win"], errors="coerce")
    return dataframe


def _apply_split(
    df: pd.DataFrame,
    method: str,
    val_ratio: float,
    cutoff_date: str,
    test_units: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    requires_station_column = method in {"station_last_k", "station_month_last_k", "leave_stations_out"}
    if requires_station_column and "station" not in df.columns:
        raise ValueError(
            f"split_method={method} requires a station column, but the current DS dataset does not contain one."
        )

    if method == "station_last_k":
        return split_station_last_k(df, k=val_ratio)
    if method == "station_month_last_k":
        return split_station_month_last_k(df, k=val_ratio)
    if method == "global_last_k":
        return split_global_last_k(df, k=val_ratio)
    if method == "fixed_date":
        if not cutoff_date:
            raise ValueError("split_cutoff_date is required when split_method=fixed_date")
        return split_by_fixed_date(df, cutoff_date=cutoff_date)
    if method == "leave_stations_out":
        if not test_units:
            raise ValueError("split_test_units is required when split_method=leave_stations_out")
        return split_leave_stations_out(df, test_station_ids=test_units)
    raise ValueError(f"Unsupported split_method: {method}")


def run(state: DGGlobalState) -> Dict:
    profile = state.read("dataset_profile", {})
    dataframe = _load_ds_split_dataframe(state)
    rows = int(len(dataframe))
    if rows <= 0:
        raise ValueError("split_strategy requires a non-empty DS dataset.")

    ratios = _normalized_ratios(profile, state)
    split_method = str(state.read("split_method") or "").strip()
    if not split_method:
        raise ValueError("split_method is missing in state. Please provide it in config_all.json or runtime args.")
    split_cutoff_date = str(state.read("split_cutoff_date") or "").strip()
    split_test_units = [item.strip() for item in str(state.read("split_test_units") or "").split(",") if item.strip()]
    window_payload = _window_config(state)

    train_df, val_df = _apply_split(
        dataframe,
        method=split_method,
        val_ratio=ratios["val_ratio"],
        cutoff_date=split_cutoff_date,
        test_units=split_test_units,
    )
    if train_df.empty or val_df.empty:
        raise ValueError("Split strategy produced an empty training or validation dataset.")

    train_index_path = write_task_dataframe_artifact(state, "data/split/train_row_ids.parquet", train_df[["__row_id__"]].reset_index(drop=True))
    val_index_path = write_task_dataframe_artifact(state, "data/split/val_row_ids.parquet", val_df[["__row_id__"]].reset_index(drop=True))

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
        },
        "row_index_paths": {
            "train": str(train_index_path),
            "val": str(val_index_path),
        },
        "dataset_paths": {},
        "pipeline_source": "data_loading_pg.data_split_metadata_only",
    }
    state.write("split_strategy_result", payload)
    write_step_artifact(state, "split_strategy", payload)
    return {"message": "split strategy created", **payload}
