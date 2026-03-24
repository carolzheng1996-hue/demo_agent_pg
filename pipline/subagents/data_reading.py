from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

try:
    from ..state import DGGlobalState
    from ..tools import (
        parse_column_selection,
        write_step_artifact,
        write_task_dataframe_artifact,
    )
    from .ds_pipeline_utils import build_ds_profile
except ImportError:
    from state import DGGlobalState
    from tools import (
        parse_column_selection,
        write_step_artifact,
        write_task_dataframe_artifact,
    )
    from subagents.ds_pipeline_utils import build_ds_profile


def _load_ods_helpers() -> tuple[Any, Any]:
    try:
        from ..data_loading_pg.data_reading_odsdata import convert_ods_to_ds, load_station_frame
    except ImportError:
        from data_loading_pg.data_reading_odsdata import convert_ods_to_ds, load_station_frame
    return convert_ods_to_ds, load_station_frame


def _parse_name_list(raw: str) -> List[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _parse_explicit_columns(
    raw: Any,
    available_columns: Sequence[str],
    field_name: str,
    *,
    allow_empty: bool = False,
) -> List[str]:
    selected = parse_column_selection(raw, available_columns)
    raw_items = _parse_name_list(raw) if isinstance(raw, str) else [str(item).strip() for item in (raw or []) if str(item).strip()]
    missing = [item for item in raw_items if item.lower() not in {str(column).lower() for column in selected}]
    if missing:
        raise ValueError(f"{field_name} contains unknown columns: {missing}. Available columns: {list(available_columns)}")
    if not selected and not allow_empty:
        raise ValueError(f"{field_name} is required and must reference existing ODS columns.")
    return selected


def _list_available_units(dataset_path: Path) -> List[str]:
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path not found or not a directory: {dataset_path}")

    units: List[str] = []
    for path in sorted(dataset_path.iterdir()):
        if path.name.startswith("station="):
            units.append(path.name.split("=", 1)[1])
    if not units:
        raise ValueError(f"No station partitions found under {dataset_path}. Expected entries like station=<unit>.")
    return units


def _resolve_units(dataset_path: Path, raw_units: str) -> List[str]:
    available_units = _list_available_units(dataset_path)
    requested = _parse_name_list(raw_units)
    if not requested:
        return available_units

    missing = [unit for unit in requested if unit not in available_units]
    if missing:
        raise ValueError(f"Unknown unit(s): {missing}. Available units: {available_units}")
    return requested


def _build_ods_to_ds_config(
    ods_df: pd.DataFrame,
    state: DGGlobalState,
) -> Dict[str, Any]:
    input_length = state.read("input_length")
    output_length = state.read("output_length")
    if input_length is None or output_length is None:
        raise ValueError("input_length/output_length is missing in state. Please provide them in config_all.json or runtime args.")

    available_columns = ods_df.columns.tolist()
    history_columns = _parse_explicit_columns(state.read("col_ls"), available_columns, "col_ls", allow_empty=True)
    predict_columns = _parse_explicit_columns(state.read("pred_col_ls"), available_columns, "pred_col_ls", allow_empty=True)
    target_columns = _parse_explicit_columns(state.read("targ_col_ls"), available_columns, "targ_col_ls")
    if not history_columns and not predict_columns:
        raise ValueError("At least one of col_ls or pred_col_ls must be provided for convert_ods_to_ds.")

    return {
        "col_ls": history_columns,
        "pred_col_ls": predict_columns,
        "targ_col_ls": target_columns,
        "hist_win_size": int(input_length),
        "forcast_win_size": int(output_length),
    }


def run(state: DGGlobalState) -> Dict:
    dataset_path_value = state.read("dataset_path")
    if not dataset_path_value:
        raise ValueError("dataset_path is required")

    dataset_path = Path(str(dataset_path_value))
    selected_units = _resolve_units(dataset_path, str(state.read("unit", "")))
    convert_ods_to_ds, load_station_frame = _load_ods_helpers()
    ods_frames = [load_station_frame(str(dataset_path), station_id) for station_id in selected_units]
    ods_df = pd.concat(ods_frames, ignore_index=True)
    ods_df["timestamp_win"] = pd.to_datetime(ods_df["timestamp_win"], errors="coerce")
    ods_df = ods_df.sort_values(["station", "timestamp_win"]).reset_index(drop=True)

    ods_to_ds_config = _build_ods_to_ds_config(ods_df, state)
    target_columns = list(ods_to_ds_config["targ_col_ls"])
    ds_df = convert_ods_to_ds(ods_dataframe=ods_df, plant_ids=selected_units, config=ods_to_ds_config)
    ds_df["timestamp_win"] = pd.to_datetime(ds_df["timestamp_win"], errors="coerce")
    ds_df = ds_df.sort_values(["station", "timestamp_win"]).reset_index(drop=True)
    ds_df["__row_id__"] = ds_df.index.astype("int64")

    derived_payload: Dict[str, Any] = {}
    if not str(state.read("target_col") or "").strip() and target_columns:
        derived_payload["target_col"] = ",".join(target_columns)
    if not str(state.read("input_feature_cols") or "").strip():
        derived_payload["input_feature_cols"] = ",".join(ods_to_ds_config["col_ls"] + ods_to_ds_config["pred_col_ls"])
    if derived_payload:
        state.update(derived_payload, persist=False)

    ds_path = write_task_dataframe_artifact(state, "data/data_reading_ds_dataset.parquet", ds_df)
    ds_dataset_paths: Dict[str, str] = {}
    if "station" in ds_df.columns:
        for station_id, station_df in ds_df.groupby(ds_df["station"].astype(str), sort=True):
            station_path = write_task_dataframe_artifact(
                state,
                f"data/ds/station_{station_id}.parquet",
                station_df.reset_index(drop=True),
            )
            ds_dataset_paths[str(station_id)] = str(station_path)
    dataset_profile = build_ds_profile(
        ds_df,
        state,
        ds_dataset_path=str(ds_path),
    )
    dataset_profile["ds_dataset_paths"] = ds_dataset_paths
    description = (
        f"已读取 {len(selected_units)} 个站点的原始数据，并完成 ODS -> DS 转换。"
        f" DS 形状为 {ds_df.shape[0]} x {ds_df.shape[1]}。"
    )

    payload = {
        "dataset_path": str(dataset_path),
        "unit": str(state.read("unit", "")),
        "selected_units": selected_units,
        "available_units": _list_available_units(dataset_path),
        "available_file_count": len(selected_units),
        "is_directory": True,
        "raw_dataset_path": str(dataset_path),
        "ds_dataset_path": str(ds_path),
        "description": description,
        "ds_shape": [int(ds_df.shape[0]), int(ds_df.shape[1])],
        "columns": [str(column) for column in ds_df.columns.tolist()],
        "ds_dataset_paths": ds_dataset_paths,
        "ods_to_ds_config": ods_to_ds_config,
        "resolved_target_col": str(state.read("target_col") or ""),
        "resolved_input_feature_cols": str(state.read("input_feature_cols") or ""),
        "target_columns": list(target_columns),
        "dataset_profile": dataset_profile,
    }
    state.update({"dataset_loading_result": payload, "dataset_profile": dataset_profile})
    write_step_artifact(state, "data_reading", payload)
    return {
        "message": "ods and ds datasets prepared successfully",
        "selected_unit_count": len(selected_units),
        "ds_shape": payload["ds_shape"],
    }
