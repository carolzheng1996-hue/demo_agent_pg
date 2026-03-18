from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

try:
    from ..state import DGGlobalState
    from ..tools import (
        parse_column_selection,
        parse_target_columns,
        write_step_artifact,
        write_task_dataframe_artifact,
    )
    from ..tools.file_tools import set_features_multi, set_target
except ImportError:
    from state import DGGlobalState
    from tools import (
        parse_column_selection,
        parse_target_columns,
        write_step_artifact,
        write_task_dataframe_artifact,
    )
    from tools.file_tools import set_features_multi, set_target


def _load_ods_helpers() -> tuple[Any, Any]:
    try:
        from ..data_loading_pg.data_reading_odsdata import convert_ods_to_ds, load_station_frame
    except ImportError:
        from data_loading_pg.data_reading_odsdata import convert_ods_to_ds, load_station_frame
    return convert_ods_to_ds, load_station_frame


def _parse_name_list(raw: str) -> List[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


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


def _resolve_targets(ods_df: pd.DataFrame, state: DGGlobalState) -> List[str]:
    explicit_targets = parse_target_columns(state.read("target_col"), ods_df.columns.tolist())
    if explicit_targets:
        return explicit_targets
    return [set_target(ods_df)]


def _resolve_input_columns(ods_df: pd.DataFrame, targets: Sequence[str], state: DGGlobalState) -> List[str]:
    explicit_features = parse_column_selection(state.read("input_feature_cols"), ods_df.columns.tolist())
    if explicit_features:
        return [column for column in explicit_features if column not in targets]
    excluded = {"station", "timestamp_win", "__index_level_0__"}
    return [column for column in set_features_multi(ods_df, targets) if column not in excluded]


def _build_ods_to_ds_config(
    ods_df: pd.DataFrame,
    state: DGGlobalState,
    targets: Sequence[str],
) -> Dict[str, Any]:
    input_columns = _resolve_input_columns(ods_df, targets, state)
    history_columns = [column for column in input_columns if not str(column).endswith("_predict")]
    predict_columns = [column for column in input_columns if str(column).endswith("_predict")]
    if not predict_columns:
        predict_columns = [column for column in targets if column in ods_df.columns]
    if not history_columns and not predict_columns:
        raise ValueError("Unable to determine DS input columns. Please provide input_feature_cols and/or target_col.")

    return {
        "col_ls": history_columns,
        "pred_col_ls": predict_columns,
        "targ_col_ls": [column for column in targets if column in ods_df.columns],
        "hist_win_size": int(state.read("input_length") or 96),
        "forcast_win_size": int(state.read("output_length") or 1),
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

    target_columns = _resolve_targets(ods_df, state)
    ods_to_ds_config = _build_ods_to_ds_config(ods_df, state, target_columns)
    ds_df = convert_ods_to_ds(ods_dataframe=ods_df, plant_ids=selected_units, config=ods_to_ds_config)
    ds_df["timestamp_win"] = pd.to_datetime(ds_df["timestamp_win"], errors="coerce")
    ds_df = ds_df.sort_values(["station", "timestamp_win"]).reset_index(drop=True)

    ods_path = write_task_dataframe_artifact(state, "data/data_reading_ods_dataset.parquet", ods_df)
    ds_path = write_task_dataframe_artifact(state, "data/data_reading_ds_dataset.parquet", ds_df)
    description = (
        f"已读取 {len(selected_units)} 个站点的原始数据，并完成 ODS -> DS 转换。"
        f" ODS 形状为 {ods_df.shape[0]} x {ods_df.shape[1]}，DS 形状为 {ds_df.shape[0]} x {ds_df.shape[1]}。"
    )

    payload = {
        "dataset_path": str(dataset_path),
        "unit": str(state.read("unit", "")),
        "selected_units": selected_units,
        "available_units": _list_available_units(dataset_path),
        "available_file_count": len(selected_units),
        "is_directory": True,
        "raw_dataset_path": str(dataset_path),
        "ods_dataset_path": str(ods_path),
        "ds_dataset_path": str(ds_path),
        "description": description,
        "ods_shape": [int(ods_df.shape[0]), int(ods_df.shape[1])],
        "ds_shape": [int(ds_df.shape[0]), int(ds_df.shape[1])],
        "columns": [str(column) for column in ds_df.columns.tolist()],
        "ods_to_ds_config": ods_to_ds_config,
        "target_columns": list(target_columns),
    }
    state.update({"dataset_loading_result": payload})
    write_step_artifact(state, "data_reading", payload)
    return {
        "message": "ods and ds datasets prepared successfully",
        "selected_unit_count": len(selected_units),
        "ods_shape": payload["ods_shape"],
        "ds_shape": payload["ds_shape"],
    }
