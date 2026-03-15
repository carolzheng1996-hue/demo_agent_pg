from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from ..state import DGGlobalState
    from ..tools import (
        infer_target_column_from_query,
        parse_column_selection,
        parse_target_columns,
        write_step_artifact,
    )
    from ..tools.file_tools import detect_date, set_features_multi, set_target
except ImportError:
    from state import DGGlobalState
    from tools import (
        infer_target_column_from_query,
        parse_column_selection,
        parse_target_columns,
        write_step_artifact,
    )
    from tools.file_tools import detect_date, set_features_multi, set_target


TIME_DERIVED_COLUMNS = ["date", "hour", "minute"]


def _ratio_payload(state: DGGlobalState) -> Dict[str, Optional[float]]:
    return {
        "train_ratio": state.read("train_ratio"),
        "val_ratio": state.read("val_ratio"),
        "test_ratio": state.read("test_ratio"),
    }


def _need_split(state: DGGlobalState) -> bool:
    return bool(state.read("plan_meta", {}).get("requires_modeling", False))


def _validate_ratios(ratios: Dict[str, Optional[float]]) -> Dict[str, float]:
    normalized = {
        "train_ratio": 0.7 if ratios.get("train_ratio") is None else float(ratios["train_ratio"]),
        "val_ratio": 0.1 if ratios.get("val_ratio") is None else float(ratios["val_ratio"]),
        "test_ratio": 0.2 if ratios.get("test_ratio") is None else float(ratios["test_ratio"]),
    }
    if any(value < 0 for value in normalized.values()):
        raise ValueError(f"Split ratios must be non-negative, got {normalized}")
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("Split ratios must contain at least one positive value.")
    return {key: value / total for key, value in normalized.items()}


def _resolve_feature_columns(df: pd.DataFrame, state: DGGlobalState, target_cols: List[str]) -> List[str]:
    explicit_features = parse_column_selection(state.read("input_feature_cols"), df.columns.tolist())
    if explicit_features:
        return [column for column in explicit_features if column not in target_cols]
    return set_features_multi(df, target_cols)


def _prepare_working_dataframe(df: pd.DataFrame, date_col: Optional[str], feature_cols: List[str], target_cols: List[str]) -> pd.DataFrame:
    working_df = df.copy()
    if date_col and date_col in working_df.columns:
        parsed_datetime = pd.to_datetime(working_df[date_col], errors="coerce")
        working_df[date_col] = parsed_datetime
        working_df = working_df.sort_values(date_col).reset_index(drop=True)
        parsed_datetime = pd.to_datetime(working_df[date_col], errors="coerce")
        if date_col != "date":
            working_df["date"] = parsed_datetime.dt.day.fillna(0).astype(int)
        if date_col != "hour":
            working_df["hour"] = parsed_datetime.dt.hour.fillna(0).astype(int)
        if date_col != "minute":
            working_df["minute"] = parsed_datetime.dt.minute.fillna(0).astype(int)

    numeric_candidates = [col for col in feature_cols + target_cols if col in working_df.columns]
    for col in numeric_candidates:
        working_df[col] = pd.to_numeric(working_df[col], errors="coerce")
    working_df = working_df.ffill().bfill().fillna(0.0)
    return working_df


def run(state: DGGlobalState) -> Dict:
    raw_df = state.read_runtime("raw_df")
    if raw_df is None:
        raise RuntimeError("Missing raw dataframe in runtime state. Run data_reading first.")

    df = raw_df.copy()
    date_col = detect_date(df)
    explicit_targets = parse_target_columns(state.read("target_col"), df.columns.tolist())
    target_preference = infer_target_column_from_query(state.read("user_query", ""), df.columns.tolist())
    if explicit_targets:
        target_cols = explicit_targets
    else:
        target_cols = [set_target(df, preferred=target_preference or "")]
    primary_target = target_cols[0]
    feature_cols = _resolve_feature_columns(df, state, target_cols)
    need_split = _need_split(state)
    ratios = _validate_ratios(_ratio_payload(state)) if need_split else _ratio_payload(state)

    working_df = _prepare_working_dataframe(df, date_col, feature_cols, target_cols)
    derived_time_columns = [column for column in TIME_DERIVED_COLUMNS if column in working_df.columns]
    selected_columns = ([date_col] if date_col and date_col in working_df.columns else []) + feature_cols + derived_time_columns + target_cols
    selected_columns = [column for idx, column in enumerate(selected_columns) if column in working_df.columns and column not in selected_columns[:idx]]
    standardized_df = working_df[selected_columns].copy()
    numeric_view = standardized_df.select_dtypes(include=["number"]).copy()
    standardized_array = numeric_view.to_numpy(dtype=float)
    target_array = working_df[target_cols].to_numpy(dtype=float)

    dataset_profile = {
        "dataset_path": state.read("dataset_path"),
        "is_directory": bool(state.read("dataset_loading_result", {}).get("is_directory", False)),
        "supported_files": state.read("dataset_loading_result", {}).get("supported_files", []),
        "loaded_files": state.read("dataset_loading_result", {}).get("loaded_files", []),
        "available_file_count": len(state.read("dataset_loading_result", {}).get("supported_files", []) or []),
        "file_type": state.read("dataset_loading_result", {}).get("file_type"),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "standardized_shape": [int(standardized_df.shape[0]), int(standardized_df.shape[1])],
        "columns": df.columns.tolist(),
        "date_column": date_col,
        "target_column": primary_target,
        "target_columns": target_cols,
        "feature_columns": feature_cols,
        "derived_time_columns": derived_time_columns,
        "input_feature_columns": feature_cols,
        "need_split": need_split,
        "split_ratios": ratios if need_split else None,
        "sorted_by_time": bool(date_col and date_col in working_df.columns),
    }
    payload = {
        "dataset_profile": dataset_profile,
        "target_preference": target_preference,
        "selected_columns": selected_columns,
        "standardized_shape": dataset_profile["standardized_shape"],
        "pipeline_source": "deterministic_formatter",
    }

    state.write_runtime("formatted_df", standardized_df.copy())
    state.write_runtime("standardized_df", standardized_df.copy())
    state.write_runtime("target_array", target_array)
    state.write_runtime("standardized_array", standardized_array)
    state.update({"dataset_profile": dataset_profile, "data_formatter_result": payload})
    write_step_artifact(state, "data_formatter", payload)
    return {
        "message": "data formatted successfully",
        "target_column": primary_target,
        "feature_count": len(feature_cols),
        "standardized_shape": dataset_profile["standardized_shape"],
    }
