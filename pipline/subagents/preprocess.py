from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from ..state import DGGlobalState
    from ..tools import write_step_artifact, write_step_dataframe_artifact
except ImportError:
    from state import DGGlobalState
    from tools import write_step_artifact, write_step_dataframe_artifact


def _load_engineered_dataframe(state: DGGlobalState) -> pd.DataFrame:
    result = state.read("feature_engineering_result", {})
    engineered_paths = result.get("engineered_dataset_paths") or {}
    if engineered_paths:
        frames = [pd.read_parquet(Path(str(path))) for path in engineered_paths.values()]
        return pd.concat(frames, axis=0, ignore_index=True)

    engineered_path = result.get("engineered_dataset_path")
    if not engineered_path:
        raise RuntimeError("Missing engineered dataset path. Run feature_engineering first.")
    return pd.read_parquet(Path(str(engineered_path)))


def _split_frames(engineered_df: pd.DataFrame, state: DGGlobalState) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    row_ids = state.read("split_strategy_result", {}).get("row_ids", {})
    if not row_ids:
        empty = engineered_df.iloc[0:0].copy()
        return engineered_df.copy().reset_index(drop=True), empty, empty
    if "__row_id__" not in engineered_df.columns:
        raise RuntimeError("Engineered dataset is missing __row_id__. Run data_formatter first.")

    train_df = engineered_df[engineered_df["__row_id__"].isin(row_ids.get("train", []))].copy()
    val_df = engineered_df[engineered_df["__row_id__"].isin(row_ids.get("val", []))].copy()
    test_df = engineered_df[engineered_df["__row_id__"].isin(row_ids.get("test", []))].copy()
    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Split row ids do not match engineered dataset rows.")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _normalize_with_train_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_cols: List[str],
    mode: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    scaling: Dict = {}
    if not numeric_cols or mode not in {"zscore", "minmax"}:
        return train_df, val_df, test_df, scaling

    scaled_train = train_df.copy()
    scaled_val = val_df.copy()
    scaled_test = test_df.copy()
    for col in numeric_cols:
        train_values = pd.to_numeric(scaled_train[col], errors="coerce").to_numpy(dtype=float)
        val_values = pd.to_numeric(scaled_val[col], errors="coerce").to_numpy(dtype=float)
        test_values = pd.to_numeric(scaled_test[col], errors="coerce").to_numpy(dtype=float)
        if mode == "minmax":
            min_value = float(np.nanmin(train_values))
            max_value = float(np.nanmax(train_values))
            scale = max(max_value - min_value, 1e-8)
            scaled_train[col] = (train_values - min_value) / scale
            scaled_val[col] = (val_values - min_value) / scale
            scaled_test[col] = (test_values - min_value) / scale
            scaling[col] = {"data_min": min_value, "data_max": max_value, "scale": scale, "mode": "minmax"}
        else:
            mean = float(np.nanmean(train_values))
            std = float(np.nanstd(train_values)) or 1.0
            scaled_train[col] = (train_values - mean) / std
            scaled_val[col] = (val_values - mean) / std
            scaled_test[col] = (test_values - mean) / std
            scaling[col] = {"mean": mean, "std": std, "mode": "zscore"}
    return scaled_train, scaled_val, scaled_test, scaling


def run(state: DGGlobalState) -> Dict:
    engineered_df = _load_engineered_dataframe(state)
    train_df, val_df, test_df = _split_frames(engineered_df, state)

    target_col = state.read("dataset_profile", {}).get("target_column")
    if not target_col:
        raise RuntimeError("Missing target column in dataset profile.")

    for frame in (train_df, val_df, test_df):
        for col in frame.columns:
            if str(frame[col].dtype) != "datetime64[ns]":
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame.ffill(inplace=True)
        frame.bfill(inplace=True)
        frame.fillna(0.0, inplace=True)

    numeric_cols = [column for column in train_df.select_dtypes(include=["number"]).columns.tolist() if column != "__row_id__"]
    scaling_policy = state.read("datanorm_result", {})
    should_normalize = bool(scaling_policy.get("should_normalize", False))
    normalization_mode = str(scaling_policy.get("recommended_mode", "skip") or "skip").lower()
    scaled_train, scaled_val, scaled_test, scaling = _normalize_with_train_statistics(
        train_df,
        val_df,
        test_df,
        numeric_cols,
        normalization_mode if should_normalize else "skip",
    )

    train_path = write_step_dataframe_artifact(state, "preprocess", scaled_train, filename="train_preprocessed.parquet")
    dataset_paths = {"train": str(train_path)}
    if not scaled_val.empty:
        val_path = write_step_dataframe_artifact(state, "preprocess", scaled_val, filename="val_preprocessed.parquet")
        dataset_paths["val"] = str(val_path)
    if not scaled_test.empty:
        test_path = write_step_dataframe_artifact(state, "preprocess", scaled_test, filename="test_preprocessed.parquet")
        dataset_paths["test"] = str(test_path)
    payload = {
        "numeric_columns": numeric_cols,
        "scaling": scaling,
        "normalization_applied": should_normalize,
        "normalization_mode": normalization_mode,
        "preprocessed_shapes": {
            "train": [int(scaled_train.shape[0]), int(scaled_train.shape[1])],
            "val": [int(scaled_val.shape[0]), int(scaled_val.shape[1])],
            "test": [int(scaled_test.shape[0]), int(scaled_test.shape[1])],
        },
        "dataset_paths": dataset_paths,
        "target_column": target_col,
        "split_applied": bool(state.read("split_strategy_result", {}).get("row_ids")),
    }
    state.write("preprocess_result", payload)
    write_step_artifact(state, "preprocess", payload)
    return {
        "message": "preprocess completed",
        "numeric_columns": len(numeric_cols),
        "normalization_applied": should_normalize,
        "normalization_mode": normalization_mode,
    }
