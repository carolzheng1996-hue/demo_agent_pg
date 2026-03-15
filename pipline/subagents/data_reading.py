from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from ..state import DGGlobalState
    from ..tools import write_step_artifact
except ImportError:
    from state import DGGlobalState
    from tools import write_step_artifact


SUPPORTED_SUFFIXES = [".csv", ".pkl", ".npy", ".parquet"]


def _find_supported_files(dataset_path: Path) -> List[Path]:
    if dataset_path.is_file():
        if dataset_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {dataset_path.suffix}. Supported: {', '.join(SUPPORTED_SUFFIXES)}")
        return [dataset_path]

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    supported = [path for path in sorted(dataset_path.iterdir()) if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES]
    if not supported:
        raise ValueError(f"No supported files found under {dataset_path}. Supported: {', '.join(SUPPORTED_SUFFIXES)}")
    return supported[:100]


def _object_to_dataframe(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, np.ndarray):
        if obj.ndim == 1:
            return pd.DataFrame({"value": obj.astype(float)})
        if obj.ndim == 2:
            return pd.DataFrame(obj, columns=[f"feature_{idx}" for idx in range(obj.shape[1])])
        raise ValueError(f"Unsupported ndarray dimension: {obj.ndim}")
    if isinstance(obj, dict):
        for value in obj.values():
            if isinstance(value, pd.DataFrame):
                return value.copy()
            if isinstance(value, np.ndarray) and value.ndim in (1, 2):
                return _object_to_dataframe(value)
        return pd.DataFrame(obj)
    raise ValueError(f"Unsupported data object type: {type(obj).__name__}")


def _load_dataset(file_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    suffix = file_path.suffix.lower()
    metadata: Dict[str, Any] = {"file_type": suffix.lstrip(".")}

    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif suffix == ".pkl":
        with file_path.open("rb") as handle:
            payload = pickle.load(handle)
        metadata["payload_type"] = type(payload).__name__
        df = _object_to_dataframe(payload)
    elif suffix == ".npy":
        payload = np.load(file_path, allow_pickle=True)
        metadata["payload_type"] = type(payload).__name__
        df = _object_to_dataframe(payload)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    metadata["shape"] = [int(df.shape[0]), int(df.shape[1])]
    metadata["columns"] = [str(column) for column in df.columns.tolist()]
    metadata["dtypes"] = {str(key): str(value) for key, value in df.dtypes.astype(str).to_dict().items()}
    return df, metadata


def _combine_dataframes(file_payloads: List[Tuple[Path, pd.DataFrame, Dict[str, Any]]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if len(file_payloads) == 1:
        file_path, df, metadata = file_payloads[0]
        metadata = {**metadata, "loaded_files": [str(file_path)], "file_shapes": {str(file_path): metadata["shape"]}}
        return df, metadata

    union_columns: List[str] = []
    for _, df, _ in file_payloads:
        for column in df.columns.tolist():
            if str(column) not in union_columns:
                union_columns.append(str(column))

    combined_frames: List[pd.DataFrame] = []
    file_shapes: Dict[str, List[int]] = {}
    for file_path, df, metadata in file_payloads:
        aligned = df.copy()
        aligned.columns = [str(column) for column in aligned.columns.tolist()]
        aligned = aligned.reindex(columns=union_columns)
        aligned["__source_file__"] = file_path.name
        combined_frames.append(aligned)
        file_shapes[str(file_path)] = metadata["shape"]

    combined_df = pd.concat(combined_frames, axis=0, ignore_index=True, sort=False)
    combined_metadata = {
        "file_type": "directory_mixed",
        "shape": [int(combined_df.shape[0]), int(combined_df.shape[1])],
        "columns": [str(column) for column in combined_df.columns.tolist()],
        "dtypes": {str(key): str(value) for key, value in combined_df.dtypes.astype(str).to_dict().items()},
        "loaded_files": [str(path) for path, _, _ in file_payloads],
        "file_shapes": file_shapes,
    }
    return combined_df, combined_metadata


def _build_description(dataset_path: Path, files: List[Path], metadata: Dict[str, Any]) -> str:
    file_count = len(files)
    shape = metadata.get("shape", ["?", "?"])
    if dataset_path.is_dir():
        return f"已加载目录数据，共拼接 {file_count} 个文件，合并后数据形状为 {shape[0]} x {shape[1]}。"
    file_name = files[0].name if files else dataset_path.name
    return f"已加载 {file_name}，类型为 {metadata.get('file_type', 'unknown')}，当前数据形状为 {shape[0]} x {shape[1]}。"


def run(state: DGGlobalState) -> Dict:
    dataset_path_value = state.read("dataset_path")
    if not dataset_path_value:
        raise ValueError("dataset_path is required")

    dataset_path = Path(str(dataset_path_value))
    files = _find_supported_files(dataset_path)
    file_payloads = []
    for file_path in files:
        df, metadata = _load_dataset(file_path)
        file_payloads.append((file_path, df, metadata))
    raw_df, metadata = _combine_dataframes(file_payloads)
    description = _build_description(dataset_path, files, metadata)

    payload = {
        "dataset_path": str(dataset_path),
        "is_directory": dataset_path.is_dir(),
        "supported_files": [str(path) for path in files],
        "loaded_files": metadata.get("loaded_files", [str(files[0])]),
        "file_shapes": metadata.get("file_shapes", {str(files[0]): metadata["shape"]}),
        "description": description,
        "preview_rows": raw_df.head(5).to_dict(orient="records"),
        "shape": metadata["shape"],
        "columns": metadata["columns"],
        "file_type": metadata["file_type"],
    }
    state.write_runtime("raw_df", raw_df.copy())
    state.write_runtime(
        "raw_file_dfs",
        [{"file_path": str(file_path), "df": df.copy()} for file_path, df, _ in file_payloads],
    )
    state.write_runtime("loaded_dataset_metadata", metadata)
    state.update({"dataset_loading_result": payload})
    write_step_artifact(state, "data_reading", payload)
    return {
        "message": "dataset loaded successfully",
        "file_type": metadata["file_type"],
        "shape": metadata["shape"],
        "loaded_file_count": len(files),
    }
