from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from ..llm_utils import invoke_text
    from ..state import DGGlobalState
    from ..tools import write_step_artifact
except ImportError:
    from llm_utils import invoke_text
    from state import DGGlobalState
    from tools import write_step_artifact


SUPPORTED_SUFFIXES = [".csv", ".pkl", ".npy"]


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
            columns = [f"feature_{idx}" for idx in range(obj.shape[1])]
            return pd.DataFrame(obj, columns=columns)
        raise ValueError(f"Unsupported ndarray dimension: {obj.ndim}")
    if isinstance(obj, dict):
        for value in obj.values():
            if isinstance(value, pd.DataFrame):
                return value.copy()
            if isinstance(value, np.ndarray) and value.ndim in (1, 2):
                return _object_to_dataframe(value)
        try:
            return pd.DataFrame(obj)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Unable to convert dict payload to DataFrame: {exc}") from exc
    raise ValueError(f"Unsupported data object type: {type(obj).__name__}")


def _load_dataset(file_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    suffix = file_path.suffix.lower()
    metadata: Dict[str, Any] = {"file_type": suffix.lstrip("."), "selected_file": str(file_path)}

    if suffix == ".csv":
        df = pd.read_csv(file_path)
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


def _describe_dataset(state: DGGlobalState, dataset_path: Path, files: List[Path], metadata: Dict[str, Any]) -> str:
    payload = {
        "task": state.read("user_query", ""),
        "dataset_path": str(dataset_path),
        "selected_file": metadata.get("selected_file"),
        "file_type": metadata.get("file_type"),
        "shape": metadata.get("shape"),
        "columns": metadata.get("columns"),
        "available_files": [path.name for path in files[:10]],
    }
    text = invoke_text(
        system_prompt="You summarize datasets in one concise Chinese sentence for a data-loading agent.",
        user_prompt=json.dumps(payload, ensure_ascii=False),
        max_tokens=120,
        temperature=0.0,
    )
    return text.strip() if text else f"已加载 {metadata.get('file_type', 'unknown')} 数据文件 {Path(str(metadata.get('selected_file', ''))).name}。"


def run(state: DGGlobalState) -> Dict:
    dataset_path_value = state.read("dataset_path")
    if not dataset_path_value:
        raise ValueError("dataset_path is required")

    dataset_path = Path(str(dataset_path_value))
    files = _find_supported_files(dataset_path)
    selected_file = files[0]
    raw_df, metadata = _load_dataset(selected_file)
    description = _describe_dataset(state, dataset_path, files, metadata)

    state.write_runtime("raw_df", raw_df.copy())
    state.write_runtime("loaded_dataset_metadata", metadata)
    state.update(
        {
            "dataset_loading_result": {
                "dataset_path": str(dataset_path),
                "is_directory": dataset_path.is_dir(),
                "selected_file": str(selected_file),
                "supported_files": [str(path) for path in files],
                "description": description,
                "preview_rows": raw_df.head(5).to_dict(orient="records"),
                "shape": metadata["shape"],
                "columns": metadata["columns"],
                "file_type": metadata["file_type"],
            }
        }
    )
    write_step_artifact(state, "data_reading", state.read("dataset_loading_result"))
    return {
        "message": "dataset loaded successfully",
        "selected_file": str(selected_file),
        "file_type": metadata["file_type"],
        "shape": metadata["shape"],
    }
