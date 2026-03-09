from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from ..tools.file_tools import detect_date, set_features, set_target
    from ..llm_utils import invoke_text
    from ..state import DGGlobalState
    from ..tools import execute_user_code_safely, write_step_artifact
except ImportError:
    from tools.file_tools import detect_date, set_features, set_target
    from llm_utils import invoke_text
    from state import DGGlobalState
    from tools import execute_user_code_safely, write_step_artifact


def _sanitize_generated_code(code: str) -> str:
    text = code.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines = lines[1:]
    return dedent("\n".join(lines)).strip() + "\n"


def _ratio_payload(state: DGGlobalState) -> Dict[str, Optional[float]]:
    return {
        "train_ratio": state.read("train_ratio"),
        "val_ratio": state.read("val_ratio"),
        "test_ratio": state.read("test_ratio"),
    }


def _need_split(state: DGGlobalState) -> bool:
    plan = state.read("plan", [])
    return any(step in plan for step in ["split_strategy", "model_selection", "model_training", "model_integration", "evaluator"])


def _validate_ratios(ratios: Dict[str, Optional[float]]) -> None:
    values = [ratios.get("train_ratio"), ratios.get("val_ratio"), ratios.get("test_ratio")]
    if any(v is None for v in values):
        raise ValueError("Model training task requires --train-ratio --val-ratio --test-ratio.")
    total = float(sum(values))
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")


def _fallback_reader_code(date_col: Optional[str], target_col: str, feature_cols: List[str], need_split: bool) -> str:
    feature_literal = json.dumps(feature_cols, ensure_ascii=False)
    lines = [
        "outputs = {}",
        "df = raw_df.copy()",
    ]
    if date_col:
        lines.append(f"df[{date_col!r}] = pd.to_datetime(df[{date_col!r}])")
    lines.extend(
        [
            f"feature_cols = {feature_literal}",
            f"target_col = {target_col!r}",
            "standardized_df = df[feature_cols + [target_col]].copy() if feature_cols else df[[target_col]].copy()",
            'standardized_df = standardized_df.apply(pd.to_numeric, errors="coerce")',
            'standardized_df = standardized_df.ffill().bfill()',
            "standardized_array = standardized_df.to_numpy(dtype=float)",
            'outputs["standardized_df"] = standardized_df',
            'outputs["standardized_array"] = standardized_array',
            'outputs["target_array"] = standardized_df[target_col].to_numpy(dtype=float)',
        ]
    )
    if need_split:
        lines.extend(
            [
                "n = len(df)",
                "train_end = int(n * train_ratio)",
                "val_end = train_end + int(n * val_ratio)",
                'outputs["splits"] = {',
                '    "train": standardized_array[:train_end].tolist(),',
                '    "val": standardized_array[train_end:val_end].tolist(),',
                '    "test": standardized_array[val_end:].tolist(),',
                "}",
                'outputs["split_index"] = {"train_end": train_end, "val_end": val_end, "test_end": n}',
            ]
        )
    return "\n".join(lines) + "\n"


def _generate_reader_code(state: DGGlobalState, df: pd.DataFrame, date_col: Optional[str], target_col: str, feature_cols: List[str], need_split: bool) -> str:
    fallback = _fallback_reader_code(date_col, target_col, feature_cols, need_split)
    payload = {
        "task_description": state.read("user_query", ""),
        "dataset_path": state.read("dataset_path"),
        "columns": df.columns.tolist(),
        "dtypes": {k: str(v) for k, v in df.dtypes.astype(str).to_dict().items()},
        "preview": df.head(3).to_dict(orient="records"),
        "date_column": date_col,
        "target_column": target_col,
        "feature_columns": feature_cols,
        "need_split": need_split,
        "split_ratios": _ratio_payload(state),
    }
    text = invoke_text(
        system_prompt=(
            "You generate Python code for a data-reading subagent. "
            "Return Python code only. The code receives raw_df, train_ratio, val_ratio, test_ratio. "
            "It must create standardized_df, standardized_array, target_array, and optionally outputs['splits']. "
            "Use pandas/numpy only. Do not read files. Do not print. Assign results into outputs dict."
        ),
        user_prompt=json.dumps(payload, ensure_ascii=False),
        max_tokens=600,
        temperature=0.0,
    )
    return _sanitize_generated_code(text) if text else fallback


def run(state: DGGlobalState) -> Dict:
    dataset_path = state.read("dataset_path")
    if not dataset_path:
        raise ValueError("dataset_path is required")

    csv_path = Path(str(dataset_path))
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    date_col = detect_date(df)
    target_col = set_target(df)
    feature_cols = set_features(df, target_col)
    need_split = _need_split(state)
    ratios = _ratio_payload(state)
    if need_split:
        _validate_ratios(ratios)

    generated_code = _generate_reader_code(state, df, date_col, target_col, feature_cols, need_split)
    state.write(
        "data_reader_proposal",
        {
            "generated_code": generated_code,
            "requires_user_confirmation": True,
            "approved": bool(state.read("approve_generated_code", False)),
        },
    )

    if not state.read("approve_generated_code", False):
        state.update(
            {
                "dataset_profile": {
                    "dataset_path": str(csv_path),
                    "shape": [int(df.shape[0]), int(df.shape[1])],
                    "columns": df.columns.tolist(),
                    "date_column": date_col,
                    "target_column": target_col,
                    "feature_columns": feature_cols,
                    "need_split": need_split,
                    "split_ratios": ratios if need_split else None,
                },
                "awaiting_user_confirmation": {
                    "subagent": "data_reading",
                    "action": "review_generated_code_and_rerun_with_approval",
                },
            }
        )
        write_step_artifact(
            state,
            "data_reading",
            {
                "approved": False,
                "dataset_path": str(csv_path),
                "proposal": state.read("data_reader_proposal"),
                "dataset_profile": state.read("dataset_profile"),
            },
        )
        return {
            "message": "generated data reading code; awaiting user confirmation",
            "approved": False,
            "dataset_path": str(csv_path),
            "pause_execution": True,
        }

    execution_context = {
        "raw_df": df.copy(),
        "train_ratio": ratios.get("train_ratio"),
        "val_ratio": ratios.get("val_ratio"),
        "test_ratio": ratios.get("test_ratio"),
    }
    try:
        executed = execute_user_code_safely(generated_code, execution_context)
        executed_code = generated_code
    except Exception:
        fallback_code = _fallback_reader_code(date_col, target_col, feature_cols, need_split)
        executed = execute_user_code_safely(fallback_code, execution_context)
        executed_code = fallback_code
    outputs_payload = executed.get("outputs", {}) if isinstance(executed.get("outputs"), dict) else {}
    standardized_df = executed.get("standardized_df", outputs_payload.get("standardized_df"))
    standardized_array = executed.get("standardized_array", outputs_payload.get("standardized_array"))
    target_array = executed.get("target_array", outputs_payload.get("target_array"))
    if standardized_df is None or standardized_array is None or target_array is None:
        raise RuntimeError("Generated data reading code did not produce required outputs.")

    standardized_df = pd.DataFrame(standardized_df)
    standardized_array = np.asarray(standardized_array, dtype=float)
    target_array = np.asarray(target_array, dtype=float)

    state.write_runtime("raw_df", df)
    state.write_runtime("standardized_df", standardized_df)
    state.write_runtime("standardized_array", standardized_array)
    state.write_runtime("target_array", target_array)
    state.write_runtime("split_payload", executed.get("splits", outputs_payload.get("splits")))
    state.update(
        {
            "dataset_profile": {
                "dataset_path": str(csv_path),
                "shape": [int(df.shape[0]), int(df.shape[1])],
                "standardized_shape": [int(standardized_array.shape[0]), int(standardized_array.shape[1])],
                "columns": df.columns.tolist(),
                "date_column": date_col,
                "target_column": target_col,
                "feature_columns": feature_cols,
                "need_split": need_split,
                "split_ratios": ratios if need_split else None,
            },
            "awaiting_user_confirmation": None,
            "data_reader_execution": {
                "code_source": "generated" if executed_code == generated_code else "fallback",
                "executed_code": executed_code,
            },
        }
    )
    write_step_artifact(
        state,
        "data_reading",
        {
            "approved": True,
            "dataset_profile": state.read("dataset_profile"),
            "standardized_preview_rows": standardized_df.head(5).to_dict(orient="records"),
        },
    )

    return {
        "message": "data standardized successfully",
        "approved": True,
        "standardized_shape": [int(standardized_array.shape[0]), int(standardized_array.shape[1])],
    }
