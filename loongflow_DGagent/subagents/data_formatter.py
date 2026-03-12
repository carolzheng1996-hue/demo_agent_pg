from __future__ import annotations

import json
from textwrap import dedent
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from ..llm_utils import invoke_text
    from ..state import DGGlobalState
    from ..tools import execute_user_code_safely, infer_target_column_from_query, parse_target_columns, write_step_artifact
    from ..tools.file_tools import detect_date, set_features_multi, set_target
except ImportError:
    from llm_utils import invoke_text
    from state import DGGlobalState
    from tools import execute_user_code_safely, infer_target_column_from_query, parse_target_columns, write_step_artifact
    from tools.file_tools import detect_date, set_features_multi, set_target


def _sanitize_generated_code(code: str) -> str:
    text = code.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return dedent(text).strip() + "\n"


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


def _fallback_formatter_code(date_col: Optional[str], target_cols: List[str], feature_cols: List[str], need_split: bool) -> str:
    feature_literal = json.dumps(feature_cols, ensure_ascii=False)
    target_literal = json.dumps(target_cols, ensure_ascii=False)
    lines = [
        "outputs = {}",
        "df = raw_df.copy()",
    ]
    if date_col:
        lines.append(f"df[{date_col!r}] = pd.to_datetime(df[{date_col!r}], errors='coerce')")
    lines.extend(
        [
            f"feature_cols = {feature_literal}",
            f"target_cols = {target_literal}",
            "primary_target_col = target_cols[0]",
            "working_df = df.copy()",
            "numeric_candidates = feature_cols + target_cols",
            "for col in numeric_candidates:",
            "    if col in working_df.columns:",
            "        working_df[col] = pd.to_numeric(working_df[col], errors='coerce')",
            "working_df = working_df.ffill().bfill()",
            "standardized_df = working_df[feature_cols + target_cols].copy() if feature_cols else working_df[target_cols].copy()",
            'standardized_df = standardized_df.fillna(0.0)',
            "standardized_array = standardized_df.to_numpy(dtype=float)",
            'outputs["standardized_df"] = standardized_df',
            'outputs["standardized_array"] = standardized_array',
            'outputs["target_array"] = standardized_df[target_cols].to_numpy(dtype=float)',
            'outputs["target_columns"] = target_cols',
            'outputs["primary_target_col"] = primary_target_col',
        ]
    )
    if need_split:
        lines.extend(
            [
                "n = len(standardized_df)",
                "train_end = int(n * train_ratio)",
                "val_end = train_end + int(n * val_ratio)",
                'outputs["split_index"] = {"train_end": train_end, "val_end": val_end, "test_end": n}',
            ]
        )
    return "\n".join(lines) + "\n"


def _generate_formatter_code(
    state: DGGlobalState,
    df: pd.DataFrame,
    date_col: Optional[str],
    target_cols: List[str],
    feature_cols: List[str],
    need_split: bool,
) -> str:
    fallback = _fallback_formatter_code(date_col, target_cols, feature_cols, need_split)
    payload = {
        "task_description": state.read("user_query", ""),
        "use_custom_processing": bool(state.read("use_custom_processing", False)),
        "custom_processing_steps": state.read("custom_processing_steps", ""),
        "dataset_loading_result": state.read("dataset_loading_result", {}),
        "columns": df.columns.tolist(),
        "dtypes": {key: str(value) for key, value in df.dtypes.astype(str).to_dict().items()},
        "preview": df.head(3).to_dict(orient="records"),
        "date_column": date_col,
        "target_column": target_cols[0],
        "target_columns": target_cols,
        "feature_columns": feature_cols,
        "need_split": need_split,
        "split_ratios": _ratio_payload(state),
        "requirements": "Normalize supported data into standardized_df, standardized_array, target_array. Keep outputs in outputs dict.",
    }
    text = invoke_text(
        system_prompt=(
            "You generate Python code for a data formatter subagent. "
            "Return Python code only. The code receives raw_df, train_ratio, val_ratio, test_ratio. "
            "It must create standardized_df, standardized_array, target_array, and optionally outputs['split_index']. "
            "Use pandas/numpy only. Do not read files. Do not print. Assign results into outputs dict. "
            "If use_custom_processing is true, prioritize custom_processing_steps. "
            f"You must use target_columns exactly as provided: {target_cols}."
        ),
        user_prompt=json.dumps(payload, ensure_ascii=False),
        max_tokens=700,
        temperature=0.0,
    )
    sanitized = _sanitize_generated_code(text) if text else fallback
    if target_cols and not all(str(column) in sanitized for column in target_cols):
        return fallback
    return sanitized


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
        target_cols = [set_target(df, preferred=target_preference or "OT")]
    primary_target = target_cols[0]
    feature_cols = set_features_multi(df, target_cols)
    need_split = _need_split(state)
    ratios = _ratio_payload(state)
    if need_split:
        _validate_ratios(ratios)

    approved_generated_codes = state.read("approved_generated_codes", {}) or {}
    proposal_context = {
        "task_description": state.read("user_query", ""),
        "dataset_loading_result": state.read("dataset_loading_result", {}),
        "columns": df.columns.tolist(),
        "dtypes": {key: str(value) for key, value in df.dtypes.astype(str).to_dict().items()},
        "date_column": date_col,
        "target_column": primary_target,
        "target_columns": target_cols,
        "target_preference": target_preference,
        "feature_columns": feature_cols,
        "need_split": need_split,
        "split_ratios": ratios,
        "use_custom_processing": bool(state.read("use_custom_processing", False)),
        "custom_processing_steps": state.read("custom_processing_steps", ""),
    }
    modified_proposals = state.read("modified_proposals", {}) or {}
    generated_code = (
        modified_proposals.get("data_formatter")
        or approved_generated_codes.get("data_formatter")
        or _generate_formatter_code(
            state,
            df,
            date_col,
            target_cols,
            feature_cols,
            need_split,
        )
    )
    state.write(
        "data_formatter_proposal",
        {
            "generated_code": generated_code,
            "requires_user_confirmation": True,
            "context": proposal_context,
            "approved": bool(approved_generated_codes.get("data_formatter")) or bool(state.read("approve_generated_code", False)),
        },
    )

    dataset_profile = {
        "dataset_path": state.read("dataset_path"),
        "selected_file": state.read("dataset_loading_result", {}).get("selected_file"),
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": df.columns.tolist(),
        "date_column": date_col,
        "target_column": primary_target,
        "target_columns": target_cols,
        "feature_columns": feature_cols,
        "use_custom_processing": bool(state.read("use_custom_processing", False)),
        "custom_processing_steps": state.read("custom_processing_steps", ""),
        "need_split": need_split,
        "split_ratios": ratios if need_split else None,
    }

    if not approved_generated_codes.get("data_formatter") and not state.read("approve_generated_code", False):
        state.update(
            {
                "dataset_profile": dataset_profile,
                "awaiting_user_confirmation": {
                    "subagent": "data_formatter",
                    "action": "review_generated_code_and_rerun_with_approval",
                },
            }
        )
        write_step_artifact(
            state,
            "data_formatter",
            {
                "approved": False,
                "proposal": state.read("data_formatter_proposal"),
                "dataset_profile": dataset_profile,
            },
        )
        return {
            "message": "generated data formatter code; awaiting user confirmation",
            "approved": False,
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
        fallback_code = _fallback_formatter_code(date_col, target_cols, feature_cols, need_split)
        executed = execute_user_code_safely(fallback_code, execution_context)
        executed_code = fallback_code

    outputs_payload = executed.get("outputs", {}) if isinstance(executed.get("outputs"), dict) else {}
    standardized_df = executed.get("standardized_df", outputs_payload.get("standardized_df"))
    standardized_array = executed.get("standardized_array", outputs_payload.get("standardized_array"))
    target_array = executed.get("target_array", outputs_payload.get("target_array"))
    if standardized_df is None or standardized_array is None or target_array is None:
        raise RuntimeError("Generated data formatter code did not produce required outputs.")

    standardized_df = pd.DataFrame(standardized_df)
    standardized_array = np.asarray(standardized_array, dtype=float)
    target_array = np.asarray(target_array, dtype=float)
    dataset_profile["standardized_shape"] = [int(standardized_array.shape[0]), int(standardized_array.shape[1])]

    state.write_runtime("standardized_df", standardized_df)
    state.write_runtime("standardized_array", standardized_array)
    state.write_runtime("target_array", target_array)
    state.write_runtime("split_payload", outputs_payload.get("splits"))
    state.update(
        {
            "dataset_profile": dataset_profile,
            "awaiting_user_confirmation": None,
            "approved_generated_codes": {
                **approved_generated_codes,
                "data_formatter": executed_code,
            },
            "modified_proposals": {
                **modified_proposals,
                "data_formatter": "",
            },
            "data_formatter_execution": {
                "code_source": "generated" if executed_code == generated_code else "fallback",
                "executed_code": executed_code,
            },
            "data_formatter_result": {
                "approved": True,
                "dataset_profile": dataset_profile,
                "standardized_preview_rows": standardized_df.head(5).to_dict(orient="records"),
            },
        }
    )
    write_step_artifact(state, "data_formatter", state.read("data_formatter_result"))
    return {
        "message": "data formatted successfully",
        "approved": True,
        "standardized_shape": dataset_profile["standardized_shape"],
    }
