from __future__ import annotations

import json
from textwrap import dedent
from typing import Dict

try:
    from ..llm_utils import invoke_json, invoke_text
    from ..state import DGGlobalState
    from ..tools import execute_user_code_safely, write_step_artifact
except ImportError:
    from llm_utils import invoke_json, invoke_text
    from state import DGGlobalState
    from tools import execute_user_code_safely, write_step_artifact


def _default_window_config(state: DGGlobalState) -> Dict:
    requires_modeling = bool(state.read("plan_meta", {}).get("requires_modeling", False))
    query = str(state.read("user_query", "")).lower()
    output_len = 1
    if any(keyword in query for keyword in ["预测", "forecast", "horizon"]):
        output_len = 24 if "24" in query else 1
    return {
        "input_length": 96 if requires_modeling else None,
        "output_length": output_len if requires_modeling else None,
        "time_increment": 1 if requires_modeling else None,
    }


def _sanitize_generated_code(code: str) -> str:
    text = str(code or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return dedent(text).strip() + "\n"


def _fallback_split_code(rows: int, ratios: Dict, window_payload: Dict) -> str:
    return (
        "outputs = {}\n"
        "rows = int(rows)\n"
        "train_ratio = float(train_ratio)\n"
        "val_ratio = float(val_ratio)\n"
        "test_ratio = float(test_ratio)\n"
        "train_end = int(rows * train_ratio)\n"
        "val_end = train_end + int(rows * val_ratio)\n"
        "outputs['strategy'] = 'time_order_split'\n"
        "outputs['rows'] = rows\n"
        "outputs['ratios'] = {\n"
        "    'train_ratio': train_ratio,\n"
        "    'val_ratio': val_ratio,\n"
        "    'test_ratio': test_ratio,\n"
        "}\n"
        f"outputs['window_config'] = {json.dumps(window_payload, ensure_ascii=False)}\n"
        "outputs['indices'] = {\n"
        "    'train': [0, train_end],\n"
        "    'val': [train_end, val_end],\n"
        "    'test': [val_end, rows],\n"
        "}\n"
    )


def _generate_split_code(state: DGGlobalState, profile: Dict, rows: int, ratios: Dict, window_payload: Dict) -> str:
    fallback = _fallback_split_code(rows, ratios, window_payload)
    payload = {
        "task_description": state.read("user_query", ""),
        "dataset_profile": profile,
        "rows": rows,
        "ratios": ratios,
        "window_config": window_payload,
    }
    text = invoke_text(
        system_prompt=(
            "You generate Python code for a time-series split strategy agent. "
            "Return Python code only. The code receives rows, train_ratio, val_ratio, test_ratio, input_length, output_length, time_increment. "
            "It must assign outputs['strategy'], outputs['rows'], outputs['ratios'], outputs['window_config'], outputs['indices']. "
            "Use pure Python only. Do not read files. Do not print."
        ),
        user_prompt=json.dumps(payload, ensure_ascii=False),
        max_tokens=420,
        temperature=0.0,
    )
    return _sanitize_generated_code(text) if text else fallback


def run(state: DGGlobalState) -> Dict:
    profile = state.read("dataset_profile", {})
    rows = int(profile.get("shape", [0, 0])[0])
    if rows <= 0:
        raise ValueError("split_strategy requires dataset_profile.shape.")

    ratios = profile.get("split_ratios") or {
        "train_ratio": state.read("train_ratio"),
        "val_ratio": state.read("val_ratio"),
        "test_ratio": state.read("test_ratio"),
    }
    if not all(ratios.get(key) is not None for key in ["train_ratio", "val_ratio", "test_ratio"]):
        raise ValueError("split_strategy requires train/val/test ratios.")

    user_window = {
        "input_length": state.read("input_length"),
        "output_length": state.read("output_length"),
        "time_increment": state.read("time_increment"),
    }
    if any(value is not None for value in user_window.values()):
        window_payload = dict(user_window)
    else:
        window_payload = invoke_json(
            system_prompt=(
                "You decide sequence/window parameters for time series splitting. "
                "Return JSON with keys: input_length, output_length, time_increment. "
                "If the task is not a modeling task, all keys can be null."
            ),
            user_prompt=str(
                {
                    "task_description": state.read("user_query", ""),
                    "plan_meta": state.read("plan_meta", {}),
                    "dataset_profile": profile,
                }
            ),
            max_tokens=120,
            temperature=0.0,
        ) or _default_window_config(state)

    proposal_context = {
        "task_description": state.read("user_query", ""),
        "dataset_profile": profile,
        "rows": rows,
        "ratios": ratios,
        "window_config": window_payload,
    }
    approved_generated_codes = state.read("approved_generated_codes", {}) or {}
    modified_proposals = state.read("modified_proposals", {}) or {}
    generated_code = (
        modified_proposals.get("split_strategy")
        or approved_generated_codes.get("split_strategy")
        or _generate_split_code(
            state,
            profile,
            rows,
            ratios,
            window_payload,
        )
    )
    state.write(
        "split_strategy_proposal",
        {
            "generated_code": generated_code,
            "requires_user_confirmation": True,
            "context": proposal_context,
            "approved": bool(approved_generated_codes.get("split_strategy")) or bool(state.read("approve_generated_code", False)),
        },
    )

    if not approved_generated_codes.get("split_strategy") and not state.read("approve_generated_code", False):
        state.update(
            {
                "awaiting_user_confirmation": {
                    "subagent": "split_strategy",
                    "action": "review_generated_code_and_rerun_with_approval",
                },
            }
        )
        write_step_artifact(
            state,
            "split_strategy",
            {
                "approved": False,
                "proposal": state.read("split_strategy_proposal"),
                "dataset_profile": profile,
                "ratios": ratios,
                "window_config": window_payload,
            },
        )
        return {
            "message": "generated split strategy code; awaiting user confirmation",
            "approved": False,
            "pause_execution": True,
        }

    execution_context = {
        "rows": rows,
        "train_ratio": ratios["train_ratio"],
        "val_ratio": ratios["val_ratio"],
        "test_ratio": ratios["test_ratio"],
        "input_length": window_payload.get("input_length"),
        "output_length": window_payload.get("output_length"),
        "time_increment": window_payload.get("time_increment"),
    }
    try:
        executed = execute_user_code_safely(generated_code, execution_context)
        executed_code = generated_code
    except Exception:
        fallback_code = _fallback_split_code(rows, ratios, window_payload)
        executed = execute_user_code_safely(fallback_code, execution_context)
        executed_code = fallback_code

    outputs_payload = executed.get("outputs", {}) if isinstance(executed.get("outputs"), dict) else {}
    payload = {
        "strategy": outputs_payload.get("strategy", "time_order_split"),
        "rows": int(outputs_payload.get("rows", rows)),
        "ratios": outputs_payload.get("ratios", ratios),
        "window_config": outputs_payload.get("window_config", window_payload),
        "indices": outputs_payload.get("indices"),
    }
    if not payload["indices"]:
        raise RuntimeError("Generated split strategy code did not produce indices.")

    state.write("split_strategy_result", payload)
    state.write_runtime("split_indices", payload["indices"])
    state.update(
        {
            "awaiting_user_confirmation": None,
            "approved_generated_codes": {
                **approved_generated_codes,
                "split_strategy": executed_code,
            },
            "modified_proposals": {
                **modified_proposals,
                "split_strategy": "",
            },
        }
    )
    write_step_artifact(state, "split_strategy", payload)
    return {
        "message": "split strategy prepared",
        "indices": payload["indices"],
        "window_config": payload["window_config"],
    }
