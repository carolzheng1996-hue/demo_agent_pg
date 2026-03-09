from __future__ import annotations

import json
from typing import Dict

from ..llm_utils import invoke_json
from ..state import DGGlobalState
from ..tools import compute_full_analysis


def run(state: DGGlobalState) -> Dict:
    df = state.read_runtime("raw_df")
    if df is None:
        raise RuntimeError("Missing raw dataframe in runtime state. Run approved data_reading first.")

    target_col = state.read("dataset_profile", {}).get("target_column")
    analysis = compute_full_analysis(df, target_col)
    llm_payload = invoke_json(
        system_prompt=(
            "You are a data analysis planner. Return JSON with keys plan (array of analysis steps), "
            "generated_code (string), findings (array). The code should be illustrative pandas/numpy code only."
        ),
        user_prompt=json.dumps(
            {
                "task_description": state.read("user_query", ""),
                "dataset_profile": state.read("dataset_profile", {}),
                "analysis_summary": analysis,
            },
            ensure_ascii=False,
        ),
        max_tokens=500,
        temperature=0.1,
    ) or {}

    analysis_payload = {
        "base_analysis": analysis,
        "llm_plan": llm_payload.get("plan", []),
        "llm_generated_code": llm_payload.get("generated_code", ""),
        "llm_findings": llm_payload.get("findings", []),
    }
    state.write("data_analysis_result", analysis_payload)
    return {
        "message": "data analysis completed",
        "target_column": target_col,
        "llm_plan_steps": len(analysis_payload["llm_plan"]),
    }
