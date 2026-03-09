from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    from ..config import OUTPUT_DIR
    from ..llm_utils import invoke_text
    from ..state import DGGlobalState
except ImportError:
    from config import OUTPUT_DIR
    from llm_utils import invoke_text
    from state import DGGlobalState


def _lines_for_models(rows: List[Dict]) -> List[str]:
    lines: List[str] = []
    for row in rows:
        metrics = row.get("metrics", {})
        lines.append(
            f"- {row.get('name')}: MSE={metrics.get('mse', 'NA')}, MAE={metrics.get('mae', 'NA')}, RMSE={metrics.get('rmse', 'NA')}, MAPE={metrics.get('mape', 'NA')}"
        )
    return lines


def run(state: DGGlobalState) -> Dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plan = state.read("plan", [])
    dataset_profile = state.read("dataset_profile", {})
    analysis_result = state.read("data_analysis_result", {})
    model_selection = state.read("model_selection_result", {})
    model_training = state.read("model_training_result", {})
    model_integration = state.read("model_integration_result", {})

    structured_lines = [
        f"# DGAgent Summary",
        "",
        f"- Task: {state.read('user_query', '')}",
        f"- Plan: {' -> '.join(plan)}",
        f"- Dataset Path: {dataset_profile.get('dataset_path', '')}",
        f"- Target Column: {dataset_profile.get('target_column', '')}",
        f"- Standardized Shape: {dataset_profile.get('standardized_shape', dataset_profile.get('shape', []))}",
        "",
        "## Analysis",
        json.dumps(analysis_result.get("base_analysis", {}), ensure_ascii=False, indent=2),
        "",
        "## Model Selection",
        json.dumps(model_selection, ensure_ascii=False, indent=2),
        "",
        "## Model Training Metrics",
    ]
    structured_lines.extend(_lines_for_models(model_training.get("results", [])))
    structured_lines.extend([
        "",
        "## Model Integration",
        json.dumps(model_integration, ensure_ascii=False, indent=2),
    ])
    structured_text = "\n".join(structured_lines)

    executive_summary = invoke_text(
        system_prompt="You are a senior data scientist. Write a concise Chinese summary of the agent workflow and conclusion in <= 8 lines.",
        user_prompt=json.dumps(
            {
                "task": state.read("user_query", ""),
                "plan": plan,
                "dataset_profile": dataset_profile,
                "analysis_result": analysis_result,
                "model_selection": model_selection,
                "model_training": model_training,
                "model_integration": model_integration,
            },
            ensure_ascii=False,
        ),
    )

    final_text = structured_text
    if executive_summary:
        final_text = executive_summary.strip() + "\n\n" + structured_text

    report_path = Path(OUTPUT_DIR) / f"dgagent_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path.write_text(final_text + "\n", encoding="utf-8")
    state.update({"summary_text": final_text, "report_path": str(report_path)})
    return {"message": "summary report generated", "report_path": str(report_path)}
