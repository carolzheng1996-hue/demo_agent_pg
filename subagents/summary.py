from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from config import OUTPUT_DIR
from global_state import GlobalState
from llm_client import LLMClient


def _fmt_metric(value: Any) -> str:
    try:
        if value is None:
            return "NA"
        return f"{float(value):.4f}"
    except Exception:
        return "NA"


def _build_structured_summary(
    query: str,
    dataset: str,
    dataset_path: str,
    plan: List[str],
    analysis: Dict,
    integration: Dict,
) -> str:
    lines: List[str] = []
    lines.append(f"Task: {query}")
    lines.append(f"Dataset: {dataset} ({dataset_path})")
    lines.append(f"Plan: {' -> '.join(plan)}")

    if analysis:
        st = analysis.get("statistics", {})
        lines.append(
            "Stats: count={count}, mean={mean:.4f}, std={std:.4f}, min={min:.4f}, max={max:.4f}".format(
                count=st.get("count", 0),
                mean=float(st.get("mean", 0.0)),
                std=float(st.get("std", 0.0)),
                min=float(st.get("min", 0.0)),
                max=float(st.get("max", 0.0)),
            )
        )

    has_model_flow = "model_training" in plan or "result_integration" in plan
    if has_model_flow and integration:
        lines.append("Model Results:")
        all_models = integration.get("all_models") or integration.get("ranking", {}).get("ranking", [])
        if all_models:
            for idx, row in enumerate(all_models, start=1):
                lines.append(
                    f"{idx}. {row.get('name')} | backend={row.get('backend')} | "
                    f"MAE={_fmt_metric(row.get('mae'))}, RMSE={_fmt_metric(row.get('rmse'))}, MAPE={_fmt_metric(row.get('mape'))}"
                )

        ens = integration.get("ensemble", {})
        ens_sel = integration.get("ensemble_selection", {})
        members = ens_sel.get("member_models") or ens.get("member_models", [])
        top_k = ens_sel.get("top_k_used") or len(members)
        strategy = ens_sel.get("strategy") or ens.get("strategy") or "simple_average"
        rule = ens_sel.get("selection_rule") or ens.get("selection_rule") or "top_k_by_lowest_mae"
        lines.append(
            f"Ensemble Selection: top_k={top_k}, strategy={strategy}, rule={rule}, members={', '.join(members) if members else 'NA'}"
        )

        member_details = ens.get("member_details", [])
        if member_details:
            lines.append("Ensemble Member Details:")
            for idx, row in enumerate(member_details, start=1):
                mt = row.get("metrics", {})
                lines.append(
                    f"{idx}. {row.get('name')} | weight={_fmt_metric(row.get('weight'))} | backend={row.get('backend')} | "
                    f"MAE={_fmt_metric(mt.get('mae'))}, RMSE={_fmt_metric(mt.get('rmse'))}, MAPE={_fmt_metric(mt.get('mape'))}"
                )

        metrics = ens.get("metrics", {})
        if metrics:
            lines.append(
                f"Ensemble Metrics: MAE={_fmt_metric(metrics.get('mae'))}, "
                f"RMSE={_fmt_metric(metrics.get('rmse'))}, MAPE={_fmt_metric(metrics.get('mape'))}"
            )
        delta = ens.get("delta_vs_best_single", {})
        if delta:
            lines.append(
                f"Delta vs Best Single: MAE={_fmt_metric(delta.get('mae'))}, "
                f"RMSE={_fmt_metric(delta.get('rmse'))}, MAPE={_fmt_metric(delta.get('mape'))}"
            )
    return "\n".join(lines)


def run(state: GlobalState) -> Dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    query = state.read("user_query", "")
    dataset = state.read("dataset_name", "")
    dataset_path = state.read("dataset_path", "")
    plan = state.read("plan", [])
    analysis = state.read("analysis", {})
    integration = state.read("integration", {})
    structured_text = _build_structured_summary(query, dataset, dataset_path, plan, analysis, integration)

    llm = LLMClient(state.read("api_config", {}))
    llm_summary = llm.complete_text(
        system_prompt=(
            "You are a senior time-series engineer. "
            "Write a concise Chinese executive summary in <= 6 lines. "
            "Do not repeat all metrics; only highlight key conclusion."
        ),
        user_prompt=json.dumps(
            {
                "query": query,
                "dataset": dataset,
                "dataset_path": dataset_path,
                "plan": plan,
                "analysis": analysis,
                "integration": integration,
                "structured_summary": structured_text,
            },
            ensure_ascii=False,
        ),
        max_tokens=220,
        temperature=0.2,
    )
    summary_text = structured_text
    if llm_summary:
        summary_text = llm_summary.strip() + "\n\nDetailed Results:\n" + structured_text

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(OUTPUT_DIR) / f"report_{ts}.md"
    report_path.write_text("# Forecast Agent Report\n\n" + summary_text + "\n", encoding="utf-8")

    state.update(
        {
            "summary_text": summary_text,
            "report_path": str(report_path),
        }
    )

    return {
        "message": "summary generated",
        "report_path": str(report_path),
    }
