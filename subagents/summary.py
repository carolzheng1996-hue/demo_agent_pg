from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from config import OUTPUT_DIR
from global_state import GlobalState
from llm_client import LLMClient


def _fallback_summary(
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

    if integration:
        ranking = integration.get("ranking", {}).get("ranking", [])
        if ranking:
            best = ranking[0]
            lines.append(
                f"Best model: {best.get('name')} (MAE={best.get('mae'):.4f}, RMSE={best.get('rmse'):.4f}, MAPE={best.get('mape'):.4f})"
            )
        ens = integration.get("ensemble", {})
        metrics = ens.get("metrics", {})
        if metrics:
            lines.append(
                f"Ensemble(top-2): MAE={metrics.get('mae'):.4f}, RMSE={metrics.get('rmse'):.4f}, MAPE={metrics.get('mape'):.4f}"
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
    fallback_text = _fallback_summary(query, dataset, dataset_path, plan, analysis, integration)

    llm = LLMClient(state.read("api_config", {}))
    llm_summary = llm.complete_text(
        system_prompt=(
            "You are a senior time-series engineer. "
            "Write a concise report in Chinese with 4 sections: 任务, 数据, 方法, 结论. "
            "Use plain text only."
        ),
        user_prompt=json.dumps(
            {
                "query": query,
                "dataset": dataset,
                "dataset_path": dataset_path,
                "plan": plan,
                "analysis": analysis,
                "integration": integration,
                "fallback_summary": fallback_text,
            },
            ensure_ascii=False,
        ),
        max_tokens=520,
        temperature=0.2,
    )
    summary_text = llm_summary or fallback_text

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
