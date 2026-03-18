from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

try:
    from ..state import DGGlobalState
    from ..tools import compute_full_analysis, write_step_artifact
except ImportError:
    from state import DGGlobalState
    from tools import compute_full_analysis, write_step_artifact


def _build_findings(analysis: Dict) -> List[str]:
    statistics = analysis.get("statistics", {})
    stationarity = analysis.get("stationarity", {})
    seasonality = analysis.get("seasonality", {})
    findings = [
        f"目标列均值为 {statistics.get('mean')}，标准差为 {statistics.get('std')}。",
        f"缺失率为 {statistics.get('missing_ratio')}，极值范围为 [{statistics.get('min')}, {statistics.get('max')}]。",
        "序列通过平稳性检验。" if stationarity.get("is_stationary") else "序列未通过平稳性检验，建模时需要依赖差分或标准化。",
    ]
    period = seasonality.get("period")
    if period not in (None, "", 0):
        findings.append(f"检测到候选季节周期约为 {period}。")
    return findings


def _build_plan(analysis: Dict) -> List[str]:
    statistics = analysis.get("statistics", {})
    stationarity = analysis.get("stationarity", {})
    seasonality = analysis.get("seasonality", {})
    plan = [
        "检查目标列缺失值并确认填补结果",
        "核对目标列统计分布与异常波动范围",
    ]
    if not stationarity.get("is_stationary"):
        plan.append("在特征工程和预处理阶段增强差分或滚动统计特征")
    if seasonality.get("period") not in (None, "", 0):
        plan.append("保留季节性相关窗口与滞后特征")
    if float(statistics.get("missing_ratio", 0.0) or 0.0) > 0:
        plan.append("复核前向/后向填补是否影响关键波动区间")
    return plan


def _build_input_statistics_text(df, analysis: Dict) -> str:
    dataset_summary = analysis.get("dataset_summary", {})
    shape = dataset_summary.get("shape", ["?", "?"])
    numeric_columns = dataset_summary.get("numeric_columns", [])
    missing_by_column = dataset_summary.get("missing_by_column", {})
    statistics = analysis.get("statistics", {})
    total_missing = sum(int(value or 0) for value in missing_by_column.values())
    top_missing = sorted(missing_by_column.items(), key=lambda item: item[1], reverse=True)[:5]
    lines = [
        f"数据维度: {shape[0]} x {shape[1]}",
        f"数值列数量: {len(numeric_columns)}",
        f"总缺失值数量: {total_missing}",
        f"目标列均值: {statistics.get('mean')}",
        f"目标列标准差: {statistics.get('std')}",
    ]
    if top_missing:
        lines.append("各列缺失值(top5):")
        lines.extend([f"- {column}: {count}" for column, count in top_missing])
    if numeric_columns:
        numeric_means = df[numeric_columns].apply(lambda series: float(pd.to_numeric(series, errors="coerce").mean()), axis=0).to_dict()
        lines.append("输入数值列均值(top8):")
        lines.extend([f"- {column}: {round(float(numeric_means.get(column, 0.0)), 6)}" for column in numeric_columns[:8]])
        lines.append("参与建模的输入列:")
        lines.extend([f"- {column}" for column in numeric_columns[:8]])
    return "\n".join(lines)


def run(state: DGGlobalState) -> Dict:
    profile = state.read("dataset_profile", {})
    station_paths = profile.get("formatted_dataset_paths") or {}
    if station_paths:
        frames = []
        for station_id, formatted_path in station_paths.items():
            frame = pd.read_parquet(Path(str(formatted_path)))
            frame["__station_id__"] = str(station_id)
            frames.append(frame)
        df = pd.concat(frames, axis=0, ignore_index=True)
    else:
        formatted_path = profile.get("formatted_dataset_path")
        if not formatted_path:
            raise RuntimeError("Missing formatted dataset path. Run data_formatter first.")
        df = pd.read_parquet(Path(str(formatted_path)))

    target_col = profile.get("target_column")
    analysis = compute_full_analysis(df, target_col)
    analysis_payload = {
        "base_analysis": analysis,
        "analysis_plan": _build_plan(analysis),
        "findings": _build_findings(analysis),
        "input_statistics_text": _build_input_statistics_text(df, analysis),
        "analysis_source": "deterministic_tools",
    }
    state.write("data_analysis_result", analysis_payload)
    write_step_artifact(state, "data_analysis", analysis_payload)
    return {
        "message": "data analysis completed",
        "target_column": target_col,
        "finding_count": len(analysis_payload["findings"]),
    }
