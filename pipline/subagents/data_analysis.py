from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

try:
    from ..state import DGGlobalState
    from ..tools import compute_full_analysis, write_step_artifact
    from .ds_pipeline_utils import (
        count_missing_values,
        is_sequence_column,
        load_ds_station_frames,
        sequence_length,
        sequence_to_scalar,
    )
except ImportError:
    from state import DGGlobalState
    from tools import compute_full_analysis, write_step_artifact
    from subagents.ds_pipeline_utils import (
        count_missing_values,
        is_sequence_column,
        load_ds_station_frames,
        sequence_length,
        sequence_to_scalar,
    )


def _build_findings(analysis: Dict, quality: Dict[str, Any]) -> List[str]:
    statistics = analysis.get("statistics", {})
    stationarity = analysis.get("stationarity", {})
    seasonality = analysis.get("seasonality", {})
    missing_summary = quality.get("missing_summary", {})
    findings = [
        f"下一时刻目标均值为 {statistics.get('mean')}，标准差为 {statistics.get('std')}。",
        f"整体缺失率为 {missing_summary.get('missing_ratio')}，极值范围为 [{statistics.get('min')}, {statistics.get('max')}]。",
        "下一时刻目标通过平稳性检验。" if stationarity.get("is_stationary") else "下一时刻目标未通过平稳性检验，建模时需要依赖差分或标准化。",
    ]
    period = seasonality.get("dominant_period")
    if period not in (None, "", 0):
        findings.append(f"检测到候选周期约为 {period}。")
    return findings


def _build_plan(analysis: Dict, quality: Dict[str, Any]) -> List[str]:
    statistics = analysis.get("statistics", {})
    stationarity = analysis.get("stationarity", {})
    seasonality = analysis.get("seasonality", {})
    dimension_summary = quality.get("dimension_summary", {})
    missing_summary = quality.get("missing_summary", {})
    sequence_summary = quality.get("sequence_summary", {})
    plan = [
        f"确认 {dimension_summary.get('row_count', 0)} 行 DS 数据覆盖的时间范围与站点数符合预期",
        "核对目标下一时刻分布与异常波动范围",
    ]
    if any(int(item.get("zero_length_rows", 0)) > 0 for item in sequence_summary.get("sequence_columns", [])):
        plan.append("关注空序列样本，避免预处理展开后出现全零窗口")
    if not stationarity.get("is_stationary"):
        plan.append("在特征工程和预处理阶段增强差分或滚动统计特征")
    if seasonality.get("dominant_period") not in (None, "", 0):
        plan.append("保留季节性相关窗口与滞后特征")
    if float(missing_summary.get("missing_ratio", 0.0) or 0.0) > 0:
        plan.append("在预处理前补齐或填补缺失，避免缺失值直接进入建模输入")
    return plan


def _build_quality_summary(station_frames: Dict[str, pd.DataFrame], target_col: str) -> Dict[str, Any]:
    row_count = 0
    column_count = 0
    station_count = len(station_frames)
    timestamp_min = None
    timestamp_max = None
    columns: List[str] = []
    missing_by_column: Dict[str, int] = {}
    total_cells = 0
    sequence_details: Dict[str, Dict[str, Any]] = {}

    for station_df in station_frames.values():
        row_count += int(station_df.shape[0])
        column_count = max(column_count, int(station_df.shape[1]))
        columns = [str(column) for column in station_df.columns.tolist()] if not columns else columns
        if "timestamp_win" in station_df.columns:
            timestamp_series = pd.to_datetime(station_df["timestamp_win"], errors="coerce").dropna()
            if not timestamp_series.empty:
                station_min = timestamp_series.min()
                station_max = timestamp_series.max()
                timestamp_min = station_min if timestamp_min is None else min(timestamp_min, station_min)
                timestamp_max = station_max if timestamp_max is None else max(timestamp_max, station_max)
        for column in station_df.columns:
            series = station_df[column]
            missing_count = int(series.apply(count_missing_values).sum())
            missing_by_column[str(column)] = missing_by_column.get(str(column), 0) + missing_count
            if is_sequence_column(series):
                total_cells += int(series.apply(sequence_length).sum())
                lengths = series.apply(sequence_length)
                detail = sequence_details.setdefault(
                    str(column),
                    {
                        "column": str(column),
                        "min_length": None,
                        "max_length": 0,
                        "length_sum": 0.0,
                        "length_count": 0,
                        "zero_length_rows": 0,
                        "missing_values": 0,
                    },
                )
                current_min = int(lengths.min()) if len(lengths) else 0
                detail["min_length"] = current_min if detail["min_length"] is None else min(int(detail["min_length"]), current_min)
                detail["max_length"] = max(int(detail["max_length"]), int(lengths.max()) if len(lengths) else 0)
                detail["length_sum"] += float(lengths.sum())
                detail["length_count"] += int(len(lengths))
                detail["zero_length_rows"] += int((lengths == 0).sum())
                detail["missing_values"] += missing_count
            else:
                total_cells += int(len(series))

    total_missing = int(sum(missing_by_column.values()))
    sequence_columns = []
    for detail in sequence_details.values():
        count = int(detail.pop("length_count", 0))
        length_sum = float(detail.pop("length_sum", 0.0))
        detail["min_length"] = int(detail["min_length"] or 0)
        detail["mean_length"] = float(length_sum / count) if count else 0.0
        sequence_columns.append(detail)
    sequence_columns.sort(key=lambda item: item["column"])
    target_sequence = next(
        (item for item in sequence_columns if item.get("column") == target_col),
        {},
    )
    return {
        "dimension_summary": {
            "row_count": int(row_count),
            "column_count": int(column_count),
            "station_count": int(station_count),
            "columns": columns,
            "timestamp_min": str(timestamp_min) if timestamp_min is not None else None,
            "timestamp_max": str(timestamp_max) if timestamp_max is not None else None,
        },
        "missing_summary": {
            "total_missing": total_missing,
            "total_cells": int(total_cells),
            "missing_ratio": float(total_missing / total_cells) if total_cells else 0.0,
            "missing_by_column": missing_by_column,
        },
        "sequence_summary": {
            "sequence_column_count": len(sequence_columns),
            "sequence_columns": sequence_columns,
        },
        "target_sequence_summary": target_sequence,
    }


def _build_input_statistics_text(quality: Dict[str, Any], target_col: str) -> str:
    dimension_summary = quality.get("dimension_summary", {})
    missing_summary = quality.get("missing_summary", {})
    target_sequence = quality.get("target_sequence_summary", {})
    top_missing = sorted(
        (missing_summary.get("missing_by_column") or {}).items(),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    lines = [
        f"DS 行数: {dimension_summary.get('row_count', 0)}",
        f"DS 列数: {dimension_summary.get('column_count', 0)}",
        f"站点数量: {dimension_summary.get('station_count', 0)}",
        f"目标序列列: {target_col}",
        f"目标序列长度范围: {target_sequence.get('min_length', 0)} - {target_sequence.get('max_length', 0)}",
        f"总缺失值数量: {missing_summary.get('total_missing', 0)}",
        f"整体缺失率: {missing_summary.get('missing_ratio', 0.0)}",
    ]
    if top_missing:
        lines.append("各列缺失值(top5):")
        lines.extend([f"- {column}: {count}" for column, count in top_missing])
    return "\n".join(lines)


def run(state: DGGlobalState) -> Dict:
    profile = state.read("dataset_profile", {})
    target_col = profile.get("target_column")
    if not target_col:
        raise RuntimeError("Missing target sequence column in dataset profile. Run data_reading first.")
    station_frames = load_ds_station_frames(state, columns=["station", "timestamp_win", target_col] + list(profile.get("feature_columns", []) or []))
    if not station_frames:
        raise RuntimeError("Missing DS station frames in state. Run data_reading first.")

    analysis_rows: List[pd.DataFrame] = []
    for station_df in station_frames.values():
        if target_col not in station_df.columns:
            continue
        analysis_rows.append(pd.DataFrame({target_col: station_df[target_col].apply(sequence_to_scalar)}))
    if not analysis_rows:
        raise RuntimeError("Target sequence column not found in DS station frames.")
    analysis_df = pd.concat(analysis_rows, axis=0, ignore_index=True)
    analysis = compute_full_analysis(analysis_df, target_col)
    quality = _build_quality_summary(station_frames, target_col)
    analysis_payload = {
        "base_analysis": analysis,
        "quality_checks": quality,
        "analysis_plan": _build_plan(analysis, quality),
        "findings": _build_findings(analysis, quality),
        "input_statistics_text": _build_input_statistics_text(quality, target_col),
        "analysis_source": "ds_direct_analysis",
    }
    state.write("data_analysis_result", analysis_payload)
    write_step_artifact(state, "data_analysis", analysis_payload)
    return {
        "message": "data analysis completed",
        "target_column": target_col,
        "finding_count": len(analysis_payload["findings"]),
    }
