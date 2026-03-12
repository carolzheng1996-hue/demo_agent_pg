from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional

try:
    from ..llm_utils import invoke_text
except ImportError:
    from llm_utils import invoke_text


def infer_target_column_from_query(query: str, columns: Iterable[str]) -> Optional[str]:
    query_text = str(query or "").lower()
    if not query_text:
        return None

    ranked_columns = sorted([str(column) for column in columns], key=len, reverse=True)
    normalized_query = re.sub(r"[^a-z0-9_]+", " ", query_text)

    for column in ranked_columns:
        column_lower = column.lower()
        if re.search(rf"(?<![a-z0-9_]){re.escape(column_lower)}(?![a-z0-9_])", normalized_query):
            return column
        if column_lower in query_text:
            return column
    return None


def parse_target_columns(raw: Any, columns: Iterable[str]) -> List[str]:
    if raw is None:
        return []
    available = {str(column).lower(): str(column) for column in columns}
    if isinstance(raw, str):
        parts = [item.strip() for item in raw.split(",")]
    elif isinstance(raw, (list, tuple, set)):
        parts = [str(item).strip() for item in raw]
    else:
        parts = [str(raw).strip()]

    selected: List[str] = []
    for part in parts:
        if not part:
            continue
        normalized = available.get(part.lower())
        if normalized and normalized not in selected:
            selected.append(normalized)
    return selected


def _fallback_data_formatter_code(context: Dict[str, Any]) -> Optional[str]:
    target_columns = [str(item) for item in (context.get("target_columns") or []) if str(item)]
    if not target_columns:
        target_column = str(context.get("target_column") or "").strip()
        if target_column:
            target_columns = [target_column]
    if not target_columns:
        return None

    feature_columns = [str(item) for item in (context.get("feature_columns") or []) if str(item)]
    date_column = str(context.get("date_column") or "").strip()
    need_split = bool(context.get("need_split", False))

    lines = [
        "outputs = {}",
        "df = raw_df.copy()",
    ]
    if date_column:
        lines.append(f"df[{date_column!r}] = pd.to_datetime(df[{date_column!r}], errors='coerce')")
    lines.extend(
        [
            f"feature_cols = {json.dumps(feature_columns, ensure_ascii=False)}",
            f"target_cols = {json.dumps(target_columns, ensure_ascii=False)}",
            "primary_target_col = target_cols[0]",
            "working_df = df.copy()",
            "numeric_candidates = feature_cols + target_cols",
            "for col in numeric_candidates:",
            "    if col in working_df.columns:",
            "        working_df[col] = pd.to_numeric(working_df[col], errors='coerce')",
            "working_df = working_df.ffill().bfill()",
            "standardized_df = working_df[feature_cols + target_cols].copy() if feature_cols else working_df[target_cols].copy()",
            "standardized_df = standardized_df.fillna(0.0)",
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
    return "\n".join(lines).strip() + "\n"


def _fallback_split_strategy_code(context: Dict[str, Any]) -> Optional[str]:
    rows = context.get("rows")
    ratios = context.get("ratios") or {}
    window_config = context.get("window_config") or {}
    if rows in (None, ""):
        return None
    if not all(key in ratios for key in ["train_ratio", "val_ratio", "test_ratio"]):
        return None
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
        f"outputs['window_config'] = {json.dumps(window_config, ensure_ascii=False)}\n"
        "outputs['indices'] = {\n"
        "    'train': [0, train_end],\n"
        "    'val': [train_end, val_end],\n"
        "    'test': [val_end, rows],\n"
        "}\n"
    )


def _fallback_regenerated_code(subagent: str, context: Dict[str, Any]) -> Optional[str]:
    if subagent == "data_formatter":
        return _fallback_data_formatter_code(context)
    if subagent == "split_strategy":
        return _fallback_split_strategy_code(context)
    return None


def regenerate_code_with_feedback(
    subagent: str,
    original_code: str,
    feedback: str,
    context: Dict[str, Any],
) -> Optional[str]:
    critique = invoke_text(
        system_prompt=(
            "You are a senior Python data engineer reviewing generated code. "
            "Provide a concise technical critique based on the user feedback and task context."
        ),
        user_prompt=json.dumps(
            {
                "subagent": subagent,
                "feedback": feedback,
                "context": context,
                "original_code": original_code,
            },
            ensure_ascii=False,
        ),
        max_tokens=220,
        temperature=0.0,
    )
    rewrite = invoke_text(
        system_prompt=(
            "You rewrite generated Python code for a data agent. "
            "Return Python code only. Do not use markdown fences. "
            "Keep the original contract and address the user feedback."
        ),
        user_prompt=json.dumps(
            {
                "subagent": subagent,
                "feedback": feedback,
                "critique": critique or "",
                "context": context,
                "original_code": original_code,
            },
            ensure_ascii=False,
        ),
        max_tokens=900,
        temperature=0.0,
    )
    if not rewrite:
        return _fallback_regenerated_code(subagent, context)
    text = rewrite.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text + "\n"
