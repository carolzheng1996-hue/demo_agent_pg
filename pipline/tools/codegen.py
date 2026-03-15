from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional


def parse_column_selection(raw: Any, columns: Iterable[str]) -> List[str]:
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
    return parse_column_selection(raw, columns)


def regenerate_code_with_feedback(
    subagent: str,
    original_code: str,
    feedback: str,
    context: Dict[str, Any],
) -> Optional[str]:
    # Deterministic pipeline no longer regenerates code through LLM.
    # Keep the function so sandbox/web code paths remain import-compatible.
    _ = (subagent, original_code, feedback, context)
    return None
