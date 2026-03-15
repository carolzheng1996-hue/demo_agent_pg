from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def list_directory(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    return sorted([x.name for x in p.iterdir()])


def detect_date(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if c.lower() in {"date", "datetime", "timestamp", "time"}:
            return c
    return None


TARGET_NAME_HINTS = (
    "target",
    "label",
    "y",
    "value",
    "values",
    "output",
    "prediction",
    "pred",
    "sales",
    "demand",
    "load",
    "price",
    "temp",
    "temperature",
    "close",
)


def set_target(df: pd.DataFrame, preferred: str = "") -> str:
    if preferred and preferred in df.columns:
        return preferred
    preferred_lower = str(preferred or "").lower()
    for column in df.columns:
        if str(column).lower() == preferred_lower and preferred_lower:
            return str(column)
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric:
        raise ValueError("No numeric columns found for target selection.")

    for hint in TARGET_NAME_HINTS:
        for column in numeric:
            if hint in str(column).lower():
                return str(column)

    date_col = detect_date(df)
    trailing_numeric = [column for column in numeric if str(column) != str(date_col)]
    if trailing_numeric:
        return str(trailing_numeric[-1])
    return str(numeric[-1])


def set_features(df: pd.DataFrame, target: str) -> List[str]:
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in numeric if c != target]


def set_features_multi(df: pd.DataFrame, targets: Sequence[str]) -> List[str]:
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    target_set = {str(item) for item in targets}
    return [column for column in numeric if column not in target_set]
