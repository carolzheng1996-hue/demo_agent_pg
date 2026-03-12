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


def set_target(df: pd.DataFrame, preferred: str = "OT") -> str:
    if preferred in df.columns:
        return preferred
    preferred_lower = str(preferred or "").lower()
    for column in df.columns:
        if str(column).lower() == preferred_lower and preferred_lower:
            return str(column)
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric:
        raise ValueError("No numeric columns found for target selection.")
    return numeric[0]


def set_features(df: pd.DataFrame, target: str) -> List[str]:
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in numeric if c != target]


def set_features_multi(df: pd.DataFrame, targets: Sequence[str]) -> List[str]:
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    target_set = {str(item) for item in targets}
    return [column for column in numeric if column not in target_set]
