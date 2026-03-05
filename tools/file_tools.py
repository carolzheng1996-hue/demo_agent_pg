from __future__ import annotations

from pathlib import Path
from typing import List, Optional

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
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric:
        raise ValueError("No numeric columns found for target selection.")
    return numeric[0]


def set_features(df: pd.DataFrame, target: str) -> List[str]:
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    return [c for c in numeric if c != target]
