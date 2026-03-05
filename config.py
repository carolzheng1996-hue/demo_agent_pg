from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
TASKS_DIR = BASE_DIR / ".tasks"
STATE_FILE = TASKS_DIR / "global_state.json"
TASKS_FILE = TASKS_DIR / "tasks.json"

def build_api_config(overrides: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, Optional[str]]:
    base_url = (
        os.getenv("AGENT_API_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_BASE")
        or os.getenv("API_BASE")
    )
    api_key = (
        os.getenv("AGENT_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("API_KEY")
    )
    cfg: Dict[str, Optional[str]] = {
        "provider": os.getenv("AGENT_API_PROVIDER", "openai"),
        "base_url": base_url,
        "api_key": api_key,
        "model": os.getenv("AGENT_MODEL", "gpt-4o-mini"),
    }
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                cfg[k] = v
    return cfg


API_CONFIG: Dict[str, Optional[str]] = build_api_config()

MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "arima": {"order": (2, 1, 2)},
    "xgboost": {"window": 48, "n_estimators": 200, "max_depth": 6, "learning_rate": 0.05},
    "lstm": {"seq_len": 96, "hidden_size": 64, "num_layers": 2, "dropout": 0.1, "epochs": 8, "lr": 1e-3, "batch_size": 64},
}


def ensure_directories() -> None:
    for p in (DATA_DIR, OUTPUT_DIR, TASKS_DIR):
        p.mkdir(parents=True, exist_ok=True)


def get_default_dataset_path(dataset_name: str) -> Optional[Path]:
    name = dataset_name.lower()
    candidates = {
        "etth": [DATA_DIR / "ETTh1.csv", DATA_DIR / "ETTh2.csv", BASE_DIR / "ETTh1.csv", BASE_DIR / "ETTh2.csv"],
        "etth1": [DATA_DIR / "ETTh1.csv", BASE_DIR / "ETTh1.csv"],
        "etth2": [DATA_DIR / "ETTh2.csv", BASE_DIR / "ETTh2.csv"],
    }
    for p in candidates.get(name, []):
        if p.exists():
            return p
    return None
