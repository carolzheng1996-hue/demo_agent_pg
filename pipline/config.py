from __future__ import annotations

from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
STATE_DIR = BASE_DIR / ".state"
STATE_FILE = STATE_DIR / "global_state.json"
TASKS_FILE = STATE_DIR / "tasks.json"
PLAN_FILE = STATE_DIR / "latest_plan.json"
DEFAULT_MAX_ITERATIONS = 3
MAX_ITERATIONS_CAP = 10

DEFAULT_ALLOWED_MODELS: List[str] = ["arima", "xgboost", "linear"]
DEFAULT_TRAINING_PARAMS: Dict[str, Dict] = {
    "arima": {"order": [2, 1, 2]},
    "xgboost": {"window": 48, "n_estimators": 200, "max_depth": 6, "learning_rate": 0.05},
    "linear": {"window": 96},
}


def ensure_directories() -> None:
    for path in (OUTPUT_DIR, STATE_DIR):
        path.mkdir(parents=True, exist_ok=True)
