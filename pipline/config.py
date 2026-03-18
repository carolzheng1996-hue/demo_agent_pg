from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
STATE_DIR = BASE_DIR / ".state"
STATE_FILE = STATE_DIR / "global_state.json"
TASKS_FILE = STATE_DIR / "tasks.json"
PLAN_FILE = STATE_DIR / "latest_plan.json"
CONFIG_ALL_FILE = BASE_DIR / "config_all.json"
DEFAULT_MAX_ITERATIONS = 3
MAX_ITERATIONS_CAP = 10

DEFAULT_ALLOWED_MODELS: List[str] = ["arima", "xgboost", "linear"]
DEFAULT_TRAINING_PARAMS: Dict[str, Dict] = {
    "arima": {"order": [2, 1, 2]},
    "xgboost": {"window": 48, "n_estimators": 200, "max_depth": 6, "learning_rate": 0.05},
    "linear": {"window": 96},
}
PIPELINE_RUNTIME_DEFAULTS: Dict[str, Any] = {
    "dataset_name": "sample_dataset",
    "unit": "",
    "formatter_unit": "",
    "start_stage": "data_reading",
    "end_stage": "summary",
    "enable_split": True,
    "skip_split": False,
    "enable_feature_engineering": True,
    "target_col": "",
    "input_feature_cols": "",
    "split_method": "global_last_k",
    "split_cutoff_date": "",
    "split_test_units": "",
    "train_ratio": 0.7,
    "val_ratio": 0.3,
    "input_length": 96,
    "output_length": 24,
    "points_per_day": 96,
    "enable_normalization": True,
    "normalization_policy": "auto",
    "use_system_random": True,
    "max_iterations": DEFAULT_MAX_ITERATIONS,
}


def _coerce_boolish(value: Any, default: bool = False) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_config_all(config_path: Path | None = None) -> Dict[str, Any]:
    target = config_path or CONFIG_ALL_FILE
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config_all must be a JSON object: {target}")
    return payload


def runtime_defaults(config_path: Path | None = None) -> Dict[str, Any]:
    payload = dict(PIPELINE_RUNTIME_DEFAULTS)
    payload.update(load_config_all(config_path))
    return payload


def build_runtime_state(
    overrides: Dict[str, Any] | None = None,
    config_path: Path | None = None,
) -> Dict[str, Any]:
    payload = runtime_defaults(config_path)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                payload[key] = value

    for key in [
        "dataset_name",
        "unit",
        "formatter_unit",
        "start_stage",
        "end_stage",
        "target_col",
        "input_feature_cols",
        "split_method",
        "split_cutoff_date",
        "split_test_units",
        "normalization_policy",
    ]:
        payload[key] = str(payload.get(key, "") or "").strip()

    payload["enable_split"] = _coerce_boolish(
        payload.get("enable_split"),
        default=not _coerce_boolish(payload.get("skip_split"), default=False),
    )
    payload["skip_split"] = not payload["enable_split"]
    payload["enable_feature_engineering"] = _coerce_boolish(payload.get("enable_feature_engineering"), default=True)
    payload["enable_normalization"] = _coerce_boolish(payload.get("enable_normalization"), default=True)
    payload["use_system_random"] = _coerce_boolish(payload.get("use_system_random"), default=True)
    payload["points_per_day"] = int(payload.get("points_per_day") or PIPELINE_RUNTIME_DEFAULTS["points_per_day"])
    payload["max_iterations"] = max(1, min(int(payload.get("max_iterations") or DEFAULT_MAX_ITERATIONS), MAX_ITERATIONS_CAP))
    return payload


def ensure_directories() -> None:
    for path in (OUTPUT_DIR, STATE_DIR):
        path.mkdir(parents=True, exist_ok=True)
