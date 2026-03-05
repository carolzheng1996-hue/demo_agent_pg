from __future__ import annotations

from pathlib import Path
from typing import Dict

from config import DATA_DIR, get_default_dataset_path
from global_state import GlobalState
from tools.file_tools import detect_date, read_csv, set_features, set_target


def run(state: GlobalState) -> Dict:
    dataset_name = str(state.read("dataset_name", "etth")).lower()
    dataset_path = state.read("dataset_path")

    if dataset_path:
        csv_path = Path(dataset_path)
    else:
        default_path = get_default_dataset_path(dataset_name)
        if default_path is not None:
            csv_path = default_path
        else:
            candidates = sorted(DATA_DIR.glob("*.csv"))
            if not candidates:
                raise FileNotFoundError("No CSV file found under data/. Please provide --dataset-path.")
            csv_path = candidates[0]

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = read_csv(str(csv_path))
    date_col = detect_date(df)
    target_col = set_target(df)
    feature_cols = set_features(df, target_col)

    split_idx = int(len(df) * 0.8)
    state.write_runtime("df", df)
    state.update(
        {
            "dataset_path": str(csv_path),
            "date_column": date_col,
            "target_column": target_col,
            "feature_columns": feature_cols,
            "train_size": split_idx,
            "test_size": len(df) - split_idx,
            "data_head": df.head(5).to_dict(orient="records"),
            "data_shape": [int(df.shape[0]), int(df.shape[1])],
        }
    )

    return {
        "message": f"loaded {csv_path.name}, rows={len(df)}, target={target_col}",
        "dataset_path": str(csv_path),
    }
