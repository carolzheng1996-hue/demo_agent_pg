from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_station_frame(station_id: str, periods: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamp = pd.date_range("2024-01-01 00:00:00", periods=periods, freq="15min")
    phase = np.linspace(0, 8 * np.pi, periods)

    observe_power = 100 + 25 * np.sin(phase) + rng.normal(0, 2, periods)
    ghi_real = np.clip(500 + 180 * np.sin(phase - 0.3) + rng.normal(0, 10, periods), 0, None)
    ghi_solargis = np.clip(ghi_real * 0.98 + rng.normal(0, 6, periods), 0, None)
    temp_solargis = 20 + 6 * np.sin(phase / 2) + rng.normal(0, 0.8, periods)
    ws_solargis = np.clip(3.5 + 1.2 * np.cos(phase / 3) + rng.normal(0, 0.2, periods), 0, None)
    wd_solargis = np.mod(180 + 30 * np.sin(phase / 4) + rng.normal(0, 5, periods), 360)

    frame = pd.DataFrame(
        {
            "__index_level_0__": timestamp,
            "observe_power": observe_power.round(4),
            "GHI_real": ghi_real.round(4),
            "GHI_SOLARGIS": ghi_solargis.round(4),
            "TEMP_SOLARGIS": temp_solargis.round(4),
            "WS_SOLARGIS": ws_solargis.round(4),
            "WD_SOLARGIS": wd_solargis.round(4),
            "GHI_SOLARGIS_predict": np.roll(ghi_solargis, -1).round(4),
            "TEMP_SOLARGIS_predict": np.roll(temp_solargis, -1).round(4),
            "WS_SOLARGIS_predict": np.roll(ws_solargis, -1).round(4),
            "WD_SOLARGIS_predict": np.roll(wd_solargis, -1).round(4),
        }
    )

    for idx in range(1, 4):
        frame[f"ssrd_pos_{idx}"] = np.clip(ghi_real * (0.8 + idx * 0.03) + rng.normal(0, 8, periods), 0, None).round(4)
        frame[f"t2m_pos_{idx}"] = (temp_solargis + idx * 0.4 + rng.normal(0, 0.4, periods)).round(4)
        frame[f"ssrd_pos_{idx}_predict"] = np.roll(frame[f"ssrd_pos_{idx}"].to_numpy(), -1).round(4)
        frame[f"t2m_pos_{idx}_predict"] = np.roll(frame[f"t2m_pos_{idx}"].to_numpy(), -1).round(4)

    frame.iloc[-1, frame.columns.get_loc("GHI_SOLARGIS_predict")] = frame.iloc[-2]["GHI_SOLARGIS_predict"]
    frame.iloc[-1, frame.columns.get_loc("TEMP_SOLARGIS_predict")] = frame.iloc[-2]["TEMP_SOLARGIS_predict"]
    frame.iloc[-1, frame.columns.get_loc("WS_SOLARGIS_predict")] = frame.iloc[-2]["WS_SOLARGIS_predict"]
    frame.iloc[-1, frame.columns.get_loc("WD_SOLARGIS_predict")] = frame.iloc[-2]["WD_SOLARGIS_predict"]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample station parquet partitions for pipeline testing")
    parser.add_argument("--output-dir", required=True, help="Output root directory")
    parser.add_argument("--stations", default="1,2,3", help="Comma-separated station ids")
    parser.add_argument("--periods", type=int, default=384, help="Rows per station")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    station_ids = [item.strip() for item in str(args.stations).split(",") if item.strip()]

    for offset, station_id in enumerate(station_ids, start=1):
        station_dir = output_dir / f"station={station_id}"
        station_dir.mkdir(parents=True, exist_ok=True)
        frame = build_station_frame(station_id=station_id, periods=args.periods, seed=2024 + offset)
        frame.to_parquet(station_dir / "part-00000.parquet", index=False)

    print(f"Generated {len(station_ids)} station partitions under {output_dir}")


if __name__ == "__main__":
    main()
