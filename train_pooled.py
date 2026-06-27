"""Train a pooled AeroClim model across quality-approved NOAA stations."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from ml_model import HazardPredictor
from noaa_client import CITIES, NOAAClient


def progress(percent: int, message: str) -> None:
    print(f"[{percent:3d}%] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-stations",
        type=int,
        default=20,
        help="Maximum stations to sample. Use 0 with --all-ready to train every ready station.",
    )
    parser.add_argument(
        "--all-ready",
        action="store_true",
        help="Use every station marked training_ready in data/data_health.json.",
    )
    parser.add_argument("--start", default="1995-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--lstm-epochs", type=int, default=15)
    parser.add_argument("--rf-estimators", type=int, default=100)
    args = parser.parse_args()

    report = json.load(open("data/data_health.json", encoding="utf-8"))
    ready = [
        row["station_id"]
        for row in report["stations"]
        if row["station_id"].startswith("USW") and row.get("training_ready")
    ]
    if args.all_ready:
        args.max_stations = 0

    # Evenly sample the sorted catalog to preserve geographic diversity.
    if args.max_stations and len(ready) > args.max_stations:
        indexes = pd.Series(range(len(ready))).sample(
            args.max_stations, random_state=42
        ).sort_values()
        ready = [ready[index] for index in indexes]

    client = NOAAClient()
    predictor = HazardPredictor()
    frames = []
    for number, station_id in enumerate(ready, start=1):
        print(f"Loading {number}/{len(ready)}: {station_id}", flush=True)
        frame = client.fetch_weather_data(station_id, args.start, args.end)
        frame = predictor._add_sst_feature(frame)
        frames.append(frame)

    pooled = pd.concat(frames, ignore_index=True).sort_values(
        ["Date", "City_ID"]
    )
    print(f"Training on {len(pooled):,} rows from {len(ready)} stations")
    predictor.train_multimodal(
        pooled,
        progress_callback=progress,
        lstm_epochs=args.lstm_epochs,
        rf_estimators=args.rf_estimators,
    )
    print(json.dumps(predictor.metrics, indent=2))


if __name__ == "__main__":
    main()
