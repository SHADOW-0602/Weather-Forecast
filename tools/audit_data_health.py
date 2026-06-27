"""Audit AeroClim's prepared datasets and model-ready station coverage."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from noaa_client import CITIES, NOAAClient


CORE_COLUMNS = [
    "TMAX",
    "TMIN",
    "PRCP",
    "DEWPOINT_F",
    "WINDSPEED_MPH",
    "PRESSURE_HPA",
    "HUMIDITY_PCT",
    "SOIL_MOISTURE_VOL",
    "SATURATION_PCT",
]


def file_summary(path: Path) -> dict:
    if not path.exists():
        return {"present": False}
    frame = pd.read_csv(path)
    date_column = "date" if "date" in frame.columns else None
    dates = pd.to_datetime(frame[date_column], errors="coerce") if date_column else None
    return {
        "present": True,
        "rows": len(frame),
        "start": str(dates.min().date()) if dates is not None else None,
        "end": str(dates.max().date()) if dates is not None else None,
    }


def main() -> None:
    client = NOAAClient()
    station_results = []
    for station_id, metadata in CITIES.items():
        try:
            frame = client.fetch_weather_data(
                station_id, "1995-01-01", "2024-12-31"
            )
            station_results.append(
                {
                    "station_id": station_id,
                    "name": metadata["name"],
                    "rows": len(frame),
                    "start": str(frame["Date"].min().date()),
                    "end": str(frame["Date"].max().date()),
                    "missing": {
                        column: (
                            int(frame[column].isna().sum())
                            if column in frame.columns
                            else len(frame)
                        )
                        for column in CORE_COLUMNS
                    },
                }
            )
            station_results[-1]["core_completeness_pct"] = round(
                100.0
                * (
                    1.0
                    - sum(station_results[-1]["missing"].values())
                    / (len(frame) * len(CORE_COLUMNS))
                ),
                3,
            )
            station_results[-1]["training_ready"] = (
                station_results[-1]["core_completeness_pct"] >= 99.0
                and len(frame) >= 3650
            )
        except Exception as exc:
            station_results.append(
                {"station_id": station_id, "name": metadata["name"], "error": str(exc)}
            )

    unique_noaa = [
        row for row in station_results if row["station_id"].startswith("USW")
    ]
    summary = {
        "configured_entries": len(CITIES),
        "unique_noaa_stations": len(unique_noaa),
        "station_errors": sum("error" in row for row in unique_noaa),
        "stations_with_complete_core_data": sum(
            "error" not in row and sum(row["missing"].values()) == 0
            for row in unique_noaa
        ),
        "stations_training_ready": sum(
            row.get("training_ready", False) for row in unique_noaa
        ),
        "auxiliary_files": {
            "climate_indices": file_summary(Path("data/climate_indices.csv")),
            "ocean": file_summary(Path("data/ocean.csv")),
        },
        "stations": station_results,
    }
    output = Path("data/data_health.json")
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "stations"}, indent=2))
    print(f"Wrote detailed audit to {output}")


if __name__ == "__main__":
    main()
