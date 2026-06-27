"""Create independent station-day labels from NOAA Storm Events.

This script downloads NOAA/NCEI Storm Events details files and maps selected
event types to station-days using event latitude/longitude and date windows.
The output is intentionally separate from the current engineered target.
"""

from __future__ import annotations

import argparse
import gzip
from io import BytesIO
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
import re
import subprocess

import pandas as pd


BASE_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
EVENT_TYPES = {
    "FLASH FLOOD",
    "FLOOD",
    "HEAVY RAIN",
    "HEAT",
    "EXCESSIVE HEAT",
    "HIGH WIND",
    "STRONG WIND",
    "THUNDERSTORM WIND",
    "TORNADO",
    "HAIL",
}


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 6371.0 * 2.0 * asin(sqrt(a))


def curl_text(url: str) -> str:
    result = subprocess.run(
        ["curl.exe", "--silent", "--show-error", "--fail", "--max-time", "60", url],
        check=True,
        capture_output=True,
        text=True,
        timeout=75,
    )
    return result.stdout


def curl_bytes(url: str) -> bytes:
    result = subprocess.run(
        ["curl.exe", "--silent", "--show-error", "--fail", "--max-time", "120", url],
        check=True,
        capture_output=True,
        timeout=150,
    )
    return result.stdout


def latest_file_for_year(listing: str, year: int) -> str:
    pattern = rf"StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz"
    matches = sorted(set(re.findall(pattern, listing)))
    if not matches:
        raise RuntimeError(f"No Storm Events details file found for {year}")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--radius-km", type=float, default=100.0)
    parser.add_argument("--output", type=Path, default=Path("data/event_labels.csv"))
    args = parser.parse_args()

    catalog = pd.read_csv("data/noaa_stations/stations.csv", dtype={"station_id": str})
    stations = catalog[["station_id", "lat", "lon"]].copy()
    listing = curl_text(BASE_URL)
    labels = []
    raw_dir = Path("data/event_labels_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    for year in range(args.start_year, args.end_year + 1):
        filename = latest_file_for_year(listing, year)
        raw_path = raw_dir / filename
        if not raw_path.exists():
            raw_path.write_bytes(curl_bytes(BASE_URL + filename))
        frame = pd.read_csv(raw_path, compression="gzip", low_memory=False)
        frame["EVENT_TYPE"] = frame["EVENT_TYPE"].astype(str).str.upper().str.strip()
        frame = frame[frame["EVENT_TYPE"].isin(EVENT_TYPES)].copy()
        frame["BEGIN_DATE_TIME"] = pd.to_datetime(
            frame["BEGIN_DATE_TIME"], format="%d-%b-%y %H:%M:%S", errors="coerce"
        )
        frame["END_DATE_TIME"] = pd.to_datetime(
            frame["END_DATE_TIME"], format="%d-%b-%y %H:%M:%S", errors="coerce"
        )
        frame["BEGIN_LAT"] = pd.to_numeric(frame["BEGIN_LAT"], errors="coerce")
        frame["BEGIN_LON"] = pd.to_numeric(frame["BEGIN_LON"], errors="coerce")
        frame = frame.dropna(subset=["BEGIN_DATE_TIME", "BEGIN_LAT", "BEGIN_LON"])

        for event in frame.itertuples():
            dates = pd.date_range(
                event.BEGIN_DATE_TIME.normalize(),
                (
                    event.END_DATE_TIME.normalize()
                    if pd.notna(event.END_DATE_TIME)
                    else event.BEGIN_DATE_TIME.normalize()
                ),
                freq="D",
            )
            nearest = []
            for station in stations.itertuples():
                distance = haversine(event.BEGIN_LAT, event.BEGIN_LON, station.lat, station.lon)
                if distance <= args.radius_km:
                    nearest.append((distance, station.station_id))
            for distance, station_id in nearest:
                for date in dates:
                    labels.append(
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "NOAA_STATION": station_id,
                            "EVENT_LABEL": 1,
                            "EVENT_TYPE": event.EVENT_TYPE,
                            "EVENT_BEGIN_TIME": event.BEGIN_DATE_TIME.isoformat(),
                            "EVENT_END_TIME": (
                                event.END_DATE_TIME.isoformat()
                                if pd.notna(event.END_DATE_TIME)
                                else event.BEGIN_DATE_TIME.isoformat()
                            ),
                            "EVENT_DISTANCE_KM": round(distance, 2),
                            "SOURCE_EVENT_ID": getattr(event, "EVENT_ID", None),
                        }
                    )

    if labels:
        output = pd.DataFrame(labels).drop_duplicates(
            subset=["date", "NOAA_STATION", "EVENT_TYPE", "SOURCE_EVENT_ID"]
        )
    else:
        output = pd.DataFrame(
            columns=[
                "date",
                "NOAA_STATION",
                "EVENT_LABEL",
                "EVENT_TYPE",
                "EVENT_BEGIN_TIME",
                "EVENT_END_TIME",
                "EVENT_DISTANCE_KM",
                "SOURCE_EVENT_ID",
            ]
        )
    output.to_csv(args.output, index=False)
    print(
        f"Wrote {len(output):,} station-event rows for "
        f"{args.start_year}-{args.end_year} -> {args.output}"
    )


if __name__ == "__main__":
    main()
