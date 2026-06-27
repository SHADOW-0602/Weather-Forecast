"""Fetch daily atmospheric features from NOAA Global Summary of the Day.

NOAA CDO tokens are useful for catalog and station discovery, but the CDO LCD
data endpoint is not reliable for bulk daily atmospheric retrieval. This tool
uses NOAA's official HTTPS GSOD archive after mapping each project's WBAN code
through the official ISD station-history catalog.
"""

from __future__ import annotations

import argparse
import io
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
GSOD_URL = (
    "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/"
    "{year}/{usaf}{wban}.csv"
)
USER_AGENT = "AeroClim-NOAA-fetcher/1.0"
MISSING_LIMITS = {
    "TEMP": 999.0,
    "DEWP": 999.0,
    "SLP": 9999.0,
    "STP": 999.0,
    "WDSP": 999.0,
}


def download_bytes(url: str, timeout: int = 20) -> bytes:
    completed = subprocess.run(
        [
            "curl.exe",
            "--silent",
            "--show-error",
            "--fail",
            "--max-time",
            str(timeout),
            "--connect-timeout",
            "10",
            "--user-agent",
            USER_AGENT,
            url,
        ],
        check=True,
        capture_output=True,
        timeout=timeout + 5,
    )
    return completed.stdout


def load_history(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(download_bytes(HISTORY_URL))
    history = pd.read_csv(cache_path, dtype={"USAF": str, "WBAN": str})
    history["USAF"] = history["USAF"].str.zfill(6)
    history["WBAN"] = history["WBAN"].str.zfill(5)
    history["BEGIN"] = pd.to_datetime(history["BEGIN"], format="%Y%m%d", errors="coerce")
    history["END"] = pd.to_datetime(history["END"], format="%Y%m%d", errors="coerce")
    return history


def station_segments(history: pd.DataFrame, station_id: str) -> pd.DataFrame:
    wban = station_id[-5:]
    rows = history[(history["WBAN"] == wban) & (history["CTRY"] == "US")].copy()
    return rows.sort_values(["BEGIN", "END"])


def choose_segment(segments: pd.DataFrame, year: int) -> pd.Series | None:
    year_start = pd.Timestamp(year=year, month=1, day=1)
    year_end = pd.Timestamp(year=year, month=12, day=31)
    matches = segments[
        (segments["BEGIN"] <= year_end) & (segments["END"] >= year_start)
    ]
    if matches.empty:
        return None
    # Prefer a real USAF code over the legacy 999999 placeholder.
    real = matches[matches["USAF"] != "999999"]
    return (real if not real.empty else matches).iloc[-1]


def relative_humidity(temp_f: pd.Series, dewpoint_f: pd.Series) -> pd.Series:
    temp_c = (temp_f - 32.0) * 5.0 / 9.0
    dew_c = (dewpoint_f - 32.0) * 5.0 / 9.0
    numerator = np.exp((17.625 * dew_c) / (243.04 + dew_c))
    denominator = np.exp((17.625 * temp_c) / (243.04 + temp_c))
    return (100.0 * numerator / denominator).clip(0.0, 100.0)


def parse_gsod(payload: bytes, station_id: str) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(payload))
    for column, missing_floor in MISSING_LIMITS.items():
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame.loc[frame[column] >= missing_floor, column] = np.nan

    station_pressure = frame["STP"].copy()
    station_pressure = station_pressure.where(
        station_pressure >= 500.0, station_pressure + 1000.0
    )
    pressure = frame["SLP"].fillna(station_pressure)
    result = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["DATE"], errors="coerce"),
            "DEWPOINT_F": frame["DEWP"],
            "WINDSPEED_MPH": frame["WDSP"] * 1.15077945,
            "PRESSURE_HPA": pressure,
            "HUMIDITY_PCT": relative_humidity(frame["TEMP"], frame["DEWP"]),
            "NOAA_STATION": station_id,
        }
    )
    return result.dropna(subset=["date"])


def fetch_year(
    station_id: str, year: int, segments: pd.DataFrame
) -> tuple[str, int, pd.DataFrame | None, str | None]:
    segment = choose_segment(segments, year)
    if segment is None:
        return station_id, year, None, "no station-history segment"
    url = GSOD_URL.format(year=year, usaf=segment["USAF"], wban=segment["WBAN"])
    try:
        return station_id, year, parse_gsod(download_bytes(url), station_id), None
    except subprocess.CalledProcessError as exc:
        if b"404" in exc.stderr:
            return station_id, year, None, "not found"
        return station_id, year, None, f"curl exit {exc.returncode}"
    except subprocess.TimeoutExpired:
        return station_id, year, None, "timeout"


def fetch_all(
    catalog_path: Path,
    output_dir: Path,
    start_year: int,
    end_year: int,
    workers: int,
    station_offset: int = 0,
    station_limit: int | None = None,
) -> None:
    catalog = pd.read_csv(catalog_path, dtype={"station_id": str})
    if station_limit is not None:
        catalog = catalog.iloc[station_offset : station_offset + station_limit].copy()
    elif station_offset:
        catalog = catalog.iloc[station_offset:].copy()
    history = load_history(output_dir / "isd-history.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "yearly"
    cache_dir.mkdir(parents=True, exist_ok=True)

    segments = {
        station_id: station_segments(history, station_id)
        for station_id in catalog["station_id"]
    }
    all_tasks = [
        (station_id, year)
        for station_id in catalog["station_id"]
        for year in range(start_year, end_year + 1)
    ]
    tasks = [
        (station_id, year)
        for station_id, year in all_tasks
        if not (cache_dir / f"{station_id.lower()}-{year}.csv").exists()
        and not (cache_dir / f"{station_id.lower()}-{year}.missing").exists()
    ]
    failures = []

    if tasks:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(fetch_year, station_id, year, segments[station_id]): (
                    station_id,
                    year,
                )
                for station_id, year in tasks
            }
            for index, future in enumerate(as_completed(futures), start=1):
                station_id, year, frame, error = future.result()
                if frame is not None and not frame.empty:
                    frame.to_csv(
                        cache_dir / f"{station_id.lower()}-{year}.csv",
                        index=False,
                        float_format="%.4f",
                    )
                elif error in {"not found", "no station-history segment"}:
                    (cache_dir / f"{station_id.lower()}-{year}.missing").write_text(
                        error, encoding="utf-8"
                    )
                else:
                    failures.append((station_id, year, error))
                if index % 25 == 0 or index == len(tasks):
                    print(
                        f"Fetched {index:,}/{len(tasks):,} uncached station-years",
                        flush=True,
                    )
    else:
        print("All requested station-years were already cached", flush=True)

    aliases = {
        "USW00024233": "seattle",
        "USW00014734": "new_york",
        "USW00023183": "phoenix",
    }
    written = 0
    for station_id in catalog["station_id"]:
        cached_files = [
            cache_dir / f"{station_id.lower()}-{year}.csv"
            for year in range(start_year, end_year + 1)
        ]
        cached_files = [path for path in cached_files if path.exists()]
        if not cached_files:
            continue
        frame = (
            pd.concat((pd.read_csv(path) for path in cached_files), ignore_index=True)
            .drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
        )
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"])
        frame.loc[
            frame["PRESSURE_HPA"].notna() & (frame["PRESSURE_HPA"] < 500.0),
            "PRESSURE_HPA",
        ] += 1000.0
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
        destination = output_dir / f"{station_id.lower()}.csv"
        frame.to_csv(destination, index=False, float_format="%.4f")
        if station_id in aliases:
            frame.to_csv(
                output_dir / f"{aliases[station_id]}.csv",
                index=False,
                float_format="%.4f",
            )
        written += 1

    print(
        f"Wrote atmospheric files for {written}/{len(catalog)} stations",
        flush=True,
    )
    if failures:
        failure_frame = pd.DataFrame(failures, columns=["station_id", "year", "error"])
        failure_frame.to_csv(output_dir / "fetch_failures.csv", index=False)
        print(f"Recorded {len(failures)} retryable failures")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=1995)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--station-offset", type=int, default=0)
    parser.add_argument("--station-limit", type=int)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/noaa_stations/stations.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/noaa_atmosphere"),
    )
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("--start-year must not exceed --end-year")
    fetch_all(
        args.catalog,
        args.output_dir,
        args.start_year,
        args.end_year,
        max(1, args.workers),
        max(0, args.station_offset),
        args.station_limit,
    )


if __name__ == "__main__":
    main()
