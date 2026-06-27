"""Build station-centered 2D CNN image chips from downloaded MRMS GRIB2 files."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


PRODUCTS = [
    "MergedReflectivityQCComposite",
    "RadarOnly_QPE_01H",
    "MergedAzShear_0-2kmAGL",
]


@dataclass
class ProductFile:
    product: str
    path: Path
    timestamp: datetime


def parse_timestamp(path: Path) -> datetime:
    match = re.search(r"(\d{8})-(\d{6})", path.name)
    if not match:
        raise ValueError(f"Could not parse MRMS timestamp from {path.name}")
    return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")


def list_product_files(raw_dir: Path) -> dict[str, list[ProductFile]]:
    files: dict[str, list[ProductFile]] = {}
    for product in PRODUCTS:
        product_dir = raw_dir / product
        product_files = [
            ProductFile(product=product, path=path, timestamp=parse_timestamp(path))
            for path in product_dir.glob("*.grib2.gz")
        ]
        files[product] = sorted(product_files, key=lambda item: item.timestamp)
    return files


def nearest_file(
    files: list[ProductFile], timestamp: datetime, tolerance_seconds: int
) -> ProductFile | None:
    if not files:
        return None
    nearest = min(files, key=lambda item: abs((item.timestamp - timestamp).total_seconds()))
    if abs((nearest.timestamp - timestamp).total_seconds()) > tolerance_seconds:
        return None
    return nearest


def decompress_if_needed(path: Path, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".gz":
        return path
    out_path = cache_dir / path.with_suffix("").name
    if not out_path.exists() or out_path.stat().st_mtime < path.stat().st_mtime:
        with gzip.open(path, "rb") as src:
            out_path.write_bytes(src.read())
    return out_path


def open_grid(path: Path, cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grib_path = decompress_if_needed(path, cache_dir)
    dataset = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    )
    variable_name = list(dataset.data_vars)[0]
    values = dataset[variable_name].values.astype(np.float32)
    latitudes = dataset["latitude"].values.astype(np.float64)
    longitudes = dataset["longitude"].values.astype(np.float64)
    dataset.close()
    return values, latitudes, longitudes


def normalize(product: str, chip: np.ndarray) -> np.ndarray:
    chip = np.nan_to_num(chip.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if product == "MergedReflectivityQCComposite":
        return np.clip((chip + 10.0) / 85.0, 0.0, 1.0)
    if product == "RadarOnly_QPE_01H":
        return np.clip(chip / 50.0, 0.0, 1.0)
    if product == "MergedAzShear_0-2kmAGL":
        return np.clip((chip + 0.02) / 0.04, 0.0, 1.0)
    lo, hi = np.nanpercentile(chip, [1, 99])
    if hi <= lo:
        return np.zeros_like(chip, dtype=np.float32)
    return np.clip((chip - lo) / (hi - lo), 0.0, 1.0)


def extract_chip(
    grid: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    lat: float,
    lon: float,
    chip_size: int,
) -> np.ndarray | None:
    lon360 = lon % 360.0
    y = int(np.abs(latitudes - lat).argmin())
    x = int(np.abs(longitudes - lon360).argmin())
    half = chip_size // 2
    y0, y1 = y - half, y + half
    x0, x1 = x - half, x + half
    if y0 < 0 or x0 < 0 or y1 > grid.shape[0] or x1 > grid.shape[1]:
        return None
    return grid[y0:y1, x0:x1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/mrms_raw")
    parser.add_argument("--out-dir", default="data/mrms_images")
    parser.add_argument("--chip-size", type=int, default=64)
    parser.add_argument("--tolerance-seconds", type=int, default=600)
    parser.add_argument(
        "--event-window-before-hours",
        type=float,
        default=2.0,
        help="Label an MRMS chip positive this many hours before an event begins.",
    )
    parser.add_argument(
        "--event-window-after-hours",
        type=float,
        default=3.0,
        help="Label an MRMS chip positive until this many hours after an event ends.",
    )
    parser.add_argument(
        "--ambiguous-window-hours",
        type=float,
        default=12.0,
        help=(
            "Mark non-positive chips as unlabeled when they are this close to a "
            "station event, avoiding noisy negative labels near storms."
        ),
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "_grib_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    stations = pd.read_csv("data/noaa_stations/stations.csv")
    event_labels_path = Path("data/event_labels.csv")
    positive_events: set[tuple[str, str]] = set()
    event_details: dict[str, pd.DataFrame] = {}
    event_stations: set[str] = set()
    event_start = None
    event_end = None
    if event_labels_path.exists():
        labels = pd.read_csv(event_labels_path)
        labels["date"] = pd.to_datetime(labels["date"], errors="coerce")
        labels = labels.dropna(subset=["date"])
        labels["date_key"] = labels["date"].dt.strftime("%Y-%m-%d")
        labels["NOAA_STATION"] = labels["NOAA_STATION"].astype(str).str.upper()
        if "EVENT_BEGIN_TIME" in labels.columns:
            labels["EVENT_BEGIN_TIME"] = pd.to_datetime(
                labels["EVENT_BEGIN_TIME"], errors="coerce"
            )
        if "EVENT_END_TIME" in labels.columns:
            labels["EVENT_END_TIME"] = pd.to_datetime(
                labels["EVENT_END_TIME"], errors="coerce"
            )
        if "EVENT_BEGIN_TIME" in labels.columns:
            labels["EVENT_END_TIME"] = labels.get(
                "EVENT_END_TIME", labels["EVENT_BEGIN_TIME"]
            ).fillna(labels["EVENT_BEGIN_TIME"])
        positive_events = set(zip(labels["NOAA_STATION"], labels["date_key"]))
        event_details = {
            station_id: group.copy()
            for station_id, group in labels.groupby("NOAA_STATION")
        }
        event_stations = set(labels["NOAA_STATION"].unique())
        event_start = labels["date"].min().date()
        event_end = labels["date"].max().date()
    product_files = list_product_files(raw_dir)
    base_files = product_files["MergedReflectivityQCComposite"]
    if not base_files:
        raise FileNotFoundError("No MergedReflectivityQCComposite files found.")

    samples = []
    arrays = []
    for base_file in base_files:
        matched = {
            product: nearest_file(
                product_files[product],
                base_file.timestamp,
                args.tolerance_seconds,
            )
            for product in PRODUCTS
        }
        if any(value is None for value in matched.values()):
            continue

        grid_cache: dict[Path, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for path in [item.path for item in matched.values() if item is not None]:
            if path not in grid_cache:
                grid_cache[path] = open_grid(path, cache_dir)

        for _, station in stations.iterrows():
            station_id = str(station["station_id"]).upper()
            date_key = base_file.timestamp.strftime("%Y-%m-%d")
            sample_date = base_file.timestamp.date()
            sample_timestamp = pd.Timestamp(base_file.timestamp)
            if (
                event_start is not None
                and event_end is not None
                and station_id in event_stations
                and event_start <= sample_date <= event_end
            ):
                label = 0
                label_note = "noaa_storm_events_time_window"
            else:
                label = -1
                label_note = "unlabeled_outside_event_coverage"
            closest_event_minutes = None
            closest_event_signed_minutes = None
            closest_event_type = ""
            closest_event_distance_km = None
            details = event_details.get(station_id)
            if label >= 0 and details is not None and "EVENT_BEGIN_TIME" in details.columns:
                begin_times = details["EVENT_BEGIN_TIME"]
                end_times = details.get("EVENT_END_TIME", begin_times).fillna(begin_times)
                positive_start = begin_times - pd.to_timedelta(
                    args.event_window_before_hours, unit="h"
                )
                positive_end = end_times + pd.to_timedelta(
                    args.event_window_after_hours, unit="h"
                )
                in_positive_window = (
                    (sample_timestamp >= positive_start)
                    & (sample_timestamp <= positive_end)
                )
                ambiguous_start = begin_times - pd.to_timedelta(
                    args.ambiguous_window_hours, unit="h"
                )
                ambiguous_end = end_times + pd.to_timedelta(
                    args.ambiguous_window_hours, unit="h"
                )
                in_ambiguous_window = (
                    (sample_timestamp >= ambiguous_start)
                    & (sample_timestamp <= ambiguous_end)
                )
                if in_positive_window.any():
                    label = 1
                    label_note = (
                        "noaa_storm_events_time_window_positive_"
                        f"-{args.event_window_before_hours:g}h_"
                        f"+{args.event_window_after_hours:g}h"
                    )
                elif in_ambiguous_window.any():
                    label = -1
                    label_note = (
                        "unlabeled_near_event_outside_positive_window_"
                        f"{args.ambiguous_window_hours:g}h"
                    )
                deltas = (
                    details["EVENT_BEGIN_TIME"] - sample_timestamp
                ).abs()
                if deltas.notna().any():
                    idx = deltas.idxmin()
                    closest_event_minutes = float(deltas.loc[idx].total_seconds() / 60.0)
                    closest_event_signed_minutes = float(
                        (
                            details.loc[idx, "EVENT_BEGIN_TIME"] - sample_timestamp
                        ).total_seconds()
                        / 60.0
                    )
                    closest_event_type = str(details.loc[idx].get("EVENT_TYPE", ""))
                    closest_event_distance_km = float(
                        details.loc[idx].get("EVENT_DISTANCE_KM", np.nan)
                    )
            channels = []
            ok = True
            for product in PRODUCTS:
                product_file = matched[product]
                assert product_file is not None
                grid, latitudes, longitudes = grid_cache[product_file.path]
                chip = extract_chip(
                    grid,
                    latitudes,
                    longitudes,
                    float(station["lat"]),
                    float(station["lon"]),
                    args.chip_size,
                )
                if chip is None:
                    ok = False
                    break
                channels.append(normalize(product, chip))
            if not ok:
                continue
            image = np.stack(channels, axis=-1).astype(np.float32)
            arrays.append(image)
            samples.append(
                {
                    "station_id": station_id,
                    "timestamp": base_file.timestamp.isoformat(),
                    "label": label,
                    "label_note": label_note,
                    "closest_event_minutes": closest_event_minutes,
                    "closest_event_signed_minutes": closest_event_signed_minutes,
                    "closest_event_type": closest_event_type,
                    "closest_event_distance_km": closest_event_distance_km,
                    "lat": float(station["lat"]),
                    "lon": float(station["lon"]),
                    "products": ",".join(PRODUCTS),
                }
            )
        del grid_cache

    if not arrays:
        raise RuntimeError("No station chips were created.")

    X = np.stack(arrays, axis=0)
    y = manifest_labels = np.array([row["label"] for row in samples], dtype=np.int8)
    npz_path = out_dir / "mrms_live_chips.npz"
    np.savez_compressed(npz_path, X=X, y=y)
    manifest = pd.DataFrame(samples)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    summary = {
        "npz": str(npz_path),
        "manifest": str(manifest_path),
        "samples": int(X.shape[0]),
        "shape": list(X.shape),
        "labeled_samples": int((manifest_labels >= 0).sum()),
        "products": PRODUCTS,
        "label_note": "Labels are -1 outside the NOAA Storm Events coverage window.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
