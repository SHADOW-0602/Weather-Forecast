"""Reduce prepared AeroClim datasets to the current 20-station model subset.

This intentionally trims generated station-level CSVs so the workspace data
matches the saved pooled model footprint. Global auxiliary files such as
climate indices and ocean SST remain unchanged because they are shared context.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


KEEP_STATIONS = {
    "USW00003813",
    "USW00004725",
    "USW00013741",
    "USW00013873",
    "USW00013893",
    "USW00013959",
    "USW00013968",
    "USW00013995",
    "USW00014607",
    "USW00014740",
    "USW00014920",
    "USW00023009",
    "USW00024121",
    "USW00024221",
    "USW00024233",
    "USW00025503",
    "USW00026510",
    "USW00026615",
    "USW00093817",
    "USW00094860",
}


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def _safe_unlink_extra_csvs(directory: Path, keep_names: set[str]) -> int:
    """Delete CSVs in a known data subdirectory that are not in keep_names."""
    directory = directory.resolve()
    if not directory.exists():
        return 0
    if DATA.resolve() not in directory.parents and directory != DATA.resolve():
        raise RuntimeError(f"Refusing to modify path outside data/: {directory}")

    removed = 0
    for path in directory.glob("*.csv"):
        if path.name.lower() not in keep_names:
            path.unlink()
            removed += 1
    return removed


def filter_station_catalog() -> int:
    catalog_path = DATA / "noaa_stations" / "stations.csv"
    catalog = pd.read_csv(catalog_path)
    subset = catalog[catalog["station_id"].str.upper().isin(KEEP_STATIONS)].copy()
    subset = subset.sort_values("station_id")
    subset.to_csv(catalog_path, index=False)
    return len(subset)


def filter_event_labels() -> int:
    labels_path = DATA / "event_labels.csv"
    if not labels_path.exists():
        return 0
    labels = pd.read_csv(labels_path)
    labels = labels[labels["NOAA_STATION"].str.upper().isin(KEEP_STATIONS)].copy()
    labels.to_csv(labels_path, index=False)
    return len(labels)


def clear_alias_locations() -> None:
    """Remove legacy alias rows so CITIES exposes only the 20 NOAA stations."""
    locations_path = DATA / "station_locations.csv"
    if locations_path.exists():
        locations_path.write_text(
            "station_id,name,region,climate,elevation,lat,lon,ocean_basin\n",
            encoding="utf-8",
        )


def update_data_health_summary() -> None:
    """If present, pre-filter stale health data before a fresh audit is run."""
    health_path = DATA / "data_health.json"
    if not health_path.exists():
        return
    report = json.loads(health_path.read_text(encoding="utf-8"))
    stations = [
        row
        for row in report.get("stations", [])
        if str(row.get("station_id", "")).upper() in KEEP_STATIONS
    ]
    report["configured_entries"] = len(stations)
    report["unique_noaa_stations"] = len(stations)
    report["stations"] = stations
    report["station_errors"] = sum("error" in row for row in stations)
    report["stations_with_complete_core_data"] = sum(
        row.get("core_completeness_pct", 0) >= 99.9 for row in stations
    )
    report["stations_training_ready"] = sum(
        bool(row.get("training_ready")) for row in stations
    )
    health_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    keep_file_names = {f"{station.lower()}.csv" for station in KEEP_STATIONS}

    catalog_count = filter_station_catalog()
    removed = {
        "noaa_stations": _safe_unlink_extra_csvs(
            DATA / "noaa_stations", keep_file_names | {"stations.csv"}
        ),
        "noaa_atmosphere": _safe_unlink_extra_csvs(
            DATA / "noaa_atmosphere", keep_file_names
        ),
        "noaa_atmosphere_repaired": _safe_unlink_extra_csvs(
            DATA / "noaa_atmosphere_repaired", keep_file_names
        ),
        "soil_moisture": _safe_unlink_extra_csvs(
            DATA / "soil_moisture", keep_file_names
        ),
    }
    event_rows = filter_event_labels()
    clear_alias_locations()
    update_data_health_summary()

    print(f"Kept {catalog_count} NOAA stations")
    print(f"Filtered event label rows: {event_rows:,}")
    for name, count in removed.items():
        print(f"Removed {count} CSV files from data/{name}")


if __name__ == "__main__":
    main()
