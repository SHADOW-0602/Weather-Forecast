"""Fetch ERA5-Land monthly soil moisture and map it to NOAA stations.

Monthly regional downloads are far more practical than thousands of daily
point requests through CDS. Monthly values are linearly interpolated to daily
resolution after nearest-grid-cell extraction.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cdsapi
import pandas as pd
import xarray as xr


DATASET = "reanalysis-era5-land-monthly-means"
VARIABLES = [
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
]
MONTHS = [f"{month:02d}" for month in range(1, 13)]
YEAR_BLOCKS = [
    list(range(1995, 2005)),
    list(range(2005, 2015)),
    list(range(2015, 2025)),
]
REGIONS = {
    "conus": [50.0, -125.0, 24.0, -66.0],
    "alaska": [72.0, -170.0, 50.0, -130.0],
}


def retrieve_regions(client: cdsapi.Client, raw_dir: Path) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for region, area in REGIONS.items():
        for years in YEAR_BLOCKS:
            target = raw_dir / f"{region}-{years[0]}-{years[-1]}.nc"
            paths.append(target)
            if target.exists():
                print(f"Reusing {target.name}", flush=True)
                continue
            request = {
                "product_type": "monthly_averaged_reanalysis",
                "variable": VARIABLES,
                "year": [str(year) for year in years],
                "month": MONTHS,
                "time": "00:00",
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": area,
            }
            print(f"Downloading {target.name}", flush=True)
            client.retrieve(DATASET, request, str(target))
    return paths


def extract_station(files: list[Path], station_id: str, lat: float, lon: float) -> pd.DataFrame:
    region = "alaska" if lat >= 50.0 and lon <= -130.0 else "conus"
    selected = [path for path in files if path.name.startswith(region + "-")]
    frames = []
    for path in selected:
        with xr.open_dataset(path) as dataset:
            time_name = "valid_time" if "valid_time" in dataset.coords else "time"
            point = dataset.sel(latitude=lat, longitude=lon, method="nearest")
            frame = pd.DataFrame(
                {
                    "date": pd.to_datetime(point[time_name].values),
                    "SOIL_MOISTURE_L1": point["swvl1"].values.reshape(-1),
                    "SOIL_MOISTURE_L2": point["swvl2"].values.reshape(-1),
                }
            )
            frames.append(frame)

    monthly = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .set_index("date")
    )
    daily_index = pd.date_range("1995-01-01", "2024-12-31", freq="D")
    daily = monthly.reindex(monthly.index.union(daily_index)).interpolate(
        method="time", limit_direction="both"
    )
    daily = daily.reindex(daily_index)
    daily.index.name = "date"
    daily["SOIL_MOISTURE_VOL"] = (
        daily["SOIL_MOISTURE_L1"] * 7.0
        + daily["SOIL_MOISTURE_L2"] * 21.0
    ) / 28.0
    daily["SATURATION_PCT"] = (
        daily["SOIL_MOISTURE_VOL"].rank(method="average", pct=True) * 100.0
    )
    daily["NOAA_STATION"] = station_id
    return daily.reset_index()


def fetch(catalog_path: Path, output_dir: Path) -> None:
    key = os.getenv("CDSAPI_KEY")
    if not key:
        raise RuntimeError("Set CDSAPI_KEY before running this downloader.")
    client = cdsapi.Client(
        url=os.getenv("CDSAPI_URL", "https://cds.climate.copernicus.eu/api"),
        key=key,
        quiet=True,
        progress=False,
    )
    files = retrieve_regions(client, output_dir / "raw")
    catalog = pd.read_csv(catalog_path, dtype={"station_id": str})
    aliases = {
        "USW00024233": "seattle",
        "USW00014734": "new_york",
        "USW00023183": "phoenix",
    }
    for number, row in enumerate(catalog.itertuples(), start=1):
        frame = extract_station(
            files, row.station_id, float(row.lat), float(row.lon)
        )
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
        destination = output_dir / f"{row.station_id.lower()}.csv"
        frame.to_csv(destination, index=False, float_format="%.6f")
        if row.station_id in aliases:
            frame.to_csv(
                output_dir / f"{aliases[row.station_id]}.csv",
                index=False,
                float_format="%.6f",
            )
        print(
            f"[{number}/{len(catalog)}] Wrote {destination.name}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/noaa_stations/stations.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/soil_moisture"),
    )
    args = parser.parse_args()
    fetch(args.catalog, args.output_dir)


if __name__ == "__main__":
    main()
