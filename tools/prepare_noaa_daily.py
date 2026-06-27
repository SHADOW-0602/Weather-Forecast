"""Prepare selected stations from a NOAA daily CSV export.

The source is expected to contain NOAA-style columns such as STATION, DATE,
PRCP, TMAX, TMIN, and their *_ATTRIBUTES quality metadata. Output values use
the normalized AeroClim schema and SI units.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CITY_ALIASES = {
    "USW00024233": "seattle",
    "USW00014734": "new_york",
    "USW00023183": "phoenix",
}

VALUE_COLUMNS = ("PRCP", "TMAX", "TMIN")
SOURCE_COLUMNS = [
    "STATION",
    "NAME",
    "LATITUDE",
    "LONGITUDE",
    "ELEVATION",
    "DATE",
    "PRCP",
    "PRCP_ATTRIBUTES",
    "TMAX",
    "TMAX_ATTRIBUTES",
    "TMIN",
    "TMIN_ATTRIBUTES",
]


def quality_flag(attributes: pd.Series) -> pd.Series:
    """Extract the NOAA quality flag (second comma-separated field)."""
    return attributes.fillna("").astype(str).str.split(",").str[1].fillna("")


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.copy()

    for column in VALUE_COLUMNS:
        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        invalid = quality_flag(chunk[f"{column}_ATTRIBUTES"]).str.strip().ne("")
        chunk.loc[invalid, column] = np.nan

    # NOAA trace precipitation is represented numerically as zero in this file.
    chunk.loc[chunk["PRCP"] < 0, "PRCP"] = np.nan
    chunk.loc[chunk["PRCP"] > 15, "PRCP"] = np.nan
    chunk.loc[(chunk["TMAX"] < -90) | (chunk["TMAX"] > 130), "TMAX"] = np.nan
    chunk.loc[(chunk["TMIN"] < -90) | (chunk["TMIN"] > 120), "TMIN"] = np.nan
    chunk.loc[chunk["TMIN"] > chunk["TMAX"], ["TMIN", "TMAX"]] = np.nan

    output = pd.DataFrame(
        {
            "date": pd.to_datetime(chunk["DATE"], errors="coerce"),
            "TMAX": (chunk["TMAX"] - 32.0) * 5.0 / 9.0,
            "TMIN": (chunk["TMIN"] - 32.0) * 5.0 / 9.0,
            "PRCP": chunk["PRCP"] * 25.4,
            "NOAA_STATION": chunk["STATION"],
            "NOAA_NAME": chunk["NAME"],
            "LATITUDE": chunk["LATITUDE"],
            "LONGITUDE": chunk["LONGITUDE"],
            "ELEVATION": chunk["ELEVATION"],
        }
    )
    output["TAVG"] = (output["TMAX"] + output["TMIN"]) / 2.0
    return output.dropna(subset=["date", "TMAX", "TMIN", "PRCP"])


def prepare(source: Path, output_dir: Path, chunksize: int = 250_000) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    station_frames: dict[str, list[pd.DataFrame]] = {}

    for chunk in pd.read_csv(
        source, usecols=SOURCE_COLUMNS, chunksize=chunksize, low_memory=False
    ):
        cleaned = clean_chunk(chunk)
        for station_id, station_frame in cleaned.groupby("NOAA_STATION"):
            station_frames.setdefault(station_id, []).append(station_frame)

    catalog = []
    for station_id, frames in sorted(station_frames.items()):
        frame = pd.concat(frames, ignore_index=True)
        frame = (
            frame.drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        name = str(frame["NOAA_NAME"].dropna().iloc[0])
        catalog.append(
            {
                "station_id": station_id,
                "name": name,
                "region": "NOAA Daily Station Network",
                "climate": "Station observations",
                "elevation": frame["ELEVATION"].dropna().iloc[0],
                "lat": frame["LATITUDE"].dropna().iloc[0],
                "lon": frame["LONGITUDE"].dropna().iloc[0],
                "ocean_basin": "",
                "csv_file": f"{station_id.lower()}.csv",
                "start_date": frame["date"].min().strftime("%Y-%m-%d"),
                "end_date": frame["date"].max().strftime("%Y-%m-%d"),
                "row_count": len(frame),
            }
        )
        frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
        destination = output_dir / f"{station_id.lower()}.csv"
        frame.to_csv(destination, index=False, float_format="%.4f")
        print(
            f"{station_id}: {len(frame):,} rows, {frame['date'].iloc[0]} to "
            f"{frame['date'].iloc[-1]} -> {destination}"
        )

    if not catalog:
        raise RuntimeError(f"No station records were found in {source}")

    catalog_frame = pd.DataFrame(catalog)
    catalog_frame.to_csv(output_dir / "stations.csv", index=False)

    # Friendly aliases preserve the existing command-line and dashboard IDs.
    for station_id, alias in CITY_ALIASES.items():
        source_file = output_dir / f"{station_id.lower()}.csv"
        if source_file.exists():
            alias_file = output_dir / f"{alias}.csv"
            alias_file.write_bytes(source_file.read_bytes())

    print(f"Prepared {len(catalog):,} stations; catalog -> {output_dir / 'stations.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Path to the NOAA CSV export")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/noaa_stations"),
        help="Destination directory (default: data/noaa_stations)",
    )
    args = parser.parse_args()
    prepare(args.source, args.output_dir)


if __name__ == "__main__":
    main()
