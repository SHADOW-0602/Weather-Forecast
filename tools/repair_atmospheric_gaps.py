"""Repair GSOD gaps using same-day observations from nearby NOAA stations."""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd


FEATURES = [
    "DEWPOINT_F",
    "WINDSPEED_MPH",
    "PRESSURE_HPA",
    "HUMIDITY_PCT",
]
MAX_DISTANCE_KM = 400.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 6371.0 * 2.0 * asin(sqrt(a))


def main() -> None:
    source_dir = Path("data/noaa_atmosphere")
    output_dir = Path("data/noaa_atmosphere_repaired")
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = pd.read_csv("data/noaa_stations/stations.csv", dtype={"station_id": str})
    metadata = catalog.set_index("station_id")
    frames = {
        station_id: pd.read_csv(source_dir / f"{station_id.lower()}.csv").set_index("date")
        for station_id in catalog["station_id"]
    }

    for number, station_id in enumerate(catalog["station_id"], start=1):
        target = frames[station_id].copy()
        missing_before = int(target[FEATURES].isna().sum().sum())
        target["ATMOSPHERE_IMPUTED"] = 0
        target["ATMOSPHERE_DONOR_KM"] = 0.0
        if missing_before:
            row = metadata.loc[station_id]
            donors = []
            for donor_id, donor_row in metadata.iterrows():
                if donor_id == station_id:
                    continue
                distance = haversine(
                    float(row.lat),
                    float(row.lon),
                    float(donor_row.lat),
                    float(donor_row.lon),
                )
                if distance <= MAX_DISTANCE_KM:
                    donors.append((distance, donor_id))
            for distance, donor_id in sorted(donors):
                missing_mask = target[FEATURES].isna()
                if not missing_mask.any().any():
                    break
                donor = frames[donor_id][FEATURES].reindex(target.index)
                fillable = missing_mask & donor.notna()
                for feature in FEATURES:
                    rows = fillable[feature]
                    if rows.any():
                        target.loc[rows, feature] = donor.loc[rows, feature]
                        target.loc[rows, "ATMOSPHERE_IMPUTED"] = 1
                        target.loc[rows, "ATMOSPHERE_DONOR_KM"] = distance

            # Interpolate only very short residual gaps.
            remaining = target[FEATURES].isna()
            interpolated = target[FEATURES].interpolate(limit=3, limit_direction="both")
            changed = remaining & interpolated.notna()
            target[FEATURES] = interpolated
            target.loc[changed.any(axis=1), "ATMOSPHERE_IMPUTED"] = 1

        missing_after = int(target[FEATURES].isna().sum().sum())
        destination = output_dir / f"{station_id.lower()}.csv"
        target.reset_index().to_csv(destination, index=False, float_format="%.4f")
        print(
            f"[{number}/100] {station_id}: missing {missing_before:,} -> "
            f"{missing_after:,}",
            flush=True,
        )

    aliases = {
        "USW00024233": "seattle",
        "USW00014734": "new_york",
        "USW00023183": "phoenix",
    }
    for station_id, alias in aliases.items():
        source = output_dir / f"{station_id.lower()}.csv"
        (output_dir / f"{alias}.csv").write_bytes(source.read_bytes())


if __name__ == "__main__":
    main()
