"""Refresh ENSO, PDO, and NAO indices from official NOAA PSL sources."""

from __future__ import annotations

import io
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


SOURCES = {
    "nino34_sst": "https://psl.noaa.gov/data/correlation/nina34.data",
    "pdo_index": "https://psl.noaa.gov/data/correlation/pdo.data",
    "nao_index": "https://psl.noaa.gov/data/correlation/nao.data",
}


def download_text(url: str) -> str:
    result = subprocess.run(
        [
            "curl.exe",
            "--silent",
            "--show-error",
            "--fail",
            "--max-time",
            "60",
            "--connect-timeout",
            "10",
            url,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=70,
    )
    return result.stdout


def parse_year_matrix(text: str, missing_floor: float = -90.0) -> pd.Series:
    records = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 13:
            continue
        try:
            year = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        for month, value in enumerate(values, start=1):
            if value > missing_floor:
                records.append((pd.Timestamp(year, month, 1), value))
    return pd.Series(dict(records)).sort_index()


def main() -> None:
    nino_sst = parse_year_matrix(download_text(SOURCES["nino34_sst"]))
    pdo = parse_year_matrix(download_text(SOURCES["pdo_index"]))
    nao = parse_year_matrix(download_text(SOURCES["nao_index"]))

    # Convert Niño 3.4 SST to monthly anomalies relative to 1991-2020.
    baseline = nino_sst.loc["1991-01-01":"2020-12-01"]
    monthly_climatology = baseline.groupby(baseline.index.month).mean()
    nino_anomaly = pd.Series(
        [
            value - monthly_climatology.loc[date.month]
            for date, value in nino_sst.items()
        ],
        index=nino_sst.index,
    )

    monthly = pd.concat(
        [
            nino_anomaly.rename("enso_nino34"),
            pdo.rename("pdo_index"),
            nao.rename("nao_index"),
        ],
        axis=1,
        sort=True,
    ).sort_index()
    monthly = monthly.loc["1995-01-01":]
    daily_index = pd.date_range(monthly.index.min(), monthly.index.max(), freq="D")
    daily = monthly.reindex(monthly.index.union(daily_index)).ffill().reindex(daily_index)
    daily.index.name = "date"
    daily["enso_phase"] = np.select(
        [daily["enso_nino34"] >= 0.5, daily["enso_nino34"] <= -0.5],
        ["El Nino", "La Nina"],
        default="Neutral",
    )
    daily["data_source"] = "NOAA_PSL"
    daily["observed_month"] = daily.index.to_period("M").astype(str)
    output = Path("data/climate_indices.csv")
    daily.reset_index().to_csv(output, index=False, float_format="%.4f")
    print(
        f"Wrote {len(daily):,} daily rows through "
        f"{daily.index.max().date()} -> {output}"
    )


if __name__ == "__main__":
    main()
