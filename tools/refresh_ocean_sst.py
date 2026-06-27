"""Refresh basin SST context from NOAA/NCEI OISST v2.1 monthly data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr


OISST_URL = (
    "https://psl.noaa.gov/thredds/dodsC/"
    "Datasets/noaa.oisst.v2.highres/sst.mon.mean.nc"
)
BASINS = {
    "sst_atlantic": (40.0, -50.0),
    "sst_pacific": (35.0, -150.0),
    "sst_gulf": (25.0, -90.0),
}


def main() -> None:
    dataset = xr.open_dataset(OISST_URL)
    series = {}
    for name, (lat, lon) in BASINS.items():
        values = (
            dataset["sst"]
            .sel(lat=lat, lon=lon % 360, method="nearest")
            .sel(time=slice("1995-01-01", None))
            .load()
            .to_series()
        )
        series[name] = values
    dataset.close()

    monthly = pd.DataFrame(series).sort_index()
    daily_index = pd.date_range(monthly.index.min(), monthly.index.max(), freq="D")
    daily = monthly.reindex(monthly.index.union(daily_index)).interpolate(
        method="time", limit_direction="both"
    ).reindex(daily_index)
    daily.index.name = "date"
    for basin in BASINS:
        daily[f"{basin}_f"] = daily[basin] * 9.0 / 5.0 + 32.0
        climatology = daily.loc["1995-01-01":"2020-12-31", basin].groupby(
            daily.loc["1995-01-01":"2020-12-31"].index.dayofyear
        ).mean()
        daily[f"{basin}_anom"] = [
            value - climatology.get(date.dayofyear, climatology.mean())
            for date, value in daily[basin].items()
        ]

    output = pd.DataFrame(
        {
            "date": daily.index,
            "sst_atlantic_f": daily["sst_atlantic_f"],
            "sst_pacific_f": daily["sst_pacific_f"],
            "sst_gulf_f": daily["sst_gulf_f"],
            "sst_atlantic_anom": daily["sst_atlantic_anom"],
            "sst_pacific_anom": daily["sst_pacific_anom"],
            "sst_gulf_anom": daily["sst_gulf_anom"],
            "data_source": "NOAA_NCEI_OISST_V2.1",
            "quality_flag": 1,
        }
    )
    path = Path("data/ocean.csv")
    output.to_csv(path, index=False, float_format="%.4f")
    print(
        f"Wrote {len(output):,} daily rows through "
        f"{output['date'].max().date()} -> {path}"
    )


if __name__ == "__main__":
    main()
