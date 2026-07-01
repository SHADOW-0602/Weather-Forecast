import os
import glob
import pandas as pd
import numpy as np

_AUX_FILES = {
    "ocean", "ocean_monthly", "atmosphere", "climate_indices",
    "region", "soil_moisture", "sst_daily_summary",
    "advanced_atmospheric",
    "station_locations",
}

# Load locations metadata CSV if present
_LOCATIONS_FILE = "data/station_locations.csv"
_LOCATIONS_DF = None
_NOAA_STATION_DIR = os.path.join("data", "noaa_stations")
_NOAA_STATION_CATALOG = os.path.join(_NOAA_STATION_DIR, "stations.csv")
_NOAA_ATMOSPHERE_DIR = os.path.join("data", "noaa_atmosphere")
_NOAA_ATMOSPHERE_REPAIRED_DIR = os.path.join(
    "data", "noaa_atmosphere_repaired"
)
if os.path.exists(_LOCATIONS_FILE):
    try:
        _LOCATIONS_DF = pd.read_csv(_LOCATIONS_FILE)
        _LOCATIONS_DF["station_id"] = _LOCATIONS_DF["station_id"].str.upper()
    except Exception as e:
        print(f"Warning: Could not load {_LOCATIONS_FILE}: {e}")

def _infer_station_details(code: str, path: str):
    preferred_path = os.path.join(_NOAA_STATION_DIR, f"{code.lower()}.csv")
    if os.path.exists(preferred_path):
        path = preferred_path
    # Try to load from dynamic locations CSV
    if _LOCATIONS_DF is not None and code in _LOCATIONS_DF["station_id"].values:
        row = _LOCATIONS_DF[_LOCATIONS_DF["station_id"] == code].iloc[0]
        return {
            "name": str(row["name"]),
            "region": str(row["region"]),
            "climate": str(row["climate"]),
            "elevation": float(row["elevation"]),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "ocean_basin": str(row["ocean_basin"]) if pd.notna(row["ocean_basin"]) else None,
            "csv_file": path
        }
    clean_name = code.replace('_', ' ').title()
    return {
        "name": f"{clean_name} ({code})",
        "region": "U.S. Climatological Grid",
        "climate": "Local Microclimate",
        "elevation": 150.0,
        "lat": 39.8,
        "lon": -98.5,  # geographic center of contiguous US
        "ocean_basin": "sst_atlantic_f",
        "csv_file": path
    }

CITIES = {}

if _LOCATIONS_DF is not None:
    for _, _row in _LOCATIONS_DF.iterrows():
        _code = str(_row["station_id"]).upper()
        _path = os.path.join(_NOAA_STATION_DIR, f"{_code.lower()}.csv")
        CITIES[_code] = _infer_station_details(_code, _path)

if os.path.exists(_NOAA_STATION_CATALOG):
    try:
        _noaa_catalog = pd.read_csv(_NOAA_STATION_CATALOG)
        for _, _row in _noaa_catalog.iterrows():
            _code = str(_row["station_id"]).upper()
            CITIES[_code] = {
                "name": str(_row["name"]),
                "region": str(_row.get("region", "NOAA Daily Station Network")),
                "climate": str(_row.get("climate", "Station observations")),
                "elevation": float(_row["elevation"]),
                "lat": float(_row["lat"]),
                "lon": float(_row["lon"]),
                "ocean_basin": (
                    str(_row["ocean_basin"])
                    if pd.notna(_row.get("ocean_basin"))
                    else None
                ),
                "csv_file": os.path.join(
                    _NOAA_STATION_DIR, str(_row["csv_file"])
                ),
            }
    except Exception as e:
        print(f"Warning: Could not load {_NOAA_STATION_CATALOG}: {e}")

# Fallback: if no CSVs are found in data/, populate with standard defaults
if not CITIES:
    CITIES["SEATTLE"] = _infer_station_details("SEATTLE", "data/noaa_stations/seattle.csv")


class NOAAClient:

    def __init__(self, token=None):
        self.token = token

    def fetch_weather_data(
        self, city_id: str, start_date: str, end_date: str,
        force_simulation: bool = False,
        allow_synthetic_fallback: bool = False,
    ) -> pd.DataFrame:
        city_key = city_id.upper()
        meta = CITIES.get(city_key, {})
        csv_file = meta.get("csv_file", f"data/{city_id.lower()}.csv")

        if not force_simulation and os.path.exists(csv_file):
            df = self._load_station_csv(csv_file, city_key, start_date, end_date)
            if df is not None and not df.empty:
                return df

        if force_simulation or allow_synthetic_fallback:
            return self._generate_synthetic(city_key, start_date, end_date)

        raise FileNotFoundError(
            f"No prepared NOAA station data is available for {city_key}. "
            "Run tools/prepare_noaa_daily.py with a NOAA daily export, or "
            "explicitly enable synthetic fallback."
        )

    def _load_station_csv(
        self, csv_file: str, city_key: str,
        start_date: str, end_date: str
    ) -> pd.DataFrame | None:
        """Loads and enriches a station CSV. Returns None on failure."""
        try:
            df = pd.read_csv(csv_file, low_memory=False)

            normalized_noaa = {"date", "TMAX", "TMIN", "PRCP"}.issubset(df.columns)

            # 1. Merge atmosphere data
            repaired_atmosphere = os.path.join(
                _NOAA_ATMOSPHERE_REPAIRED_DIR, f"{city_key.lower()}.csv"
            )
            station_atmosphere = (
                repaired_atmosphere
                if os.path.exists(repaired_atmosphere)
                else os.path.join(
                    _NOAA_ATMOSPHERE_DIR, f"{city_key.lower()}.csv"
                )
            )
            if os.path.exists(station_atmosphere):
                atmosphere = pd.read_csv(station_atmosphere)
                keep = [
                    "date",
                    "DEWPOINT_F",
                    "WINDSPEED_MPH",
                    "PRESSURE_HPA",
                    "HUMIDITY_PCT",
                    "ATMOSPHERE_IMPUTED",
                    "ATMOSPHERE_DONOR_KM",
                ]
                atmosphere = atmosphere[
                    [column for column in keep if column in atmosphere.columns]
                ]
                existing = [
                    column
                    for column in (
                        keep[1:]
                        + [
                            "dewpoint_f",
                            "windspeed_mph",
                            "pressure_hpa",
                            "humidity_pct",
                        ]
                    )
                    if column in df.columns
                ]
                df = df.drop(columns=existing)
                df = pd.merge(df, atmosphere, on="date", how="left")

            # 2. Merge soil moisture data
            station_soil = os.path.join(
                "data", "soil_moisture", f"{city_key.lower()}.csv"
            )
            if os.path.exists(station_soil):
                soil = pd.read_csv(station_soil)
                soil_keep = ["date", "SOIL_MOISTURE_VOL", "SATURATION_PCT"]
                soil = soil[[column for column in soil_keep if column in soil.columns]]
                soil["SOIL_MOISTURE_OBSERVED"] = (
                    soil[
                        [
                            column
                            for column in ["SOIL_MOISTURE_VOL", "SATURATION_PCT"]
                            if column in soil.columns
                        ]
                    ]
                    .notna()
                    .any(axis=1)
                    .astype(int)
                )
                df = df.drop(
                    columns=[
                        column
                        for column in soil_keep[1:] + ["SOIL_MOISTURE_OBSERVED"]
                        if column in df.columns
                    ],
                    errors="ignore",
                )
                df = pd.merge(df, soil, on="date", how="left")
            elif "SOIL_MOISTURE_OBSERVED" not in df.columns:
                df["SOIL_MOISTURE_OBSERVED"] = 0

            # 3. Merge climate indices
            if os.path.exists("data/climate_indices.csv"):
                ci = pd.read_csv("data/climate_indices.csv")
                # drop enso_phase text column - keeps numeric indices only
                ci = ci.drop(columns=[c for c in ci.columns if "phase" in c.lower()], errors="ignore")
                df = pd.merge(df, ci, on="date", how="left")
                index_columns = [
                    column
                    for column in ["enso_nino34", "pdo_index", "nao_index"]
                    if column in df.columns
                ]
                # Climate indices are lower-frequency context variables. Carry
                # their most recent observation within the available archive,
                # while retaining an explicit freshness indicator.
                if index_columns:
                    observed = df[index_columns].notna().any(axis=1)
                    df["CLIMATE_INDEX_OBSERVED"] = observed.astype(int)
                    df[index_columns] = df[index_columns].ffill().bfill()

            # 4. Convert dates
            df["Date"] = pd.to_datetime(df["date"])

            for yr_col, new_col in [
                ("record_max_temp_year", "REC_MAX_YR"),
                ("record_min_temp_year", "REC_MIN_YR"),
            ]:
                if yr_col in df.columns:
                    df[new_col] = df[yr_col]

            # 5. Normalise atmosphere column names
            rename_map = {
                "dewpoint_f":      "DEWPOINT_F",
                "windspeed_mph":   "WINDSPEED_MPH",
                "pressure_hpa":    "PRESSURE_HPA",
                "humidity_pct":    "HUMIDITY_PCT",
                "soil_moisture_vol": "SOIL_MOISTURE_VOL",
                "saturation_pct":  "SATURATION_PCT",
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

            # 6. Date filter
            mask = (
                (df["Date"] >= pd.to_datetime(start_date)) &
                (df["Date"] <= pd.to_datetime(end_date))
            )
            out = df.loc[mask].copy().sort_values("Date").reset_index(drop=True)
            archived_window = False
            if out.empty and normalized_noaa:
                requested_days = max(
                    1, (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
                )
                latest_date = df["Date"].max()
                archive_start = latest_date - pd.Timedelta(days=requested_days)
                out = (
                    df.loc[df["Date"].between(archive_start, latest_date)]
                    .copy()
                    .sort_values("Date")
                    .reset_index(drop=True)
                )
                archived_window = not out.empty
            if out.empty:
                return None

            # 7. Housekeeping
            source = "NOAA Daily Station Archive"
            if archived_window:
                source += " (latest available window)"
            out["Source"] = source
            out["Data_End_Date"] = df["Date"].max().strftime("%Y-%m-%d")
            out["City_ID"] = city_key
            out["City_Name"] = CITIES.get(city_key, {}).get("name", city_key)

            drop_cols = [
                "actual_max_temp", "actual_min_temp", "actual_precipitation",
                "average_max_temp", "average_min_temp",
                "record_max_temp", "record_min_temp",
                "record_max_temp_year", "record_min_temp_year",
                "record_precipitation", "date", "data_source",
            ]
            out = out.drop(columns=[c for c in drop_cols if c in out.columns])
            return out

        except Exception as exc:
            print(f"[NOAAClient] Error loading {csv_file}: {exc}")
            return None

    def _generate_synthetic(
        self, city_key: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Physics-informed seasonal synthetic generator.
        Includes a +0.02 C / year warming trend per IPCC AR6.
        """
        dates = pd.date_range(
            pd.to_datetime(start_date), pd.to_datetime(end_date)
        )
        meta = CITIES.get(city_key, {})
        np.random.seed(hash(city_key) % (2 ** 32))

        # Determine generator parameters dynamically based on inferred climate
        climate_str = meta.get("climate", "").lower()
        basin_str = meta.get("ocean_basin", "")
        basin_str = basin_str.lower() if basin_str else ""
        
        if "desert" in climate_str or "hot" in climate_str:
            p = {"base": 26.0, "amp": 11.0, "prcp_p": 0.08, "prcp_scale": 5.0}
        elif "oceanic" in climate_str or "temperate" in climate_str or "pacific" in basin_str:
            p = {"base": 11.0, "amp": 7.0, "prcp_p": 0.52, "prcp_scale": 3.0}
        elif "continental" in climate_str or "atlantic" in basin_str:
            p = {"base": 13.0, "amp": 12.0, "prcp_p": 0.36, "prcp_scale": 4.0}
        else:
            p = {"base": 15.0, "amp": 9.0, "prcp_p": 0.28, "prcp_scale": 4.0}

        records = []
        ref_year = dates[0].year
        for date in dates:
            doy = date.dayofyear
            yr_offset = (date.year - ref_year) + doy / 365.0
            seasonal = -p["amp"] * np.cos(2 * np.pi * (doy - 15) / 365)
            warming   = 0.02 * yr_offset
            t_base    = p["base"] + seasonal + warming

            tmax = round(t_base + 6.0 + 3.0 * np.random.randn(), 1)
            tmin = round(t_base - 6.0 + 3.0 * np.random.randn(), 1)
            prcp = round(
                np.random.exponential(p["prcp_scale"]) if np.random.rand() < p["prcp_p"] else 0.0,
                1,
            )
            records.append({
                "Date": date,
                "TMAX": tmax, "TMIN": tmin, "PRCP": prcp,
                "AVG_MAX": round(t_base + 5.0, 1),
                "AVG_MIN": round(t_base - 5.0, 1),
                "REC_MAX": round(t_base + 15.0, 1),
                "REC_MIN": round(t_base - 15.0, 1),
                "REC_MAX_YR": 2012, "REC_MIN_YR": 1985,
                "REC_PRCP": 25.4,
                # Synthetic defaults
                "DEWPOINT_F":     round((t_base * 9/5 + 32) - 11.0, 1),
                "WINDSPEED_MPH":  round(abs(np.random.normal(9, 3)), 1),
                "PRESSURE_HPA":   round(1013.25 + np.random.normal(0, 6), 1),
                "HUMIDITY_PCT":   round(np.clip(60 + np.random.normal(0, 12), 10, 100), 1),
                "SOIL_MOISTURE_VOL": round(np.clip(0.25 + np.random.normal(0, 0.07), 0.02, 0.50), 4),
                "SATURATION_PCT":    round(np.clip(50 + np.random.normal(0, 15), 4, 100), 1),
                "enso_nino34": round(np.random.normal(0, 0.8), 3),
                "pdo_index":   round(np.random.normal(0, 0.6), 3),
                "nao_index":   round(np.random.normal(0, 0.7), 3),
                "Source": "Synthetic Climate Generator",
                "City_ID": city_key,
                "City_Name": meta.get("name", city_key),
            })

        return pd.DataFrame(records)
