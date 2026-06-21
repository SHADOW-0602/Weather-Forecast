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
if os.path.exists(_LOCATIONS_FILE):
    try:
        _LOCATIONS_DF = pd.read_csv(_LOCATIONS_FILE)
        _LOCATIONS_DF["station_id"] = _LOCATIONS_DF["station_id"].str.upper()
    except Exception as e:
        print(f"Warning: Could not load {_LOCATIONS_FILE}: {e}")

def _infer_station_details(code: str, path: str):
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

for _path in glob.glob("data/*.csv"):
    _code = os.path.splitext(os.path.basename(_path))[0].upper()
    if _code.lower() in _AUX_FILES:
        continue
    CITIES[_code] = _infer_station_details(_code, _path)

# Fallback: if no CSVs are found in data/, populate with standard defaults
if not CITIES:
    CITIES["SEATTLE"] = _infer_station_details("SEATTLE", "data/seattle.csv")


class NOAAClient:

    def __init__(self, token=None):
        self.token = token

    def fetch_weather_data(
        self, city_id: str, start_date: str, end_date: str,
        force_simulation: bool = False
    ) -> pd.DataFrame:
        city_key = city_id.upper()
        meta = CITIES.get(city_key, {})
        csv_file = meta.get("csv_file", f"data/{city_id.lower()}.csv")

        if not force_simulation and os.path.exists(csv_file):
            df = self._load_station_csv(csv_file, city_key, start_date, end_date)
            if df is not None and not df.empty:
                return df

        return self._generate_synthetic(city_key, start_date, end_date)

    def _load_station_csv(
        self, csv_file: str, city_key: str,
        start_date: str, end_date: str
    ) -> pd.DataFrame | None:
        """Loads and enriches a station CSV. Returns None on failure."""
        try:
            df = pd.read_csv(csv_file, low_memory=False)

            # -- 1. Merge atmosphere.csv (dew point, wind, pressure, humidity) --
            df = self._merge_aux(df, "data/atmosphere.csv", city_key)

            # -- 2. Merge soil_moisture.csv (or legacy region.csv) --
            if os.path.exists("data/soil_moisture.csv"):
                df = self._merge_aux(df, "data/soil_moisture.csv", city_key)
            elif os.path.exists("data/region.csv"):
                df = self._merge_aux(df, "data/region.csv", city_key)

            # -- 3. Merge climate_indices.csv (ENSO, PDO, NAO) --
            if os.path.exists("data/climate_indices.csv"):
                ci = pd.read_csv("data/climate_indices.csv")
                # drop enso_phase text column - keeps numeric indices only
                ci = ci.drop(columns=[c for c in ci.columns if "phase" in c.lower()], errors="ignore")
                df = pd.merge(df, ci, on="date", how="left")

            # -- 4. Convert dates and shift to requested window --
            df["Date"] = pd.to_datetime(df["date"])
            latest_csv = df["Date"].max()
            requested_end = pd.to_datetime(end_date)
            df["Date"] = df["Date"] + (requested_end - latest_csv)

            # -- 5. Unit conversion: Fahrenheit -> Celsius, inches -> mm --
            for col_f, col_c in [
                ("actual_max_temp", "TMAX"),
                ("actual_min_temp", "TMIN"),
                ("average_max_temp", "AVG_MAX"),
                ("average_min_temp", "AVG_MIN"),
                ("record_max_temp", "REC_MAX"),
                ("record_min_temp", "REC_MIN"),
            ]:
                if col_f in df.columns:
                    df[col_c] = (df[col_f] - 32) * 5.0 / 9.0

            if "actual_precipitation" in df.columns:
                df["PRCP"] = df["actual_precipitation"] * 25.4   # inch -> mm
            if "record_precipitation" in df.columns:
                df["REC_PRCP"] = df["record_precipitation"] * 25.4

            for yr_col, new_col in [
                ("record_max_temp_year", "REC_MAX_YR"),
                ("record_min_temp_year", "REC_MIN_YR"),
            ]:
                if yr_col in df.columns:
                    df[new_col] = df[yr_col]

            # -- 6. Normalise atmosphere column names --
            rename_map = {
                "dewpoint_f":      "DEWPOINT_F",
                "windspeed_mph":   "WINDSPEED_MPH",
                "pressure_hpa":    "PRESSURE_HPA",
                "humidity_pct":    "HUMIDITY_PCT",
                "soil_moisture_vol": "SOIL_MOISTURE_VOL",
                "saturation_pct":  "SATURATION_PCT",
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

            # -- 7. Date filter --
            mask = (
                (df["Date"] >= pd.to_datetime(start_date)) &
                (df["Date"] <= pd.to_datetime(end_date))
            )
            out = df.loc[mask].copy().sort_values("Date").reset_index(drop=True)
            if out.empty:
                return None

            # -- 8. Housekeeping --
            out["Source"] = "Kaggle US Weather Dataset"
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

    def _merge_aux(
        self, df: pd.DataFrame, aux_file: str, city_key: str
    ) -> pd.DataFrame:
        try:
            aux = pd.read_csv(aux_file)
            prefix = city_key.lower() + "_"
            cols = ["date"] + [
                c for c in aux.columns
                if c.lower().startswith(prefix)
            ]
            if len(cols) <= 1:
                return df   # no city-specific columns found

            aux_sub = aux[cols].copy()
            # strip prefix from column names
            aux_sub = aux_sub.rename(columns={
                c: c[len(prefix):] for c in aux_sub.columns if c != "date"
            })
            return pd.merge(df, aux_sub, on="date", how="left")
        except Exception as exc:
            print(f"[NOAAClient] Could not merge {aux_file}: {exc}")
            return df

    def _generate_synthetic(
        self, city_key: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Physics-informed seasonal synthetic generator.
        Includes a +0.02  C / year warming trend per IPCC AR6.
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
                # Synthetic atmospheric defaults
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
