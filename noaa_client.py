import os
import glob
import datetime
import pandas as pd
import numpy as np

# Detailed metadata for the standard stations
STATION_METADATA = {
    "SEATTLE": {"name": "Seattle, WA (SEATTLE)", "region": "Pacific Northwest", "climate": "Temperate Oceanic", "elevation": 137, "lat": 47.45, "lon": -122.31},
    "NEW_YORK": {"name": "New York, NY (NEW_YORK)", "region": "Mid-Atlantic", "climate": "Humid Subtropical", "elevation": 10, "lat": 40.78, "lon": -73.97},
    "PHOENIX": {"name": "Phoenix, AZ (PHOENIX)", "region": "Southwest Desert", "climate": "Hot Desert", "elevation": 344, "lat": 33.43, "lon": -112.01},
    "KMDW": {"name": "Chicago, IL (KMDW)", "region": "Midwest", "climate": "Humid Continental", "elevation": 186, "lat": 41.78, "lon": -87.75},
    "KHOU": {"name": "Houston, TX (KHOU)", "region": "Gulf Coast", "climate": "Humid Subtropical", "elevation": 14, "lat": 29.65, "lon": -95.28},
    "KJAX": {"name": "Jacksonville, FL (KJAX)", "region": "Southeast", "climate": "Humid Subtropical", "elevation": 9, "lat": 30.49, "lon": -81.69},
    "KCLT": {"name": "Charlotte, NC (KCLT)", "region": "Southeast", "climate": "Humid Subtropical", "elevation": 222, "lat": 35.21, "lon": -80.94},
    "KCQT": {"name": "Los Angeles, CA (KCQT)", "region": "West Coast", "climate": "Mediterranean", "elevation": 87, "lat": 34.02, "lon": -118.29},
    "KIND": {"name": "Indianapolis, IN (KIND)", "region": "Midwest", "climate": "Humid Continental", "elevation": 242, "lat": 39.72, "lon": -86.27},
    "KPHL": {"name": "Philadelphia, PA (KPHL)", "region": "Mid-Atlantic", "climate": "Humid Subtropical", "elevation": 9, "lat": 39.87, "lon": -75.24}
}

# DYNAMIC LOADER: Scan the workspace directory for CSV datasets on-the-fly
CITIES = {}
for path in glob.glob("*.csv"):
    code = os.path.splitext(os.path.basename(path))[0].upper()
    if code in ["OCEAN", "OCEAN_MONTHLY", "ATMOSPHERE", "CLIMATE_INDICES", "REGION"]:
        continue
    
    if code in STATION_METADATA:
        CITIES[code] = STATION_METADATA[code]
    else:
        CITIES[code] = {
            "name": f"Station {code}",
            "region": "US Climatological Grid",
            "climate": "Local Microclimate",
            "elevation": 150,
            "lat": 35.0,
            "lon": -95.0
        }

# Fallback to defaults if no station CSVs are present
if not CITIES:
    CITIES = STATION_METADATA

class NOAAClient:
    def __init__(self, token=None):
        self.token = token

    def fetch_weather_data(self, city_id, start_date, end_date, force_simulation=False):
        """Loads weather data dynamically from discovered local CSV datasets."""
        csv_file = f"{city_id.lower()}.csv"
        if not force_simulation and os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                # Merge auxiliary datasets dynamically
                if os.path.exists("atmosphere.csv"):
                    adv_df = pd.read_csv("atmosphere.csv")
                    cols = ["date"] + [c for c in adv_df.columns if c.lower().startswith(f"{city_id.lower()}_")]
                    if len(cols) > 1:
                        adv_df = adv_df[cols].rename(columns={c: c.split("_", 1)[1].upper() for c in cols if c != "date"})
                        df = pd.merge(df, adv_df, on="date", how="left")
                        
                if os.path.exists("region.csv"):
                    soil_df = pd.read_csv("region.csv")
                    cols = ["date"] + [c for c in soil_df.columns if c.lower().startswith(f"{city_id.lower()}_")]
                    if len(cols) > 1:
                        soil_df = soil_df[cols].rename(columns={c: c.split("_", 1)[1].upper() for c in cols if c != "date"})
                        df = pd.merge(df, soil_df, on="date", how="left")
                        
                if os.path.exists("climate_indices.csv"):
                    clim_df = pd.read_csv("climate_indices.csv")
                    df = pd.merge(df, clim_df, on="date", how="left")

                # Parse Date and shift timeline to match requested range (dynamic projections)
                df["Date"] = pd.to_datetime(df["date"])
                latest_csv_date = df["Date"].max()
                requested_end = pd.to_datetime(end_date)
                
                # Shift dates so the end of the historical CSV aligns with the requested end date
                df["Date"] = df["Date"] + (requested_end - latest_csv_date)
                
                # Convert Fahrenheit (Kaggle standard) to Celsius
                df["TMAX"] = (df["actual_max_temp"] - 32) * 5.0 / 9.0
                df["TMIN"] = (df["actual_min_temp"] - 32) * 5.0 / 9.0
                df["PRCP"] = df["actual_precipitation"] * 25.4
                
                # Parse additional interesting historical metrics
                df["AVG_MAX"] = (df["average_max_temp"] - 32) * 5.0 / 9.0
                df["AVG_MIN"] = (df["average_min_temp"] - 32) * 5.0 / 9.0
                df["REC_MAX"] = (df["record_max_temp"] - 32) * 5.0 / 9.0
                df["REC_MIN"] = (df["record_min_temp"] - 32) * 5.0 / 9.0
                df["REC_MAX_YR"] = df["record_max_temp_year"]
                df["REC_MIN_YR"] = df["record_min_temp_year"]
                df["REC_PRCP"] = df["record_precipitation"] * 25.4
                
                # Filter by date range
                mask = (df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))
                filtered = df.loc[mask].copy().sort_values("Date").reset_index(drop=True)
                
                if not filtered.empty:
                    filtered["Source"] = "Kaggle US Weather Dataset"
                    filtered["City_ID"] = city_id
                    filtered["City_Name"] = CITIES[city_id.upper()]["name"]
                    
                    cols_to_drop = [
                        "actual_max_temp", "actual_min_temp", "actual_precipitation", 
                        "average_max_temp", "average_min_temp", "record_max_temp", "record_min_temp",
                        "record_max_temp_year", "record_min_temp_year", "record_precipitation", "date",
                        "data_source"
                    ]
                    filtered = filtered.drop(columns=[c for c in cols_to_drop if c in filtered.columns])
                    return filtered
            except Exception:
                pass
                
        return self.generate_synthetic_data(city_id, start_date, end_date)

    def generate_synthetic_data(self, city_id, start_date, end_date):
        """Fallback generator in case a local CSV is missing or corrupted."""
        dates = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date))
        city = CITIES.get(city_id.upper(), {"name": f"Station {city_id}"})
        
        np.random.seed(hash(city_id) % (2**32))
        records = []
        
        for date in dates:
            day_of_year = date.dayofyear
            seasonal_shift = -8.0 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
            t_base = 15.0 + seasonal_shift
            
            tmax = t_base + 6.0 + 3.0 * np.random.randn()
            tmin = t_base - 6.0 + 3.0 * np.random.randn()
            prcp = round(np.random.exponential(4.0), 1) if np.random.rand() < 0.25 else 0.0
            
            records.append({
                "Date": date, "TMAX": round(tmax, 1), "TMIN": round(tmin, 1), "PRCP": prcp,
                "Source": "Simulated Weather", "City_ID": city_id, "City_Name": city["name"],
                "AVG_MAX": round(t_base + 5.0, 1), "AVG_MIN": round(t_base - 5.0, 1),
                "REC_MAX": round(t_base + 15.0, 1), "REC_MIN": round(t_base - 15.0, 1),
                "REC_MAX_YR": 2012, "REC_MIN_YR": 1985, "REC_PRCP": 25.4
            })
            
        return pd.DataFrame(records)
