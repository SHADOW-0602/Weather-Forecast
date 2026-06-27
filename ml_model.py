import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings/infos
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)

RF_FEATURES = [
    "HEAT_INDEX", "WIND_CHILL", "WET_BULB", "DROUGHT_IDX",
    "TEMP_ANOM", "PRCP_ANOM", "EXTREME_HEAT", "EXTREME_COLD",
    "SST", "SST_AIR_DIFF", "FLOOD_RISK_IDX", "WIND_RISK_IDX",
    "PRESSURE_HPA", "HUMIDITY_PCT", "SOIL_MOISTURE_VOL",
    "SATURATION_PCT", "enso_nino34", "pdo_index", "nao_index",
    "LATITUDE", "LONGITUDE", "ELEVATION",
    "ATMOSPHERE_IMPUTED", "ATMOSPHERE_DONOR_KM",
    "SOIL_MOISTURE_OBSERVED", "CLIMATE_INDEX_OBSERVED",
    "PRCP_3D", "PRCP_7D", "PRCP_14D", "PRCP_30D", "PRCP_MAX_7D",
    "WIND_MAX_7D", "PRESSURE_DROP_3D", "SATURATION_MAX_7D",
    "SOIL_MOISTURE_ANOM_30D", "TEMP_ANOM_7D",
]

TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TensorFlow is not available ({e}). The system will fallback to pure Random Forest predictions.")
class HazardPredictor:
    def __init__(self, model_dir="saved_models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.rf_model = None
        self.lstm_model = None
        self.fusion_model = None
        self.scaler_lstm = None
        self.scaler_rf = None
        self.event_model = None
        self.scaler_event = None
        self.event_threshold = 0.5
        self.metrics = None
        self._load_models()

    def _load_models(self):
        """Loads the trained models and scalers from disk."""
        rf_path = os.path.join(self.model_dir, "rf_model.pkl")
        lstm_path = os.path.join(self.model_dir, "lstm_final.keras")
        fusion_path = os.path.join(self.model_dir, "fusion_final.keras")
        sl_path = os.path.join(self.model_dir, "scaler_lstm.pkl")
        sr_path = os.path.join(self.model_dir, "scaler_rf.pkl")
        event_path = os.path.join(self.model_dir, "event_rf_model.pkl")
        event_scaler_path = os.path.join(self.model_dir, "scaler_event.pkl")
        event_threshold_path = os.path.join(self.model_dir, "event_threshold.json")
        metrics_path = os.path.join(self.model_dir, "metrics.json")

        if all(os.path.exists(p) for p in [rf_path, sl_path, sr_path]):
            try:
                self.rf_model = joblib.load(rf_path)
                self.scaler_lstm = joblib.load(sl_path)
                self.scaler_rf = joblib.load(sr_path)
            except Exception as e:
                print(f"Error loading Random Forest artifacts: {e}")
                self.rf_model = None

            if TF_AVAILABLE and os.path.exists(lstm_path) and os.path.exists(fusion_path):
                try:
                    tf.get_logger().setLevel('ERROR')
                    self.lstm_model = tf.keras.models.load_model(lstm_path)
                    self.fusion_model = tf.keras.models.load_model(fusion_path)
                except Exception as e:
                    print(f"Neural models unavailable; using Random Forest only: {e}")
                    self.lstm_model = None
                    self.fusion_model = None

            try:
                if os.path.exists(metrics_path):
                    import json
                    with open(metrics_path, "r") as f:
                        self.metrics = json.load(f)
                if os.path.exists(event_path) and os.path.exists(event_scaler_path):
                    self.event_model = joblib.load(event_path)
                    self.scaler_event = joblib.load(event_scaler_path)
                if os.path.exists(event_threshold_path):
                    import json
                    with open(event_threshold_path, "r") as f:
                        self.event_threshold = float(
                            json.load(f).get("threshold", 0.5)
                        )
            except Exception as e:
                print(f"Error loading model metrics: {e}")
        else:
            self.rf_model = None

    def _add_sst_feature(self, df):
        """Merges SST time-series data into the weather dataframe."""
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"])
            
        daily_path = "data/ocean.csv"
        monthly_path = "data/ocean_monthly.csv"
        
        # 1. Try Daily Resolution Merge
        if os.path.exists(daily_path):
            try:
                sst_df = pd.read_csv(daily_path)
                
                # Check column casing for date
                sst_date_col = "date" if "date" in sst_df.columns else "Date" if "Date" in sst_df.columns else sst_df.columns[0]
                sst_df["Date_Str"] = pd.to_datetime(sst_df[sst_date_col]).dt.strftime("%Y-%m-%d")
                
                # Determine City_ID
                city_id = "SEATTLE" # Default fallback
                if "City_ID" in df.columns and len(df) > 0:
                    city_id = str(df["City_ID"].iloc[0]).upper()
                
                # Map City_ID to appropriate ocean basin column
                basin_mapping = {
                    "SEATTLE":  "sst_pacific_f",
                    "KCQT":    "sst_pacific_f",
                    "NEW_YORK": "sst_atlantic_f",
                    "KPHL":    "sst_atlantic_f",
                    "KJAX":    "sst_atlantic_f",
                    "KCLT":    "sst_atlantic_f",
                    "KHOU":    "sst_gulf_f",
                    "PHOENIX": "sst_gulf_f",
                    "KIND":    "sst_atlantic_f",
                    "KMDW":    "sst_atlantic_f",
                }
                
                target_col = basin_mapping.get(city_id)
                if target_col is None and {"LATITUDE", "LONGITUDE"}.issubset(df.columns):
                    latitude = float(pd.to_numeric(df["LATITUDE"], errors="coerce").median())
                    longitude = float(pd.to_numeric(df["LONGITUDE"], errors="coerce").median())
                    if longitude <= -100.0:
                        target_col = "sst_pacific_f"
                    elif latitude <= 31.5 and longitude <= -80.0:
                        target_col = "sst_gulf_f"
                    else:
                        target_col = "sst_atlantic_f"
                if target_col and target_col in sst_df.columns:
                    # Convert Fahrenheit to Celsius
                    sst_df["SST_Mean_Celsius"] = (sst_df[target_col] - 32) * 5.0 / 9.0
                else:
                    # Fallback: average all available ocean basins and convert to Celsius
                    avail_cols = [c for c in ["sst_pacific_f", "sst_atlantic_f", "sst_gulf_f"] if c in sst_df.columns]
                    if avail_cols:
                        avg_f = sst_df[avail_cols].mean(axis=1)
                        sst_df["SST_Mean_Celsius"] = (avg_f - 32) * 5.0 / 9.0
                    elif "SST_Mean_Celsius" in sst_df.columns:
                        pass
                    else:
                        sst_df["SST_Mean_Celsius"] = 18.0 # ultimate constant fallback
                
                sst_df = sst_df[["Date_Str", "SST_Mean_Celsius"]]
                
                df["Date_Str"] = df["Date"].dt.strftime("%Y-%m-%d")
                merged = df.merge(sst_df, on="Date_Str", how="left")
                merged["SST"] = merged["SST_Mean_Celsius"].ffill().bfill()
                
                if merged["SST"].isna().any():
                    fallback_mean = sst_df["SST_Mean_Celsius"].mean()
                    if pd.isna(fallback_mean):
                        fallback_mean = 18.0
                    merged["SST"] = merged["SST"].fillna(fallback_mean)
                    
                merged = merged.drop(columns=["Date_Str", "SST_Mean_Celsius"])
                return merged
            except Exception as e:
                print(f"Error merging daily SST data: {e}")
                
        # 2. Fallback to Monthly Resolution Merge
        elif os.path.exists(monthly_path):
            try:
                sst_df = pd.read_csv(monthly_path)
                sst_df["Year"] = sst_df["Year"].astype(int)
                sst_df["Month"] = sst_df["Month"].astype(int)
                sst_df = sst_df[["Year", "Month", "SST_Mean_Celsius"]]
                
                df["Year"] = df["Date"].dt.year
                df["Month"] = df["Date"].dt.month
                merged = df.merge(sst_df, on=["Year", "Month"], how="left")
                merged["SST"] = merged["SST_Mean_Celsius"].ffill().bfill()
                
                if merged["SST"].isna().any():
                    fallback_mean = sst_df["SST_Mean_Celsius"].mean()
                    if pd.isna(fallback_mean):
                        fallback_mean = 18.0
                    merged["SST"] = merged["SST"].fillna(fallback_mean)
                    
                merged = merged.drop(columns=["Year", "Month", "SST_Mean_Celsius"])
                return merged
            except Exception as e:
                print(f"Error merging monthly SST data: {e}")
                
        # 3. Ultimate Fallback (Constants)
        df["SST"] = 18.0
        return df

    def _add_event_labels(self, df):
        """Attach independent NOAA Storm Events station-day labels when present.

        The labels are sparse and currently cover the 2022-2024 test period.
        Training can still use the engineered next-day proxy before that period,
        while the test split can report metrics against these independent events.
        """
        df = df.copy()
        df["EVENT_LABEL_AVAILABLE"] = 0
        df["EVENT_LABEL"] = np.nan
        df["EVENT_TYPE"] = ""
        df["EVENT_DISTANCE_KM"] = np.nan

        path = os.path.join("data", "event_labels.csv")
        if not os.path.exists(path) or "City_ID" not in df.columns:
            return df

        try:
            labels = pd.read_csv(path)
            required = {"date", "NOAA_STATION", "EVENT_LABEL"}
            if not required.issubset(labels.columns):
                return df

            labels["Date"] = pd.to_datetime(labels["date"], errors="coerce")
            labels["City_ID"] = labels["NOAA_STATION"].astype(str).str.upper()
            labels["EVENT_LABEL"] = pd.to_numeric(
                labels["EVENT_LABEL"], errors="coerce"
            ).fillna(0).astype(int)

            aggregations = {"EVENT_LABEL": "max"}
            if "EVENT_TYPE" in labels.columns:
                aggregations["EVENT_TYPE"] = (
                    lambda values: "; ".join(
                        sorted({str(value) for value in values if pd.notna(value)})
                    )
                )
            if "EVENT_DISTANCE_KM" in labels.columns:
                labels["EVENT_DISTANCE_KM"] = pd.to_numeric(
                    labels["EVENT_DISTANCE_KM"], errors="coerce"
                )
                aggregations["EVENT_DISTANCE_KM"] = "min"

            labels = (
                labels.dropna(subset=["Date"])
                .groupby(["Date", "City_ID"], as_index=False)
                .agg(aggregations)
            )
            covered_stations = set(labels["City_ID"].unique())
            coverage_start = labels["Date"].min()
            coverage_end = labels["Date"].max()
            labels["EVENT_LABEL_AVAILABLE"] = 1

            merged = df.merge(
                labels[
                    [
                        column
                        for column in [
                            "Date",
                            "City_ID",
                            "EVENT_LABEL",
                            "EVENT_LABEL_AVAILABLE",
                            "EVENT_TYPE",
                            "EVENT_DISTANCE_KM",
                        ]
                        if column in labels.columns
                    ]
                ],
                on=["Date", "City_ID"],
                how="left",
                suffixes=("", "_EVENT"),
            )
            merged = merged.assign(
                EVENT_LABEL=lambda out: out["EVENT_LABEL_EVENT"].combine_first(
                    out["EVENT_LABEL"]
                )
                if "EVENT_LABEL_EVENT" in out.columns
                else out["EVENT_LABEL"],
                EVENT_LABEL_AVAILABLE=lambda out: out[
                    "EVENT_LABEL_AVAILABLE_EVENT"
                ].combine_first(out["EVENT_LABEL_AVAILABLE"])
                if "EVENT_LABEL_AVAILABLE_EVENT" in out.columns
                else out["EVENT_LABEL_AVAILABLE"],
                EVENT_TYPE=lambda out: out["EVENT_TYPE_EVENT"].combine_first(
                    out["EVENT_TYPE"]
                )
                if "EVENT_TYPE_EVENT" in out.columns
                else out["EVENT_TYPE"],
                EVENT_DISTANCE_KM=lambda out: out[
                    "EVENT_DISTANCE_KM_EVENT"
                ].combine_first(out["EVENT_DISTANCE_KM"])
                if "EVENT_DISTANCE_KM_EVENT" in out.columns
                else out["EVENT_DISTANCE_KM"],
            )
            coverage_mask = (
                merged["City_ID"].astype(str).str.upper().isin(covered_stations)
                & (merged["Date"] >= coverage_start)
                & (merged["Date"] <= coverage_end)
            )
            merged.loc[coverage_mask, "EVENT_LABEL_AVAILABLE"] = 1
            merged.loc[coverage_mask, "EVENT_LABEL"] = merged.loc[
                coverage_mask, "EVENT_LABEL"
            ].fillna(0)
            merged["EVENT_LABEL_AVAILABLE"] = pd.to_numeric(
                merged["EVENT_LABEL_AVAILABLE"], errors="coerce"
            ).fillna(0).astype(int)
            return merged.drop(
                columns=[
                    "EVENT_LABEL_EVENT",
                    "EVENT_LABEL_AVAILABLE_EVENT",
                    "EVENT_TYPE_EVENT",
                    "EVENT_DISTANCE_KM_EVENT",
                ],
                errors="ignore",
            )
        except Exception as exc:
            print(f"Warning: could not merge event labels: {exc}")
            return df

    @staticmethod
    def _safe_auc(y_true, y_score):
        y_true = np.asarray(y_true)
        if len(np.unique(y_true)) < 2:
            return 0.5
        return float(roc_auc_score(y_true, y_score))

    @staticmethod
    def _safe_average_precision(y_true, y_score):
        y_true = np.asarray(y_true)
        if len(np.unique(y_true)) < 2:
            return float(np.mean(y_true)) if len(y_true) else 0.0
        return float(average_precision_score(y_true, y_score))

    @staticmethod
    def _best_threshold_by_f1(y_true, y_score):
        y_true = np.asarray(y_true).astype(int)
        y_score = np.asarray(y_score)
        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            return 0.5, 0.0
        thresholds = np.linspace(0.05, 0.95, 91)
        scores = [
            f1_score(y_true, (y_score >= threshold).astype(int), zero_division=0)
            for threshold in thresholds
        ]
        best_index = int(np.argmax(scores))
        return float(thresholds[best_index]), float(scores[best_index])

    def _train_event_model(self, df_rf, rf_feats, rf_estimators):
        """Train a separate NOAA event/no-event classifier on coverage years."""
        event_df = df_rf[
            pd.to_numeric(
                df_rf.get("EVENT_LABEL_AVAILABLE", pd.Series(0, index=df_rf.index)),
                errors="coerce",
            )
            .fillna(0)
            .astype(bool)
        ].copy()
        if event_df.empty or "Date" not in event_df.columns:
            return {
                "event_model_trained": False,
                "event_model_reason": "No NOAA event coverage rows available.",
            }

        event_df["Date"] = pd.to_datetime(event_df["Date"])
        event_df["EVENT_LABEL"] = pd.to_numeric(
            event_df["EVENT_LABEL"], errors="coerce"
        ).fillna(0).astype(int)

        train_event = event_df[event_df["Date"] < "2023-01-01"].copy()
        val_event = event_df[
            (event_df["Date"] >= "2023-01-01")
            & (event_df["Date"] < "2024-01-01")
        ].copy()
        test_event = event_df[event_df["Date"] >= "2024-01-01"].copy()

        if (
            min(len(train_event), len(val_event), len(test_event)) < 30
            or train_event["EVENT_LABEL"].nunique() < 2
        ):
            return {
                "event_model_trained": False,
                "event_model_reason": (
                    "NOAA event split is too small or lacks both classes."
                ),
                "event_model_rows": int(len(event_df)),
            }

        self.scaler_event = MinMaxScaler()
        X_train = self.scaler_event.fit_transform(train_event[rf_feats])
        X_val = self.scaler_event.transform(val_event[rf_feats])
        X_test = self.scaler_event.transform(test_event[rf_feats])
        y_train = train_event["EVENT_LABEL"].values
        y_val = val_event["EVENT_LABEL"].values
        y_test = test_event["EVENT_LABEL"].values

        self.event_model = RandomForestClassifier(
            n_estimators=rf_estimators,
            max_depth=12,
            min_samples_leaf=3,
            random_state=143,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        self.event_model.fit(X_train, y_train)

        val_prob = self.event_model.predict_proba(X_val)[:, 1]
        threshold, val_f1 = self._best_threshold_by_f1(y_val, val_prob)
        self.event_threshold = threshold

        test_prob = self.event_model.predict_proba(X_test)[:, 1]
        test_pred = (test_prob >= threshold).astype(int)
        return {
            "event_model_trained": True,
            "event_model_threshold": threshold,
            "event_model_val_f1": val_f1,
            "event_model_train_rows": int(len(train_event)),
            "event_model_validation_rows": int(len(val_event)),
            "event_model_test_rows": int(len(test_event)),
            "event_model_test_accuracy": float(accuracy_score(y_test, test_pred)),
            "event_model_test_precision": float(
                precision_score(y_test, test_pred, zero_division=0)
            ),
            "event_model_test_recall": float(
                recall_score(y_test, test_pred, zero_division=0)
            ),
            "event_model_test_f1": float(
                f1_score(y_test, test_pred, zero_division=0)
            ),
            "event_model_test_auc": self._safe_auc(y_test, test_prob),
            "event_model_test_average_precision": self._safe_average_precision(
                y_test, test_prob
            ),
            "event_model_test_positive_rate": float(np.mean(y_test)),
        }

    def _engineer_features_bulk(self, df):
        """Engineers features for the entire dataframe for training RF."""
        df = df.copy()
        
        # Ensure base columns exist
        if 'TAVG' not in df.columns:
            df['TAVG'] = (df['TMAX'] + df['TMIN']) / 2
        
        T = df['TAVG']
        
        # Fill missing new features with defaults so it doesn't break if files are missing
        defaults = {
            "DEWPOINT_F": (T * 9/5 + 32) - 15.0,  # synthetic: ~15 F below air temp ( F)
            "WINDSPEED_MPH": 5.0,
            "PRESSURE_HPA": 1013.25,
            "HUMIDITY_PCT": 50.0,
            "SOIL_MOISTURE_VOL": 0.2,
            "SATURATION_PCT": 40.0,
            "enso_nino34": 0.0,
            "pdo_index": 0.0,
            "nao_index": 0.0,
            "ATMOSPHERE_IMPUTED": 0.0,
            "ATMOSPHERE_DONOR_KM": 0.0,
            "SOIL_MOISTURE_OBSERVED": 1.0,
            "CLIMATE_INDEX_OBSERVED": 1.0,
        }
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val
            else:
                df[col] = df[col].fillna(val)

        for column in ["LATITUDE", "LONGITUDE", "ELEVATION"]:
            if column not in df.columns:
                df[column] = 0.0
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

        # Real Dew Point (Celsius) and Wind (km/h)
        D = (df['DEWPOINT_F'] - 32) * 5.0 / 9.0 
        W = df['WINDSPEED_MPH'] * 1.60934
        P = df['PRCP'].fillna(0.0)

        monthly_t = T.rolling(30, min_periods=1).mean()
        monthly_p = P.rolling(30, min_periods=1).mean()
        wind = pd.to_numeric(df["WINDSPEED_MPH"], errors="coerce").fillna(5.0)
        pressure = pd.to_numeric(df["PRESSURE_HPA"], errors="coerce").fillna(1013.25)
        saturation = pd.to_numeric(df["SATURATION_PCT"], errors="coerce").fillna(40.0)
        soil = pd.to_numeric(df["SOIL_MOISTURE_VOL"], errors="coerce").fillna(0.2)

        df["PRCP_3D"] = P.rolling(3, min_periods=1).sum()
        df["PRCP_7D"] = P.rolling(7, min_periods=1).sum()
        df["PRCP_14D"] = P.rolling(14, min_periods=1).sum()
        df["PRCP_30D"] = P.rolling(30, min_periods=1).sum()
        df["PRCP_MAX_7D"] = P.rolling(7, min_periods=1).max()
        df["WIND_MAX_7D"] = wind.rolling(7, min_periods=1).max()
        df["PRESSURE_DROP_3D"] = (
            pressure.rolling(3, min_periods=1).max() - pressure
        ).clip(lower=0)
        df["SATURATION_MAX_7D"] = saturation.rolling(7, min_periods=1).max()
        df["SOIL_MOISTURE_ANOM_30D"] = soil - soil.rolling(
            30, min_periods=1
        ).mean()
        df["TEMP_ANOM_7D"] = T - T.rolling(7, min_periods=1).mean()
        
        df['DROUGHT_IDX'] = (monthly_p.mean() - monthly_p).clip(lower=0)
        
        # Enhanced Flood Risk = High precip combined with high soil saturation
        df['FLOOD_RISK_IDX'] = df['PRCP'] * (df['SATURATION_PCT'] / 100.0)
        
        # Enhanced Wind Risk = High wind speed + low pressure
        df['WIND_RISK_IDX'] = df['WINDSPEED_MPH'] * (1013.25 / df['PRESSURE_HPA'].clip(lower=900))

        df['TEMP_ANOM'] = T - monthly_t
        df['PRCP_ANOM'] = P - monthly_p
        
        t_95 = T.quantile(0.95)
        t_05 = T.quantile(0.05)
        df['EXTREME_HEAT'] = (T > t_95).astype(int)
        df['EXTREME_COLD'] = (T < t_05).astype(int)

        df['HEAT_INDEX'] = (
            -8.78469475556 + 1.61139411*T + 2.33854883889*D 
            - 0.14611605*T*D - 0.012308094*T**2 - 0.016424828*D**2 
            + 0.002211732*T**2*D + 0.00072546*T*D**2 - 0.000003582*T**2*D**2
        )
        df['WIND_CHILL'] = 13.12 + 0.6215*T - 11.37*(W)**0.16 + 0.3965*T*(W)**0.16
        df['WET_BULB'] = (
            T * np.arctan(0.151977 * (D + 8.313659)**0.5) + np.arctan(T + D) 
            - np.arctan(D - 1.676331) + 0.00391838 * D**1.5 * np.arctan(0.023101 * D) - 4.686035
        )
        
        # Add SST air difference as a feature
        df['SST_AIR_DIFF'] = df['TAVG'] - df['SST']

        # Predict the following day's hazardous state rather than reconstructing
        # a label generated from the same row's predictors.
        current_risk = (
            (df["EXTREME_HEAT"] == 1)
            | (df["DROUGHT_IDX"] > df["DROUGHT_IDX"].quantile(0.75))
            | (df["FLOOD_RISK_IDX"] > df["FLOOD_RISK_IDX"].quantile(0.90))
            | (df["WIND_RISK_IDX"] > df["WIND_RISK_IDX"].quantile(0.90))
        ).astype(float)
        df["RISK_ENGINEERED"] = current_risk.shift(-1)
        df["RISK"] = df["RISK_ENGINEERED"]
        if {"EVENT_LABEL", "EVENT_LABEL_AVAILABLE"}.issubset(df.columns):
            event_available = pd.to_numeric(
                df["EVENT_LABEL_AVAILABLE"], errors="coerce"
            ).fillna(0).astype(bool)
            event_label = pd.to_numeric(df["EVENT_LABEL"], errors="coerce")
            df.loc[event_available & event_label.notna(), "RISK"] = event_label[
                event_available & event_label.notna()
            ].astype(float)
        feature_columns = [column for column in df.columns if column != "RISK"]
        df[feature_columns] = df[feature_columns].fillna(0)
        return df

    def engineer_rf_features(self, T, D, W, P, weather_df):
        """Engineers the features needed for the Random Forest model for a single prediction."""
        hist_T = weather_df['TMAX'] if (weather_df is not None and 'TMAX' in weather_df.columns) else pd.Series([T])
        hist_P = weather_df['PRCP'] if (weather_df is not None and 'PRCP' in weather_df.columns) else pd.Series([P])
        
        def latest_valid(column, default):
            if weather_df is None or column not in weather_df.columns:
                return default
            values = pd.to_numeric(weather_df[column], errors="coerce").dropna()
            return float(values.iloc[-1]) if not values.empty else default

        pressure = latest_valid("PRESSURE_HPA", 1013.25)
        humidity = latest_valid("HUMIDITY_PCT", 50.0)
        soil_moisture = latest_valid("SOIL_MOISTURE_VOL", 0.2)
        saturation = latest_valid("SATURATION_PCT", 40.0)
        enso = latest_valid("enso_nino34", 0.0)
        pdo = latest_valid("pdo_index", 0.0)
        nao = latest_valid("nao_index", 0.0)
        latitude = latest_valid("LATITUDE", 0.0)
        longitude = latest_valid("LONGITUDE", 0.0)
        elevation = latest_valid("ELEVATION", 0.0)
        atmosphere_imputed = latest_valid("ATMOSPHERE_IMPUTED", 0.0)
        atmosphere_donor_km = latest_valid("ATMOSPHERE_DONOR_KM", 0.0)
        soil_observed = latest_valid("SOIL_MOISTURE_OBSERVED", 1.0)
        climate_observed = latest_valid("CLIMATE_INDEX_OBSERVED", 1.0)

        wind_mph = W / 1.60934
        
        monthly_t = hist_T.mean()
        monthly_p = hist_P.mean()
        t_95 = hist_T.quantile(0.95) if len(hist_T) > 0 else T
        t_05 = hist_T.quantile(0.05) if len(hist_T) > 0 else T
        
        recent_prcp = hist_P.tail(30).mean() if len(hist_P) >= 30 else P
        drought_idx = max(0.0, float(monthly_p - recent_prcp))
        hist_wind = (
            pd.to_numeric(weather_df["WINDSPEED_MPH"], errors="coerce")
            if weather_df is not None and "WINDSPEED_MPH" in weather_df.columns
            else pd.Series([wind_mph])
        )
        hist_pressure = (
            pd.to_numeric(weather_df["PRESSURE_HPA"], errors="coerce")
            if weather_df is not None and "PRESSURE_HPA" in weather_df.columns
            else pd.Series([pressure])
        )
        hist_saturation = (
            pd.to_numeric(weather_df["SATURATION_PCT"], errors="coerce")
            if weather_df is not None and "SATURATION_PCT" in weather_df.columns
            else pd.Series([saturation])
        )
        hist_soil = (
            pd.to_numeric(weather_df["SOIL_MOISTURE_VOL"], errors="coerce")
            if weather_df is not None and "SOIL_MOISTURE_VOL" in weather_df.columns
            else pd.Series([soil_moisture])
        )
        current_prcp_series = pd.concat(
            [pd.to_numeric(hist_P, errors="coerce"), pd.Series([P])],
            ignore_index=True,
        ).fillna(0.0)
        current_wind_series = pd.concat(
            [hist_wind, pd.Series([wind_mph])], ignore_index=True
        ).fillna(wind_mph)
        current_pressure_series = pd.concat(
            [hist_pressure, pd.Series([pressure])], ignore_index=True
        ).fillna(pressure)
        current_saturation_series = pd.concat(
            [hist_saturation, pd.Series([saturation])], ignore_index=True
        ).fillna(saturation)
        current_soil_series = pd.concat(
            [hist_soil, pd.Series([soil_moisture])], ignore_index=True
        ).fillna(soil_moisture)
        current_temp_series = pd.concat(
            [pd.to_numeric(hist_T, errors="coerce"), pd.Series([T])],
            ignore_index=True,
        ).fillna(T)

        prcp_3d = float(current_prcp_series.tail(3).sum())
        prcp_7d = float(current_prcp_series.tail(7).sum())
        prcp_14d = float(current_prcp_series.tail(14).sum())
        prcp_30d = float(current_prcp_series.tail(30).sum())
        prcp_max_7d = float(current_prcp_series.tail(7).max())
        wind_max_7d = float(current_wind_series.tail(7).max())
        pressure_drop_3d = max(
            0.0,
            float(current_pressure_series.tail(3).max() - current_pressure_series.iloc[-1]),
        )
        saturation_max_7d = float(current_saturation_series.tail(7).max())
        soil_moisture_anom_30d = float(
            current_soil_series.iloc[-1] - current_soil_series.tail(30).mean()
        )
        temp_anom_7d = float(
            current_temp_series.iloc[-1] - current_temp_series.tail(7).mean()
        )
        
        flood_risk_idx = P * (saturation / 100.0)
        wind_risk_idx = wind_mph * (1013.25 / max(900.0, pressure))

        temp_anom = T - monthly_t
        prcp_anom = P - monthly_p
        ext_heat = 1 if T > t_95 else 0
        ext_cold = 1 if T < t_05 else 0

        heat_idx = (
            -8.78469475556 + 1.61139411*T + 2.33854883889*D 
            - 0.14611605*T*D - 0.012308094*T**2 - 0.016424828*D**2 
            + 0.002211732*T**2*D + 0.00072546*T*D**2 - 0.000003582*T**2*D**2
        )
        wind_chill = 13.12 + 0.6215*T - 11.37*(W)**0.16 + 0.3965*T*(W)**0.16
        wet_bulb = (
            T * np.arctan(0.151977 * (D + 8.313659)**0.5) + np.arctan(T + D) 
            - np.arctan(D - 1.676331)
            + 0.00391838 * max(D, 0.0)**1.5 * np.arctan(0.023101 * D)
            - 4.686035
        )

        sst_val = weather_df['SST'].iloc[-1] if (weather_df is not None and 'SST' in weather_df.columns) else 18.0
        sst_air_diff = T - sst_val

        features = {
            "HEAT_INDEX": heat_idx, "WIND_CHILL": wind_chill, "WET_BULB": wet_bulb, 
            "DROUGHT_IDX": drought_idx, "TEMP_ANOM": temp_anom, "PRCP_ANOM": prcp_anom, 
            "EXTREME_HEAT": ext_heat, "EXTREME_COLD": ext_cold, "SST": sst_val, "SST_AIR_DIFF": sst_air_diff,
            "FLOOD_RISK_IDX": flood_risk_idx, "WIND_RISK_IDX": wind_risk_idx,
            "PRESSURE_HPA": pressure, "HUMIDITY_PCT": humidity, "SOIL_MOISTURE_VOL": soil_moisture, 
            "SATURATION_PCT": saturation, "enso_nino34": enso, "pdo_index": pdo, "nao_index": nao
            , "LATITUDE": latitude, "LONGITUDE": longitude, "ELEVATION": elevation,
            "ATMOSPHERE_IMPUTED": atmosphere_imputed,
            "ATMOSPHERE_DONOR_KM": atmosphere_donor_km,
            "SOIL_MOISTURE_OBSERVED": soil_observed,
            "CLIMATE_INDEX_OBSERVED": climate_observed,
            "PRCP_3D": prcp_3d,
            "PRCP_7D": prcp_7d,
            "PRCP_14D": prcp_14d,
            "PRCP_30D": prcp_30d,
            "PRCP_MAX_7D": prcp_max_7d,
            "WIND_MAX_7D": wind_max_7d,
            "PRESSURE_DROP_3D": pressure_drop_3d,
            "SATURATION_MAX_7D": saturation_max_7d,
            "SOIL_MOISTURE_ANOM_30D": soil_moisture_anom_30d,
            "TEMP_ANOM_7D": temp_anom_7d,
        }
        return pd.DataFrame([features])

    def get_lstm_sequence(self, weather_df, current_tmax, current_tmin, current_prcp):
        """Constructs the 30-day sequence for the LSTM."""
        needed = ["TMAX", "TMIN", "TAVG", "PRCP", "SST"]
        
        if weather_df is not None and not weather_df.empty:
            df = weather_df.copy()
            if "TAVG" not in df.columns:
                df["TAVG"] = (df["TMAX"] + df["TMIN"]) / 2
        else:
            df = pd.DataFrame(columns=needed)
            
        current_tavg = (current_tmax + current_tmin) / 2
        current_sst = df["SST"].iloc[-1] if "SST" in df.columns and len(df) > 0 else 18.0
        new_row = pd.DataFrame([[current_tmax, current_tmin, current_tavg, current_prcp, current_sst]], columns=needed)
        df = pd.concat([df[needed], new_row], ignore_index=True)
        
        seq = df[needed].tail(30).ffill().bfill().fillna(18.0).values
        if len(seq) < 30:
            pad = np.tile(seq[0], (30 - len(seq), 1))
            seq = np.vstack([pad, seq])
            
        return seq

    def _make_lstm_data(self, df, seq_len=30):
        needed = ["TMAX", "TMIN", "TAVG", "PRCP", "SST"]
        sub = df[needed].ffill().bfill().fillna(18.0).values
        X, y = [], []
        for i in range(seq_len, len(sub)):
            X.append(sub[i-seq_len:i])
            y.append(sub[i, 2]) # predict TAVG
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def _make_lstm_data_grouped(self, df, seq_len=30):
        if "City_ID" not in df.columns or df["City_ID"].nunique() <= 1:
            return self._make_lstm_data(df, seq_len)
        sequences, targets = [], []
        for _, group in df.groupby("City_ID", sort=False):
            group = group.sort_values("Date")
            X, y = self._make_lstm_data(group, seq_len)
            if len(X):
                sequences.append(X)
                targets.append(y)
        if not sequences:
            return np.empty((0, seq_len, 5), dtype=np.float32), np.empty(
                (0,), dtype=np.float32
            )
        return np.concatenate(sequences), np.concatenate(targets)

    def train_multimodal(
        self,
        weather_df,
        progress_callback=None,
        lstm_epochs=15,
        rf_estimators=100,
    ):
        """Trains the LSTM, RF, and Fusion models on the provided dataframe."""
        if weather_df is None or len(weather_df) < 100:
            raise ValueError("Not enough historical data to train the models. Please select a longer time period.")

        # Merge SST data
        if "SST" not in weather_df.columns:
            weather_df = self._add_sst_feature(weather_df)

        df = weather_df.copy()
        if "TAVG" not in df.columns:
            df["TAVG"] = (df["TMAX"] + df["TMIN"]) / 2
        df = self._add_event_labels(df)
            
        # 1. Feature Engineering
        if progress_callback: progress_callback(10, "Engineering features...")
        if "City_ID" in df.columns and df["City_ID"].nunique() > 1:
            df_rf = pd.concat(
                [
                    self._engineer_features_bulk(group.sort_values("Date"))
                    for _, group in df.groupby("City_ID", sort=False)
                ],
                ignore_index=True,
            )
        else:
            df_rf = self._engineer_features_bulk(df)
        df_rf = df_rf.dropna(subset=["RISK"]).copy()

        # Leakage-safe chronological split. Use fixed calendar boundaries when
        # the full 1995-2024 station archive is available.
        if "Date" in df_rf.columns:
            df_rf["Date"] = pd.to_datetime(df_rf["Date"])
        if (
            "Date" in df_rf.columns
            and df_rf["Date"].min() <= pd.Timestamp("1995-01-01")
            and df_rf["Date"].max() >= pd.Timestamp("2024-01-01")
        ):
            df_tr = df_rf[df_rf["Date"] < "2019-01-01"].copy()
            df_va = df_rf[
                (df_rf["Date"] >= "2019-01-01")
                & (df_rf["Date"] < "2022-01-01")
            ].copy()
            df_te = df_rf[df_rf["Date"] >= "2022-01-01"].copy()
        else:
            n = len(df_rf)
            train_end = int(n * 0.7)
            val_end = int(n * 0.85)
            df_tr = df_rf.iloc[:train_end].copy()
            df_va = df_rf.iloc[train_end:val_end].copy()
            df_te = df_rf.iloc[val_end:].copy()

        if min(len(df_tr), len(df_va), len(df_te)) <= 30:
            raise ValueError("Chronological train/validation/test splits are too small.")

        # 2. Train LSTM
        if TF_AVAILABLE:
            if progress_callback: progress_callback(25, "Training LSTM model (this may take a minute)...")
            self.scaler_lstm = MinMaxScaler()
            # Scale whole dataset for LSTM
            lstm_feats = ["TMAX", "TMIN", "TAVG", "PRCP", "SST"]
            scaled_tr = df_tr.copy()
            scaled_tr[lstm_feats] = self.scaler_lstm.fit_transform(df_tr[lstm_feats])
            scaled_va = df_va.copy()
            scaled_va[lstm_feats] = self.scaler_lstm.transform(df_va[lstm_feats])
            scaled_te = df_te.copy()
            scaled_te[lstm_feats] = self.scaler_lstm.transform(df_te[lstm_feats])

            X_tr_l, y_tr_l = self._make_lstm_data_grouped(scaled_tr)
            X_va_l, y_va_l = self._make_lstm_data_grouped(scaled_va)
            X_te_l, y_te_l = self._make_lstm_data_grouped(scaled_te)

            inp_lstm = Input(shape=(30, 5))
            x = LSTM(64, return_sequences=True)(inp_lstm)
            x = Dropout(0.2)(x)
            x = LSTM(32)(x)
            x = Dropout(0.2)(x)
            lstm_out = Dense(1)(x)

            self.lstm_model = Model(inp_lstm, lstm_out)
            self.lstm_model.compile(optimizer=Adam(0.001), loss="mse")
            
            cb = [EarlyStopping(patience=3, restore_best_weights=True)]
            self.lstm_model.fit(
                X_tr_l, y_tr_l, validation_data=(X_va_l, y_va_l),
                epochs=lstm_epochs, batch_size=32, callbacks=cb, verbose=0
            )
        else:
            if progress_callback: progress_callback(25, "Skipping LSTM training (TensorFlow unavailable).")
            self.scaler_lstm = MinMaxScaler()
            # Dummy fit just to have the scaler available
            lstm_feats = ["TMAX", "TMIN", "TAVG", "PRCP", "SST"]
            self.scaler_lstm.fit(df_tr[lstm_feats])

        # 3. Train Random Forest
        if progress_callback: progress_callback(60, "Training Random Forest classifier...")
        rf_feats = RF_FEATURES
        
        self.scaler_rf = MinMaxScaler()
        X_tr_r = self.scaler_rf.fit_transform(df_tr[rf_feats])
        X_va_r = self.scaler_rf.transform(df_va[rf_feats])
        X_te_r = self.scaler_rf.transform(df_te[rf_feats])

        y_tr_r = df_tr["RISK"].values
        y_va_r = df_va["RISK"].values
        y_te_r = df_te["RISK"].values

        self.rf_model = RandomForestClassifier(
            n_estimators=rf_estimators,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        self.rf_model.fit(X_tr_r, y_tr_r)

        # 4. Late Fusion. Train only from validation predictions; the test
        # period remains untouched until final evaluation.
        if TF_AVAILABLE and self.lstm_model is not None:
            if progress_callback: progress_callback(80, "Training Late Fusion Meta-Learner...")
            y_pred_lstm_val = self.lstm_model.predict(X_va_l, verbose=0).flatten()
            y_proba_rf_val = self.rf_model.predict_proba(X_va_r)[:, 1]

            n_fuse = min(len(y_pred_lstm_val), len(y_proba_rf_val))
            lstm_feat = y_pred_lstm_val[-n_fuse:].reshape(-1, 1)
            rf_feat = y_proba_rf_val[-n_fuse:].reshape(-1, 1)
            fused_X = np.hstack([lstm_feat, rf_feat])

            inp_fuse = Input(shape=(2,))
            fx = Dense(16, activation="relu")(inp_fuse)
            fx = Dense(8, activation="relu")(fx)
            fuse_out = Dense(1, activation="sigmoid")(fx)

            self.fusion_model = Model(inp_fuse, fuse_out)
            self.fusion_model.compile(optimizer=Adam(0.001), loss="binary_crossentropy")
            
            y_fuse = y_va_r[-n_fuse:]
            self.fusion_model.fit(
                fused_X, y_fuse, epochs=20, batch_size=32, verbose=0
            )
        else:
            if progress_callback: progress_callback(80, "Skipping Late Fusion (TensorFlow unavailable).")

        # 4.5 Compute and save validation accuracy metrics
        if progress_callback: progress_callback(90, "Computing validation metrics...")
        metrics = {}
        
        # RF Metrics
        y_pred_rf = self.rf_model.predict(X_te_r)
        y_proba_rf = self.rf_model.predict_proba(X_te_r)[:, 1]
        metrics["rf_accuracy"] = float(accuracy_score(y_te_r, y_pred_rf))
        metrics["rf_auc"] = self._safe_auc(y_te_r, y_proba_rf)
        
        # LSTM and Fusion Metrics
        if TF_AVAILABLE and self.lstm_model is not None and self.fusion_model is not None:
            y_pred_lstm = self.lstm_model.predict(X_te_l, verbose=0).flatten()
            metrics["lstm_mse_scaled"] = float(
                mean_squared_error(y_te_l, y_pred_lstm)
            )
            metrics["lstm_r2"] = float(r2_score(y_te_l, y_pred_lstm))

            target_min = float(self.scaler_lstm.data_min_[2])
            target_range = float(self.scaler_lstm.data_range_[2])
            y_te_c = y_te_l * target_range + target_min
            y_pred_lstm_c = y_pred_lstm * target_range + target_min
            metrics["lstm_mae_c"] = float(
                np.mean(np.abs(y_te_c - y_pred_lstm_c))
            )
            metrics["lstm_rmse_c"] = float(
                np.sqrt(mean_squared_error(y_te_c, y_pred_lstm_c))
            )
            
            n_fuse = min(len(y_pred_lstm), len(y_proba_rf))
            lstm_feat = y_pred_lstm[-n_fuse:].reshape(-1, 1)
            rf_feat = y_proba_rf[-n_fuse:].reshape(-1, 1)
            fused_X_te = np.hstack([lstm_feat, rf_feat])
            
            y_pred_fusion_proba = self.fusion_model.predict(fused_X_te, verbose=0).flatten()
            y_pred_fusion = (y_pred_fusion_proba >= 0.5).astype(int)
            y_fuse = y_te_r[-n_fuse:]
            
            metrics["fusion_accuracy"] = float(accuracy_score(y_fuse, y_pred_fusion))
            metrics["fusion_auc"] = self._safe_auc(y_fuse, y_pred_fusion_proba)
        else:
            metrics["lstm_mse_scaled"] = 0.0
            metrics["lstm_mae_c"] = 0.0
            metrics["lstm_rmse_c"] = 0.0
            metrics["lstm_r2"] = 0.0
            metrics["fusion_accuracy"] = metrics["rf_accuracy"]
            metrics["fusion_auc"] = metrics["rf_auc"]

        event_mask = (
            pd.to_numeric(
                df_te.get("EVENT_LABEL_AVAILABLE", pd.Series(0, index=df_te.index)),
                errors="coerce",
            )
            .fillna(0)
            .astype(bool)
            .values
        )
        event_count = int(event_mask.sum())
        metrics["event_labeled_test_rows"] = event_count
        if event_count:
            event_y = pd.to_numeric(
                df_te.loc[event_mask, "EVENT_LABEL"], errors="coerce"
            ).fillna(0).astype(int).values
            metrics["rf_event_accuracy"] = float(
                accuracy_score(event_y, y_pred_rf[event_mask])
            )
            metrics["rf_event_auc"] = self._safe_auc(event_y, y_proba_rf[event_mask])
        else:
            metrics["rf_event_accuracy"] = 0.0
            metrics["rf_event_auc"] = 0.5

        event_model_metrics = self._train_event_model(
            df_rf, rf_feats, rf_estimators
        )
        metrics.update(event_model_metrics)
            
        metrics_path = os.path.join(self.model_dir, "metrics.json")
        try:
            import json
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
            manifest = {
                "forecast_horizon_days": 1,
                "sequence_length_days": 30,
                "train_rows": int(len(df_tr)),
                "validation_rows": int(len(df_va)),
                "test_rows": int(len(df_te)),
                "train_start": str(pd.to_datetime(df_tr["Date"]).min().date()),
                "train_end": str(pd.to_datetime(df_tr["Date"]).max().date()),
                "validation_start": str(pd.to_datetime(df_va["Date"]).min().date()),
                "validation_end": str(pd.to_datetime(df_va["Date"]).max().date()),
                "test_start": str(pd.to_datetime(df_te["Date"]).min().date()),
                "test_end": str(pd.to_datetime(df_te["Date"]).max().date()),
                "minimum_recommended_history_years": 10,
                "preferred_history_years": 20,
                "station_count": int(
                    df_rf["City_ID"].nunique()
                    if "City_ID" in df_rf.columns
                    else 1
                ),
                "target": (
                    "Hybrid next-day target: engineered climate-hazard proxy "
                    "where independent NOAA Storm Events labels are unavailable; "
                    "NOAA event/no-event labels replace the proxy on event "
                    "coverage days."
                ),
                "event_labeled_rows": int(
                    pd.to_numeric(
                        df_rf.get(
                            "EVENT_LABEL_AVAILABLE",
                            pd.Series(0, index=df_rf.index),
                        ),
                        errors="coerce",
                    )
                    .fillna(0)
                    .sum()
                ),
                "rf_features": rf_feats,
                "operational_limit": (
                    "This artifact predicts one day ahead. Forecasts beyond "
                    "7-14 days require numerical-weather ensemble inputs; "
                    "monthly outlooks require seasonal ensemble data."
                ),
            }
            with open(
                os.path.join(self.model_dir, "training_manifest.json"), "w"
            ) as f:
                json.dump(manifest, f, indent=4)
            self.metrics = metrics
        except Exception as e:
            print(f"Error saving metrics: {e}")

        # 5. Save Models
        if progress_callback: progress_callback(95, "Saving models...")
        if TF_AVAILABLE and self.lstm_model is not None and self.fusion_model is not None:
            self.lstm_model.save(os.path.join(self.model_dir, "lstm_final.keras"))
            self.fusion_model.save(os.path.join(self.model_dir, "fusion_final.keras"))
        joblib.dump(self.rf_model, os.path.join(self.model_dir, "rf_model.pkl"))
        joblib.dump(self.scaler_lstm, os.path.join(self.model_dir, "scaler_lstm.pkl"))
        joblib.dump(self.scaler_rf, os.path.join(self.model_dir, "scaler_rf.pkl"))
        if self.event_model is not None and self.scaler_event is not None:
            joblib.dump(
                self.event_model, os.path.join(self.model_dir, "event_rf_model.pkl")
            )
            joblib.dump(
                self.scaler_event, os.path.join(self.model_dir, "scaler_event.pkl")
            )
            with open(
                os.path.join(self.model_dir, "event_threshold.json"), "w"
            ) as f:
                import json
                json.dump({"threshold": self.event_threshold}, f, indent=4)

        if progress_callback: progress_callback(100, "Training complete!")
        return True

    def predict(self, weather_df, current_tmax, current_tmin, dew_point, wind_speed, current_prcp):
        """Runs the full multimodal prediction pipeline."""
        if self.rf_model is None:
            return {
                "Probability": 0.0, "Category": "Model Not Trained",
                "Description": "Models are not trained yet. Please click the train button.",
                "Explanation": "Models missing.",
                "RF_Prob": 0.0,
                "LSTM_Temp": 0.0
            }

        # Merge SST data
        weather_df = self._add_sst_feature(weather_df)

        current_temp = (current_tmax + current_tmin) / 2
        sst_val = weather_df['SST'].iloc[-1] if (weather_df is not None and 'SST' in weather_df.columns and len(weather_df) > 0) else 18.0
        sst_air_diff = current_temp - sst_val

        rf_df = self.engineer_rf_features(current_temp, dew_point, wind_speed, current_prcp, weather_df)
        rf_features = RF_FEATURES
        X_rf = self.scaler_rf.transform(rf_df[rf_features].fillna(0))
        
        rf_prob = self.rf_model.predict_proba(X_rf)[0, 1]
        event_prob = None
        event_alert = False
        if self.event_model is not None and self.scaler_event is not None:
            X_event = self.scaler_event.transform(rf_df[rf_features].fillna(0))
            event_prob = float(self.event_model.predict_proba(X_event)[0, 1])
            event_alert = event_prob >= self.event_threshold

        if TF_AVAILABLE and self.lstm_model is not None and self.fusion_model is not None:
            seq_2d = self.get_lstm_sequence(weather_df, current_tmax, current_tmin, current_prcp)
            lstm_feats = ["TMAX", "TMIN", "TAVG", "PRCP", "SST"]
            seq_df = pd.DataFrame(seq_2d, columns=lstm_feats)
            seq_scaled = self.scaler_lstm.transform(seq_df)
            if hasattr(seq_scaled, "values"):
                seq_scaled = seq_scaled.values
            X_lstm = np.expand_dims(seq_scaled, axis=0)

            lstm_pred_scaled = self.lstm_model.predict(X_lstm, verbose=0).flatten()
            
            dummy_df = pd.DataFrame(np.zeros((1, 5)), columns=lstm_feats)
            dummy_df.iloc[0, 2] = lstm_pred_scaled[0]
            inverse_result = self.scaler_lstm.inverse_transform(dummy_df)
            if isinstance(inverse_result, pd.DataFrame):
                lstm_temp_c = inverse_result.iloc[0, 2]
            else:
                lstm_temp_c = inverse_result[0, 2]

            lstm_feat = lstm_pred_scaled.reshape(-1, 1)
            rf_feat = np.array([[rf_prob]])
            fused_X = np.hstack([lstm_feat, rf_feat])
            
            fusion_prob = float(self.fusion_model.predict(fused_X, verbose=0).flatten()[0])
            prob_pct = min(max(fusion_prob * 100.0, 0.0), 100.0)
            
            explanation = (
                f"**Late Fusion Details**:\n"
                f"- **LSTM Model** predicts a TAVG of **{lstm_temp_c:.1f} C** tomorrow based on the last 30 days of land + SST observations.\n"
                f"- **Random Forest Classifier** gives a raw risk of **{rf_prob*100.0:.1f}%** incorporating land indices and sea surface temperature (**{sst_val:.2f} C**).\n"
                f"- **Air-SST Difference**: **{sst_air_diff:.2f} C** (Air TAVG: {current_temp:.1f} C vs. SST: {sst_val:.1f} C)."
            )
        else:
            lstm_temp_c = current_temp
            prob_pct = min(max(rf_prob * 100.0, 0.0), 100.0)
            explanation = (
                f"**Pure Random Forest Model**:\n"
                f"- **LSTM / Fusion Model Unavailable** due to environment constraints. Using Random Forest alone.\n"
                f"- **Random Forest Classifier** gives a raw risk of **{rf_prob*100.0:.1f}%** incorporating land indices, soil moisture, climate indices and sea surface temperature (**{sst_val:.2f} C**).\n"
                f"- **Air-SST Difference**: **{sst_air_diff:.2f} C** (Air TAVG: {current_temp:.1f} C vs. SST: {sst_val:.1f} C)."
            )

        if event_prob is not None:
            explanation += (
                "\n"
                f"- **NOAA Event Model** estimates a reported-event probability of "
                f"**{event_prob*100.0:.1f}%** using lagged rain/wind/pressure/soil features "
                f"(alert threshold {self.event_threshold*100.0:.1f}%)."
            )

        # Calculate local contributions
        if prob_pct < 30.0:
            cat, desc = "Low Risk", "Weather parameters are within safe bounds."
        elif prob_pct < 60.0:
            cat, desc = "Moderate Alert", "Elevated climatic factors. Stay updated."
        elif prob_pct < 85.0:
            cat, desc = "Severe Warning", "High hazard risk. Extremes detected."
        else:
            cat, desc = "Extreme Danger", "CRITICAL conditions! Extreme risk profile."
            
        if self.rf_model is not None:
            importances = self.rf_model.feature_importances_
        else:
            importances = np.array([0.1] * len(rf_features))
        
        scaled_vals = X_rf[0]
        contrib_vals = scaled_vals * importances
        if contrib_vals.sum() > 0:
            contrib_vals = contrib_vals / contrib_vals.sum()
        else:
            contrib_vals = np.array([0.1] * len(rf_features))
            
        contributions = {feat: float(val) for feat, val in zip(rf_features, contrib_vals)}

        return {
            "Probability": prob_pct,
            "Category": cat,
            "Description": desc,
            "Explanation": explanation,
            "RF_Prob": float(rf_prob * 100.0),
            "LSTM_Temp": float(lstm_temp_c),
            "Event_Prob": None if event_prob is None else float(event_prob * 100.0),
            "Event_Alert": bool(event_alert),
            "Event_Threshold": float(self.event_threshold * 100.0),
            "Contributions": contributions
        }

    def get_feature_importances(self):
        """Returns the Gini feature importances of the Random Forest model."""
        rf_feats = RF_FEATURES
        if self.rf_model is not None:
            importances = self.rf_model.feature_importances_
        else:
            importances = [0.1] * len(rf_feats)
        df = pd.DataFrame({
            "Feature": rf_feats,
            "Gini Importance": importances
        })
        return df.sort_values("Gini Importance", ascending=True)
