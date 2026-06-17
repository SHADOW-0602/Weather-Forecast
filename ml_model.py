import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
import streamlit as st

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
        self.metrics = None
        self._load_models()

    def _load_models(self):
        """Loads the trained models and scalers from disk."""
        rf_path = os.path.join(self.model_dir, "rf_model.pkl")
        lstm_path = os.path.join(self.model_dir, "lstm_final.keras")
        fusion_path = os.path.join(self.model_dir, "fusion_final.keras")
        sl_path = os.path.join(self.model_dir, "scaler_lstm.pkl")
        sr_path = os.path.join(self.model_dir, "scaler_rf.pkl")
        metrics_path = os.path.join(self.model_dir, "metrics.json")

        if all(os.path.exists(p) for p in [rf_path, sl_path, sr_path]):
            try:
                self.rf_model = joblib.load(rf_path)
                self.scaler_lstm = joblib.load(sl_path)
                self.scaler_rf = joblib.load(sr_path)
                if TF_AVAILABLE and os.path.exists(lstm_path) and os.path.exists(fusion_path):
                    tf.get_logger().setLevel('ERROR')
                    self.lstm_model = tf.keras.models.load_model(lstm_path)
                    self.fusion_model = tf.keras.models.load_model(fusion_path)
                if os.path.exists(metrics_path):
                    import json
                    with open(metrics_path, "r") as f:
                        self.metrics = json.load(f)
            except Exception as e:
                print(f"Error loading models: {e}")
                self.rf_model = None
        else:
            self.rf_model = None

    def _add_sst_feature(self, df):
        """Merges SST time-series data into the weather dataframe."""
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
            df["Date"] = pd.to_datetime(df["Date"])
            
        daily_path = "ocean.csv"
        monthly_path = "ocean_monthly.csv"
        
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
                    "SEATTLE": "sst_pacific_f",
                    "KCQT": "sst_pacific_f",
                    "NEW_YORK": "sst_atlantic_f",
                    "KPHL": "sst_atlantic_f",
                    "KJAX": "sst_atlantic_f",
                    "KCLT": "sst_atlantic_f",
                    "KHOU": "sst_gulf_f",
                }
                
                target_col = basin_mapping.get(city_id)
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

    def _engineer_features_bulk(self, df):
        """Engineers features for the entire dataframe for training RF."""
        df = df.copy()
        
        # Ensure base columns exist
        if 'TAVG' not in df.columns:
            df['TAVG'] = (df['TMAX'] + df['TMIN']) / 2
        
        T = df['TAVG']
        
        # Fill missing new features with defaults so it doesn't break if files are missing
        defaults = {
            "DEWPOINT_F": (T * 9/5 + 32) - 10.8,
            "WINDSPEED_MPH": 5.0,
            "PRESSURE_HPA": 1013.25,
            "HUMIDITY_PCT": 50.0,
            "SOIL_MOISTURE_VOL": 0.2,
            "SATURATION_PCT": 40.0,
            "enso_nino34": 0.0,
            "pdo_index": 0.0,
            "nao_index": 0.0
        }
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val
            else:
                df[col] = df[col].fillna(val)

        # Real Dew Point (Celsius) and Wind (km/h)
        D = (df['DEWPOINT_F'] - 32) * 5.0 / 9.0 
        W = df['WINDSPEED_MPH'] * 1.60934
        P = df['PRCP'].fillna(0.0)

        monthly_t = T.rolling(30, min_periods=1).mean()
        monthly_p = P.rolling(30, min_periods=1).mean()
        
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

        # Risk label for training (High Risk = extreme heat OR drought OR high flood risk OR high wind risk)
        df["RISK"] = ((df["EXTREME_HEAT"] == 1) | 
                      (df["DROUGHT_IDX"] > df["DROUGHT_IDX"].quantile(0.75)) |
                      (df["FLOOD_RISK_IDX"] > df["FLOOD_RISK_IDX"].quantile(0.90)) |
                      (df["WIND_RISK_IDX"] > df["WIND_RISK_IDX"].quantile(0.90))).astype(int)
        return df.fillna(0)

    def engineer_rf_features(self, T, D, W, P, weather_df):
        """Engineers the features needed for the Random Forest model for a single prediction."""
        hist_T = weather_df['TMAX'] if (weather_df is not None and 'TMAX' in weather_df.columns) else pd.Series([T])
        hist_P = weather_df['PRCP'] if (weather_df is not None and 'PRCP' in weather_df.columns) else pd.Series([P])
        
        pressure = weather_df['PRESSURE_HPA'].iloc[-1] if (weather_df is not None and 'PRESSURE_HPA' in weather_df.columns) else 1013.25
        humidity = weather_df['HUMIDITY_PCT'].iloc[-1] if (weather_df is not None and 'HUMIDITY_PCT' in weather_df.columns) else 50.0
        soil_moisture = weather_df['SOIL_MOISTURE_VOL'].iloc[-1] if (weather_df is not None and 'SOIL_MOISTURE_VOL' in weather_df.columns) else 0.2
        saturation = weather_df['SATURATION_PCT'].iloc[-1] if (weather_df is not None and 'SATURATION_PCT' in weather_df.columns) else 40.0
        enso = weather_df['enso_nino34'].iloc[-1] if (weather_df is not None and 'enso_nino34' in weather_df.columns) else 0.0
        pdo = weather_df['pdo_index'].iloc[-1] if (weather_df is not None and 'pdo_index' in weather_df.columns) else 0.0
        nao = weather_df['nao_index'].iloc[-1] if (weather_df is not None and 'nao_index' in weather_df.columns) else 0.0

        wind_mph = W / 1.60934
        
        monthly_t = hist_T.mean()
        monthly_p = hist_P.mean()
        t_95 = hist_T.quantile(0.95) if len(hist_T) > 0 else T
        t_05 = hist_T.quantile(0.05) if len(hist_T) > 0 else T
        
        recent_prcp = hist_P.tail(30).mean() if len(hist_P) >= 30 else P
        drought_idx = max(0.0, float(monthly_p - recent_prcp))
        
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
            - np.arctan(D - 1.676331) + 0.00391838 * D**1.5 * np.arctan(0.023101 * D) - 4.686035
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
        
        seq = df[needed].tail(30).ffill().fillna(18.0).values
        if len(seq) < 30:
            pad = np.tile(seq[0], (30 - len(seq), 1))
            seq = np.vstack([pad, seq])
            
        return seq

    def _make_lstm_data(self, df, seq_len=30):
        needed = ["TMAX", "TMIN", "TAVG", "PRCP", "SST"]
        sub = df[needed].ffill().fillna(18.0).values
        X, y = [], []
        for i in range(seq_len, len(sub)):
            X.append(sub[i-seq_len:i])
            y.append(sub[i, 2]) # predict TAVG
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def train_multimodal(self, weather_df, progress_callback=None):
        """Trains the LSTM, RF, and Fusion models on the provided dataframe."""
        if weather_df is None or len(weather_df) < 100:
            raise ValueError("Not enough historical data to train the models. Please select a longer time period.")

        # Merge SST data
        weather_df = self._add_sst_feature(weather_df)

        df = weather_df.copy()
        if "TAVG" not in df.columns:
            df["TAVG"] = (df["TMAX"] + df["TMIN"]) / 2
            
        # 1. Feature Engineering
        if progress_callback: progress_callback(10, "Engineering features...")
        df_rf = self._engineer_features_bulk(df)
        
        # Split data (70% train, 15% val, 15% test)
        n = len(df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        df_tr = df_rf.iloc[:train_end]
        df_va = df_rf.iloc[train_end:val_end]
        df_te = df_rf.iloc[val_end:]

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

            X_tr_l, y_tr_l = self._make_lstm_data(scaled_tr)
            X_va_l, y_va_l = self._make_lstm_data(scaled_va)
            X_te_l, y_te_l = self._make_lstm_data(scaled_te)

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
                epochs=15, batch_size=32, callbacks=cb, verbose=0
            )
        else:
            if progress_callback: progress_callback(25, "Skipping LSTM training (TensorFlow unavailable).")
            self.scaler_lstm = MinMaxScaler()
            # Dummy fit just to have the scaler available
            lstm_feats = ["TMAX", "TMIN", "TAVG", "PRCP", "SST"]
            self.scaler_lstm.fit(df_tr[lstm_feats])

        # 3. Train Random Forest
        if progress_callback: progress_callback(60, "Training Random Forest classifier...")
        rf_feats = ["HEAT_INDEX", "WIND_CHILL", "WET_BULB", "DROUGHT_IDX", 
                    "TEMP_ANOM", "PRCP_ANOM", "EXTREME_HEAT", "EXTREME_COLD", "SST", "SST_AIR_DIFF",
                    "FLOOD_RISK_IDX", "WIND_RISK_IDX", "PRESSURE_HPA", "HUMIDITY_PCT", 
                    "SOIL_MOISTURE_VOL", "SATURATION_PCT", "enso_nino34", "pdo_index", "nao_index"]
        
        self.scaler_rf = MinMaxScaler()
        X_tr_r = self.scaler_rf.fit_transform(df_tr[rf_feats])
        X_va_r = self.scaler_rf.transform(df_va[rf_feats])
        X_te_r = self.scaler_rf.transform(df_te[rf_feats])

        y_tr_r = df_tr["RISK"].values
        y_va_r = df_va["RISK"].values
        y_te_r = df_te["RISK"].values

        self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced")
        self.rf_model.fit(X_tr_r, y_tr_r)

        # 4. Late Fusion
        if TF_AVAILABLE and self.lstm_model is not None:
            if progress_callback: progress_callback(80, "Training Late Fusion Meta-Learner...")
            y_pred_lstm = self.lstm_model.predict(X_te_l, verbose=0).flatten()
            y_proba_rf = self.rf_model.predict_proba(X_te_r)[:, 1]

            n_fuse = min(len(y_pred_lstm), len(y_proba_rf))
            lstm_feat = y_pred_lstm[-n_fuse:].reshape(-1, 1)
            rf_feat = y_proba_rf[-n_fuse:].reshape(-1, 1)
            fused_X = np.hstack([lstm_feat, rf_feat])

            inp_fuse = Input(shape=(2,))
            fx = Dense(16, activation="relu")(inp_fuse)
            fx = Dense(8, activation="relu")(fx)
            fuse_out = Dense(1, activation="sigmoid")(fx)

            self.fusion_model = Model(inp_fuse, fuse_out)
            self.fusion_model.compile(optimizer=Adam(0.001), loss="binary_crossentropy")
            
            y_fuse = y_te_r[-n_fuse:]
            self.fusion_model.fit(fused_X, y_fuse, epochs=10, batch_size=16, verbose=0)
        else:
            if progress_callback: progress_callback(80, "Skipping Late Fusion (TensorFlow unavailable).")

        # 4.5 Compute and save validation accuracy metrics
        if progress_callback: progress_callback(90, "Computing validation metrics...")
        metrics = {}
        
        # RF Metrics
        y_pred_rf = self.rf_model.predict(X_te_r)
        y_proba_rf = self.rf_model.predict_proba(X_te_r)[:, 1]
        metrics["rf_accuracy"] = float(accuracy_score(y_te_r, y_pred_rf))
        metrics["rf_auc"] = float(roc_auc_score(y_te_r, y_proba_rf))
        
        # LSTM and Fusion Metrics
        if TF_AVAILABLE and self.lstm_model is not None and self.fusion_model is not None:
            y_pred_lstm = self.lstm_model.predict(X_te_l, verbose=0).flatten()
            metrics["lstm_mse"] = float(mean_squared_error(y_te_l, y_pred_lstm))
            metrics["lstm_r2"] = float(r2_score(y_te_l, y_pred_lstm))
            
            n_fuse = min(len(y_pred_lstm), len(y_proba_rf))
            lstm_feat = y_pred_lstm[-n_fuse:].reshape(-1, 1)
            rf_feat = y_proba_rf[-n_fuse:].reshape(-1, 1)
            fused_X_te = np.hstack([lstm_feat, rf_feat])
            
            y_pred_fusion_proba = self.fusion_model.predict(fused_X_te, verbose=0).flatten()
            y_pred_fusion = (y_pred_fusion_proba >= 0.5).astype(int)
            y_fuse = y_te_r[-n_fuse:]
            
            metrics["fusion_accuracy"] = float(accuracy_score(y_fuse, y_pred_fusion))
            metrics["fusion_auc"] = float(roc_auc_score(y_fuse, y_pred_fusion_proba))
        else:
            metrics["lstm_mse"] = 0.0
            metrics["lstm_r2"] = 0.0
            metrics["fusion_accuracy"] = metrics["rf_accuracy"]
            metrics["fusion_auc"] = metrics["rf_auc"]
            
        metrics_path = os.path.join(self.model_dir, "metrics.json")
        try:
            import json
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)
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
        rf_features = ["HEAT_INDEX", "WIND_CHILL", "WET_BULB", "DROUGHT_IDX", 
                       "TEMP_ANOM", "PRCP_ANOM", "EXTREME_HEAT", "EXTREME_COLD", "SST", "SST_AIR_DIFF",
                       "FLOOD_RISK_IDX", "WIND_RISK_IDX", "PRESSURE_HPA", "HUMIDITY_PCT", 
                       "SOIL_MOISTURE_VOL", "SATURATION_PCT", "enso_nino34", "pdo_index", "nao_index"]
        X_rf = self.scaler_rf.transform(rf_df[rf_features].fillna(0))
        
        rf_prob = self.rf_model.predict_proba(X_rf)[0, 1]

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
                f"- **LSTM Model** predicts a TAVG of **{lstm_temp_c:.1f}°C** tomorrow based on the last 30 days of land + SST observations.\n"
                f"- **Random Forest Classifier** gives a raw risk of **{rf_prob*100.0:.1f}%** incorporating land indices and sea surface temperature (**{sst_val:.2f}°C**).\n"
                f"- **Air-SST Difference**: **{sst_air_diff:.2f}°C** (Air TAVG: {current_temp:.1f}°C vs. SST: {sst_val:.1f}°C)."
            )
        else:
            lstm_temp_c = current_temp
            prob_pct = min(max(rf_prob * 100.0, 0.0), 100.0)
            explanation = (
                f"**Pure Random Forest Model**:\n"
                f"- **LSTM / Fusion Model Unavailable** due to environment constraints. Using Random Forest alone.\n"
                f"- **Random Forest Classifier** gives a raw risk of **{rf_prob*100.0:.1f}%** incorporating land indices, soil moisture, climate indices and sea surface temperature (**{sst_val:.2f}°C**).\n"
                f"- **Air-SST Difference**: **{sst_air_diff:.2f}°C** (Air TAVG: {current_temp:.1f}°C vs. SST: {sst_val:.1f}°C)."
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
            "RF_Prob": rf_prob * 100.0,
            "LSTM_Temp": lstm_temp_c,
            "Contributions": contributions
        }

    def get_feature_importances(self):
        """Returns the Gini feature importances of the Random Forest model."""
        rf_feats = ["HEAT_INDEX", "WIND_CHILL", "WET_BULB", "DROUGHT_IDX", 
                    "TEMP_ANOM", "PRCP_ANOM", "EXTREME_HEAT", "EXTREME_COLD", "SST", "SST_AIR_DIFF",
                    "FLOOD_RISK_IDX", "WIND_RISK_IDX", "PRESSURE_HPA", "HUMIDITY_PCT", 
                    "SOIL_MOISTURE_VOL", "SATURATION_PCT", "enso_nino34", "pdo_index", "nao_index"]
        if self.rf_model is not None:
            importances = self.rf_model.feature_importances_
        else:
            importances = [0.1] * len(rf_feats)
        df = pd.DataFrame({
            "Feature": rf_feats,
            "Gini Importance": importances
        })
        return df.sort_values("Gini Importance", ascending=True)
