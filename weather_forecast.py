# weather_forecast.py
import pandas as pd
import numpy as np
import requests
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeatherForecastSystem:
    def __init__(self):
        self.latitude = None
        self.longitude = None
        self.location_name = None
        self.temperature_model = None
        self.temperature_features = None
        self.condition_model = None
        self.condition_features = None
        self.historical_data = pd.DataFrame()
        self.last_update = None
        self.openmeteo_base_url = "https://api.open-meteo.com/v1/forecast"
        self.geolocator = Nominatim(user_agent="weather_forecast_app")

    def set_location(self, location_input: str) -> bool:
        """Set location using geopy"""
        if not location_input:
            logger.error("Empty location input")
            return False

        try:
            logger.info(f"Searching for coordinates of: {location_input}")
            location = self.geolocator.geocode(location_input)

            if location:
                self.latitude = location.latitude
                self.longitude = location.longitude
                self.location_name = location.address
                logger.info(f"Location set: {self.location_name} ({self.latitude}, {self.longitude})")
                return True
            else:
                logger.warning("Location not found")
                return False
                
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.error(f"Geocoding service error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error setting location: {e}")
            return False

    def fetch_weather_data(self, days: int = 30, forecast: bool = False) -> pd.DataFrame:
        try:
            if self.latitude is None or self.longitude is None:
                raise ValueError("Location not set.")

            end_date = datetime.now()
            start_date = end_date - timedelta(days=days-1)

            params = {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "hourly": ("temperature_2m,relative_humidity_2m,"
                          "dew_point_2m,pressure_msl,cloud_cover,"
                          "wind_speed_10m,wind_direction_10m,precipitation"),
                "timezone": "auto",
                "start_date": start_date.strftime('%Y-%m-%d'),
                "end_date": end_date.strftime('%Y-%m-%d')
            }

            if forecast:
                params = {
                    "latitude": self.latitude,
                    "longitude": self.longitude,
                    "hourly": params["hourly"],
                    "forecast_days": days,
                    "timezone": "auto"
                }

            logger.info(f"Fetching {'forecast' if forecast else 'historical'} data for {self.location_name}")
            response = requests.get(self.openmeteo_base_url, params=params)
            response.raise_for_status()

            data = response.json()
            df = pd.DataFrame(data["hourly"])
            df['time'] = pd.to_datetime(df['time'])

            column_map = {
                "temperature_2m": "temperature",
                "relative_humidity_2m": "relative_humidity",
                "dew_point_2m": "dew_point",
                "pressure_msl": "pressure_msl (hPa)",
                "cloud_cover": "cloud_cover (%)",
                "wind_speed_10m": "wind_speed_10m (km/h)",
                "wind_direction_10m": "wind_direction",
                "precipitation": "precipitation (mm)"
            }
            df = df.rename(columns=column_map)

            expected_cols = list(column_map.values()) + ['time']
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns in fetched data: {missing_cols}")
                for col in missing_cols:
                    if col != 'time':
                        df[col] = 0

            required_cols = ['time', 'temperature']
            missing_required = [col for col in required_cols if col not in df.columns]
            if missing_required:
                raise ValueError(f"Missing required columns: {missing_required}")

            logger.info(f"Fetched data with columns: {df.columns.tolist()}")
            return df

        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            raise

    def update_historical_data(self, new_data: pd.DataFrame) -> None:
        if self.historical_data.empty:
            self.historical_data = new_data
        else:
            combined = pd.concat([self.historical_data, new_data])
            combined = combined.sort_values('time')
            combined = combined.drop_duplicates(subset=['time'], keep='last')
            self.historical_data = combined

        logger.info(f"Historical data updated. Records: {len(self.historical_data)}")
        logger.info(f"Date range: {self.historical_data['time'].min()} to {self.historical_data['time'].max()}")

    def categorize_weather(self, row: pd.Series) -> str:
        if 'precipitation (mm)' in row and row['precipitation (mm)'] > 10:
            return 'Heavy Rain'
        elif 'precipitation (mm)' in row and row['precipitation (mm)'] > 2.5:
            return 'Moderate Rain'
        elif 'precipitation (mm)' in row and row['precipitation (mm)'] > 0:
            return 'Light Rain'
        elif 'temperature' in row and row['temperature'] > 40:
            return 'Heat Wave'
        elif 'cloud_cover (%)' in row and row['cloud_cover (%)'] > 80:
            return 'Overcast'
        elif 'cloud_cover (%)' in row and row['cloud_cover (%)'] > 50:
            return 'Cloudy'
        elif 'cloud_cover (%)' in row and row['cloud_cover (%)'] > 20:
            return 'Partly Cloudy'
        else:
            return 'Clear'

    def explore_data(self) -> Dict:
        """Generate exploratory data analysis as text"""
        df = self.historical_data.copy()
        if df.empty:
            logger.warning("No data to explore")
            return {}

        df['weather_condition'] = df.apply(self.categorize_weather, axis=1)
        
        # Basic statistics
        stats = {
            'temperature': {
                'min': round(df['temperature'].min()),
                'max': round(df['temperature'].max()),
                'mean': round(df['temperature'].mean(), 1),
                'median': round(df['temperature'].median(), 1)
            },
            'humidity': {
                'min': round(df['relative_humidity'].min()),
                'max': round(df['relative_humidity'].max()),
                'mean': round(df['relative_humidity'].mean(), 1),
                'median': round(df['relative_humidity'].median(), 1)
            },
            'precipitation': {
                'total': round(df['precipitation (mm)'].sum(), 1),
                'max': round(df['precipitation (mm)'].max(), 1),
                'days_with_rain': len(df[df['precipitation (mm)'] > 0])
            },
            'weather_conditions': dict(df['weather_condition'].value_counts())
        }
        
        return stats

    def view_historical_trends(self, start_date: str = None, end_date: str = None, metrics: List[str] = None) -> Dict:
        """Return historical trends as text data"""
        logger.info("Generating historical trends data")

        try:
            if self.historical_data.empty:
                raise ValueError("No historical data available.")

            if not start_date or not end_date:
                end_date = self.historical_data['time'].max()
                start_date = end_date - timedelta(days=30)
            else:
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)

            mask = (self.historical_data['time'] >= start_date) & (self.historical_data['time'] <= end_date)
            df = self.historical_data[mask].copy()

            if df.empty:
                raise ValueError("No data available for the specified date range")

            available_metrics = [
                'temperature', 'relative_humidity', 'precipitation (mm)',
                'cloud_cover (%)', 'wind_speed_10m (km/h)', 'pressure_msl (hPa)'
            ]
            if not metrics:
                metrics = ['temperature', 'relative_humidity', 'precipitation (mm)']

            metrics = [m for m in metrics if m in df.columns and m in available_metrics]
            if not metrics:
                raise ValueError("No valid metrics selected")

            results = {}
            for metric in metrics:
                results[metric] = {
                    'min': round(df[metric].min(), 2),
                    'max': round(df[metric].max(), 2),
                    'mean': round(df[metric].mean(), 2),
                    'median': round(df[metric].median(), 2),
                    'latest': round(df[metric].iloc[-1], 2) if len(df) > 0 else None
                }

            return {
                'location': self.location_name,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'metrics': results
            }

        except Exception as e:
            logger.error(f"Error generating historical trends: {e}")
            raise

    def create_features(self, data: Optional[pd.DataFrame] = None, target: str = 'temperature', hours_ahead: int = 24) -> Tuple[pd.DataFrame, pd.Series]:
        """Create features for weather prediction with improved handling for single-row inputs"""
        df = data.copy() if data is not None else self.historical_data.copy()
        
        if df.empty:
            logger.error("Input data is empty")
            raise ValueError("No data available for feature creation")
            
        if 'time' not in df.columns:
            logger.error("'time' column missing in input data")
            raise ValueError("'time' column is required")
            
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time')
        
        # Ensure essential columns exist, fill missing with defaults
        essential_cols = ['temperature', 'relative_humidity', 'precipitation (mm)', 
                        'cloud_cover (%)', 'wind_speed_10m (km/h)', 'pressure_msl (hPa)']
        for col in essential_cols:
            if col not in df.columns:
                logger.warning(f"Missing column {col}, filling with 0")
                df[col] = 0
        
        # Handle single-row input by avoiding invalid operations
        is_single_row = len(df) == 1
        
        # Time-based features (safe for single row)
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['month'] = df['time'].dt.month
        df['day_of_year'] = df['time'].dt.dayofyear
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        if target == 'temperature':
            # Temperature-specific features
            if not is_single_row:
                # Lagged and rolling features for multi-row data
                for lag in [1, 6, 12, 24]:
                    df[f'temperature_lag_{lag}'] = df['temperature'].shift(lag)
                df['temperature_rolling_avg_24h'] = df['temperature'].rolling(window=24, min_periods=1).mean()
                df['temperature_rolling_std_24h'] = df['temperature'].rolling(window=24, min_periods=1).std()
            else:
                # For single row, use current values or historical means
                for lag in [1, 6, 12, 24]:
                    df[f'temperature_lag_{lag}'] = df['temperature'].iloc[0] if not self.historical_data.empty else 0
                df['temperature_rolling_avg_24h'] = df['temperature'].iloc[0] if not self.historical_data.empty else 0
                df['temperature_rolling_std_24h'] = 0  # No variation for single point
            
            df['target'] = df['temperature'] if is_single_row else df['temperature'].shift(-hours_ahead)
            
            feature_cols = [
                'relative_humidity', 'pressure_msl (hPa)', 'cloud_cover (%)', 
                'wind_speed_10m (km/h)', 'hour', 'day_of_week', 'month', 
                'day_of_year', 'is_weekend', 'temperature_lag_1', 'temperature_lag_6', 
                'temperature_lag_12', 'temperature_lag_24', 'temperature_rolling_avg_24h', 
                'temperature_rolling_std_24h'
            ]
            
        elif target == 'weather_condition':
            # Weather condition features
            df['weather_condition'] = df.apply(self.categorize_weather, axis=1)
            
            if not is_single_row:
                # Lagged and rolling features for multi-row data
                for lag in [1, 6, 12, 24]:
                    df[f'precipitation_lag_{lag}'] = df['precipitation (mm)'].shift(lag)
                    df[f'cloud_cover_lag_{lag}'] = df['cloud_cover (%)'].shift(lag)
                    df[f'temperature_lag_{lag}'] = df['temperature'].shift(lag)
                    df[f'humidity_lag_{lag}'] = df['relative_humidity'].shift(lag)
                df['precipitation_rolling_sum_6h'] = df['precipitation (mm)'].rolling(window=6, min_periods=1).sum()
                df['precipitation_rolling_sum_24h'] = df['precipitation (mm)'].rolling(window=24, min_periods=1).sum()
                df['cloud_cover_rolling_avg_24h'] = df['cloud_cover (%)'].rolling(window=24, min_periods=1).mean()
                df['temperature_rolling_avg_24h'] = df['temperature'].rolling(window=24, min_periods=1).mean()
            else:
                # For single row, use current values or defaults
                for lag in [1, 6, 12, 24]:
                    df[f'precipitation_lag_{lag}'] = df['precipitation (mm)'].iloc[0]
                    df[f'cloud_cover_lag_{lag}'] = df['cloud_cover (%)'].iloc[0]
                    df[f'temperature_lag_{lag}'] = df['temperature'].iloc[0]
                    df[f'humidity_lag_{lag}'] = df['relative_humidity'].iloc[0]
                df['precipitation_rolling_sum_6h'] = df['precipitation (mm)'].iloc[0]
                df['precipitation_rolling_sum_24h'] = df['precipitation (mm)'].iloc[0]
                df['cloud_cover_rolling_avg_24h'] = df['cloud_cover (%)'].iloc[0]
                df['temperature_rolling_avg_24h'] = df['temperature'].iloc[0]
            
            df['target'] = df['weather_condition'] if is_single_row else df['weather_condition'].shift(-hours_ahead)
            
            feature_cols = [
                'relative_humidity', 'pressure_msl (hPa)', 'cloud_cover (%)', 
                'wind_speed_10m (km/h)', 'hour', 'day_of_week', 'month', 
                'day_of_year', 'is_weekend', 'precipitation (mm)', 
                'precipitation_lag_1', 'precipitation_lag_6', 'precipitation_lag_12', 
                'precipitation_lag_24', 'cloud_cover_lag_1', 'cloud_cover_lag_6', 
                'cloud_cover_lag_12', 'cloud_cover_lag_24', 'temperature_lag_1', 
                'temperature_lag_6', 'temperature_lag_12', 'temperature_lag_24', 
                'humidity_lag_1', 'humidity_lag_6', 'humidity_lag_12', 'humidity_lag_24', 
                'precipitation_rolling_sum_6h', 'precipitation_rolling_sum_24h', 
                'cloud_cover_rolling_avg_24h', 'temperature_rolling_avg_24h'
            ]
        
        # Filter available feature columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        if not feature_cols:
            logger.error("No valid feature columns available")
            raise ValueError("No valid features generated")
        
        # Handle missing values
        numeric_cols = df[feature_cols].select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            if df[col].isna().any():
                logger.warning(f"Filling NaN in {col} with 0")
                df[col] = df[col].fillna(0)
        
        # Select features and target
        X = df[feature_cols]
        y = df['target']
        
        # Ensure non-empty output
        if X.empty or y.empty:
            logger.error(f"Feature DataFrame empty: X shape={X.shape}, y shape={y.shape}")
            raise ValueError("Feature creation resulted in empty data")
        
        # Final NaN check
        if X.isna().any().any():
            logger.warning(f"Found {X.isna().sum().sum()} NaN values in features - filling with 0")
            X = X.fillna(0)
        
        if target == 'weather_condition':
            y = y.fillna('Clear').astype(str)
        else:
            y = y.fillna(0 if is_single_row else y.mean())
        
        logger.info(f"Created features for {target}. Shape: {X.shape}, Features: {feature_cols}")
        return X, y

    def train_models(self) -> None:
        try:
            X_temp, y_temp = self.create_features(target='temperature')
            self.train_temperature_model(X_temp, y_temp)
            
            X_cond, y_cond = self.create_features(target='weather_condition')
            self.train_condition_model(X_cond, y_cond)
            
            logger.info("Both temperature and condition models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training models: {e}", exc_info=True)
            raise

    def train_temperature_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train temperature prediction model with improved validation"""
        X=X.fillna(0)
        y=y.fillna(y.mean())
        tscv = TimeSeriesSplit(n_splits=5)  # More splits for better validation
        
        models = {
            'RandomForest': make_pipeline(
                SimpleImputer(strategy='mean'),
                StandardScaler(),
                RandomForestRegressor(
                    n_estimators=200,
                    min_samples_leaf=3,
                    max_features=0.5,
                    random_state=42,
                    n_jobs=-1
                )
            ),
            'GradientBoosting': make_pipeline(
                SimpleImputer(strategy='mean'),
                StandardScaler(),
                GradientBoostingRegressor(
                    n_estimators=150,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                )
            ),
            'RidgeRegression': make_pipeline(
                StandardScaler(),
                SimpleImputer(strategy='mean'),
                Ridge(alpha=1.0)
            )
        }
        
        best_score = float('inf')
        best_model = None
        
        for name, model in models.items():
            logger.info(f"Training {name} for temperature prediction...")
            maes = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mae = mean_absolute_error(y_test, preds)
                maes.append(mae)
            
            avg_mae = np.mean(maes)
            std_mae = np.std(maes)
            logger.info(f"{name} MAE: {avg_mae:.2f}°C ± {std_mae:.2f}")
            
            if avg_mae < best_score:
                best_score = avg_mae
                best_model = model
        
        if best_model is None:
            raise ValueError("No valid model trained")
        
        # Final training on all data
        best_model.fit(X, y)
        self.temperature_model = best_model
        self.temperature_features = X.columns.tolist()
        self.last_update = datetime.now()
        
        logger.info(f"Best temperature model trained with MAE: {best_score:.2f}°C")
        logger.debug(f"Feature importances: {self.get_feature_importances()}")

    def train_condition_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train weather condition classifier with improved validation"""
        tscv = TimeSeriesSplit(n_splits=5)
        
        model = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=3,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        )
        
        logger.info("Training weather condition classifier...")
        accuracies = []
        f1_scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            accuracies.append(accuracy_score(y_test, preds))
            f1_scores.append(f1_score(y_test, preds, average='weighted'))
        
        avg_accuracy = np.mean(accuracies)
        avg_f1 = np.mean(f1_scores)
        
        logger.info(f"Condition model accuracy: {avg_accuracy:.2f} ± {np.std(accuracies):.2f}")
        logger.info(f"Condition model F1 score: {avg_f1:.2f} ± {np.std(f1_scores):.2f}")
        
        # Final training on all data
        model.fit(X, y)
        self.condition_model = model
        self.condition_features = X.columns.tolist()
        
        logger.debug(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        logger.debug(f"Feature importances: {self.get_feature_importances(target='condition')}")

    def predict_weather(self, date_time: datetime, hours_ahead: int = 24) -> Dict:
        """Make weather prediction with robust data validation and error handling"""
        logger.info(f"Attempting weather prediction for {date_time}")
        
        if not self.temperature_model or not self.condition_model:
            error_msg = "Models not trained. Please set a location and update models first."
            logger.error(error_msg)
            return {
                'error': error_msg,
                'success': False,
                'datetime': date_time.strftime('%Y-%m-%d %H:%M:%S') if isinstance(date_time, datetime) else str(date_time),
                'prediction_made_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        try:
            # Validate and parse input time
            prediction_time = pd.to_datetime(date_time)
            current_time = datetime.now()
            
            if prediction_time <= current_time:
                error_msg = "Prediction time must be in the future"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Calculate needed forecast period with buffer
            hours_diff = (prediction_time - current_time).total_seconds() / 3600
            days_needed = min(14, int(np.ceil(hours_diff / 24) + 2))  # Add buffer day
            
            # Fetch forecast data with validation
            logger.info(f"Fetching forecast data for {days_needed} days")
            forecast_data = self.fetch_weather_data(days=days_needed, forecast=True)
            if forecast_data.empty:
                error_msg = "No forecast data available from API"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            forecast_data['time'] = pd.to_datetime(forecast_data['time'])
            
            # Find closest matching time point
            time_diffs = abs((forecast_data['time'] - prediction_time).dt.total_seconds())
            if time_diffs.empty:
                error_msg = "No matching time points found in forecast data"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            closest_idx = time_diffs.idxmin()
            closest_data = forecast_data.loc[[closest_idx]].copy()
            
            # Validate we have required data columns
            required_columns = {'time', 'temperature'}
            if not required_columns.issubset(closest_data.columns):
                missing = required_columns - set(closest_data.columns)
                error_msg = f"Missing required columns in forecast data: {missing}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Create features with validation
            logger.info("Creating features for prediction")
            X_temp, _ = self.create_features(data=closest_data, target='temperature')
            X_cond, _ = self.create_features(data=closest_data, target='weather_condition')
            
            if X_temp.empty or X_cond.empty:
                error_msg = "Feature creation returned empty data"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Ensure feature alignment with training
            X_temp = X_temp.reindex(columns=self.temperature_features, fill_value=0)
            X_cond = X_cond.reindex(columns=self.condition_features, fill_value=0)
            
            # Validate data shape before prediction
            if X_temp.shape[0] == 0 or X_cond.shape[0] == 0:
                error_msg = "No samples available for prediction after feature processing"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Apply preprocessing transformations
            try:
                if hasattr(self.temperature_model.steps[0][1], 'transform'):
                    X_temp = self.temperature_model.steps[0][1].transform(X_temp)
                if hasattr(self.condition_model.steps[0][1], 'transform'):
                    X_cond = self.condition_model.steps[0][1].transform(X_cond)
            except Exception as e:
                error_msg = f"Preprocessing failed: {str(e)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Make predictions
            try:
                temp_pred = float(self.temperature_model.predict(X_temp)[0])
                cond_pred = self.condition_model.predict(X_cond)[0]
                cond_proba = self.condition_model.predict_proba(X_cond)[0]
                classes = self.condition_model.named_steps['randomforestclassifier'].classes_
            except Exception as e:
                error_msg = f"Prediction failed: {str(e)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Format and validate results
            if not isinstance(temp_pred, (int, float)):
                error_msg = "Invalid temperature prediction"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            if cond_pred not in classes:
                error_msg = "Invalid weather condition prediction"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            proba_dict = {
                cls: round(float(prob), 3) 
                for cls, prob in zip(classes, cond_proba)
                if isinstance(prob, (int, float))
            }
            
            logger.info(f"Prediction successful for {prediction_time}")
            return {
                'location': self.location_name,
                'latitude': self.latitude,
                'longitude': self.longitude,
                'datetime': prediction_time.strftime('%Y-%m-%d %H:%M:%S'),
                'prediction_made_at': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'temperature': round(temp_pred, 1),
                'temperature_unit': '°C',
                'weather_condition': cond_pred,
                'condition_probabilities': proba_dict,
                'prediction_horizon_hours': hours_ahead,
                'model_last_updated': self.last_update.strftime('%Y-%m-%d %H:%M:%S') if self.last_update else None,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Prediction failed for {date_time}: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'success': False,
                'datetime': date_time.strftime('%Y-%m-%d %H:%M:%S') if isinstance(date_time, datetime) else str(date_time),
                'prediction_made_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        
    def get_feature_importances(self, target: str = 'temperature') -> Dict:
        """Get feature importances from trained models"""
        try:
            if target == 'temperature' and self.temperature_model:
                # Access the RandomForestRegressor from the pipeline
                model = self.temperature_model.named_steps.get('randomforestregressor')
                if model is None:
                    # Handle case where the model is GradientBoosting or Ridge
                    model = self.temperature_model.named_steps.get('gradientboostingregressor') or \
                            self.temperature_model.named_steps.get('ridge')
                    if model is None:
                        return {}
                features = self.temperature_features
            elif target == 'condition' and self.condition_model:
                # Access the RandomForestClassifier from the pipeline
                model = self.condition_model.named_steps.get('randomforestclassifier')
                if model is None:
                    return {}
                
                features = self.condition_features
            else:
                return {}
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return dict(sorted(zip(features, importances), key=lambda x: x[1], reverse=True))
            elif hasattr(model, 'coef_'):  # For Ridge regression
                importances = np.abs(model.coef_)
                return dict(sorted(zip(features, importances), key=lambda x: x[1], reverse=True))
            return {}

        except Exception as e:
            logger.error(f"Error getting feature importances: {e}")
            return {}

    def update_model(self) -> None:
        logger.info("Starting model update")
        
        try:
            historical_data = self.fetch_weather_data(days=90)
            logger.info(f"Historical data shape: {historical_data.shape}")
            self.update_historical_data(historical_data)
            
            logger.info("Training models...")
            self.train_models()
            
            logger.info("Model update completed")
            
        except Exception as e:
            logger.error(f"Update failed: {e}", exc_info=True)
            raise