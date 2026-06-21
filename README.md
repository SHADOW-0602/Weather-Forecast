# AeroClim: Climate Intelligence Platform

AeroClim is an interactive weather analytics, climate risk prediction, and machine learning dashboard designed to model extreme hazards (such as flash floods or severe storm fronts) through a physics-informed multimodal late-fusion framework.

---

## 🚀 Key Features

### 1. Multimodal Late Fusion Modeling
AeroClim couples sequential temporal forecasting with tabular risk classification using three models:
* **LSTM Network**: Analyzes a sliding window sequence of `(Batch, 30 Days, 5 Features)` (max/min temps, precipitation, and sea surface temperature) to forecast tomorrow's land temperature.
* **Random Forest Classifier**: Integrates 19 atmospheric and oceanic teleconnection parameters to calculate raw hazard classification scores.
* **Late Fusion Meta-Learner**: Concatenates both prediction streams and passes them to a Dense Feed-Forward Neural Network to compute the final **Fused Hazard Probability**.

### 2. Interactive Analytics Dashboard
Built entirely in Streamlit, the platform features:
* **Live NOAA Station Data**: View historical weather records, accumulated precipitation, and monthly summary tables for discovered stations.
* **Sea Surface Temperature (SST)**: Visualizes historical trends of thermal gradients between land temperatures and ocean basins.
* **Extreme Hazard Predictor**: Run live predictions using interactive sliders (max/min temp, dew point, wind, rain) with hazard ratings displayed on a custom Plotly **Speedometer Gauge**.
* **Model Explanability**: Includes **SHAP relative attributions** (local factor influences) and **Gini global feature importances** (model weights).
* **Radar & CNN Activations**: Generates side-by-side **Seaborn heatmaps** visualizing NEXRAD reflectivity grids and CNN Grad-CAM class activations.
* **Data Lineage & Specifications**: Outlines feature engineering, coupled indices (heat index, wind chill, wet bulb, drought indexes), and model configuration.

---

## 📁 Repository Structure

```text
├── assets/
│   └── logo.png              # Application logo icon
├── data/
│   ├── station_locations.csv # Registry mapping coordinates, climate zones, elevation
│   ├── seattle.csv           # Ingested daily station records (Seattle)
│   ├── new_york.csv          # Ingested daily station records (New York)
│   ├── phoenix.csv           # Ingested daily station records (Phoenix)
│   ├── atmosphere.csv        # Auxiliary dew point, wind, humidity, pressure data
│   ├── soil_moisture.csv     # Auxiliary soil parameters (volume, saturation)
│   ├── climate_indices.csv   # Climate indexes (ENSO, PDO, NAO)
│   └── ocean.csv             # Copernicus SST climate records
├── saved_models/
│   ├── lstm_final.keras      # Trained LSTM weights
│   ├── rf_model.pkl          # Trained Random Forest classifier
│   ├── fusion_final.keras    # Trained Late Fusion Meta-Learner
│   ├── scaler_*.pkl          # MinMax scaling files
│   └── metrics.json          # Compiled test set performance metrics
├── app.py                    # Streamlit Dashboard application
├── noaa_client.py            # Dynamic data parser & simulation engine
├── ml_model.py               # Multimodal prediction & training pipeline
├── gradcam_radar.py          # CNN Grad-CAM activation generator
├── train_now.py              # CLI training automation script
├── requirements.txt          # Python dependency list
└── .gitignore                # Version control excludes
```

---

## 🛠️ Getting Started

### 1. Install Dependencies
Make sure you have Python 3.10+ installed. In your terminal, install the dependencies listed in `requirements.txt`:
```powershell
pip install -r requirements.txt
```

*Note: Streamlit, TensorFlow, Scikit-Learn, Plotly, Seaborn, and Matplotlib are required.*

### 2. Train the Models (Optional)
Pre-trained models are stored in `saved_models/`. However, you can re-train the models for any station using `train_now.py`:
```powershell
python train_now.py --city SEATTLE --days 1000
```
Parameters:
* `--city`: Choose between `SEATTLE`, `NEW_YORK`, or `PHOENIX`.
* `--days`: Historical lookback window range (defaults to 10000 days).

### 3. Launch the Dashboard
Run the Streamlit application through the Python interpreter:
```powershell
python -m streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`.

---

## 📊 Feature Engineering & Calculations

The raw weather feeds are combined daily to engineer coupled indices:
* **HEAT_INDEX**: Humiture index (apparent temperature felt by the human body).
* **WIND_CHILL**: Perceived cold temperature due to wind flow.
* **WET_BULB**: The evaporative limit of the air.
* **DROUGHT_IDX**: Rolling 30-day precipitation deficits.
* **SST_AIR_DIFF**: Coupled thermal gradient ($T_{AVG} - SST$) between land air temperature and sea surface temperature.
