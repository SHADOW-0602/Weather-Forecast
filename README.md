# AeroClim: Climate Intelligence Platform

AeroClim is an interactive weather analytics, climate risk prediction, and machine learning dashboard designed to model extreme hazards (such as flash floods or severe storm fronts) through a physics-informed multimodal late-fusion framework.

---

## 🚀 Key Features

### 1. Multimodal Late Fusion Modeling
AeroClim couples sequential temporal forecasting with tabular risk classification using three models:
* **LSTM Network**: Analyzes a sliding window sequence of `(Batch, 30 Days, 5 Features)` (max/min temps, precipitation, and sea surface temperature) to forecast tomorrow's land temperature.
* **Random Forest Classifier**: Integrates 36 atmospheric, soil, oceanic, teleconnection, station-geometry, lagged-storm, and data-quality parameters to calculate raw hazard classification scores.
* **Late Fusion Meta-Learner**: Concatenates both prediction streams and passes them to a Dense Feed-Forward Neural Network to compute the final **Fused Hazard Probability**.
* **Separate NOAA Event Model**: Trains an event-focused Random Forest on Storm Events coverage years, tunes its alert threshold on a validation year, and reports a separate event probability.

### 2. Interactive Analytics Dashboard
Built entirely in Streamlit, the platform features:
* **Prepared NOAA Station Archive**: View historical records, accumulated precipitation, and monthly summaries for the reduced 20-station NOAA subset used by the saved model artifacts.
* **Sea Surface Temperature (SST)**: Visualizes historical trends of thermal gradients between land temperatures and ocean basins.
* **Extreme Hazard Predictor**: Run live predictions using interactive sliders (max/min temp, dew point, wind, rain) with hazard ratings displayed on a custom Plotly **Speedometer Gauge**.
* **Model Explainability**: Includes a local relative-contribution heuristic and Random Forest Gini feature importances.
* **Radar & CNN Activations**: Generates side-by-side **Seaborn heatmaps** visualizing NEXRAD reflectivity grids and CNN Grad-CAM class activations.
* **Data Lineage & Specifications**: Outlines feature engineering, coupled indices (heat index, wind chill, wet bulb, drought indexes), and model configuration.

---

## 📁 Repository Structure

```text
├── assets/
│   ├── logo.png              # Application logo icon
│   └── mrms_gallery/         # Static MRMS radar images and manifest json
├── data/
│   ├── noaa_stations/        # Cleaned daily station files and station catalog
│   ├── noaa_atmosphere/      # NOAA GSOD dew point, wind, pressure, humidity
│   ├── soil_moisture/        # ERA5-Land layer and root-zone moisture
│   ├── climate_indices.csv   # Climate indexes (ENSO, PDO, NAO)
│   └── ocean.csv             # Basin-level SST context
├── report_tools/             # Tools for generating term reports and dashboard evidence
│   ├── build_ieee_report.py  # Generates an academic IEEE-styled report
│   ├── build_ntcc_term_report.py # Generates a term report
│   ├── generate_report_assets.py # Renders report charts and figures
│   └── report_dashboard.py   # Streamlit evidence dashboard for reports
├── saved_models/
│   ├── lstm_final.keras      # Trained LSTM weights
│   ├── rf_model.pkl          # Trained Random Forest classifier
│   ├── fusion_final.keras    # Trained Late Fusion Meta-Learner
│   ├── scaler_*.pkl          # MinMax scaling files
│   └── metrics.json          # Compiled test set performance metrics
├── tests/                    # Unit and integration test suite
│   ├── test_data_pipeline.py # Tests station catalog and data integrity
│   └── test_models.py        # Tests model artifact reload and metrics
├── tools/                    # Core pipeline and data fetching utilities
│   ├── audit_data_health.py  # Evaluates availability of station datasets
│   ├── build_mrms_image_chips.py # Tensors build tool for radar composite grids
│   ├── fetch_noaa_event_labels.py # Downloads NOAA Storm Events labels
│   └── ...                   # Additional fetch/process CLI utilities
├── app.py                    # Streamlit Dashboard application
├── explanation.html          # Static scientific formulation & architecture dashboard
├── gradcam_radar.py          # CNN Grad-CAM activation generator
├── ml_model.py               # Multimodal prediction & training pipeline
├── mrms_gallery.html         # Static interactive MRMS radar dataset gallery
├── noaa_client.py            # Dynamic data parser & simulation engine
├── pytest.ini                # Pytest configuration file
├── requirements.txt          # Python dependency list
├── train_now.py              # CLI training automation script
├── train_pooled.py           # Pooled model training across ready stations
├── .gitignore                # Version control excludes
└── README.md                 # Project documentation
```

---

## 🛠️ Getting Started

### 1. Set Up Virtual Environment & Install Dependencies
Make sure you have Python 3.10+ installed. We recommend setting up a virtual environment to avoid package conflicts (such as numpy 2.x incompatibilities with older TensorFlow versions):
```powershell
# Create a virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

*Note: Streamlit, TensorFlow, Scikit-Learn, Plotly, Seaborn, and Matplotlib are required.*

### 2. Train the Models (Optional)
Pre-trained models are stored in `saved_models/`. However, you can re-train the models for any station using `train_now.py`:
```powershell
python train_now.py --city USW00024233 --days 1000
```

To prepare the larger NOAA daily export for the three configured stations:

```powershell
python tools/prepare_noaa_daily.py C:\path\to\4339672.csv
```

This writes cleaned, SI-unit files for every station under
`data/noaa_stations/`, plus a `stations.csv` catalog. AeroClim automatically
adds every cataloged station to the dashboard. Root-level city CSVs are no
longer required.

Fetch daily dew point, wind, pressure, and derived relative humidity from the
official NOAA Global Summary of the Day archive:

```powershell
python tools/fetch_noaa_atmosphere.py --start-year 1995 --end-year 2024
```

The downloader maps the NOAA WBAN identifiers, fetches station-years in
parallel, records retryable failures, and writes merge-ready files under
`data/noaa_atmosphere/`.

Repair short and medium atmospheric gaps from nearby same-day stations:

```powershell
python tools/repair_atmospheric_gaps.py
```

Fetch ERA5-Land soil moisture after setting `CDSAPI_KEY`:

```powershell
$env:CDSAPI_URL="https://cds.climate.copernicus.eu/api"
$env:CDSAPI_KEY="<your-personal-access-token>"
python tools/fetch_era5_soil_moisture.py
```

The output contains layer-1, layer-2, root-zone volumetric soil moisture, and a
station-relative saturation percentile under `data/soil_moisture/`.

Audit all prepared sources before training:

```powershell
python tools/audit_data_health.py
```

Or refresh all lightweight auxiliary sources and audits together:

```powershell
python tools/refresh_all_data.py
```

Create independent NOAA Storm Events station-day labels. The trainer now uses
these labels as an independent 2022-2024 event/no-event target wherever the
Storm Events coverage window is available:

```powershell
python tools/fetch_noaa_event_labels.py --start-year 2022 --end-year 2024 --radius-km 100
```

Important modeling note: Storm Events labels are sparse and represent reported
events, not a dense physical hazard field. Current saved artifacts report the
broad hazard-proxy metrics separately from a NOAA event-specific model, so you
can see whether the proxy and reported-event tasks diverge.

The normal loader does not silently substitute synthetic weather when a
prepared station file is missing. Synthetic generation is used only when
explicitly requested.
Parameters:
* `--city`: Use any NOAA station ID in the reduced
  `data/noaa_stations/stations.csv` catalog.
* `--days`: Historical lookback window range (defaults to 12000 days so the
  complete 1995-2024 archive is included).

To train a broader pooled model across quality-approved stations:

```powershell
python train_pooled.py --max-stations 20
```

For a heavier research run across every station marked ready by the health
audit:

```powershell
python train_pooled.py --all-ready --lstm-epochs 15 --rf-estimators 100
```

For a faster schema-valid refresh:

```powershell
python train_pooled.py --max-stations 20 --lstm-epochs 6 --rf-estimators 60
```

### 4. MRMS 2D CNN image pipeline

Download a bounded live MRMS sample for the three CNN image channels:

```powershell
python tools/fetch_mrms_recent.py --max-files 2
```

Download a small historical MRMS sample that overlaps NOAA Storm Events labels:

```powershell
python tools/fetch_mrms_historical_sample.py --max-dates 12 --target-hour 18
```

For a stronger image model, collect more dates, multiple UTC hours, and quiet
non-event dates for negative examples:

```powershell
python tools/fetch_mrms_historical_sample.py --max-dates 100 --negative-dates 30 --target-hours 0,6,12,18 --skip-errors
```

Convert the downloaded GRIB2 files into station-centered CNN tensors:

```powershell
python tools/build_mrms_image_chips.py --chip-size 64
```

For historical CNN training chips:

```powershell
python tools/build_mrms_image_chips.py --raw-dir data\mrms_historical_raw --out-dir data\mrms_historical_images --chip-size 64 --event-window-before-hours 6 --event-window-after-hours 6 --ambiguous-window-hours 12
```

You can retune the event-timing labels without re-decoding the GRIB2 files:

```powershell
python tools/relabel_mrms_chips.py --event-window-before-hours 6 --event-window-after-hours 6 --ambiguous-window-hours 12
```

Train the CNN once you have historical MRMS chips that overlap
`data/event_labels.csv`:

```powershell
python tools/train_mrms_cnn_event_model.py --dataset data\mrms_historical_images\mrms_live_chips.npz --manifest data\mrms_historical_images\manifest.csv --epochs 30 --batch-size 32 --no-balance-train --loss bce
```

Train the optional point-9 fusion model that combines image probability from
the MRMS CNN with the NOAA tabular event model:

```powershell
python tools/train_cnn_tabular_fusion.py
```

The current 2D CNN pipeline uses three image channels:
`MergedReflectivityQCComposite`, `RadarOnly_QPE_01H`, and
`MergedAzShear_0-2kmAGL`. The live MRMS directory is useful for pipeline
testing, but supervised CNN training needs historical MRMS files from the NOAA
archive so image timestamps can be matched to NOAA Storm Events labels. Event
labels include begin/end timestamps. Chips are positive only when the MRMS
timestamp falls inside the configured event window, near-event chips outside
that positive window are marked unlabeled, and CNN/fusion evaluation uses a
chronological 2022 train, 2023 validation, and 2024 test split when enough
samples exist.

### 5. Launch the Dashboard
Run the Streamlit application through the Python interpreter:
```powershell
python -m streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`.

### 6. Run the Test Suite
Ensure the dataset structure, data health audits, and model artifacts reload cleanly by running the pytest suite:
```powershell
pytest
```
*(If pytest has path collection errors, you can run `python -m pytest` or use the active virtual environment: `.venv\Scripts\python -m pytest`)*

### 7. Academic Reports & Evidence Dashboard
AeroClim includes scripts to generate publication-ready IEEE papers and a dedicated evidence dashboard:
* **Generate Report Assets**: Renders the figures and analysis results required for reports:
  ```powershell
  python report_tools/generate_report_assets.py
  ```
* **Build IEEE Report**: Compiles the LaTeX/PDF/HTML source structure of the academic paper:
  ```powershell
  python report_tools/build_ieee_report.py
  ```
* **Launch Report Dashboard**: Renders the standalone Streamlit evidence dashboard containing regional climate analyses and training summaries:
  ```powershell
  python -m streamlit run report_tools/report_dashboard.py
  ```

### 8. Interactive Static Dashboards (No Server Required)
AeroClim features two client-side interactive HTML files that can be opened directly in any web browser without running streamlit:
* **`explanation.html`**: A visually rich dashboard outlining the system architecture, mathematical formulations (e.g., wet bulb temperature, heat index, and teleconnection coefficients), and dynamic mathematical explanations.
* **`mrms_gallery.html`**: An interactive image viewer for checking Multi-Radar Multi-Sensor (MRMS) composite radar chips, NEXRAD reflectivities, and CNN model Grad-CAM activation examples.

---

## 📊 Feature Engineering & Calculations

The raw weather feeds are combined daily to engineer coupled indices:
* **HEAT_INDEX**: Humiture index (apparent temperature felt by the human body).
* **WIND_CHILL**: Perceived cold temperature due to wind flow.
* **WET_BULB**: The evaporative limit of the air.
* **DROUGHT_IDX**: Rolling 30-day precipitation deficits.
* **SST_AIR_DIFF**: Coupled thermal gradient ($T_{AVG} - SST$) between land air temperature and sea surface temperature.
