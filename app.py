import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import datetime
import json
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Import custom modules
from noaa_client import NOAAClient, CITIES
from ml_model import HazardPredictor
from gradcam_radar import make_gradcam_heatmap, generate_synthetic_heatmap

# 1. Page Configuration
st.set_page_config(
    page_title="AeroClim: Climate Intelligence Platform",
    page_icon="assets/logo.png",
    layout="wide"
)

# Load environment variable
load_dotenv(dotenv_path=".env")
NOAA_TOKEN = os.getenv("NOAA_API_TOKEN")

# Initialize client and ML model
@st.cache_resource
def load_resources():
    client = NOAAClient(token=NOAA_TOKEN)
    predictor = HazardPredictor()
    return client, predictor

noaa_client, predictor = load_resources()

@st.cache_data
def load_data_health():
    path = "data/data_health.json"
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    return {
        row["station_id"]: row
        for row in report.get("stations", [])
    }

data_health = load_data_health()

# Sidebar Control Layout
with st.sidebar:
    st.image("assets/logo.png", width=64)
    st.title("AeroClim Console")
    st.write("Climate Intelligence Platform")
    
    # Station Settings
    st.subheader("Station Settings")
    selected_city_id = st.selectbox(
        "Select Meteorological Station",
        options=list(CITIES.keys()),
        format_func=lambda cid: CITIES[cid]["name"]
    )
    
    city_data = CITIES[selected_city_id]
    station_health = data_health.get(selected_city_id, {})
    
    # Render station parameters
    st.info(f"""
    **Station Metadata:**
    - **Region:** {city_data["region"]}
    - **Climate Zone:** {city_data["climate"]}
    - **Elevation:** {city_data["elevation"]} m
    - **Coordinates:** {city_data["lat"]}° N, {abs(city_data["lon"])}° W
    """)
    if station_health and not station_health.get("training_ready", True):
        st.warning(
            "This station has substantial atmospheric gaps "
            f"({station_health.get('core_completeness_pct', 0):.1f}% core completeness). "
            "Use predictions cautiously or repair its GSOD mapping."
        )
    
    # Temporal Window Presets
    st.subheader("Temporal Window")
    time_preset = st.radio(
        "Climatic Window Range",
        options=["Past 30 Days", "Past 6 Months", "Full 12 Months"]
    )
    
    today = datetime.date.today()
    days_map = {"Past 30 Days": 30, "Past 6 Months": 180, "Full 12 Months": 365}
    start_date = today - datetime.timedelta(days=days_map.get(time_preset, 30))
    end_date = today
    
    # Query force simulation option
    force_sim = st.toggle("Override with Synthetic Generator", value=False)
    st.caption("AeroClim v1.3 - Dynamic Ingestion Engine")

# Main Application Content
st.title("AeroClim Analytics Dashboard")
st.markdown("An interactive platform evaluating weather anomalies, sea surface temperature dynamics, and machine learning models to assess hazards and predict extreme environmental risk indices.")

tab_live, tab_sst, tab_ml, tab_radar, tab_data = st.tabs([
    " NOAA Station Archive",
    " Sea Surface Temperature",
    " Flash Flood & Hazard Predictor",
    " Radar Heatmap & CNN Activations",
    " Data Lineage & Specifications"
])

# Fetch data globally to share between tabs
with st.spinner("Loading prepared meteorological records..."):
    weather_df = noaa_client.fetch_weather_data(
        selected_city_id, 
        start_date.strftime("%Y-%m-%d"), 
        end_date.strftime("%Y-%m-%d"),
        force_simulation=force_sim
    )

# Tab 1: Live NOAA Explorer
with tab_live:
    st.header("Historical Weather Station Explorer")
    
    if weather_df is not None and not weather_df.empty:
        st.write(f"Showing weather records from **{weather_df['Source'].iloc[0]}**.")
        
        # Summary metrics
        avg_temp = (weather_df["TMAX"].mean() + weather_df["TMIN"].mean()) / 2
        total_rain = weather_df["PRCP"].sum()
        max_temp = weather_df["TMAX"].max()
        rainy_days = (weather_df["PRCP"] > 0.1).sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Temp", f"{avg_temp:.1f} °C", f"Peak: {max_temp:.1f} °C")
        col2.metric("Min Recorded Temp", f"{weather_df['TMIN'].min():.1f} °C")
        col3.metric("Accumulated Rainfall", f"{total_rain:.1f} mm", f"Max Daily: {weather_df['PRCP'].max():.1f} mm")
        col4.metric("Rainy Days Count", f"{rainy_days} Days", f"{rainy_days/len(weather_df)*100.0:.1f}% of window")
            
        # Draw Plotly Climatic chart
        fig_weather = go.Figure()
        
        fig_weather.add_trace(go.Scatter(
            x=weather_df["Date"], y=weather_df["TMAX"],
            mode='lines', line=dict(color='rgba(239, 68, 68, 0.8)', width=1.5),
            name='Daily Max Temp (°C)'
        ))
        fig_weather.add_trace(go.Scatter(
            x=weather_df["Date"], y=weather_df["TMIN"],
            mode='lines', line=dict(color='rgba(56, 189, 248, 0.8)', width=1.5),
            fill='tonexty', fillcolor='rgba(139, 92, 246, 0.05)',
            name='Daily Min Temp (°C)'
        ))
        fig_weather.add_trace(go.Bar(
            x=weather_df["Date"], y=weather_df["PRCP"],
            name="Daily Precipitation (mm)",
            marker_color='rgba(167, 139, 250, 0.6)',
            yaxis="y2"
        ))
        
        fig_weather.update_layout(
            title=f"Climatic Trends & Precipitation Dynamics - {city_data['name']}",
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            yaxis2=dict(title="Precipitation (mm)", overlaying="y", side="right", showgrid=False),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(fig_weather, width="stretch")
        
        # Monthly summaries table
        st.subheader("Climatological Monthly Summaries")
        monthly_df = weather_df.copy()
        monthly_df["Month"] = monthly_df["Date"].dt.to_period("M").astype(str)
        
        month_summary = monthly_df.groupby("Month").agg(
            Avg_Max_Temp=("TMAX", "mean"),
            Avg_Min_Temp=("TMIN", "mean"),
            Peak_Daily_Temp=("TMAX", "max"),
            Total_Rainfall=("PRCP", "sum"),
            Rainy_Days_Count=("PRCP", lambda r: (r > 0.1).sum())
        ).reset_index()
        
        month_summary.columns = [
            "Period (Month)", "Avg Max Temp (°C)", "Avg Min Temp (°C)", 
            "Peak Temp (°C)", "Total Rainfall (mm)", "Rainy Days Count"
        ]
        st.dataframe(month_summary.round(1), hide_index=True, width="stretch")
    else:
        st.error("Severe network failure or empty dataset encountered. Please check fallback simulation settings.")

# Tab 2: Sea Surface Temperature
with tab_sst:
    st.header("Ocean Surface Dynamics (SST Climate Records)")
    st.write("Warm Sea Surface Temperature (SST) anomalies transfer latent heat and moisture, driving extreme weather anomalies over adjacent regions.")
    
    daily_path = "data/ocean.csv"
    if os.path.exists(daily_path):
        sst_df = pd.read_csv(daily_path)
        sst_date_col = "date" if "date" in sst_df.columns else sst_df.columns[0]
        sst_df["Date"] = pd.to_datetime(sst_df[sst_date_col])
        
        # Determine target column
        basin_mapping = {
            "SEATTLE": "sst_pacific_f", "KCQT": "sst_pacific_f",
            "NEW_YORK": "sst_atlantic_f", "KPHL": "sst_atlantic_f", "KJAX": "sst_atlantic_f", 
            "KCLT": "sst_atlantic_f", "KIND": "sst_atlantic_f", "KMDW": "sst_atlantic_f",
            "KHOU": "sst_gulf_f", "PHOENIX": "sst_gulf_f"
        }
        target_col = basin_mapping.get(selected_city_id) or (
            "sst_pacific_f" if city_data["lon"] <= -100 else
            "sst_gulf_f" if city_data["lat"] <= 31.5 and city_data["lon"] <= -80 else
            "sst_atlantic_f"
        )
        if target_col in sst_df.columns:
            sst_df["SST_Mean_Celsius"] = (sst_df[target_col] - 32) * 5.0 / 9.0
        else:
            avail_cols = [c for c in ["sst_pacific_f", "sst_atlantic_f", "sst_gulf_f"] if c in sst_df.columns]
            sst_df["SST_Mean_Celsius"] = (sst_df[avail_cols].mean(axis=1) - 32) * 5.0 / 9.0 if avail_cols else 18.0
        
        latest_row = sst_df.iloc[-1]
        col1, col2, col3 = st.columns(3)
        col1.metric("Latest SST Compiled", f"{latest_row['SST_Mean_Celsius']:.2f} °C", f"Date: {latest_row['Date'].strftime('%Y-%m-%d')}")
        col2.metric("Mean Historical SST", f"{sst_df['SST_Mean_Celsius'].mean():.2f} °C")
        col3.metric("Peak SST Temperature", f"{sst_df['SST_Mean_Celsius'].max():.2f} °C")
        
        # Trend line
        fig_sst = px.line(
            sst_df, x="Date", y="SST_Mean_Celsius",
            title="Satellite Sea Surface Temperature (1980 - 2025)",
            labels={"SST_Mean_Celsius": "SST (°C)"}
        )
        st.plotly_chart(fig_sst, width="stretch")
    else:
        st.warning("Ocean summary dataset `data/ocean.csv` not found.")

# Tab 3: Extreme Weather & Flood Risk Predictor
with tab_ml:
    st.header("Prototype Next-Day Environmental Hazard Predictor")
    st.caption("Research output only — not an official weather warning or emergency product.")
    
    col_inputs, col_results = st.columns(2)
    
    with col_inputs:
        st.subheader("Atmospheric Controls")
        current_tmax = st.slider("Today's Max Temp (°C)", -20.0, 50.0, 25.0, 0.5)
        current_tmin = st.slider("Today's Min Temp (°C)", -30.0, 40.0, 15.0, 0.5)
        dew_point = st.slider("Dew Point (°C)", -20.0, 30.0, 10.0, 0.5)
        wind_speed = st.slider("Wind Speed (km/h)", 0, 120, 15, 5)
        current_prcp = st.slider("Today's Rainfall (mm)", 0.0, 100.0, 5.0, 1.0)
        
        if predictor.rf_model is None:
            st.warning("Multimodal models are not trained yet. Train them below.")
        elif predictor.lstm_model is None or predictor.fusion_model is None:
            st.warning(
                "Only the Random Forest artifact is currently usable. "
                "Retrain on the full archive to restore LSTM and fusion output."
            )
            
        if st.button("Train Models (30 Years of Daily Coupled Data)", width="stretch"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            def progress_cb(pct, msg):
                progress_bar.progress(pct)
                status_text.text(msg)
            
            try:
                train_start = datetime.date(1995, 1, 1)
                train_df = noaa_client.fetch_weather_data(
                    selected_city_id, train_start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d"), force_simulation=force_sim
                )
                predictor.train_multimodal(train_df, progress_callback=progress_cb)
                st.success("Training complete! Models saved.")
                st.rerun()
            except Exception as e:
                st.error(f"Error during training: {e}")
                
        metrics = getattr(predictor, "metrics", None)
        if metrics:
            st.subheader("Model Validation Accuracy")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Meta-Learner Fusion", f"{metrics.get('fusion_accuracy', 0.0)*100.0:.1f}%")
            col_m2.metric("Random Forest", f"{metrics.get('rf_accuracy', 0.0)*100.0:.1f}%")
            col_m3.metric("LSTM R² Score", f"{metrics.get('lstm_r2', 0.0):.3f}")
            if metrics.get("event_labeled_test_rows", 0):
                st.caption(
                    "Independent NOAA Storm Events check: "
                    f"{metrics.get('event_labeled_test_rows', 0):,} labeled test rows, "
                    f"RF event AUC {metrics.get('rf_event_auc', 0.5):.3f}."
                )
            if metrics.get("event_model_trained"):
                st.caption(
                    "Separate NOAA event model: "
                    f"F1 {metrics.get('event_model_test_f1', 0.0):.3f}, "
                    f"recall {metrics.get('event_model_test_recall', 0.0):.3f}, "
                    f"threshold {metrics.get('event_model_threshold', 0.5)*100.0:.1f}%."
                )

    prediction = predictor.predict(
        weather_df=weather_df, current_tmax=current_tmax, current_tmin=current_tmin,
        dew_point=dew_point, wind_speed=wind_speed, current_prcp=current_prcp
    )
    
    with col_results:
        st.subheader("Computed Hazard Rating")
        
        # Speedometer Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction['Probability'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fused Hazard Probability", 'font': {'size': 18}},
            number={'suffix': "%", 'font': {'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "gray"},
                'bar': {'color': "#1f77b4"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#10b981'},     # Low Risk (Green)
                    {'range': [30, 60], 'color': '#f59e0b'},    # Moderate Alert (Yellow)
                    {'range': [60, 85], 'color': '#f97316'},    # Severe Warning (Orange)
                    {'range': [85, 100], 'color': '#ef4444'}    # Extreme Danger (Red)
                ],
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=260,
            margin=dict(l=30, r=30, t=50, b=10)
        )
        st.plotly_chart(fig_gauge, width="stretch")
        
        # Severity alert box
        severity, desc = prediction["Category"], prediction["Description"]
        if severity == "Low Risk": st.success(f"**{severity}**: {desc}")
        elif severity == "Moderate Alert": st.warning(f"**{severity}**: {desc}")
        else: st.error(f"**{severity}**: {desc}")
            
        st.markdown(f"**Diagnostic Explanation:**\n{prediction['Explanation']}")
        if prediction.get("Event_Prob") is not None:
            st.metric(
                "Separate NOAA Event Probability",
                f"{prediction['Event_Prob']:.1f}%",
                delta=(
                    "above tuned threshold"
                    if prediction.get("Event_Alert")
                    else "below tuned threshold"
                ),
            )

    st.divider()
    
    col_feat, col_shap = st.columns(2)
    with col_feat:
        st.subheader("Local Relative Contribution Heuristic")
        if "Contributions" in prediction:
            contrib_df = pd.DataFrame(list(prediction["Contributions"].items()), columns=["Factor", "Weight"])
            contrib_df = contrib_df.sort_values("Weight", ascending=True)
            fig_contrib = px.bar(contrib_df, x="Weight", y="Factor", orientation="h", title="Relative Feature Contributions")
            st.plotly_chart(fig_contrib, width="stretch")
        else:
            st.info("Relative contribution estimates will display here when models are trained.")
            
    with col_shap:
        st.subheader("Global Feature Importance (Gini)")
        try:
            importances_df = predictor.get_feature_importances()
            fig_global = px.bar(importances_df, x="Gini Importance", y="Feature", orientation="h", title="Global Model Importances")
            st.plotly_chart(fig_global, width="stretch")
        except Exception as e:
            st.error(f"Could not load global importances: {e}")

# Tab 4: Radar Heatmap & CNN Activations
with tab_radar:
    st.header("Precipitation Radar & CNN Activations (Seaborn Heatmaps)")
    st.write("This tab visualizes MRMS radar image chips and Convolutional Neural Network (CNN) class activations (Grad-CAM).")

    st.subheader("Real MRMS CNN Sample")
    try:
        mrms_npz = "data/mrms_historical_images/mrms_live_chips.npz"
        mrms_manifest = "data/mrms_historical_images/manifest.csv"
        cnn_path = "saved_models/cnn_event_model.keras"
        cnn_metrics_path = "saved_models/cnn_event_metrics.json"
        fusion_metrics_path = "saved_models/cnn_tabular_fusion_metrics.json"
        if os.path.exists(mrms_npz) and os.path.exists(mrms_manifest) and os.path.exists(cnn_path):
            import tensorflow as tf
            mrms = np.load(mrms_npz)
            mrms_meta = pd.read_csv(mrms_manifest)
            cnn_model = tf.keras.models.load_model(cnn_path)
            labels = mrms["y"]
            positive_indexes = np.where(labels == 1)[0]
            default_index = 0
            if len(positive_indexes):
                default_index = next((int(idx) for idx in positive_indexes if mrms["X"][idx][:, :, 0].max() > 0.15), int(positive_indexes[0]))
            sample_index = st.slider(
                "MRMS sample index",
                0,
                len(labels) - 1,
                default_index,
                key="mrms_sample_index",
            )
            chip = mrms["X"][sample_index]
            meta_row = mrms_meta.iloc[sample_index]
            probability = float(
                cnn_model.predict(chip[np.newaxis, ...], verbose=0).flatten()[0]
            )
            threshold = 0.5
            if os.path.exists(cnn_metrics_path):
                with open(cnn_metrics_path, "r", encoding="utf-8") as handle:
                    threshold = float(json.load(handle).get("threshold", 0.5))

            st.metric(
                "CNN MRMS Event Probability",
                f"{probability * 100.0:.1f}%",
                delta=(
                    "above tuned threshold"
                    if probability >= threshold
                    else "below tuned threshold"
                ),
            )
            st.caption(
                f"Station {meta_row['station_id']} · {meta_row['timestamp']} · "
                f"Label {int(meta_row['label'])} · Threshold {threshold*100.0:.1f}%"
            )

            if os.path.exists(fusion_metrics_path):
                with open(fusion_metrics_path, "r", encoding="utf-8") as handle:
                    fusion_metrics = json.load(handle)
                cols = st.columns(3)
                cols[0].metric(
                    "CNN + Tabular Fusion AUC",
                    f"{fusion_metrics.get('test_auc', 0.0):.3f}",
                )
                cols[1].metric(
                    "Fusion F1",
                    f"{fusion_metrics.get('test_f1', 0.0):.3f}",
                )
                cols[2].metric(
                    "Fusion Test Accuracy",
                    f"{fusion_metrics.get('test_accuracy', 0.0) * 100.0:.1f}%",
                )
                st.caption(
                    "Fusion combines MRMS CNN event probability with the NOAA tabular event model "
                    f"using {fusion_metrics.get('split_strategy', 'the saved split')}."
                )

            heatmap = make_gradcam_heatmap(chip[np.newaxis, ...], cnn_model, "final_conv")
            channel_titles = [
                "Reflectivity Composite",
                "Radar QPE 1H",
                "0-2 km AzShear",
                "CNN Grad-CAM",
            ]
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            for axis, title, image in zip(
                axes,
                channel_titles,
                [chip[:, :, 0], chip[:, :, 1], chip[:, :, 2], heatmap],
            ):
                sns.heatmap(image, ax=axis, cmap="viridis", cbar=False, xticklabels=False, yticklabels=False)
                axis.set_title(title)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Real MRMS CNN artifacts are not available yet; use the synthetic demo below.")
    except Exception as e:
        st.warning(f"Could not render real MRMS CNN sample: {e}")

    st.subheader("Synthetic CNN Activation Demo")
    
    grid_size = st.slider("Radar Visual Resolution (Grid Size)", 32, 128, 64, 16)
    
    if st.button("Generate Seaborn Radar Heatmaps", type="primary", width="stretch"):
        with st.spinner("Generating CNN activations..."):
            try:
                # Generate synthetic storm cells
                radar_grid = generate_synthetic_heatmap(height=grid_size, width=grid_size, n_storm_cells=3, seed=42)
                # scale to dBZ ranges (0 to 75 dBZ)
                radar_dbz = radar_grid * 75.0
                
                # Setup dummy CNN Grad-CAM
                import tensorflow as tf
                inputs = tf.keras.Input(shape=(grid_size, grid_size, 3))
                x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", name="conv1")(inputs)
                x = tf.keras.layers.MaxPooling2D(2)(x)
                x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="final_conv")(x)
                x = tf.keras.layers.GlobalAveragePooling2D()(x)
                outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
                cnn = tf.keras.Model(inputs, outputs)
                
                # Reshape grid as 3-channel input
                input_img = np.zeros((1, grid_size, grid_size, 3), dtype=np.float32)
                for c in range(3):
                    input_img[0, :, :, c] = radar_grid
                
                # Get Grad-CAM activations
                activation_grid = make_gradcam_heatmap(input_img, cnn, "final_conv")
                
                # Render using Seaborn Heatmaps in Matplotlib
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Radar Reflectivity
                sns.heatmap(
                    radar_dbz, ax=ax1, cmap="viridis", cbar=True,
                    cbar_kws={'label': 'Reflectivity (dBZ)'}, xticklabels=False, yticklabels=False
                )
                ax1.set_title("NEXRAD Composite Reflectivity Grid")
                
                # CNN Grad-CAM Heatmap
                sns.heatmap(
                    activation_grid, ax=ax2, cmap="inferno", cbar=True,
                    cbar_kws={'label': 'Grad-CAM Activation weight'}, xticklabels=False, yticklabels=False
                )
                ax2.set_title("CNN Class Activation Heatmap")
                
                plt.tight_layout()
                st.pyplot(fig)
                st.success("Successfully generated Seaborn radar and activation heatmaps.")
            except Exception as e:
                st.error(f"Failed to generate heatmaps: {e}")

# Tab 5: Data Lineage & Specifications
with tab_data:
    st.header("Data Lineage & Specifications")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Meteorological Weather Ingestion")
        if weather_df is not None and not weather_df.empty:
            st.write(f"- **Dimensions:** {len(weather_df)} records x {len(weather_df.columns)} variables")
            st.write(f"- **Ingested features:** `TMAX`, `TMIN`, `PRCP`, `DEWPOINT_F`, `WINDSPEED_MPH`, `PRESSURE_HPA`, `HUMIDITY_PCT`, `SOIL_MOISTURE_VOL`, `SATURATION_PCT`")
            st.markdown("**Ingestion Dataset Preview (First 5 Rows):**")
            st.dataframe(weather_df.head(5), width="stretch")
            
    with col2:
        st.subheader("Feature Engineering & Coupling")
        st.markdown("""
        The raw meteorological feeds are combined daily to engineer coupled indicators:
        - **HEAT_INDEX**: Humiture index (apparent temp felt by body).
        - **WIND_CHILL**: Perceived cold temp due to wind flow.
        - **WET_BULB**: Evaporative limits of the air index.
        - **DROUGHT_IDX**: Rolling 30-day precipitation anomaly deficits.
        - **SST_AIR_DIFF**: Coupled thermal gradient `(TAVG - SST)` between land air temperature and sea surface temperature.
        """)
        
    st.subheader("Late Fusion Modeling Framework")
    st.markdown("""
    The multimodal framework couples sequential temporal forecasting with tabular classification:
    1. **LSTM Network**: Analyzes a sliding window sequence of `(Batch, 30 Days, 5 Features)` containing `[TMAX, TMIN, TAVG, PRCP, SST]` to forecast tomorrow's land temperature.
    2. **Random Forest Classifier**: Integrates today's 19 atmospheric and ocean teleconnections parameters to output raw hazard probability.
    3. **Late Fusion Meta-Learner**: Concatenates both prediction feeds and feeds them to a Dense Feed-Forward Neural Network to compute the final hazard probability.
    """)
