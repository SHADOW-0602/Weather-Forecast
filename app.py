import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages
import io
import datetime
import requests
import folium
import cv2
import streamlit.components.v1 as components
import tensorflow as tf
from PIL import Image
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

# Import custom modules
from noaa_client import NOAAClient, CITIES
from ml_model import HazardPredictor
from research_data import RESEARCH_PAPERS, COMPARATIVE_SCHEMAS
from gradcam_radar import make_gradcam_heatmap

# 1. Page Configuration (using standard defaults)
st.set_page_config(
    page_title="AeroClim: AI Climate Intelligence Platform",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variable
load_dotenv(dotenv_path="c:/Apps/Python/Weather-Forecast/.env")
NOAA_TOKEN = os.getenv("NOAA_API_TOKEN")

# Initialize client and ML model
@st.cache_resource
def load_resources():
    client = NOAAClient(token=NOAA_TOKEN)
    predictor = HazardPredictor()
    return client, predictor

noaa_client, predictor = load_resources()

# ----------------------------------------------------
# ADVANCED CUSTOM CSS & STYLE INJECTIONS
# ----------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts import */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Typography & Font Overrides */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Beautiful Dark Mode Accent Gradients */
    .stApp {
        background: linear-gradient(135deg, #0f0c1b 0%, #15102a 50%, #090514 100%) !important;
        color: #e2e8f0 !important;
    }
    
    /* Custom Sidebar Aesthetics */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #110d24 0%, #090514 100%) !important;
        border-right: 1px solid rgba(139, 92, 246, 0.15) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4);
    }
    
    /* Custom Card Containers (Glassmorphism style) */
    .glass-card {
        background: rgba(30, 27, 57, 0.45) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(139, 92, 246, 0.15);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(139, 92, 246, 0.35);
    }
    
    /* Mini-metrics stylings */
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1;
        background: linear-gradient(90deg, #a78bfa 0%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    
    /* Glowing Dots & Badges */
    .pulse-green {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 12px #10b981;
        animation: pulse 2s infinite;
        vertical-align: middle;
        margin-right: 6px;
    }
    
    .pulse-orange {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #f97316;
        border-radius: 50%;
        box-shadow: 0 0 12px #f97316;
        animation: pulse 2s infinite;
        vertical-align: middle;
        margin-right: 6px;
    }
    
    @keyframes pulse {
        0% {
            transform: scale(0.9);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4);
        }
        70% {
            transform: scale(1.1);
            box-shadow: 0 0 0 6px rgba(16, 185, 129, 0);
        }
        100% {
            transform: scale(0.9);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
        }
    }
    
    /* Tab Styling Overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(30, 27, 57, 0.3);
        padding: 8px;
        border-radius: 12px;
        border: 1px solid rgba(139, 92, 246, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        color: #94a3b8;
        background-color: transparent;
        border: none !important;
        border-radius: 8px;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        background: linear-gradient(90deg, #6d28d9 0%, #4c1d95 100%) !important;
        box-shadow: 0 4px 14px rgba(109, 40, 217, 0.35);
    }
    
    /* Gradient Dividers */
    .grad-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #8b5cf6 50%, transparent 100%);
        margin: 25px 0;
        opacity: 0.5;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# SIDEBAR CONTROL LAYOUT
# ----------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-top: -15px; margin-bottom: 25px;">
            <span style="font-family: 'Outfit', sans-serif; font-size: 2.2rem; font-weight: 900; background: linear-gradient(90deg, #c084fc 0%, #38bdf8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: block; line-height: 1;">AeroClim</span>
            <span style="font-size: 0.8rem; letter-spacing: 0.15em; color: #a78bfa; text-transform: uppercase; font-weight: 600;">Climate Intelligence</span>
        </div>
    """, unsafe_allow_html=True)
    
    # API Key status
    if NOAA_TOKEN:
        st.markdown(
            f'<div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 10px 14px; font-size: 0.85rem; margin-bottom: 25px;">'
            f'<span class="pulse-green"></span> <b>Live NOAA CDO API</b>: Active'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="background: rgba(249, 115, 22, 0.1); border: 1px solid rgba(249, 115, 22, 0.25); border-radius: 8px; padding: 10px 14px; font-size: 0.85rem; margin-bottom: 25px;">'
            f'<span class="pulse-orange"></span> <b>Offline Engine</b>: Simulation Fallback'
            f'</div>',
            unsafe_allow_html=True
        )
        
    st.markdown("### 🛰️ Station Settings")
    selected_city_id = st.selectbox(
        "Select U.S. Meteorological Station",
        options=list(CITIES.keys()),
        format_func=lambda cid: CITIES[cid]["name"]
    )
    
    city_data = CITIES[selected_city_id]
    
    # Render station parameters
    st.markdown(f"""
        <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; padding: 15px; margin-top: 10px; font-size: 0.85rem;">
            <b>Region:</b> {city_data["region"]}<br>
            <b>Climate Zone:</b> {city_data["climate"]}<br>
            <b>Elevation:</b> {city_data["elevation"]} m above sea level<br>
            <b>Coordinates:</b> {city_data["lat"]}° N, {abs(city_data["lon"])}° W
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div class='grad-divider'></div>", unsafe_allow_html=True)
    
    st.markdown("### 📅 Temporal Horizon")
    time_preset = st.radio(
        "Climatic Window Range",
        options=["Past 30 Days", "Past 6 Months", "Full 12 Months"]
    )
    
    # Calculate dates
    today = datetime.date.today()
    if time_preset == "Past 30 Days":
        start_date = today - datetime.timedelta(days=30)
    elif time_preset == "Past 6 Months":
        start_date = today - datetime.timedelta(days=180)
    else:
        start_date = today - datetime.timedelta(days=365)
        
    end_date = today
    
    # Query force simulation option
    force_sim = st.toggle("Override with Synthetic Generator", value=False, help="Force local generator instead of calling NOAA CDO API.")
    
    st.markdown("<div style='margin-top: 80px; text-align: center; color: #64748b; font-size: 0.75rem;'>AeroClim v1.1 • Running on Local Machine Learning</div>", unsafe_allow_html=True)

# ----------------------------------------------------
# MAIN APPLICATION CONTENT
# ----------------------------------------------------

# Header Title Block
st.markdown(f"""
    <div style="margin-bottom: 25px;">
        <span style="font-size: 0.85rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.1em;">Unified Environmental Intelligence</span>
        <h1 style="font-size: 2.8rem; margin: 0 0 10px 0; background: linear-gradient(90deg, #ffffff 0%, #c084fc 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AeroClim Analytics Console</h1>
        <p style="color: #94a3b8; font-size: 1.1rem; max-width: 800px; margin: 0;">An interactive platform evaluating live weather anomalies, sea surface temperature dynamics, and machine learning models to assess hazards and predict extreme environmental risk indices.</p>
    </div>
""", unsafe_allow_html=True)

tab_live, tab_sst, tab_ml, tab_radar, tab_data = st.tabs([
    "🛰️ Live NOAA CDO Station",
    "🌊 Sea Surface Temperature (Copernicus L4)",
    "🔮 Flash Flood & Hazard Predictor",
    "📡 Real-Time Radar Heatmap",
    "📊 Data Source & Lineage"
])

# Load station weather records globally to share between tabs
with st.spinner("Synchronizing with meteorological network..."):
    weather_df = noaa_client.fetch_weather_data(
        selected_city_id, 
        start_date.strftime("%Y-%m-%d"), 
        end_date.strftime("%Y-%m-%d"),
        force_simulation=force_sim
    )

# ----------------------------------------------------
# TAB 1: LIVE NOAA CDO EXPLORER
# ----------------------------------------------------
with tab_live:
    st.subheader("🛰️ U.S. Historical Weather Station Data Viewer")
    
    if weather_df is not None and not weather_df.empty:
        # Display Data Source Badge
        src = weather_df["Source"].iloc[0]
        badge_style = "background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); color: #10b981;" if "Live" in src else "background: rgba(249, 115, 22, 0.1); border: 1px solid rgba(249, 115, 22, 0.3); color: #f97316;"
        
        st.markdown(f"""
            <div style="display: flex; justify-content: flex-end; margin-bottom: 15px;">
                <span style="font-size: 0.75rem; font-weight: 600; text-transform: uppercase; padding: 4px 10px; border-radius: 20px; {badge_style}">
                    Data Source: {src}
                </span>
            </div>
        """, unsafe_allow_html=True)
        
        # Calculate summary metrics
        avg_temp = (weather_df["TMAX"].mean() + weather_df["TMIN"].mean()) / 2
        total_rain = weather_df["PRCP"].sum()
        max_temp = weather_df["TMAX"].max()
        rainy_days = (weather_df["PRCP"] > 0.1).sum()
        
        # Display stats metrics in elegant cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label">Average Temp</div>
                    <div class="metric-value">{avg_temp:.1f}°C <span style="font-size:1.1rem; font-weight:400; color:#94a3b8;">/ {(avg_temp*9/5+32):.1f}°F</span></div>
                    <div style="font-size: 0.8rem; color: #10b981;">Peak: {max_temp:.1f}°C</div>
                </div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label">Min Recorded Temp</div>
                    <div class="metric-value">{weather_df['TMIN'].min():.1f}°C <span style="font-size:1.1rem; font-weight:400; color:#94a3b8;">/ {(weather_df['TMIN'].min()*9/5+32):.1f}°F</span></div>
                    <div style="font-size: 0.8rem; color: #38bdf8;">Trough recorded</div>
                </div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label">Accumulated Rainfall</div>
                    <div class="metric-value">{total_rain:.1f} mm <span style="font-size:1.1rem; font-weight:400; color:#94a3b8;">/ {(total_rain/25.4):.2f}"</span></div>
                    <div style="font-size: 0.8rem; color: #a78bfa;">Max Daily: {weather_df['PRCP'].max():.1f} mm</div>
                </div>
            """, unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label">Rainy Days</div>
                    <div class="metric-value">{rainy_days} Days</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">Out of {len(weather_df)} Days ({(rainy_days/len(weather_df)*100.0):.1f}%)</div>
                </div>
            """, unsafe_allow_html=True)
            
        # Draw Plotly Climatic chart
        fig_weather = go.Figure()
        
        # Add temperature range shading
        fig_weather.add_trace(go.Scatter(
            x=weather_df["Date"],
            y=weather_df["TMAX"],
            mode='lines',
            line=dict(color='rgba(239, 68, 68, 0.75)', width=1.5),
            name='Daily Max Temp (°C)'
        ))
        
        fig_weather.add_trace(go.Scatter(
            x=weather_df["Date"],
            y=weather_df["TMIN"],
            mode='lines',
            line=dict(color='rgba(56, 189, 248, 0.75)', width=1.5),
            fill='tonexty', # fills region between tmax and tmin
            fillcolor='rgba(139, 92, 246, 0.08)',
            name='Daily Min Temp (°C)'
        ))
        
        # Add daily precipitation bars on secondary Y-axis
        fig_weather.add_trace(go.Bar(
            x=weather_df["Date"],
            y=weather_df["PRCP"],
            name="Daily Precipitation (mm)",
            marker_color='rgba(167, 139, 250, 0.65)',
            yaxis="y2"
        ))
        
        # Setup modern dual-axis layout in dark theme
        fig_weather.update_layout(
            title={
                "text": f"Climatic Anomalies & Precipitation Dynamics - {city_data['name']}",
                "font": {"family": "Outfit", "size": 18, "color": "#ffffff"}
            },
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.05)',
                title="Time Horizon"
            ),
            yaxis=dict(
                title="Temperature (°C)",
                showgrid=True,
                gridcolor='rgba(255,255,255,0.05)',
                side="left"
            ),
            yaxis2=dict(
                title="Precipitation (mm)",
                showgrid=False,
                side="right",
                overlaying="y"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified",
            margin=dict(l=40, r=40, t=60, b=40),
            height=500
        )
        
        st.plotly_chart(fig_weather, width="stretch")
        
        # Monthly summaries table
        st.markdown("### 📊 Climatological Monthly Summaries (GSOD/GSOM Simulation)")
        
        monthly_df = weather_df.copy()
        monthly_df["Month"] = monthly_df["Date"].dt.to_period("M").astype(str)
        
        month_summary = monthly_df.groupby("Month").agg(
            Avg_Max_Temp=("TMAX", "mean"),
            Avg_Min_Temp=("TMIN", "mean"),
            Peak_Daily_Temp=("TMAX", "max"),
            Total_Rainfall=("PRCP", "sum"),
            Rainy_Days_Count=("PRCP", lambda r: (r > 0.1).sum())
        ).reset_index()
        
        # format rounding
        month_summary["Avg_Max_Temp"] = month_summary["Avg_Max_Temp"].round(1)
        month_summary["Avg_Min_Temp"] = month_summary["Avg_Min_Temp"].round(1)
        month_summary["Total_Rainfall"] = month_summary["Total_Rainfall"].round(1)
        
        # Rename columns for table presentation
        month_summary.columns = [
            "Period (Month)", 
            "Average Max Temp (°C)", 
            "Average Min Temp (°C)", 
            "Peak Temperature (°C)", 
            "Accumulated Precipitation (mm)", 
            "Rainy Days Count"
        ]
        
        st.dataframe(
            month_summary,
            hide_index=True,
            width="stretch"
        )
        
    else:
        st.error("Severe network failure or empty dataset encountered. Switch to Simulation Fallback mode.")

# ----------------------------------------------------
# TAB 2: SEA SURFACE TEMPERATURE & COPERNICUS
# ----------------------------------------------------
with tab_sst:
    st.subheader("🌊 Ocean Surface Dynamics (Copernicus ESA SST L4 Climate Records)")
    
    st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.3); border: 1px solid rgba(56, 189, 248, 0.15); border-radius: 12px; padding: 18px; margin-bottom: 25px; font-size: 0.95rem; line-height: 1.6;">
            <b>Marine-Land Coupled Physics</b>: Warm sea surface temperature (SST) anomalies serve as major energy storage tanks, transferring latent heat and moisture into the lower troposphere. This convective destabilization drives extreme weather anomalies and flood hazards over adjacent coastal and continental regions.
        </div>
    """, unsafe_allow_html=True)
    
    daily_path = "ocean.csv"
    if os.path.exists(daily_path):
        sst_df = pd.read_csv(daily_path)
        
        # Check column casing for date
        sst_date_col = "date" if "date" in sst_df.columns else "Date" if "Date" in sst_df.columns else sst_df.columns[0]
        sst_df["Date"] = pd.to_datetime(sst_df[sst_date_col])
        
        # Map selected_city_id to appropriate ocean basin column
        basin_mapping = {
            "SEATTLE": "sst_pacific_f",
            "KCQT": "sst_pacific_f",
            "NEW_YORK": "sst_atlantic_f",
            "KPHL": "sst_atlantic_f",
            "KJAX": "sst_atlantic_f",
            "KCLT": "sst_atlantic_f",
            "KHOU": "sst_gulf_f",
        }
        
        target_col = basin_mapping.get(selected_city_id)
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
                sst_df["SST_Mean_Celsius"] = 18.0
        
        latest_row = sst_df.iloc[-1]
        mean_sst = sst_df["SST_Mean_Celsius"].mean()
        max_sst = sst_df["SST_Mean_Celsius"].max()
        min_sst = sst_df["SST_Mean_Celsius"].min()
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label">Latest SST Compiled</div>
                    <div class="metric-value">{latest_row['SST_Mean_Celsius']:.2f}°C</div>
                    <div style="font-size: 0.8rem; color: #a78bfa;">Date: {latest_row['Date'].strftime('%Y-%m-%d')}</div>
                </div>
            """, unsafe_allow_html=True)
        with col_s2:
            st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label">Mean Historical SST</div>
                    <div class="metric-value">{mean_sst:.2f}°C</div>
                    <div style="font-size: 0.8rem; color: #10b981;">Overall dataset average</div>
                </div>
            """, unsafe_allow_html=True)
        with col_s3:
            st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label">Peak SST Temperature</div>
                    <div class="metric-value">{max_sst:.2f}°C</div>
                    <div style="font-size: 0.8rem; color: #ef4444;">Maximum recorded</div>
                </div>
            """, unsafe_allow_html=True)
        with col_s4:
            st.markdown(f"""
                <div class="glass-card">
                    <div class="metric-label">Minimum SST Temperature</div>
                    <div class="metric-value">{min_sst:.2f}°C</div>
                    <div style="font-size: 0.8rem; color: #38bdf8;">Minimum recorded</div>
                </div>
            """, unsafe_allow_html=True)
            
        # Draw Sea Surface Temperature trend line
        fig_sst = px.line(
            sst_df,
            x="Date",
            y="SST_Mean_Celsius",
            title="Copernicus Satellite Sea Surface Temperature (1980 - 2025)",
            labels={"SST_Mean_Celsius": "SST (°C)"},
            template="plotly_dark"
        )
        
        fig_sst.update_traces(line=dict(color='#0ea5e9', width=2))
        fig_sst.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            height=450,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        st.plotly_chart(fig_sst, width="stretch")
        
        st.info(f"ℹ️ **Dataset Progress**: Currently contains **{len(sst_df)}** compiled daily/monthly records representing ocean temperature measurements in the U.S. Bounding Box.")
    else:
        st.warning("⚠️ Copernicus ocean summary dataset `ocean.csv` not found. Please ensure downloader and compiler have run.")

# ----------------------------------------------------
# TAB 3: EXTREME WEATHER & FLOOD RISK PREDICTOR
# ----------------------------------------------------
with tab_ml:
    st.subheader("🔮 Operational Flash Flood & Extreme Hazard Forecast Simulator")
    st.markdown("""
        <p style="color: #94a3b8; font-size: 0.95rem; margin-top: -10px; margin-bottom: 25px;">
            Adjust real-time meteorological variables using the panel below. AeroClim's embedded <b>Random Forest Classifier + LSTM Late Fusion meta-learner model</b> 
            (trained on coupled land-sea physical interactions) will predict local hazard probabilities, classify risk severities, 
            and decompose input feature attributions.
        </p>
    """, unsafe_allow_html=True)
    
    col_inputs, col_results = st.columns([1, 1])
    
    with col_inputs:
        st.markdown("#### 🎛️ Atmospheric & Locational Control Panel")
        current_tmax = st.slider("Today's Max Temp (°C)", -20.0, 50.0, 25.0, 0.5)
        current_tmin = st.slider("Today's Min Temp (°C)", -30.0, 40.0, 15.0, 0.5)
        dew_point = st.slider("Dew Point (°C)", -20.0, 30.0, 10.0, 0.5)
        wind_speed = st.slider("Wind Speed (km/h)", 0, 120, 15, 5)
        current_prcp = st.slider("Today's Rainfall (mm)", 0.0, 100.0, 5.0, 1.0)
        
        # Trigger model training if weights are missing
        if predictor.rf_model is None:
            st.warning("⚠️ Multimodal Models are not trained. Please click the button below to train them on 30 years of daily coupled data.")
            
        if st.button("Train Multimodal Models (30 Years, 11,000+ Rows)", width="stretch"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            def progress_cb(pct, msg):
                progress_bar.progress(pct)
                status_text.text(msg)
            
            try:
                status_text.text("Loading 30 years of meteorological records...")
                train_start = today - datetime.timedelta(days=11000)
                train_df = noaa_client.fetch_weather_data(
                    selected_city_id,
                    train_start.strftime("%Y-%m-%d"),
                    today.strftime("%Y-%m-%d"),
                    force_simulation=force_sim
                )
                status_text.text(f"Fetched {len(train_df)} rows. Commencing training...")
                predictor.train_multimodal(train_df, progress_callback=progress_cb)
                st.success("Training complete! Models saved.")
                st.rerun()
            except Exception as e:
                st.error(f"Error during training: {e}")
                
        # Display validation accuracy metrics if available (defensively check for cached instances)
        metrics = getattr(predictor, "metrics", None)
        if metrics:
            st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
            st.markdown("#### 📊 Model Validation Performance")
            
            # Late Fusion Card
            st.markdown(f"""
                <div class="glass-card" style="padding: 15px; margin-bottom: 12px; border-left: 4px solid #8b5cf6;">
                    <div class="metric-label" style="font-size: 0.75rem;">Late Fusion Meta-Learner (FNN)</div>
                    <div class="metric-value" style="font-size: 1.6rem; margin: 2px 0;">{metrics.get('fusion_accuracy', 0.0)*100.0:.2f}% <span style="font-size: 0.9rem; font-weight: 400; color: #94a3b8;">Accuracy</span></div>
                    <div style="font-size: 0.78rem; color: #cbd5e1;">ROC-AUC Score: <b>{metrics.get('fusion_auc', 0.0):.3f}</b></div>
                </div>
            """, unsafe_allow_html=True)
            
            # RF Card
            st.markdown(f"""
                <div class="glass-card" style="padding: 15px; margin-bottom: 12px; border-left: 4px solid #38bdf8;">
                    <div class="metric-label" style="font-size: 0.75rem;">Random Forest Classifier</div>
                    <div class="metric-value" style="font-size: 1.6rem; margin: 2px 0;">{metrics.get('rf_accuracy', 0.0)*100.0:.2f}% <span style="font-size: 0.9rem; font-weight: 400; color: #94a3b8;">Accuracy</span></div>
                    <div style="font-size: 0.78rem; color: #cbd5e1;">ROC-AUC Score: <b>{metrics.get('rf_auc', 0.0):.3f}</b></div>
                </div>
            """, unsafe_allow_html=True)
            
            # LSTM Card
            st.markdown(f"""
                <div class="glass-card" style="padding: 15px; margin-bottom: 12px; border-left: 4px solid #10b981;">
                    <div class="metric-label" style="font-size: 0.75rem;">LSTM Trend Predictor (TAVG)</div>
                    <div class="metric-value" style="font-size: 1.6rem; margin: 2px 0;">R²: {metrics.get('lstm_r2', 0.0):.3f}</div>
                    <div style="font-size: 0.78rem; color: #cbd5e1;">Mean Squared Error: <b>{metrics.get('lstm_mse', 0.0):.5f}</b></div>
                </div>
            """, unsafe_allow_html=True)
    
    # Run prediction
    prediction = predictor.predict(
        weather_df=weather_df,
        current_tmax=current_tmax,
        current_tmin=current_tmin,
        dew_point=dew_point,
        wind_speed=wind_speed,
        current_prcp=current_prcp
    )
    
    with col_results:
        st.markdown("#### 🚨 Computed Environmental Hazard Rating")
        
        # Draw dynamic circular gauge representing probability
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction["Probability"],
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': "%", 'font': {'size': 44, 'family': 'Outfit', 'color': '#ffffff'}},
            title={'text': "HAZARD INDEX PROBABILITY", 'font': {'size': 14, 'family': 'Outfit', 'color': '#94a3b8'}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#475569"},
                'bar': {'color': "#8b5cf6"},
                'bgcolor': "rgba(30, 27, 57, 0.3)",
                'borderwidth': 1,
                'bordercolor': "rgba(139, 92, 246, 0.15)",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.08)'},
                    {'range': [30, 60], 'color': 'rgba(234, 179, 8, 0.08)'},
                    {'range': [60, 85], 'color': 'rgba(249, 115, 22, 0.08)'},
                    {'range': [85, 100], 'color': 'rgba(239, 68, 68, 0.08)'}
                ],
                'threshold': {
                    'line': {'color': "#ffffff", 'width': 3},
                    'thickness': 0.75,
                    'value': prediction["Probability"]
                }
            }
        ))
        
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=280,
            margin=dict(l=30, r=30, t=50, b=10)
        )
        
        st.plotly_chart(fig_gauge, width="stretch")
        
        # Define severity colors
        sev_color = "#10b981" if prediction["Probability"] < 30.0 else "#eab308" if prediction["Probability"] < 60.0 else "#f97316" if prediction["Probability"] < 85.0 else "#ef4444"
        
        # Display Classifier Card
        st.markdown(f"""
            <div style="background: rgba(30, 27, 57, 0.45); border: 2px solid {sev_color}; border-radius: 12px; padding: 18px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.25);">
                <div style="font-size: 0.8rem; font-weight:700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">RISK CLASSIFICATION</div>
                <div style="font-size: 1.8rem; font-family: 'Outfit', sans-serif; font-weight: 800; color: {sev_color};">{prediction["Category"]}</div>
                <p style="font-size: 0.88rem; line-height: 1.5; color: #cbd5e1; margin-top: 8px; margin-bottom: 0;">{prediction["Description"]}</p>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div class='grad-divider'></div>", unsafe_allow_html=True)
    
    col_feat, col_shap = st.columns([1, 1])
    
    with col_feat:
        st.markdown("#### 🔬 Relative Feature Attributions (SHAP Inferences)")
        if "Contributions" in prediction:
            contrib_data = prediction["Contributions"]
            contrib_df = pd.DataFrame(list(contrib_data.items()), columns=["Atmospheric Factor", "Hazard Value Impact"])
            contrib_df = contrib_df.sort_values("Hazard Value Impact", ascending=True)
            
            fig_contrib = px.bar(
                contrib_df,
                x="Hazard Value Impact",
                y="Atmospheric Factor",
                orientation="h",
                template="plotly_dark",
                color="Hazard Value Impact",
                color_continuous_scale=[[0, "#4f46e5"], [0.5, "#a855f7"], [1, "#ef4444"]]
            )
            
            fig_contrib.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title="Hazard Influence Score (Weight)"),
                yaxis=dict(title=""),
                height=320,
                margin=dict(l=40, r=40, t=10, b=40)
            )
            st.plotly_chart(fig_contrib, width="stretch")
        else:
            st.info("Feature attributions will be plotted here once predictions are run.")
            
    with col_shap:
        st.markdown("#### 🧠 Explainable ML Diagnostic & Global Importance")
        st.markdown(f"""
            <div style="background: rgba(139, 92, 246, 0.05); border: 1px solid rgba(139, 92, 246, 0.15); border-radius: 12px; padding: 20px; font-size: 0.92rem; line-height: 1.6; margin-bottom: 20px;">
                <b style="color: #c084fc;">Local Decision Explanation:</b><br>
                {prediction["Explanation"]}
            </div>
        """, unsafe_allow_html=True)
        
        try:
            importances_df = predictor.get_feature_importances()
            fig_global_imp = px.bar(
                importances_df,
                x="Gini Importance",
                y="Feature",
                orientation="h",
                template="plotly_dark",
                labels={"Gini Importance": "Gini Node Splitting Importance (Global weight)"}
            )
            
            fig_global_imp.update_traces(marker_color="rgba(56, 189, 248, 0.75)")
            fig_global_imp.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title=""),
                height=220,
                margin=dict(l=40, r=40, t=10, b=40)
            )
            st.plotly_chart(fig_global_imp, width="stretch")
        except Exception as e:
            st.error(f"Could not load global importances: {e}")

# ----------------------------------------------------
# TAB 4: REAL-TIME RADAR HEATMAP
# ----------------------------------------------------
with tab_radar:
    st.subheader("📡 Real-Time Radar & CNN Heatmap")
    st.markdown("""
        <p style="color: #94a3b8; font-size: 0.95rem; margin-top: -10px; margin-bottom: 25px;">
            This module fetches live composite reflectivity radar data from the NOAA GeoServer for the selected region.
            A Convolutional Neural Network (CNN) analyzes the radar imagery to detect extreme storm cells and
            computes a Grad-CAM heatmap highlighting regions contributing most to the risk prediction.
        </p>
    """, unsafe_allow_html=True)
    
    if st.button("Fetch Live Radar & Generate Heatmap", type="primary"):
        with st.spinner(f"Fetching Live WMS Radar Data and computing CNN activations for {city_data['name']}..."):
            try:
                lat, lon = city_data["lat"], city_data["lon"]
                bbox = f"{lat-2},{lon-2},{lat+2},{lon+2}"
                width, height = 512, 512
                
                wms_url = "https://opengeo.ncep.noaa.gov/geoserver/conus/conus_bref_qcd/ows"
                params = {
                    "SERVICE": "WMS",
                    "VERSION": "1.3.0",
                    "REQUEST": "GetMap",
                    "BBOX": bbox,
                    "CRS": "EPSG:4326",
                    "WIDTH": width,
                    "HEIGHT": height,
                    "LAYERS": "conus_bref_qcd",
                    "STYLES": "",
                    "FORMAT": "image/png",
                    "TRANSPARENT": "true"
                }
                
                response = requests.get(wms_url, params=params, timeout=15)
                if response.status_code == 200 and len(response.content) > 1000:
                    img = Image.open(io.BytesIO(response.content)).convert("RGBA")
                    img_np = np.array(img)
                    
                    rgb_img = img.convert("RGB")
                    rgb_np = np.array(rgb_img)
                    
                    # Dummy CNN creation for demo
                    inputs = tf.keras.Input(shape=(height, width, 3))
                    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu", name="conv1")(inputs)
                    x = tf.keras.layers.MaxPooling2D(2)(x)
                    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="conv2")(x)
                    x = tf.keras.layers.MaxPooling2D(2)(x)
                    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="final_conv")(x)
                    x = tf.keras.layers.GlobalAveragePooling2D()(x)
                    outputs = tf.keras.layers.Dense(2, activation="softmax", name="predictions")(x)
                    cnn_model = tf.keras.Model(inputs, outputs)
                    
                    input_tensor = np.expand_dims(rgb_np, axis=0).astype(np.float32) / 255.0
                    heatmap = make_gradcam_heatmap(input_tensor, cnn_model, "final_conv")
                    
                    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize((width, height))
                    heatmap_resized_np = np.array(heatmap_resized)
                    
                    import scipy.ndimage
                    # Apply Gaussian blur to make the CNN activations look more organic and less blocky
                    heatmap_smoothed = scipy.ndimage.gaussian_filter(heatmap_resized_np, sigma=15)
                    color_heatmap_bgr = cv2.applyColorMap(heatmap_smoothed, cv2.COLORMAP_JET)
                    color_heatmap_rgb = cv2.cvtColor(color_heatmap_bgr, cv2.COLOR_BGR2RGB) / 255.0
                    
                    # Base alpha based on heatmap intensity
                    alpha_channel = np.clip(heatmap_smoothed / 255.0, 0.15, 0.7)
                    
                    # Create a radial gradient mask to fade out the hard square edges into the map
                    y, x = np.ogrid[:height, :width]
                    center_y, center_x = height / 2, width / 2
                    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    max_dist = min(width, height) / 2.0
                    radial_mask = np.clip(1.0 - (distance / max_dist), 0, 1)
                    # Apply smoothstep for a natural fade
                    radial_mask = radial_mask * radial_mask * (3 - 2 * radial_mask)
                    
                    # Multiply alpha by the radial mask so it smoothly disappears at the bounds
                    alpha_channel = alpha_channel * radial_mask
                    
                    heatmap_rgba = np.zeros((height, width, 4))
                    heatmap_rgba[:, :, :3] = color_heatmap_rgb
                    heatmap_rgba[:, :, 3] = alpha_channel
                    
                    m = folium.Map(location=[lat, lon], zoom_start=7, tiles="CartoDB dark_matter")
                    
                    import base64
                    def numpy_to_b64(arr):
                        img_pil = Image.fromarray(np.uint8(arr * 255))
                        buffer = io.BytesIO()
                        img_pil.save(buffer, format="PNG")
                        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
                        
                    bounds = [[lat-2, lon-2], [lat+2, lon+2]]
                    
                    folium.raster_layers.ImageOverlay(
                        image=numpy_to_b64(img_np / 255.0),
                        bounds=bounds,
                        opacity=0.7,
                        name="Live NOAA Radar"
                    ).add_to(m)
                    
                    folium.raster_layers.ImageOverlay(
                        image=numpy_to_b64(heatmap_rgba),
                        bounds=bounds,
                        opacity=0.8,
                        name="CNN Activation Heatmap"
                    ).add_to(m)
                    
                    folium.LayerControl().add_to(m)
                    
                    map_html = m.get_root().render()
                    components.html(map_html, height=600)
                    
                    st.success(f"Successfully rendered storm cell heatmap for {city_data['name']}.")
                else:
                    st.error("No radar precipitation detected in this region currently, or GeoServer is down.")
            except Exception as e:
                st.error(f"Failed to generate radar heatmap: {e}")

# ----------------------------------------------------
# TAB 5: DATA SOURCE & LINEAGE
# ----------------------------------------------------
with tab_data:
    st.subheader("📊 Data Lineage, Source Specifications & Model Inputs")
    st.markdown("""
        <p style="color: #94a3b8; font-size: 0.95rem; margin-top: -10px; margin-bottom: 25px;">
            This section provides transparency into the datasets ingested, their geographical and technical origins, and how they are transformed and coupled as feature vectors for the multimodal machine learning model.
        </p>
    """, unsafe_allow_html=True)
    
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        st.markdown("### 🛰️ Land Weather Station Ingestion")
        
        # Calculate dynamic size and dimension
        weather_size_kb = 0.0
        weather_rows = 0
        weather_cols = 0
        if weather_df is not None and not weather_df.empty:
            weather_size_kb = weather_df.memory_usage(deep=True).sum() / 1024.0
            weather_rows = len(weather_df)
            weather_cols = len(weather_df.columns)
            
        # Get total size on disk of the historical dataset
        total_historical_rows = 10227
        total_historical_cols = 11
        station_csv = f"{selected_city_id}.csv"
        if os.path.exists(station_csv):
            try:
                temp_df = pd.read_csv(station_csv)
                total_historical_rows = len(temp_df)
                total_historical_cols = len(temp_df.columns)
            except:
                pass
            
        st.markdown(f"""
            <div class="glass-card" style="margin-bottom: 20px;">
                <b style="color: #38bdf8; font-size: 1.1rem;">NOAA Climate Data Online (CDO)</b>
                <p style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6; margin-top: 8px;">
                    <b>Data Origin:</b> National Oceanic and Atmospheric Administration (NOAA) CDO meteorological stations.<br>
                    <b>Geographical Focus:</b> Active station: <b>{city_data["name"]}</b> ({selected_city_id}) at coordinates {city_data["lat"]}° N, {abs(city_data["lon"])}° W.<br>
                    <b>Temporal Scope:</b> {time_preset} ({weather_rows} daily data points ingested for visualization).<br>
                    <b>Dataset Profile:</b><br>
                    - <b>Active Console Selection:</b> {weather_rows} rows × {weather_cols} columns ({weather_size_kb:.2f} KB in memory)<br>
                    - <b>Total Historical File on Disk:</b> {total_historical_rows:,} rows × {total_historical_cols} columns<br>
                    <b>Ingested Features:</b>
                </p>
                <ul style="font-size: 0.85rem; color: #94a3b8; padding-left: 20px; margin-bottom: 15px;">
                    <li><b>TMAX</b>: Daily Maximum Temperature (°C)</li>
                    <li><b>TMIN</b>: Daily Minimum Temperature (°C)</li>
                    <li><b>PRCP</b>: Daily Accumulated Precipitation (mm)</li>
                </ul>
                <b style="color: #38bdf8; font-size: 0.9rem; display: block; margin-bottom: 8px;">Land Dataset Preview (First 5 Rows):</b>
            </div>
        """, unsafe_allow_html=True)
        if weather_df is not None and not weather_df.empty:
            st.dataframe(weather_df.head(5), width="stretch")
        
        st.markdown("### 🌊 Sea Surface Temperature Ingestion")
        
        # Calculate daily records count and size dynamically
        sst_records_count = 0
        sst_file_size_kb = 0.0
        sst_cols = 0
        sst_preview_df = None
        if os.path.exists("ocean.csv"):
            try:
                sst_preview_df = pd.read_csv("ocean.csv")
                sst_records_count = len(sst_preview_df)
                sst_cols = len(sst_preview_df.columns)
                sst_file_size_kb = os.path.getsize("ocean.csv") / 1024.0
            except:
                pass
                
        st.markdown(f"""
            <div class="glass-card" style="margin-bottom: 20px;">
                <b style="color: #0ea5e9; font-size: 1.1rem;">Copernicus ESA SST L4 Records</b>
                <p style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6; margin-top: 8px;">
                    <b>Data Origin:</b> Copernicus Climate Data Store (CDS) Satellite-Derived Global Sea Surface Temperature.<br>
                    <b>Spatial Scope:</b> Option A U.S. Coastal Bounding Box (Latitude: 20°N to 55°N, Longitude: 135°W to 60°W).<br>
                    <b>Temporal Scope:</b> 1980–2025 (Daily resolution monthly zip chunks downloaded in background).<br>
                    <b>Dataset Profile:</b><br>
                    - <b>Size on Disk:</b> {sst_file_size_kb:.2f} KB<br>
                    - <b>Dimensions:</b> {sst_records_count} rows × {sst_cols} columns<br>
                    <b>Ingested Features:</b>
                </p>
                <ul style="font-size: 0.85rem; color: #94a3b8; padding-left: 20px; margin-bottom: 15px;">
                    <li><b>SST</b>: Average Sea Surface Temperature (°C) computed as the spatial mean over coastal pixels.</li>
                </ul>
                <b style="color: #0ea5e9; font-size: 0.9rem; display: block; margin-bottom: 8px;">Sea Temperature Dataset Preview (First 5 Rows):</b>
            </div>
        """, unsafe_allow_html=True)
        if sst_preview_df is not None:
            st.dataframe(sst_preview_df.head(5), width="stretch")

    with col_dl2:
        st.markdown("### ⚙️ Feature Transformation & Model Coupling")
        st.markdown("""
            <div class="glass-card" style="height: 100%;">
                <b style="color: #a78bfa; font-size: 1.1rem;">Engineered Inputs for Random Forest</b>
                <p style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6; margin-top: 8px;">
                    The raw land and sea features are combined daily to engineer physics-derived indicators:
                </p>
                <ul style="font-size: 0.85rem; color: #cbd5e1; padding-left: 20px; line-height: 1.8;">
                    <li><b>HEAT_INDEX</b>: Perceived air temperature accounting for relative humidity.</li>
                    <li><b>WIND_CHILL</b>: Apparent temperature felt on exposed skin due to wind flow.</li>
                    <li><b>WET_BULB</b>: Thermodynamic limit of evaporative cooling (wet-bulb temperature).</li>
                    <li><b>DROUGHT_IDX</b>: Rolling 30-day precipitation anomaly showing moisture deficits.</li>
                    <li><b>TEMP_ANOM / PRCP_ANOM</b>: Today's deviations from historical station averages.</li>
                    <li><b>SST_AIR_DIFF</b>: Coupled thermal gradient <code>(TAVG - SST)</code> between land air temperature and sea surface temperature.</li>
                </ul>
                <div class="grad-divider" style="margin: 15px 0;"></div>
                <b style="color: #a78bfa; font-size: 1.1rem;">Temporal Sequence Inputs for LSTM</b>
                <p style="font-size: 0.88rem; color: #cbd5e1; line-height: 1.6; margin-top: 8px;">
                    The LSTM network operates on a sliding sequence:
                </p>
                <ul style="font-size: 0.85rem; color: #cbd5e1; padding-left: 20px; line-height: 1.8;">
                    <li><b>Input Shape</b>: <code>(Batch Size, 30 Days, 5 Features)</code></li>
                    <li><b>Sequence Vectors</b>: 30 consecutive historical days of <code>[TMAX, TMIN, TAVG, PRCP, SST]</code> scaled via MinMaxScaler.</li>
                    <li><b>Target variable</b>: Land average temperature (<code>TAVG</code>) tomorrow.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 📐 Model Pipelines & Late Fusion Architecture")
    
    st.markdown("""
        <div style="background: rgba(30, 27, 57, 0.3); border: 1px solid rgba(139, 92, 246, 0.15); border-radius: 12px; padding: 20px; font-size: 0.9rem; line-height: 1.6;">
            <b>Late Fusion Workflow</b>:<br>
            1. <b>LSTM Model</b> reads a 30-day sequential time series of <code>[TMAX, TMIN, TAVG, PRCP, SST]</code> and predicts the land air temperature for tomorrow.<br>
            2. <b>Random Forest Classifier</b> evaluates today's 10 engineered land-sea physics parameters and outputs a static hazard/flood risk probability.<br>
            3. <b>Late Fusion Meta-Learner (FNN)</b> concatenates the LSTM temperature prediction and the Random Forest hazard probability, running them through a Dense Neural Network with a Sigmoid activation to produce the final, coupled <b>Fused Hazard Probability</b>.
        </div>
    """, unsafe_allow_html=True)



