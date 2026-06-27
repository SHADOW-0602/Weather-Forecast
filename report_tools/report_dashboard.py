from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "report_output" / "assets"

st.set_page_config(
    page_title="AeroClim Evidence Dashboard",
    page_icon=str(ROOT / "assets" / "logo.png"),
    layout="wide",
)

CITY_FILES = {
    "Seattle": "seattle.csv",
    "New York": "new_york.csv",
    "Phoenix": "phoenix.csv",
}
RF_FEATURES = [
    "HEAT_INDEX", "WIND_CHILL", "WET_BULB", "DROUGHT_IDX", "TEMP_ANOM",
    "PRCP_ANOM", "EXTREME_HEAT", "EXTREME_COLD", "SST", "SST_AIR_DIFF",
    "FLOOD_RISK_IDX", "WIND_RISK_IDX", "PRESSURE_HPA", "HUMIDITY_PCT",
    "SOIL_MOISTURE_VOL", "SATURATION_PCT", "enso_nino34", "pdo_index",
    "nao_index",
]


@st.cache_data
def load_city(city: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "noaa_stations" / CITY_FILES[city].lower())
    df["date"] = pd.to_datetime(df["date"])
    # No unit conversion needed, columns are already TMAX, TMIN, TAVG, PRCP!
    return df


@st.cache_resource
def load_rf():
    return joblib.load(ROOT / "saved_models" / "rf_model.pkl")


stats = json.loads((ASSETS / "analysis_summary.json").read_text(encoding="utf-8"))
metrics = stats["model_metrics"]
rf_model = load_rf()

with st.sidebar:
    st.image(str(ROOT / "assets" / "logo.png"), width=72)
    st.title("AeroClim")
    st.caption("Report evidence dashboard")
    requested_city = st.query_params.get("city", "Seattle")
    city_names = list(CITY_FILES)
    default_index = city_names.index(requested_city) if requested_city in city_names else 0
    city = st.selectbox("Study region", city_names, index=default_index)
    st.info(
        "This dashboard renders the repository's actual station data, saved "
        "Random Forest artifact, and historical model metrics."
    )

st.title("AeroClim Climate Intelligence Platform")
st.markdown(
    "**Multimodal assessment of U.S. environmental change using station weather, "
    "sea-surface temperature, teleconnection indices, and machine learning.**"
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Regional climate", "Ocean coupling", "Model evidence", "Radar explainability"]
)

df = load_city(city)

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Daily records", f"{len(df):,}")
    c2.metric("Mean temperature", f"{df['TAVG'].mean():.2f} °C")
    c3.metric("Annual rainfall", f"{df.set_index('date')['PRCP'].resample('YE').sum().mean():.0f} mm")
    c4.metric("Maximum daily rainfall", f"{df['PRCP'].max():.1f} mm")

    recent = df[df["date"] >= "2018-01-01"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent["date"], y=recent["TMAX"], name="Maximum temperature"))
    fig.add_trace(go.Scatter(x=recent["date"], y=recent["TMIN"], name="Minimum temperature"))
    fig.add_trace(go.Bar(x=recent["date"], y=recent["PRCP"], name="Precipitation", yaxis="y2", opacity=0.35))
    fig.update_layout(
        title=f"Recent Weather Variability: {city}",
        yaxis_title="Temperature (°C)",
        yaxis2=dict(title="Precipitation (mm)", overlaying="y", side="right"),
        height=460,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, width="stretch")

    annual = df.set_index("date").resample("YE").agg(TAVG=("TAVG", "mean"), PRCP=("PRCP", "sum")).reset_index()
    st.dataframe(annual.tail(8).round(2), width="stretch", hide_index=True)

with tab2:
    ocean = pd.read_csv(ROOT / "data" / "ocean.csv")
    ocean["date"] = pd.to_datetime(ocean["date"])
    for col in ["sst_atlantic_f", "sst_pacific_f", "sst_gulf_f"]:
        ocean[col.replace("_f", "_c")] = (ocean[col] - 32) * 5 / 9
    long = ocean.melt(
        id_vars="date",
        value_vars=["sst_atlantic_c", "sst_pacific_c", "sst_gulf_c"],
        var_name="Basin",
        value_name="SST",
    )
    long["Basin"] = long["Basin"].str.replace("sst_", "").str.replace("_c", "").str.title()
    annual_sst = long.assign(Year=long["date"].dt.year).groupby(["Year", "Basin"], as_index=False)["SST"].mean()
    fig = px.line(annual_sst, x="Year", y="SST", color="Basin", markers=True,
                  title="Annual Sea-Surface Temperature by Basin")
    st.plotly_chart(fig, width="stretch")
    st.image(str(ASSETS / "fig05_climate_indices.png"), caption="ENSO, PDO, and NAO indices used by AeroClim")

with tab3:
    c1, c2, c3 = st.columns(3)
    c1.metric("Random Forest accuracy", f"{metrics['rf_accuracy'] * 100:.2f}%")
    c2.metric("Fusion accuracy", f"{metrics['fusion_accuracy'] * 100:.2f}%")
    c3.metric("LSTM R²", f"{metrics['lstm_r2']:.3f}")
    st.warning(
        "The near-perfect classification metrics are historical saved values. "
        "They require cautious interpretation because the target is derived from "
        "engineered threshold features that also appear among the predictors."
    )
    imp = pd.DataFrame({"Feature": RF_FEATURES, "Importance": rf_model.feature_importances_})
    imp = imp.sort_values("Importance", ascending=True).tail(12)
    fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                 title="Saved Random Forest Global Feature Importance")
    st.plotly_chart(fig, width="stretch")
    st.image(str(ASSETS / "fig08_architecture.png"), caption="Repository-verified late-fusion design")

with tab4:
    st.subheader("Radar and CNN Activation Demonstration")
    st.write(
        "The current repository contains Grad-CAM utilities but no trained radar CNN. "
        "Accordingly, the visualization below is explicitly synthetic and is used only "
        "to demonstrate the explainability workflow."
    )
    st.image(str(ASSETS / "fig11_radar_gradcam.png"), width="stretch")
