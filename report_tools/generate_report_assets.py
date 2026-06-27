from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_output" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tmp" / "matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


CITY_META = {
    "Seattle": {"file": "seattle.csv", "lat": 47.45, "lon": -122.31, "basin": "sst_pacific_f"},
    "New York": {"file": "new_york.csv", "lat": 40.78, "lon": -73.97, "basin": "sst_atlantic_f"},
    "Phoenix": {"file": "phoenix.csv", "lat": 33.43, "lon": -112.01, "basin": "sst_gulf_f"},
}

RF_FEATURES = [
    "HEAT_INDEX", "WIND_CHILL", "WET_BULB", "DROUGHT_IDX", "TEMP_ANOM",
    "PRCP_ANOM", "EXTREME_HEAT", "EXTREME_COLD", "SST", "SST_AIR_DIFF",
    "FLOOD_RISK_IDX", "WIND_RISK_IDX", "PRESSURE_HPA", "HUMIDITY_PCT",
    "SOIL_MOISTURE_VOL", "SATURATION_PCT", "enso_nino34", "pdo_index",
    "nao_index",
]


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_city(name: str) -> pd.DataFrame:
    meta = CITY_META[name]
    df = pd.read_csv(ROOT / "data" / "noaa_stations" / meta["file"])
    df["date"] = pd.to_datetime(df["date"])
    df["tmax_c"] = df["TMAX"]
    df["tmin_c"] = df["TMIN"]
    df["tavg_c"] = df["TAVG"]
    df["prcp_mm"] = df["PRCP"]
    return df


def build_stats(city_frames: dict[str, pd.DataFrame]) -> dict:
    stats: dict[str, object] = {"cities": {}, "files": {}, "model_metrics": {}}
    for name, df in city_frames.items():
        annual = df.set_index("date").resample("YE").agg(
            tavg_c=("tavg_c", "mean"), prcp_mm=("prcp_mm", "sum")
        )
        slope = np.polyfit(np.arange(len(annual)), annual["tavg_c"], 1)[0] * 10
        stats["cities"][name] = {
            "rows": int(len(df)),
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
            "mean_tavg_c": round(float(df["tavg_c"].mean()), 3),
            "mean_annual_prcp_mm": round(float(annual["prcp_mm"].mean()), 3),
            "temperature_trend_c_per_decade": round(float(slope), 4),
            "max_daily_prcp_mm": round(float(df["prcp_mm"].max()), 3),
        }
    for path in sorted((ROOT / "data").glob("*.csv")):
        stats["files"][path.name] = {
            "rows": int(len(pd.read_csv(path))),
            "bytes": int(path.stat().st_size),
        }
    metrics_path = ROOT / "saved_models" / "metrics.json"
    if metrics_path.exists():
        stats["model_metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
    return stats


def regional_map() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.5))
    ax.set_facecolor("#eef4f8")
    ax.set_xlim(-127, -66)
    ax.set_ylim(24, 51)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("AeroClim Study Regions Across the Contiguous United States", weight="bold")
    colors = {"Seattle": "#2563eb", "New York": "#dc2626", "Phoenix": "#f59e0b"}
    for name, meta in CITY_META.items():
        ax.scatter(meta["lon"], meta["lat"], s=110, color=colors[name], edgecolor="white", linewidth=1.2)
        dx = 1.2 if name != "New York" else -8
        ax.annotate(name, (meta["lon"], meta["lat"]), xytext=(meta["lon"] + dx, meta["lat"] + 1.2),
                    arrowprops=dict(arrowstyle="-", color="#475569"), fontsize=9, weight="bold")
    ax.grid(alpha=0.25)
    save(fig, "fig01_study_regions.png")


def annual_temperature(city_frames: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    colors = {"Seattle": "#2563eb", "New York": "#dc2626", "Phoenix": "#f59e0b"}
    for name, df in city_frames.items():
        annual = df.set_index("date")["tavg_c"].resample("YE").mean()
        years = annual.index.year
        ax.plot(years, annual, lw=1.6, alpha=0.65, color=colors[name], label=name)
        z = np.polyfit(years, annual.values, 1)
        ax.plot(years, np.polyval(z, years), lw=2.2, color=colors[name])
    ax.set_title("Annual Mean Air Temperature and Linear Trends (1995-2022)", weight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean temperature (°C)")
    ax.legend(ncol=3, frameon=False)
    ax.grid(alpha=0.25)
    save(fig, "fig02_annual_temperature.png")


def annual_precipitation(city_frames: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.3), sharex=True)
    colors = {"Seattle": "#2563eb", "New York": "#0f766e", "Phoenix": "#f59e0b"}
    for ax, (name, df) in zip(axes, city_frames.items()):
        annual = df.set_index("date")["prcp_mm"].resample("YE").sum()
        ax.bar(annual.index.year, annual.values, color=colors[name], alpha=0.8)
        ax.set_ylabel("mm")
        ax.set_title(name, loc="left", fontsize=10, weight="bold")
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_title("Annual Accumulated Precipitation by Study Region", weight="bold", pad=18)
    axes[-1].set_xlabel("Year")
    save(fig, "fig03_annual_precipitation.png")


def sst_trends() -> None:
    df = pd.read_csv(ROOT / "data" / "ocean.csv")
    df["date"] = pd.to_datetime(df["date"])
    basins = {
        "Atlantic": "sst_atlantic_f",
        "Pacific": "sst_pacific_f",
        "Gulf": "sst_gulf_f",
    }
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for label, col in basins.items():
        s = ((df[col] - 32.0) * 5.0 / 9.0).groupby(df["date"].dt.year).mean()
        ax.plot(s.index, s.values, lw=1.8, label=label)
    ax.set_title("Annual Mean Sea-Surface Temperature by Coupled Ocean Basin", weight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("SST (°C)")
    ax.legend(ncol=3, frameon=False)
    ax.grid(alpha=0.25)
    save(fig, "fig04_sst_trends.png")


def climate_indices() -> None:
    df = pd.read_csv(ROOT / "data" / "climate_indices.csv")
    df["date"] = pd.to_datetime(df["date"])
    monthly = df.set_index("date")[["enso_nino34", "pdo_index", "nao_index"]].resample("ME").mean()
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 6.7), sharex=True)
    colors = ["#dc2626", "#2563eb", "#7c3aed"]
    labels = ["ENSO Niño 3.4", "Pacific Decadal Oscillation", "North Atlantic Oscillation"]
    for ax, col, color, label in zip(axes, monthly.columns, colors, labels):
        ax.plot(monthly.index, monthly[col], color=color, lw=0.8)
        ax.axhline(0, color="#334155", lw=0.7)
        ax.fill_between(monthly.index, monthly[col], 0, where=monthly[col] >= 0, color=color, alpha=0.18)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.2)
    axes[0].set_title("Atmospheric-Oceanic Teleconnection Indices", weight="bold")
    axes[-1].set_xlabel("Date")
    save(fig, "fig05_climate_indices.png")


def seasonal_climatology(city_frames: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
    for name, df in city_frames.items():
        monthly = df.groupby(df["date"].dt.month).agg(tavg=("tavg_c", "mean"), prcp=("prcp_mm", "mean"))
        axes[0].plot(monthly.index, monthly["tavg"], marker="o", label=name)
        axes[1].plot(monthly.index, monthly["prcp"], marker="o", label=name)
    axes[0].set_title("Monthly Mean Temperature", weight="bold")
    axes[0].set_ylabel("°C")
    axes[1].set_title("Monthly Mean Daily Precipitation", weight="bold")
    axes[1].set_ylabel("mm/day")
    for ax in axes:
        ax.set_xlabel("Month")
        ax.set_xticks(range(1, 13))
        ax.grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, "fig06_seasonal_climatology.png")


def correlation_heatmap() -> None:
    weather = load_city("Seattle")[["date", "tmax_c", "tmin_c", "tavg_c", "prcp_mm"]]
    atmosphere = pd.read_csv(ROOT / "data" / "noaa_atmosphere" / "seattle.csv")
    atmosphere["date"] = pd.to_datetime(atmosphere["date"])
    climate = pd.read_csv(ROOT / "data" / "climate_indices.csv")
    climate["date"] = pd.to_datetime(climate["date"])
    ocean = pd.read_csv(ROOT / "data" / "ocean.csv")
    ocean["date"] = pd.to_datetime(ocean["date"])
    ocean["sst_c"] = (ocean["sst_pacific_f"] - 32.0) * 5.0 / 9.0
    merged = weather.merge(atmosphere, on="date", how="left").merge(
        climate.drop(columns=["enso_phase"], errors="ignore"), on="date", how="left"
    ).merge(ocean[["date", "sst_c"]], on="date", how="left")
    numeric = merged.select_dtypes(include=[np.number]).copy()
    rename = {
        "tmax_c": "Tmax", "tmin_c": "Tmin", "tavg_c": "Tavg", "prcp_mm": "Precip.",
        "DEWPOINT_F": "Dew point", "WINDSPEED_MPH": "Wind", "PRESSURE_HPA": "Pressure",
        "HUMIDITY_PCT": "Humidity", "enso_nino34": "ENSO", "pdo_index": "PDO",
        "nao_index": "NAO", "sst_c": "SST",
    }
    numeric = numeric[[c for c in rename if c in numeric.columns]].rename(columns=rename)
    fig, ax = plt.subplots(figsize=(8.4, 6.3))
    sns.heatmap(numeric.corr(), cmap="vlag", center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.4, annot=False, cbar_kws={"shrink": 0.75}, ax=ax)
    ax.set_title("Correlation Structure of Coupled Seattle Predictors", weight="bold")
    save(fig, "fig07_feature_correlation.png")


def architecture_diagram() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    boxes = [
        (0.3, 4.9, 2.0, 1.2, "30-day sequence\nTmax, Tmin, Tavg,\nprecipitation, SST", "#dbeafe"),
        (3.0, 4.9, 2.0, 1.2, "LSTM branch\n64 units → Dropout\n32 units → Dense", "#bfdbfe"),
        (0.3, 1.2, 2.0, 1.4, "Current conditions\n+ engineered indices\n+ teleconnections", "#dcfce7"),
        (3.0, 1.2, 2.0, 1.4, "Random Forest\n100 trees\nmax depth 10", "#bbf7d0"),
        (6.0, 3.1, 1.7, 1.3, "Late-fusion\nmeta-learner\n16 → 8 → 1", "#fef3c7"),
        (8.3, 3.1, 1.4, 1.3, "Hazard\nprobability\n0–100%", "#fee2e2"),
    ]
    for x, y, w, h, text, color in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                               fc=color, ec="#334155", lw=1.2)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, weight="bold")
    arrows = [((2.3, 5.5), (3.0, 5.5)), ((2.3, 1.9), (3.0, 1.9)),
              ((5.0, 5.5), (6.0, 4.1)), ((5.0, 1.9), (6.0, 3.4)),
              ((7.7, 3.75), (8.3, 3.75))]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                     lw=1.3, color="#475569"))
    ax.set_title("AeroClim Multimodal Late-Fusion Architecture", weight="bold", pad=8)
    save(fig, "fig08_architecture.png")


def model_metrics(stats: dict) -> None:
    metrics = stats["model_metrics"]
    names = ["RF accuracy", "RF AUC", "Fusion accuracy", "Fusion AUC", "LSTM R²"]
    vals = [
        metrics.get("rf_accuracy", 0), metrics.get("rf_auc", 0),
        metrics.get("fusion_accuracy", 0), metrics.get("fusion_auc", 0),
        metrics.get("lstm_r2", 0),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    bars = ax.barh(names, vals, color=["#16a34a", "#15803d", "#2563eb", "#1d4ed8", "#f59e0b"])
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Score")
    ax.set_title("Saved Validation Metrics (Historical Model Artifacts)", weight="bold")
    for bar, val in zip(bars, vals):
        ax.text(val + 0.012, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    save(fig, "fig09_saved_metrics.png")


def rf_importance() -> None:
    model = joblib.load(ROOT / "saved_models" / "rf_model.pkl")
    vals = np.asarray(model.feature_importances_)
    order = np.argsort(vals)[-12:]
    fig, ax = plt.subplots(figsize=(8.4, 5.3))
    ax.barh(np.array(RF_FEATURES)[order], vals[order], color="#2563eb")
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title("Random Forest Global Feature Importance", weight="bold")
    ax.grid(axis="x", alpha=0.25)
    save(fig, "fig10_rf_importance.png")


def radar_heatmap() -> None:
    rng = np.random.default_rng(42)
    h = w = 96
    y, x = np.ogrid[:h, :w]
    radar = np.zeros((h, w), dtype=float)
    activation = np.zeros((h, w), dtype=float)
    for cy, cx, radius, intensity in [(35, 40, 17, 48), (62, 67, 21, 58), (48, 72, 11, 36)]:
        blob = intensity * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * radius ** 2))
        radar = np.maximum(radar, blob)
    radar += rng.normal(0, 1.2, radar.shape)
    radar = np.clip(radar, 0, 65)
    for cy, cx, radius in [(61, 67, 15), (37, 42, 11)]:
        activation = np.maximum(
            activation,
            np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * radius ** 2)),
        )
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.4))
    axes[0].imshow(radar, cmap="turbo", vmin=0, vmax=65)
    axes[0].set_title("Synthetic radar (dBZ)")
    axes[1].imshow(activation, cmap="inferno", vmin=0, vmax=1)
    axes[1].set_title("Synthetic activation")
    axes[2].imshow(radar, cmap="gray", vmin=0, vmax=65)
    axes[2].imshow(activation, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title("Grad-CAM-style overlay")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Radar Explainability Demonstration (Explicitly Synthetic)", weight="bold")
    save(fig, "fig11_radar_gradcam.png")


def hazard_proxies(city_frames: dict[str, pd.DataFrame]) -> None:
    rows = []
    for city, df in city_frames.items():
        t95 = df["tavg_c"].quantile(0.95)
        p90 = df["prcp_mm"].quantile(0.90)
        rows.append({
            "City": city,
            "Extreme heat days (%)": 100 * (df["tavg_c"] > t95).mean(),
            "Heavy precipitation days (%)": 100 * (df["prcp_mm"] > p90).mean(),
            "Dry days (%)": 100 * (df["prcp_mm"] < 0.1).mean(),
        })
    table = pd.DataFrame(rows).set_index("City")
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    table.plot(kind="bar", ax=ax, color=["#ef4444", "#2563eb", "#f59e0b"])
    ax.set_ylabel("Share of records (%)")
    ax.set_title("Descriptive Hazard Proxies by Region", weight="bold")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig12_hazard_proxies.png")


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    city_frames = {name: load_city(name) for name in CITY_META}
    stats = build_stats(city_frames)
    (OUT / "analysis_summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    regional_map()
    annual_temperature(city_frames)
    annual_precipitation(city_frames)
    sst_trends()
    climate_indices()
    seasonal_climatology(city_frames)
    correlation_heatmap()
    architecture_diagram()
    model_metrics(stats)
    rf_importance()
    radar_heatmap()
    hazard_proxies(city_frames)
    print(json.dumps({"output": str(OUT), "figures": 12, "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
