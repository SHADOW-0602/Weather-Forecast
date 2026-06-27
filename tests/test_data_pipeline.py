import json
from pathlib import Path

import pandas as pd
import pytest

from noaa_client import CITIES, NOAAClient


def test_station_catalog_has_20_unique_noaa_stations():
    stations = [station for station in CITIES if station.startswith("USW")]
    assert len(stations) == 20
    assert len(set(stations)) == 20


def test_reduced_station_loads_complete_model_inputs():
    frame = NOAAClient().fetch_weather_data(
        "USW00024233", "2022-01-01", "2024-12-31"
    )
    required = [
        "TMAX",
        "TMIN",
        "PRCP",
        "DEWPOINT_F",
        "WINDSPEED_MPH",
        "PRESSURE_HPA",
        "HUMIDITY_PCT",
        "SOIL_MOISTURE_VOL",
        "SATURATION_PCT",
        "enso_nino34",
        "pdo_index",
        "nao_index",
        "ATMOSPHERE_IMPUTED",
        "SOIL_MOISTURE_OBSERVED",
        "CLIMATE_INDEX_OBSERVED",
    ]
    assert not frame[required].isna().any().any()
    assert frame["Date"].min() == pd.Timestamp("2022-01-01")
    assert frame["Date"].max() == pd.Timestamp("2024-12-31")


def test_missing_station_does_not_silently_generate_synthetic_data():
    with pytest.raises(FileNotFoundError):
        NOAAClient().fetch_weather_data(
            "DOES_NOT_EXIST", "2024-01-01", "2024-12-31"
        )


def test_auxiliary_sources_cover_station_test_period():
    for filename in ["data/climate_indices.csv", "data/ocean.csv"]:
        frame = pd.read_csv(filename)
        dates = pd.to_datetime(frame["date"])
        assert dates.min() <= pd.Timestamp("1995-01-01")
        assert dates.max() >= pd.Timestamp("2024-12-31")


def test_data_health_report_is_current_and_training_ready():
    report = json.loads(Path("data/data_health.json").read_text(encoding="utf-8"))
    assert report["unique_noaa_stations"] == 20
    assert report["station_errors"] == 0
    assert report["stations_training_ready"] == 20


def test_independent_event_labels_exist_for_validation_period():
    labels = pd.read_csv("data/event_labels.csv")
    assert len(labels) > 0
    assert labels["NOAA_STATION"].nunique() >= 15
    assert {"date", "NOAA_STATION", "EVENT_LABEL", "EVENT_TYPE"}.issubset(
        labels.columns
    )
