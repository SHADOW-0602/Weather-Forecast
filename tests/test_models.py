import json
from pathlib import Path

from ml_model import HazardPredictor


def test_all_model_artifacts_reload():
    predictor = HazardPredictor()
    assert predictor.rf_model is not None
    assert predictor.lstm_model is not None
    assert predictor.fusion_model is not None
    assert predictor.event_model is not None
    assert predictor.scaler_event is not None


def test_training_manifest_has_leakage_safe_boundaries():
    manifest = json.loads(
        Path("saved_models/training_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["train_end"] < manifest["validation_start"]
    assert manifest["validation_end"] < manifest["test_start"]
    assert manifest["forecast_horizon_days"] == 1
    assert manifest["train_rows"] >= 3650
    assert manifest["station_count"] >= 20
    assert manifest["event_labeled_rows"] > 0
    assert "ATMOSPHERE_IMPUTED" in manifest["rf_features"]
    assert "PRCP_7D" in manifest["rf_features"]


def test_saved_metrics_are_plausible():
    metrics = json.loads(Path("saved_models/metrics.json").read_text(encoding="utf-8"))
    assert 0.5 <= metrics["rf_auc"] <= 1.0
    assert 0.5 <= metrics["fusion_auc"] <= 1.0
    assert metrics["event_labeled_test_rows"] > 0
    assert 0.0 <= metrics["rf_event_auc"] <= 1.0
    assert metrics["event_model_trained"] is True
    assert 0.0 <= metrics["event_model_test_f1"] <= 1.0
    assert 0.0 < metrics["event_model_threshold"] < 1.0
    assert 0.0 < metrics["lstm_mae_c"] < 10.0
