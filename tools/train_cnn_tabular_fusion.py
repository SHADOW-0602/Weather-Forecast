"""Train a small fusion model combining MRMS CNN and tabular event probabilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_model import HazardPredictor, RF_FEATURES
from noaa_client import NOAAClient


def safe_auc(y_true, y_score) -> float:
    return 0.5 if len(np.unique(y_true)) < 2 else float(roc_auc_score(y_true, y_score))


def best_threshold_by_f1(y_true, y_score) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, y_score >= t, zero_division=0) for t in thresholds]
    index = int(np.argmax(scores))
    return float(thresholds[index]), float(scores[index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/mrms_historical_images/mrms_live_chips.npz")
    parser.add_argument("--manifest", default="data/mrms_historical_images/manifest.csv")
    parser.add_argument("--cnn-model", default="saved_models/cnn_event_model.keras")
    parser.add_argument("--model-out", default="saved_models/cnn_tabular_fusion.pkl")
    parser.add_argument("--metrics-out", default="saved_models/cnn_tabular_fusion_metrics.json")
    args = parser.parse_args()

    data = np.load(args.dataset)
    X_img = data["X"].astype(np.float32)
    y = data["y"].astype(int)
    manifest = pd.read_csv(args.manifest)
    labeled = y >= 0
    X_img = X_img[labeled]
    y = y[labeled]
    manifest = manifest.loc[labeled].reset_index(drop=True)
    manifest["timestamp"] = pd.to_datetime(manifest["timestamp"])
    manifest["date"] = manifest["timestamp"].dt.normalize()

    cnn = tf.keras.models.load_model(args.cnn_model)
    cnn_prob = cnn.predict(X_img, batch_size=64, verbose=0).flatten()

    predictor = HazardPredictor()
    if predictor.event_model is None or predictor.scaler_event is None:
        raise RuntimeError("Tabular event model artifacts are missing.")

    client = NOAAClient()
    tabular_cache: dict[str, pd.DataFrame] = {}
    tabular_probs = []
    for row in manifest.itertuples():
        station_id = str(row.station_id).upper()
        if station_id not in tabular_cache:
            frame = client.fetch_weather_data(station_id, "2022-01-01", "2024-12-31")
            frame = predictor._add_sst_feature(frame)
            frame = predictor._add_event_labels(frame)
            engineered = predictor._engineer_features_bulk(frame.sort_values("Date"))
            engineered["Date"] = pd.to_datetime(engineered["Date"]).dt.normalize()
            tabular_cache[station_id] = engineered
        station_frame = tabular_cache[station_id]
        match = station_frame[station_frame["Date"] == row.date]
        if match.empty:
            tabular_probs.append(np.nan)
            continue
        rf_row = match.iloc[[0]][RF_FEATURES].fillna(0)
        scaled = predictor.scaler_event.transform(rf_row)
        tabular_probs.append(float(predictor.event_model.predict_proba(scaled)[0, 1]))

    tabular_prob = np.array(tabular_probs, dtype=np.float32)
    valid = ~np.isnan(tabular_prob)
    y = y[valid]
    cnn_prob = cnn_prob[valid]
    tabular_prob = tabular_prob[valid]
    years = manifest.loc[valid, "timestamp"].dt.year.values

    X = np.column_stack([cnn_prob, tabular_prob])
    train = years <= 2022
    val = years == 2023
    test = years >= 2024
    if min(train.sum(), val.sum(), test.sum()) < 20:
        train = np.zeros_like(years, dtype=bool)
        val = np.zeros_like(years, dtype=bool)
        test = np.zeros_like(years, dtype=bool)
        train[: int(len(years) * 0.7)] = True
        val[int(len(years) * 0.7): int(len(years) * 0.85)] = True
        test[int(len(years) * 0.85):] = True
        split_strategy = "fallback_ordered"
    else:
        split_strategy = "chronological_2022_2023_2024"

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train])
    X_val = scaler.transform(X[val])
    X_test = scaler.transform(X[test])
    model = LogisticRegression(class_weight="balanced", random_state=42)
    model.fit(X_train, y[train])

    val_prob = model.predict_proba(X_val)[:, 1]
    threshold, val_f1 = best_threshold_by_f1(y[val], val_prob)
    test_prob = model.predict_proba(X_test)[:, 1]
    test_pred = test_prob >= threshold
    metrics = {
        "trained": True,
        "samples": int(len(y)),
        "train_samples": int(train.sum()),
        "validation_samples": int(val.sum()),
        "test_samples": int(test.sum()),
        "split_strategy": split_strategy,
        "threshold": threshold,
        "validation_f1_at_threshold": val_f1,
        "test_accuracy": float(accuracy_score(y[test], test_pred)),
        "test_f1": float(f1_score(y[test], test_pred, zero_division=0)),
        "test_auc": safe_auc(y[test], test_prob),
        "test_average_precision": float(average_precision_score(y[test], test_prob)),
        "features": ["cnn_event_probability", "tabular_event_probability"],
    }

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "threshold": threshold}, args.model_out)
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
