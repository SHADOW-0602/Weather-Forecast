"""Train a 2D CNN event classifier from MRMS image chips.

The input NPZ must contain:
  X: [samples, height, width, channels]
  y: [samples], where 0/1 are labels and -1 means unlabeled
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def best_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return 0.5, 0.0
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [
        f1_score(y_true, (y_score >= threshold).astype(int), zero_division=0)
        for threshold in thresholds
    ]
    index = int(np.argmax(scores))
    return float(thresholds[index]), float(scores[index])


def build_model(input_shape: tuple[int, int, int], loss_name: str) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(24, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(48, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(96, 3, padding="same", activation="relu", name="final_conv")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    if loss_name == "focal":
        loss = tf.keras.losses.BinaryFocalCrossentropy(
            gamma=2.0,
            apply_class_balancing=False,
        )
    else:
        loss = "binary_crossentropy"
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/mrms_images/mrms_live_chips.npz")
    parser.add_argument("--model-out", default="saved_models/cnn_event_model.keras")
    parser.add_argument("--metrics-out", default="saved_models/cnn_event_metrics.json")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--loss", choices=["bce", "focal"], default="focal")
    parser.add_argument(
        "--no-balance-train",
        action="store_true",
        help="Disable class-balanced undersampling of the CNN training split.",
    )
    args = parser.parse_args()

    dataset = np.load(args.dataset)
    X = dataset["X"].astype(np.float32)
    y = dataset["y"].astype(np.int8)
    labeled = y >= 0
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else Path(args.dataset).with_name("manifest.csv")
    )
    manifest = None
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        if len(manifest) == len(labeled):
            manifest = manifest.loc[labeled].reset_index(drop=True)
        else:
            manifest = None
    X = X[labeled]
    y = y[labeled].astype(int)

    if len(y) < 20 or len(np.unique(y)) < 2:
        summary = {
            "trained": False,
            "reason": (
                "Not enough labeled MRMS image chips with both classes. "
                "The live MRMS sample is expected to be unlabeled; build "
                "historical chips overlapping data/event_labels.csv."
            ),
            "labeled_samples": int(len(y)),
            "classes": sorted(np.unique(y).tolist()) if len(y) else [],
        }
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    split_strategy = "random_stratified"
    if manifest is not None and "timestamp" in manifest.columns:
        years = pd.to_datetime(manifest["timestamp"], errors="coerce").dt.year
        train_mask = years <= 2022
        val_mask = years == 2023
        test_mask = years >= 2024
        if (
            train_mask.sum() >= 20
            and val_mask.sum() >= 20
            and test_mask.sum() >= 20
            and len(np.unique(y[train_mask])) == 2
            and len(np.unique(y[val_mask])) == 2
            and len(np.unique(y[test_mask])) == 2
        ):
            X_train, y_train = X[train_mask], y[train_mask]
            X_val, y_val = X[val_mask], y[val_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            split_strategy = "chronological_2022_2023_2024"
        else:
            manifest = None

    if split_strategy == "random_stratified":
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train,
        )

    original_train_samples = int(len(y_train))
    original_train_positive_rate = float(np.mean(y_train))
    balanced_train = False
    if not args.no_balance_train:
        class_indexes = {
            cls: np.flatnonzero(y_train == cls)
            for cls in np.unique(y_train)
        }
        if len(class_indexes) == 2:
            rng = np.random.default_rng(42)
            target_count = min(len(indexes) for indexes in class_indexes.values())
            if target_count > 0:
                selected = np.concatenate(
                    [
                        rng.choice(indexes, size=target_count, replace=False)
                        for indexes in class_indexes.values()
                    ]
                )
                rng.shuffle(selected)
                X_train = X_train[selected]
                y_train = y_train[selected]
                balanced_train = True

    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {int(cls): float(weight) for cls, weight in zip(classes, weights)}

    model = build_model(tuple(X.shape[1:]), args.loss)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=3, restore_best_weights=True, monitor="val_auc", mode="max"
        )
    ]
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    val_probabilities = model.predict(X_val, verbose=0).flatten()
    threshold, validation_f1 = best_threshold_by_f1(y_val, val_probabilities)
    probabilities = model.predict(X_test, verbose=0).flatten()
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "trained": True,
        "samples": int(len(y)),
        "train_samples": int(len(y_train)),
        "original_train_samples": original_train_samples,
        "validation_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "split_strategy": split_strategy,
        "balanced_train": balanced_train,
        "loss": args.loss,
        "original_train_positive_rate": original_train_positive_rate,
        "train_positive_rate": float(np.mean(y_train)),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
        "test_auc": safe_auc(y_test, probabilities),
        "test_average_precision": float(average_precision_score(y_test, probabilities)),
        "threshold": threshold,
        "validation_f1_at_threshold": validation_f1,
        "positive_rate": float(np.mean(y)),
        "class_weight": class_weight,
        "epochs_ran": int(len(history.history.get("loss", []))),
    }

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
