"""Relabel existing MRMS image chips with time-aware NOAA Storm Events windows.

This updates the NPZ ``y`` array and manifest label metadata without re-decoding
the GRIB2 radar files. Use it to tune event-window assumptions after
``tools/build_mrms_image_chips.py`` has already created image chips.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_station_events(event_labels_path: Path) -> tuple[dict[str, pd.DataFrame], object, object]:
    labels = pd.read_csv(event_labels_path)
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce")
    labels = labels.dropna(subset=["date"])
    labels["NOAA_STATION"] = labels["NOAA_STATION"].astype(str).str.upper()
    labels["EVENT_BEGIN_TIME"] = pd.to_datetime(
        labels["EVENT_BEGIN_TIME"], errors="coerce"
    )
    labels["EVENT_END_TIME"] = pd.to_datetime(
        labels.get("EVENT_END_TIME", labels["EVENT_BEGIN_TIME"]), errors="coerce"
    )
    labels["EVENT_END_TIME"] = labels["EVENT_END_TIME"].fillna(labels["EVENT_BEGIN_TIME"])
    labels = labels.dropna(subset=["EVENT_BEGIN_TIME"])
    station_events = {
        station_id: group.copy()
        for station_id, group in labels.groupby("NOAA_STATION")
    }
    return station_events, labels["date"].min().date(), labels["date"].max().date()


def relabel_row(
    row: pd.Series,
    station_events: dict[str, pd.DataFrame],
    event_start,
    event_end,
    before_hours: float,
    after_hours: float,
    ambiguous_hours: float,
) -> dict[str, object]:
    station_id = str(row["station_id"]).upper()
    timestamp = pd.Timestamp(row["timestamp"])
    sample_date = timestamp.date()
    if station_id not in station_events or not (event_start <= sample_date <= event_end):
        return {
            "label": -1,
            "label_note": "unlabeled_outside_event_coverage",
            "closest_event_minutes": np.nan,
            "closest_event_signed_minutes": np.nan,
            "closest_event_type": "",
            "closest_event_distance_km": np.nan,
        }

    events = station_events[station_id]
    begin_times = events["EVENT_BEGIN_TIME"]
    end_times = events["EVENT_END_TIME"]
    deltas = begin_times - timestamp
    closest_index = deltas.abs().idxmin()
    closest_minutes = float(deltas.abs().loc[closest_index].total_seconds() / 60.0)
    closest_signed = float(deltas.loc[closest_index].total_seconds() / 60.0)
    closest_type = str(events.loc[closest_index].get("EVENT_TYPE", ""))
    closest_distance = float(events.loc[closest_index].get("EVENT_DISTANCE_KM", np.nan))

    positive_start = begin_times - pd.to_timedelta(before_hours, unit="h")
    positive_end = end_times + pd.to_timedelta(after_hours, unit="h")
    if ((timestamp >= positive_start) & (timestamp <= positive_end)).any():
        label = 1
        label_note = f"noaa_storm_events_time_window_positive_-{before_hours:g}h_+{after_hours:g}h"
    else:
        ambiguous_start = begin_times - pd.to_timedelta(ambiguous_hours, unit="h")
        ambiguous_end = end_times + pd.to_timedelta(ambiguous_hours, unit="h")
        if ((timestamp >= ambiguous_start) & (timestamp <= ambiguous_end)).any():
            label = -1
            label_note = f"unlabeled_near_event_outside_positive_window_{ambiguous_hours:g}h"
        else:
            label = 0
            label_note = "noaa_storm_events_time_window"

    return {
        "label": label,
        "label_note": label_note,
        "closest_event_minutes": closest_minutes,
        "closest_event_signed_minutes": closest_signed,
        "closest_event_type": closest_type,
        "closest_event_distance_km": closest_distance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/mrms_historical_images/mrms_live_chips.npz")
    parser.add_argument("--manifest", default="data/mrms_historical_images/manifest.csv")
    parser.add_argument("--event-labels", default="data/event_labels.csv")
    parser.add_argument("--event-window-before-hours", type=float, default=6.0)
    parser.add_argument("--event-window-after-hours", type=float, default=6.0)
    parser.add_argument("--ambiguous-window-hours", type=float, default=12.0)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    manifest_path = Path(args.manifest)
    data = np.load(dataset_path)
    arrays = {name: data[name] for name in data.files}
    manifest = pd.read_csv(manifest_path)
    station_events, event_start, event_end = build_station_events(Path(args.event_labels))

    updates = manifest.apply(
        lambda row: relabel_row(
            row,
            station_events,
            event_start,
            event_end,
            args.event_window_before_hours,
            args.event_window_after_hours,
            args.ambiguous_window_hours,
        ),
        axis=1,
        result_type="expand",
    )
    for column in updates.columns:
        manifest[column] = updates[column]
    arrays["y"] = manifest["label"].astype(np.int8).to_numpy()
    np.savez_compressed(dataset_path, **arrays)
    manifest.to_csv(manifest_path, index=False)

    summary = {
        "dataset": str(dataset_path),
        "manifest": str(manifest_path),
        "samples": int(len(manifest)),
        "labeled_samples": int((manifest["label"] >= 0).sum()),
        "label_counts": {
            str(key): int(value)
            for key, value in manifest["label"].value_counts().sort_index().items()
        },
        "event_window_before_hours": args.event_window_before_hours,
        "event_window_after_hours": args.event_window_after_hours,
        "ambiguous_window_hours": args.ambiguous_window_hours,
    }
    (manifest_path.parent / "relabel_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
