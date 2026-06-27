"""Download a small labeled historical MRMS sample from the public S3 archive.

The script selects event-rich dates from data/event_labels.csv and downloads one
CONUS MRMS timestep per date for the three image channels used by the CNN.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests


BUCKET_URL = "https://noaa-mrms-pds.s3.amazonaws.com"
PRODUCT_PREFIXES = {
    "MergedReflectivityQCComposite": "CONUS/MergedReflectivityQCComposite_00.50",
    "RadarOnly_QPE_01H": "CONUS/RadarOnly_QPE_01H_00.00",
    "MergedAzShear_0-2kmAGL": "CONUS/MergedAzShear_0-2kmAGL_00.50",
}
S3_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


@dataclass
class DownloadedHistoricalFile:
    product: str
    date: str
    target_hour: int
    key: str
    url: str
    local_path: str
    bytes: int


def parse_key_timestamp(key: str) -> datetime | None:
    match = re.search(r"(\d{8})-(\d{6})", key)
    if not match:
        return None
    return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")


def list_s3_keys(prefix: str) -> list[str]:
    keys: list[str] = []
    token = None
    while True:
        params = f"list-type=2&prefix={quote(prefix)}&max-keys=1000"
        if token:
            params += f"&continuation-token={quote(token)}"
        url = f"{BUCKET_URL}/?{params}"
        response = request_with_retries(url, timeout=60)
        root = ET.fromstring(response.text)
        keys.extend(
            item.find("s:Key", S3_NS).text
            for item in root.findall("s:Contents", S3_NS)
            if item.find("s:Key", S3_NS) is not None
        )
        next_token = root.find("s:NextContinuationToken", S3_NS)
        if next_token is None or not next_token.text:
            break
        token = next_token.text
    return keys


def pick_key_for_date(product_prefix: str, date_key: str, target_hour: int) -> str:
    prefix = f"{product_prefix}/{date_key}/"
    keys = list_grib_keys_for_date(product_prefix, date_key)
    return pick_key_from_keys(keys, date_key, target_hour, prefix)


def list_grib_keys_for_date(product_prefix: str, date_key: str) -> list[str]:
    prefix = f"{product_prefix}/{date_key}/"
    return [
        key for key in list_s3_keys(prefix)
        if key.endswith(".grib2.gz") or key.endswith(".grib2")
    ]


def pick_key_from_keys(
    keys: list[str],
    date_key: str,
    target_hour: int,
    prefix: str,
) -> str:
    if not keys:
        raise FileNotFoundError(f"No MRMS keys under {prefix}")
    target = datetime.strptime(f"{date_key}{target_hour:02d}0000", "%Y%m%d%H%M%S")
    timestamped = [
        (key, parse_key_timestamp(key))
        for key in keys
    ]
    timestamped = [(key, ts) for key, ts in timestamped if ts is not None]
    if not timestamped:
        return sorted(keys)[len(keys) // 2]
    return min(timestamped, key=lambda item: abs((item[1] - target).total_seconds()))[0]


def write_manifest(
    out_dir: Path,
    dates: list[str],
    target_hours: list[int],
    downloaded: list[DownloadedHistoricalFile],
) -> None:
    manifest = {
        "source": "s3://noaa-mrms-pds",
        "products": PRODUCT_PREFIXES,
        "dates": dates,
        "target_hours_utc": target_hours,
        "files": [asdict(item) for item in downloaded],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def download_key(key: str, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BUCKET_URL}/{key}"
    if out_path.exists():
        return out_path.stat().st_size
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with request_with_retries(url, timeout=120, stream=True) as response:
        total = 0
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    total += len(chunk)
    tmp_path.replace(out_path)
    return total


def request_with_retries(
    url: str,
    timeout: int,
    stream: bool = False,
    attempts: int = 4,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(url, timeout=timeout, stream=stream)
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(min(2 ** attempt, 15))
    assert last_error is not None
    raise last_error


def choose_dates(max_event_dates: int, negative_dates: int) -> list[str]:
    labels = pd.read_csv("data/event_labels.csv")
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce")
    labels = labels.dropna(subset=["date"])
    counts = (
        labels.groupby(labels["date"].dt.strftime("%Y%m%d"))
        .size()
        .sort_values(ascending=False)
    )
    event_dates = counts.head(max_event_dates).index.tolist()
    if negative_dates <= 0:
        return event_dates

    all_days = pd.date_range(labels["date"].min(), labels["date"].max(), freq="D")
    event_date_set = set(labels["date"].dt.strftime("%Y%m%d"))
    quiet_days = [
        day.strftime("%Y%m%d")
        for day in all_days
        if day.strftime("%Y%m%d") not in event_date_set
    ]
    quiet_sample = (
        pd.Series(quiet_days)
        .sample(min(negative_dates, len(quiet_days)), random_state=42)
        .sort_values()
        .tolist()
        if quiet_days
        else []
    )
    return sorted(set(event_dates + quiet_sample))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/mrms_historical_raw")
    parser.add_argument("--max-dates", type=int, default=12)
    parser.add_argument("--target-hour", type=int, default=18)
    parser.add_argument(
        "--target-hours",
        default=None,
        help="Comma-separated UTC hours. Overrides --target-hour when provided.",
    )
    parser.add_argument(
        "--negative-dates",
        type=int,
        default=0,
        help="Number of quiet non-event dates to add.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue when a date/product cannot be listed or downloaded.",
    )
    args = parser.parse_args()

    target_hours = (
        [int(value) for value in args.target_hours.split(",")]
        if args.target_hours
        else [args.target_hour]
    )
    dates = choose_dates(args.max_dates, args.negative_dates)
    out_dir = Path(args.out_dir)
    downloaded: list[DownloadedHistoricalFile] = []
    print(f"Selected dates: {', '.join(dates)}")

    for date_key in dates:
        for product, product_prefix in PRODUCT_PREFIXES.items():
            prefix = f"{product_prefix}/{date_key}/"
            try:
                print(f"Listing {product} {date_key}", flush=True)
                keys = list_grib_keys_for_date(product_prefix, date_key)
            except Exception as exc:
                print(f"  ERROR listing {product} {date_key}: {exc}")
                if not args.skip_errors:
                    raise
                continue
            for target_hour in target_hours:
                try:
                    key = pick_key_from_keys(keys, date_key, target_hour, prefix)
                    filename = Path(key).name
                    local_path = out_dir / product / filename
                    print(f"  Download {key}")
                    size = download_key(key, local_path)
                    downloaded.append(
                        DownloadedHistoricalFile(
                            product=product,
                            date=date_key,
                            target_hour=target_hour,
                            key=key,
                            url=f"{BUCKET_URL}/{key}",
                            local_path=str(local_path),
                            bytes=size,
                        )
                    )
                    write_manifest(out_dir, dates, target_hours, downloaded)
                except Exception as exc:
                    print(f"  ERROR {product} {date_key} {target_hour:02d}Z: {exc}")
                    if not args.skip_errors:
                        raise

    write_manifest(out_dir, dates, target_hours, downloaded)
    print(f"Wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
