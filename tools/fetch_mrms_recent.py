"""Download a bounded recent sample of MRMS 2D products.

The live MRMS directory can contain many frequently-updated GRIB2 files, so this
script intentionally defaults to a small recent sample instead of mirroring an
entire product folder.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import requests


PRODUCT_URLS = {
    "RadarOnly_QPE_01H": "https://mrms.ncep.noaa.gov/2D/RadarOnly_QPE_01H/",
    "MergedReflectivityQCComposite": "https://mrms.ncep.noaa.gov/2D/MergedReflectivityQCComposite/",
    "MergedAzShear_0-2kmAGL": "https://mrms.ncep.noaa.gov/2D/MergedAzShear_0-2kmAGL/",
}


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


@dataclass
class DownloadedFile:
    product: str
    filename: str
    url: str
    local_path: str
    bytes: int


def fetch_file_list(base_url: str, timeout: int = 60) -> list[str]:
    response = requests.get(base_url, timeout=timeout)
    response.raise_for_status()
    parser = LinkParser()
    parser.feed(response.text)
    links = []
    for href in parser.links:
        if href.startswith(("?", "/", "../")):
            continue
        if href.endswith("/"):
            continue
        # MRMS data files are usually GRIB2, often gzip-compressed.
        if not re.search(r"\.(grib2|grib2\.gz|gz)$", href, re.IGNORECASE):
            continue
        links.append(href)
    return sorted(set(links))


def download_file(url: str, path: Path, timeout: int = 120) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = 0
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                total += len(chunk)
    tmp_path.replace(path)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="data/mrms_raw",
        help="Target directory for downloaded product files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=6,
        help="Number of newest files to download per product.",
    )
    parser.add_argument(
        "--product",
        action="append",
        choices=sorted(PRODUCT_URLS),
        help="Product to download. Repeatable. Defaults to all three products.",
    )
    args = parser.parse_args()

    products = args.product or list(PRODUCT_URLS)
    out_dir = Path(args.out_dir)
    downloaded: list[DownloadedFile] = []

    for product in products:
        base_url = PRODUCT_URLS[product]
        print(f"Listing {product}: {base_url}", flush=True)
        files = fetch_file_list(base_url)
        selected = files[-args.max_files :]
        print(f"  Found {len(files)} files; downloading {len(selected)} newest")

        for filename in selected:
            url = urljoin(base_url, filename)
            local_path = out_dir / product / filename
            if local_path.exists():
                size = local_path.stat().st_size
                print(f"  Skip existing {filename} ({size:,} bytes)")
            else:
                print(f"  Download {filename}")
                size = download_file(url, local_path)
            downloaded.append(
                DownloadedFile(
                    product=product,
                    filename=filename,
                    url=url,
                    local_path=str(local_path),
                    bytes=size,
                )
            )

    manifest = {
        "source": "https://mrms.ncep.noaa.gov/2D/",
        "note": (
            "Live MRMS sample. For model training over historical years, use the "
            "NOAA MRMS archive rather than only this live directory."
        ),
        "max_files_per_product": args.max_files,
        "files": [asdict(item) for item in downloaded],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
