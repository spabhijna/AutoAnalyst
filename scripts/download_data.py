"""
Download the Superstore Sales sample dataset.

The Superstore dataset is a widely-used public domain sales dataset with
~10,000 rows covering orders, customers, products, and financials.
We fetch the CSV from a reliable GitHub mirror and save it to data/raw/.

Usage:
  python scripts/download_data.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# Public mirror of the Superstore dataset (Kaggle-derived, CC0 license)
_DATASET_URL = (
    "https://raw.githubusercontent.com/nicholasdavies/superstore-sales/"
    "main/Sample%20-%20Superstore.csv"
)

_OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "superstore_sales.csv"


def download_dataset(url: str = _DATASET_URL, output: Path = _OUTPUT_PATH) -> Path:
    """
    Download the Superstore Sales CSV to *output*.

    Args:
        url:    URL to download from.
        output: Destination file path (parent dirs are created if needed).

    Returns:
        The path the file was saved to.

    Raises:
        urllib.error.URLError: If the download fails.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists():
        print(f"Dataset already exists at {output} — skipping download.")
        return output

    print(f"Downloading dataset from:\n  {url}")
    print(f"Saving to: {output}")

    urllib.request.urlretrieve(url, output)  # noqa: S310

    size_kb = output.stat().st_size / 1024
    print(f"Done! Downloaded {size_kb:.1f} KB")
    return output


if __name__ == "__main__":
    try:
        path = download_dataset()
        print(f"\nDataset ready at: {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
