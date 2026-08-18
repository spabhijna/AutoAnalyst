"""
Synthetic sales dataset generator.

Two public functions:
- :func:`generate_sales_data` — produces a clean, realistic sales DataFrame.
- :func:`inject_corruption`   — takes the clean DataFrame and returns a
  corrupted copy + a manifest dict describing every injected defect.

The manifest is the ground truth for Phase 5 evaluation: when the critic
flags data-quality issues, you can check them against the manifest to score
the agent's detection accuracy.

Usage
-----
    from data.generate_data import generate_sales_data, inject_corruption

    clean_df = generate_sales_data(n_rows=700, seed=42)
    corrupt_df, manifest = inject_corruption(clean_df, seed=42)
"""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd


# ── Schema constants ──────────────────────────────────────────────────────────

REGIONS: list[str] = ["North", "South", "East", "West"]

PRODUCT_CATEGORIES: list[str] = [
    "Electronics",
    "Clothing",
    "Food & Beverage",
    "Home & Garden",
    "Sports & Outdoors",
    "Beauty & Health",
]

# Realistic per-unit price ranges (min, max) for each category
_PRICE_RANGES: dict[str, tuple[float, float]] = {
    "Electronics": (49.99, 899.99),
    "Clothing": (14.99, 149.99),
    "Food & Beverage": (3.99, 59.99),
    "Home & Garden": (9.99, 299.99),
    "Sports & Outdoors": (19.99, 349.99),
    "Beauty & Health": (7.99, 89.99),
}

# Typical units sold per transaction (mean, std)
_UNITS_PARAMS: dict[str, tuple[float, float]] = {
    "Electronics": (1.5, 0.8),
    "Clothing": (2.5, 1.5),
    "Food & Beverage": (4.0, 2.0),
    "Home & Garden": (2.0, 1.2),
    "Sports & Outdoors": (1.8, 1.0),
    "Beauty & Health": (3.0, 1.8),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_sales_data(
    n_rows: int = 700,
    seed: int = 42,
    start_date: date = date(2024, 1, 1),
    end_date: date = date(2024, 4, 30),
) -> pd.DataFrame:
    """
    Generate a clean synthetic sales dataset.

    Each row represents one sales transaction with realistic correlation
    between product category, unit price, and units sold.

    Schema
    ------
    - ``transaction_id`` : str    — unique UUID4 hex string
    - ``date``           : date   — random date within [start_date, end_date]
    - ``region``         : str    — one of REGIONS
    - ``product_category``: str   — one of PRODUCT_CATEGORIES
    - ``units_sold``     : int    — positive integer
    - ``unit_price``     : float  — category-appropriate price, 2 d.p.
    - ``revenue``        : float  — units_sold × unit_price, 2 d.p.

    Args:
        n_rows:     Number of transaction rows to generate.
        seed:       NumPy random seed for reproducibility.
        start_date: Earliest transaction date (inclusive).
        end_date:   Latest transaction date (inclusive).

    Returns:
        Clean :class:`pandas.DataFrame` with no nulls, no duplicates,
        and all values in valid ranges.
    """
    rng = np.random.default_rng(seed)

    # Date column: random dates in range
    date_range_days = (end_date - start_date).days
    offsets = rng.integers(0, date_range_days + 1, size=n_rows)
    dates = [start_date + timedelta(days=int(d)) for d in offsets]

    # Region and category
    regions = rng.choice(REGIONS, size=n_rows)
    categories = rng.choice(PRODUCT_CATEGORIES, size=n_rows)

    # Units sold and unit price — category-correlated
    units_sold = np.zeros(n_rows, dtype=int)
    unit_prices = np.zeros(n_rows, dtype=float)

    for i, cat in enumerate(categories):
        mean_units, std_units = _UNITS_PARAMS[cat]
        raw_units = rng.normal(mean_units, std_units)
        units_sold[i] = max(1, int(round(raw_units)))

        lo, hi = _PRICE_RANGES[cat]
        unit_prices[i] = round(rng.uniform(lo, hi), 2)

    revenues = np.round(units_sold * unit_prices, 2)

    # Unique transaction IDs
    transaction_ids = [uuid.uuid4().hex[:12].upper() for _ in range(n_rows)]

    df = pd.DataFrame(
        {
            "transaction_id": transaction_ids,
            "date": dates,
            "region": regions,
            "product_category": categories,
            "units_sold": units_sold,
            "unit_price": unit_prices,
            "revenue": revenues,
        }
    )

    return df.reset_index(drop=True)


def inject_corruption(
    df: pd.DataFrame,
    seed: int = 99,
    duplicate_frac: float = 0.03,
    null_revenue_frac: float = 0.02,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Inject known data-quality defects into a copy of *df*.

    Three categories of corruption are applied in order:

    1. **Duplicate transaction IDs** (~3% of rows): randomly selected rows
       have their ``transaction_id`` replaced with one already present in the
       dataset, creating ID collisions that invalidate any per-transaction
       aggregation.

    2. **Null revenue** (~2% of rows): ``revenue`` is set to ``NaN`` for a
       random subset of rows. These will silently drop from ``sum()`` /
       ``mean()`` calls unless the analyst explicitly handles them.

    3. **Impossible values** (2–3 rows): injected at the end of the DataFrame:
       - One row with **negative revenue** (−999.99) in a random category.
       - One row with a **50× outlier** revenue for the day's mean in a
         random category (flags statistical outlier detection).
       - One row with **zero units sold but positive revenue** (logical
         inconsistency).

    The returned manifest dict is the ground truth for Phase 5 scoring.

    Args:
        df:                  Clean DataFrame (output of :func:`generate_sales_data`).
        seed:                NumPy random seed for reproducibility.
        duplicate_frac:      Fraction of rows whose transaction_id to duplicate.
        null_revenue_frac:   Fraction of rows whose revenue to null.

    Returns:
        A 2-tuple of:
        - ``corrupt_df``: :class:`pandas.DataFrame` with injected defects.
        - ``manifest``:   Dict describing every injected defect (see below).

    Manifest structure
    ------------------
    .. code-block:: python

        {
          "duplicated_transaction_ids": {
              "<txn_id>": [original_idx, duplicate_idx, ...],
              ...
          },
          "null_revenue_indices": [int, ...],
          "impossible_rows": [
              {
                "index": int,
                "type": "negative_revenue" | "revenue_outlier" | "zero_units_positive_revenue",
                "injected_values": {...},
              },
              ...
          ],
          "total_defects": int,
        }
    """
    rng = np.random.default_rng(seed)
    corrupt = df.copy()
    manifest: dict[str, Any] = {
        "duplicated_transaction_ids": {},
        "null_revenue_indices": [],
        "impossible_rows": [],
        "total_defects": 0,
    }

    n = len(corrupt)

    # ── 1. Duplicate transaction IDs ─────────────────────────────────────────
    n_dupes = max(1, int(n * duplicate_frac))
    # Pick source rows whose IDs will be cloned
    source_indices = rng.choice(n, size=n_dupes, replace=False).tolist()
    # Pick target rows (different from sources) that will receive the cloned IDs
    remaining = list(set(range(n)) - set(source_indices))
    target_indices = rng.choice(remaining, size=n_dupes, replace=False).tolist()

    dupe_map: dict[str, list[int]] = {}
    for src_idx, tgt_idx in zip(source_indices, target_indices):
        original_id = corrupt.at[src_idx, "transaction_id"]
        corrupt.at[tgt_idx, "transaction_id"] = original_id
        if original_id not in dupe_map:
            dupe_map[original_id] = [src_idx]
        dupe_map[original_id].append(tgt_idx)

    manifest["duplicated_transaction_ids"] = dupe_map

    # ── 2. Null out revenue ───────────────────────────────────────────────────
    n_nulls = max(1, int(n * null_revenue_frac))
    null_indices = rng.choice(n, size=n_nulls, replace=False).tolist()
    corrupt.loc[null_indices, "revenue"] = np.nan
    manifest["null_revenue_indices"] = sorted(null_indices)

    # ── 3. Impossible values (appended rows) ──────────────────────────────────
    impossible_rows: list[dict[str, Any]] = []
    base_date = df["date"].iloc[0]

    # 3a. Negative revenue
    neg_cat = rng.choice(PRODUCT_CATEGORIES)
    neg_row = {
        "transaction_id": "CORRUPT_NEG001",
        "date": base_date,
        "region": rng.choice(REGIONS),
        "product_category": neg_cat,
        "units_sold": 5,
        "unit_price": 199.99,
        "revenue": -999.99,
    }
    neg_idx = len(corrupt)
    corrupt = pd.concat([corrupt, pd.DataFrame([neg_row])], ignore_index=True)
    impossible_rows.append(
        {
            "index": neg_idx,
            "type": "negative_revenue",
            "injected_values": {"revenue": -999.99, "product_category": neg_cat},
        }
    )

    # 3b. 50× revenue outlier (use median revenue for the category as baseline)
    outlier_cat = rng.choice(PRODUCT_CATEGORIES)
    cat_median = float(
        df.loc[df["product_category"] == outlier_cat, "revenue"].median()
    )
    outlier_revenue = round(cat_median * 50, 2)
    outlier_row = {
        "transaction_id": "CORRUPT_OUT001",
        "date": base_date,
        "region": rng.choice(REGIONS),
        "product_category": outlier_cat,
        "units_sold": 1,
        "unit_price": outlier_revenue,
        "revenue": outlier_revenue,
    }
    out_idx = len(corrupt)
    corrupt = pd.concat([corrupt, pd.DataFrame([outlier_row])], ignore_index=True)
    impossible_rows.append(
        {
            "index": out_idx,
            "type": "revenue_outlier_50x",
            "injected_values": {
                "revenue": outlier_revenue,
                "category_median_revenue": cat_median,
                "product_category": outlier_cat,
            },
        }
    )

    # 3c. Zero units sold with positive revenue (logical inconsistency)
    logic_cat = rng.choice(PRODUCT_CATEGORIES)
    logic_row = {
        "transaction_id": "CORRUPT_LOG001",
        "date": base_date,
        "region": rng.choice(REGIONS),
        "product_category": logic_cat,
        "units_sold": 0,
        "unit_price": 59.99,
        "revenue": 59.99,  # should be 0 if units_sold == 0
    }
    log_idx = len(corrupt)
    corrupt = pd.concat([corrupt, pd.DataFrame([logic_row])], ignore_index=True)
    impossible_rows.append(
        {
            "index": log_idx,
            "type": "zero_units_positive_revenue",
            "injected_values": {"units_sold": 0, "revenue": 59.99},
        }
    )

    manifest["impossible_rows"] = impossible_rows
    manifest["total_defects"] = len(dupe_map) + len(null_indices) + len(impossible_rows)

    return corrupt, manifest


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("Generating clean dataset …")
    clean = generate_sales_data(n_rows=700, seed=42)
    clean.to_csv("data/sales_clean.csv", index=False)
    print(f"  Saved data/sales_clean.csv  ({len(clean)} rows)")

    print("Injecting corruption …")
    corrupted, mf = inject_corruption(clean, seed=99)
    corrupted.to_csv("data/sales_corrupted.csv", index=False)
    print(f"  Saved data/sales_corrupted.csv  ({len(corrupted)} rows)")

    with open("data/corruption_manifest.json", "w") as f:
        json.dump(mf, f, indent=2, default=str)
    print(
        f"  Saved data/corruption_manifest.json  (total_defects={mf['total_defects']})"
    )
