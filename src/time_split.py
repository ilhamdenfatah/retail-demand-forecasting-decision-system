
"""
Time-based splitting for panel weekly forecasting.

We create train/val/test splits purely by time (no random split) to avoid leakage.

Typical usage:
    from src.time_split import compute_cutoffs, apply_time_split, save_cutoffs
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import json
import pandas as pd
import numpy as np


def compute_cutoffs(
    df: pd.DataFrame,
    date_col: str,
    val_weeks: int = 8,
    test_weeks: int = 4,
) -> Dict[str, str]:
    """
    We include the last `test_weeks` timestamps in the test set.
    Therefore test_start = max_date - (test_weeks - 1) * 7d
    and val_start = test_start - val_weeks * 7d
    """
    d = pd.to_datetime(df[date_col], errors="coerce")
    max_date = d.max()
    if pd.isna(max_date):
        raise ValueError(f"Could not compute cutoffs: {date_col} has no valid dates.")

    test_start = max_date - pd.Timedelta(days=7 * (test_weeks - 1))
    val_start = test_start - pd.Timedelta(days=7 * val_weeks)

    return {
        "date_min": str(d.min().date()),
        "date_max": str(max_date.date()),
        "val_start": str(pd.Timestamp(val_start).date()),
        "test_start": str(pd.Timestamp(test_start).date()),
        "val_weeks": int(val_weeks),
        "test_weeks": int(test_weeks),
    }


def apply_time_split(
    df: pd.DataFrame,
    date_col: str,
    val_start: str,
    test_start: str,
    split_col: str = "split",
) -> pd.DataFrame:
    """
    Adds a column {split_col} with values: train / val / test.
    """
    out = df.copy()
    d = pd.to_datetime(out[date_col], errors="coerce")
    vs = pd.to_datetime(val_start)
    ts = pd.to_datetime(test_start)

    out[split_col] = np.where(
        d < vs, "train",
        np.where(d < ts, "val", "test")
    )
    out[split_col] = out[split_col].astype("category")
    return out


def panel_coverage_report(
    df: pd.DataFrame,
    keys: List[str],
    split_col: str = "split",
) -> pd.Series:
    """
    For each panel entity, count how many splits are present (1/2/3).
    Ideal: everyone has 3 splits.
    """
    present = df.groupby(keys, dropna=False)[split_col].nunique()
    return present.value_counts().sort_index()


def save_cutoffs(cutoffs: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cutoffs, f, indent=2)
