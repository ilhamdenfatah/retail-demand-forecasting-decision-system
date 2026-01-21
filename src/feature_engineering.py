"""
Feature engineering utilities for panel weekly forecasting.

Design goals:
- Time-safe: only use information available up to time t to predict y(t+h).
- Panel-aware: compute lags/rolling per entity (Store-Dept).
- Reproducible: all parameters live in fe_config.json.
"""

from __future__ import annotations
from typing import Dict, List, Any
import numpy as np
import pandas as pd

# ================= Config helpers =================

def load_fe_config(path: str) -> Dict[str, Any]:
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ================= Core helpers =================

def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def make_calendar_spine(
    df: pd.DataFrame,
    keys: List[str],
    date_col: str,
    freq: str = "W-FRI",
) -> pd.DataFrame:
    df = _ensure_datetime(df, date_col)
    g = df.groupby(keys, dropna=False)[date_col].agg(["min", "max"]).reset_index()
    out = []

    for _, row in g.iterrows():
        if pd.isna(row["min"]) or pd.isna(row["max"]):
            continue
        dates = pd.date_range(start=row["min"], end=row["max"], freq=freq)
        tmp = {k: row[k] for k in keys}
        tmp_df = pd.DataFrame(tmp, index=range(len(dates)))
        tmp_df[date_col] = dates
        out.append(tmp_df)

    return pd.concat(out, ignore_index=True)

def add_datetime_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    d = df[date_col]
    df = df.copy()
    df["year"] = d.dt.year.astype("int16")
    df["month"] = d.dt.month.astype("int8")
    df["weekofyear"] = d.dt.isocalendar().week.astype("int16")
    df["quarter"] = d.dt.quarter.astype("int8")
    df["is_month_start"] = d.dt.is_month_start.astype("int8")
    df["is_month_end"] = d.dt.is_month_end.astype("int8")
    return df

def standardize_is_holiday(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.lower()
    return (
        s.map({"true": 1, "false": 0})
         .fillna(pd.to_numeric(series, errors="coerce"))
         .fillna(0)
         .astype("int8")
    )

def build_target(
    df: pd.DataFrame,
    keys: List[str],
    sales_col: str,
    horizon: int,
    target_col: str,
) -> pd.DataFrame:
    df = df.copy()
    df[target_col] = df.groupby(keys, dropna=False)[sales_col].shift(-horizon)
    return df

# ================= Feature blocks =================

def add_missing_flags(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[f"{c}_missing"] = df[c].isna().astype("int8")
    return df

def add_lag_features(
    df: pd.DataFrame,
    keys: List[str],
    sales_col: str,
    lags: List[int],
) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby(keys, dropna=False)[sales_col]
    for k in lags:
        col = f"sales_lag_{k}"
        df[col] = g.shift(k)
        df[f"{col}_missing"] = df[col].isna().astype("int8")
    return df

def add_rolling_features(
    df: pd.DataFrame,
    keys: List[str],
    sales_col: str,
    windows: List[int],
) -> pd.DataFrame:
    df = df.copy()
    past = df.groupby(keys, dropna=False)[sales_col].shift(1)
    df["_past"] = past

    for w in windows:
        grp = df.groupby(keys, dropna=False)["_past"]
        df[f"roll_mean_{w}"] = grp.rolling(w, 1).mean().reset_index(level=keys, drop=True)
        df[f"roll_std_{w}"]  = grp.rolling(w, 2).std().reset_index(level=keys, drop=True)
        df[f"roll_min_{w}"]  = grp.rolling(w, 1).min().reset_index(level=keys, drop=True)
        df[f"roll_max_{w}"]  = grp.rolling(w, 1).max().reset_index(level=keys, drop=True)

        for c in [f"roll_mean_{w}", f"roll_std_{w}", f"roll_min_{w}", f"roll_max_{w}"]:
            df[f"{c}_missing"] = df[c].isna().astype("int8")

    return df.drop(columns="_past")

def add_baselines(df: pd.DataFrame, sales_col: str) -> pd.DataFrame:
    df = df.copy()

    past_store = df.groupby("Store")[sales_col].shift(1)
    df["store_baseline_mean"] = past_store.groupby(df["Store"]).expanding(2).mean().reset_index(0, drop=True)
    df["store_baseline_mean_missing"] = df["store_baseline_mean"].isna().astype("int8")

    store_std = past_store.groupby(df["Store"]).expanding(2).std().reset_index(0, drop=True)
    df["store_cv"] = store_std / df["store_baseline_mean"].replace(0, np.nan)
    df["store_cv_missing"] = df["store_cv"].isna().astype("int8")

    past_dept = df.groupby("Dept")[sales_col].shift(1)
    df["dept_baseline_mean"] = past_dept.groupby(df["Dept"]).expanding(2).mean().reset_index(0, drop=True)
    df["dept_baseline_std"]  = past_dept.groupby(df["Dept"]).expanding(2).std().reset_index(0, drop=True)

    df["dept_baseline_mean_missing"] = df["dept_baseline_mean"].isna().astype("int8")
    df["dept_baseline_std_missing"]  = df["dept_baseline_std"].isna().astype("int8")

    df["dept_cv"] = df["dept_baseline_std"] / df["dept_baseline_mean"].replace(0, np.nan)
    df["dept_cv_missing"] = df["dept_cv"].isna().astype("int8")

    return df

def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    dept_proxy = df["sales_lag_1"]
    store_total = df.groupby(["Store", "Date"])["sales_lag_1"].transform("sum")

    df["share_of_store_demand"] = dept_proxy / store_total.replace(0, np.nan)
    df["share_of_store_demand_missing"] = df["share_of_store_demand"].isna().astype("int8")

    df["demand_vs_store_baseline"] = dept_proxy / df["store_baseline_mean"].replace(0, np.nan)
    df["demand_vs_store_baseline_missing"] = df["demand_vs_store_baseline"].isna().astype("int8")

    df["demand_vs_dept_baseline"] = dept_proxy / df["dept_baseline_mean"].replace(0, np.nan)
    df["demand_vs_dept_baseline_missing"] = df["demand_vs_dept_baseline"].isna().astype("int8")

    df["mom_wow"] = (df["sales_lag_1"] / df["sales_lag_2"].replace(0, np.nan)) - 1
    df["mom_wow_missing"] = df["mom_wow"].isna().astype("int8")

    df["mom_4w"] = (df["sales_lag_1"] / df["sales_lag_4"].replace(0, np.nan)) - 1
    df["mom_4w_missing"] = df["mom_4w"].isna().astype("int8")

    df["dev_from_roll4"] = df["sales_lag_1"] - df["roll_mean_4"]
    df["dev_from_roll8"] = df["sales_lag_1"] - df["roll_mean_8"]

    df["dev_from_roll4_missing"] = df["dev_from_roll4"].isna().astype("int8")
    df["dev_from_roll8_missing"] = df["dev_from_roll8"].isna().astype("int8")

    df["pct_rank_in_store"] = df.groupby(["Store", "Date"])["sales_lag_1"].rank(pct=True)
    df["pct_rank_in_store_missing"] = df["pct_rank_in_store"].isna().astype("int8")

    return df

# ================= MAIN PIPELINE =================

def build_features(raw_df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    keys = cfg["panel_keys"]
    date_col = cfg["time_col"]
    sales_col = cfg["sales_col"]
    dim_cols = cfg.get("dim_cols", [])
    is_holiday_col = cfg.get("is_holiday_col", "IsHoliday")
    horizon = int(cfg.get("horizon", 1))
    target_col = cfg["target_col"]

    freq = cfg.get("freq", "W-FRI")
    lags = cfg.get("lags", [1, 2, 4, 8])
    roll_windows = cfg.get("roll_windows", [4, 8])
    raw_numeric_cols = cfg.get("raw_numeric_cols", [])

    df = _ensure_datetime(raw_df.copy(), date_col)

    spine = make_calendar_spine(df, keys, date_col, freq)
    df = spine.merge(df, on=keys + [date_col], how="left")
    df = df.sort_values(keys + [date_col]).reset_index(drop=True)

    df = add_datetime_features(df, date_col)

    df = add_missing_flags(df, raw_numeric_cols + dim_cols + ["Size", is_holiday_col])

    if is_holiday_col in df.columns:
        df["is_holiday"] = standardize_is_holiday(df[is_holiday_col])
        df["IsHoliday_missing"] = df[is_holiday_col].isna().astype("int8")

    for c in dim_cols:
        if c in df.columns:
            df[c] = df[c].astype("string")

    df = build_target(df, keys, sales_col, horizon, target_col)
    df = add_lag_features(df, keys, sales_col, lags)
    df = add_rolling_features(df, keys, sales_col, roll_windows)
    df = add_baselines(df, sales_col)
    df = add_business_features(df)

    return df
