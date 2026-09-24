"""Time-series feature engineering.

Given a daily sales series (``date``, ``sales``) this module builds calendar,
lag and rolling-window features.

Avoiding data leakage
---------------------
Every predictive feature is derived **only from past observations**:

* Lag features use ``shift(k)`` (strictly the value ``k`` days earlier).
* Rolling means are computed on the lag-1 series
  (``sales.shift(1).rolling(w)``) so the current day's value is *never*
  included in its own feature. This is the crucial detail: a naive
  ``sales.rolling(w).mean()`` would leak the target into the features.

Rows whose lag/rolling features cannot be fully computed (the initial warm-up
window) are dropped, so the model only trains on complete feature vectors.
"""

from __future__ import annotations

import pandas as pd

from app.core.logging_config import get_logger

logger = get_logger(__name__)

LAG_DAYS: tuple[int, ...] = (1, 7, 14, 30)
ROLLING_WINDOWS: tuple[int, ...] = (7, 14, 30)
TARGET_COLUMN = "sales"

# Column order the model consumes (kept stable for train/predict parity).
FEATURE_COLUMNS: tuple[str, ...] = (
    "day",
    "week",
    "month",
    "quarter",
    "year",
    "day_of_week",
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_30",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day/week/month/quarter/year/day_of_week from the ``date`` column."""

    out = df.copy()
    dt = pd.to_datetime(out["date"])
    out["day"] = dt.dt.day
    out["week"] = dt.dt.isocalendar().week.astype(int)
    out["month"] = dt.dt.month
    out["quarter"] = dt.dt.quarter
    out["year"] = dt.dt.year
    out["day_of_week"] = dt.dt.dayofweek
    return out


def add_lag_features(df: pd.DataFrame, target: str = TARGET_COLUMN) -> pd.DataFrame:
    """Add ``sales_lag_{k}`` columns using strictly past values."""

    out = df.copy()
    for k in LAG_DAYS:
        out[f"sales_lag_{k}"] = out[target].shift(k)
    return out


def add_rolling_features(df: pd.DataFrame, target: str = TARGET_COLUMN) -> pd.DataFrame:
    """Add ``rolling_mean_{w}`` computed on the *lag-1* series (no leakage)."""

    out = df.copy()
    shifted = out[target].shift(1)  # exclude current day
    for w in ROLLING_WINDOWS:
        out[f"rolling_mean_{w}"] = shifted.rolling(window=w, min_periods=w).mean()
    return out


def build_features(daily: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """Full feature pipeline for a daily series.

    Returns a DataFrame containing ``date``, all :data:`FEATURE_COLUMNS` and the
    ``sales`` target. When ``dropna`` is True the warm-up rows with incomplete
    features are removed.
    """

    df = add_calendar_features(daily)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    if dropna:
        before = len(df)
        df = df.dropna(subset=list(FEATURE_COLUMNS)).reset_index(drop=True)
        logger.info("Feature engineering dropped %d warm-up rows", before - len(df))
    return df
