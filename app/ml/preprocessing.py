"""Preprocessing utilities for the forecasting pipeline.

The forecasting target is *daily total revenue*. The raw sales table holds one
row per order line, so the first step is always to aggregate to a continuous
daily time series (filling gaps for days with no sales as 0 revenue).
"""

from __future__ import annotations

import pandas as pd

from app.core.exceptions import InsufficientDataError
from app.core.logging_config import get_logger

logger = get_logger(__name__)


def aggregate_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate order-level rows into a continuous daily revenue series.

    Parameters
    ----------
    df:
        Must contain ``order_date`` (datetime-like) and ``revenue`` columns.

    Returns
    -------
    DataFrame with a complete daily ``DatetimeIndex`` and columns
    ``["date", "sales"]`` where ``sales`` is total revenue that day (0 for
    days without orders, so the series is regularly spaced).
    """

    if df.empty:
        raise InsufficientDataError("Cannot build a time series from empty data.")

    work = df.copy()
    work["order_date"] = pd.to_datetime(work["order_date"])
    daily = work.groupby(work["order_date"].dt.normalize())["revenue"].sum().rename("sales")

    # Reindex to a complete daily range so lag/rolling windows are meaningful.
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index, fill_value=0.0)
    daily.index.name = "date"

    result = daily.reset_index()
    logger.info(
        "Built daily series: %d days from %s to %s",
        len(result),
        result["date"].min().date(),
        result["date"].max().date(),
    )
    return result
