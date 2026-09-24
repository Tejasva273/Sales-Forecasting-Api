"""Reusable, framework-agnostic validation helpers.

These functions operate on plain data / DataFrames and raise
:class:`~app.core.exceptions.DataValidationError` with actionable messages.
They are used by the data ingestion pipeline (where Pydantic is not involved)
and are independently unit-tested.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from app.core.exceptions import DataValidationError

# Canonical schema expected in raw sales CSVs.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "order_id",
    "order_date",
    "product_id",
    "product_name",
    "category",
    "region",
    "quantity",
    "unit_price",
    "discount",
)

NUMERIC_COLUMNS: tuple[str, ...] = ("quantity", "unit_price", "discount")


def validate_columns(df: pd.DataFrame, required: tuple[str, ...] = REQUIRED_COLUMNS) -> None:
    """Ensure every required column is present.

    Raises
    ------
    DataValidationError
        Listing exactly which columns are missing (never silently ignored).
    """

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"CSV is missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(required)}."
        )


def parse_date(value: object) -> date:
    """Parse a value into a :class:`datetime.date`.

    Accepts ``date``/``datetime`` objects and ISO-like strings. Raises a clear
    error rather than returning ``NaT`` so bad rows are surfaced.
    """

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        parsed = pd.to_datetime(value, errors="raise")
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"Invalid date value: {value!r} ({exc})") from exc
    return parsed.date()


def validate_numeric_ranges(quantity: float, unit_price: float, discount: float) -> None:
    """Validate the core numeric business rules for a single record."""

    errors: list[str] = []
    if quantity <= 0:
        errors.append(f"quantity must be > 0 (got {quantity})")
    if unit_price < 0:
        errors.append(f"unit_price must be >= 0 (got {unit_price})")
    if not (0 <= discount <= 1):
        errors.append(f"discount must be between 0 and 1 (got {discount})")
    if errors:
        raise DataValidationError("; ".join(errors))


def compute_revenue(quantity: float, unit_price: float, discount: float) -> float:
    """Return ``quantity * unit_price * (1 - discount)`` rounded to 2 dp."""

    return round(quantity * unit_price * (1 - discount), 2)
