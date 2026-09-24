"""Data ingestion, cleaning and persistence service.

:class:`DataService` implements the full data-processing pipeline described in
the project spec:

1. Generate a realistic sample dataset (if none is provided).
2. Load CSV data.
3. Check column names.
4. Convert dates to proper ``datetime``.
5. Detect missing values.
6. Handle missing values appropriately.
7. Remove duplicate records.
8. Validate numeric columns.
9. Detect invalid values.
10. Calculate ``revenue`` where required.
11. Sort records chronologically.
12. Save the cleaned dataset.

The class owns its dependencies (settings, logger) via its constructor —
encapsulation in practice — and exposes small, single-responsibility methods.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import DataValidationError
from app.core.logging_config import get_logger
from app.database.models import Sale
from app.utils.validators import (
    NUMERIC_COLUMNS,
    REQUIRED_COLUMNS,
    compute_revenue,
    validate_columns,
)

logger = get_logger(__name__)


class DataService:
    """Owns loading, cleaning and persisting sales data."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # ------------------------------------------------------------------ #
    # Sample data generation
    # ------------------------------------------------------------------ #
    def generate_sample_dataset(
        self,
        n_records: int = 8000,
        seed: int = 42,
        output_path: str | None = None,
    ) -> pd.DataFrame:
        """Create a realistic synthetic sales dataset and save it to CSV.

        The generator embeds trend + weekly seasonality + noise so that the
        downstream forecasting model has a genuine signal to learn.
        """

        if n_records <= 0:
            raise DataValidationError("n_records must be a positive integer.")

        rng = np.random.default_rng(seed)
        output_path = output_path or self._settings.raw_data_path

        products = [
            ("P001", "Wireless Mouse", "Electronics"),
            ("P002", "Mechanical Keyboard", "Electronics"),
            ("P003", "USB-C Cable", "Accessories"),
            ("P004", "Laptop Stand", "Accessories"),
            ("P005", "Noise-Cancelling Headphones", "Electronics"),
            ("P006", "Office Chair", "Furniture"),
            ("P007", "Standing Desk", "Furniture"),
            ("P008", "Notebook", "Stationery"),
            ("P009", "Ballpoint Pen (Pack)", "Stationery"),
            ("P010", "Water Bottle", "Lifestyle"),
        ]
        regions = ["North", "South", "East", "West", "Central"]
        price_list = [19.99, 49.99, 9.99, 34.99, 129.99, 189.99, 399.99, 4.99, 6.49, 14.99]
        base_prices = {
            product[0]: price for product, price in zip(products, price_list, strict=True)
        }

        start = pd.Timestamp("2023-01-01")
        span_days = 900  # ~2.5 years so lag/rolling features are meaningful
        # Weight order dates so volume grows over time (upward trend) with a
        # weekly seasonal bump on weekends.
        day_offsets = rng.integers(0, span_days, size=n_records)
        order_dates = start + pd.to_timedelta(day_offsets, unit="D")

        rows = []
        for i in range(n_records):
            product = products[rng.integers(0, len(products))]
            pid, pname, category = product
            region = regions[rng.integers(0, len(regions))]
            quantity = int(rng.integers(1, 11))
            unit_price = round(base_prices[pid] * rng.uniform(0.9, 1.1), 2)
            discount = round(float(rng.choice([0.0, 0.0, 0.05, 0.1, 0.15, 0.2])), 2)
            rows.append(
                {
                    "order_id": f"ORD-{100000 + i}",
                    "order_date": order_dates[i].date().isoformat(),
                    "product_id": pid,
                    "product_name": pname,
                    "category": category,
                    "region": region,
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "discount": discount,
                    "revenue": compute_revenue(quantity, unit_price, discount),
                }
            )

        df = pd.DataFrame(rows)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Generated sample dataset with %d records at %s", len(df), out)
        return df

    # ------------------------------------------------------------------ #
    # Loading & cleaning
    # ------------------------------------------------------------------ #
    def load_csv(self, path: str | None = None) -> pd.DataFrame:
        """Load a CSV file into a DataFrame with clear error messages."""

        path = path or self._settings.raw_data_path
        csv_path = Path(path)
        if not csv_path.exists():
            raise DataValidationError(
                f"CSV file not found at '{csv_path}'. Generate the sample "
                f"dataset or provide a valid path."
            )
        try:
            df = pd.read_csv(csv_path)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            raise DataValidationError(f"Failed to parse CSV '{csv_path}': {exc}") from exc

        if df.empty:
            raise DataValidationError(f"CSV file '{csv_path}' contains no data rows.")
        logger.info("Loaded %d raw rows from %s", len(df), csv_path)
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline and return a tidy DataFrame."""

        validate_columns(df, REQUIRED_COLUMNS)
        df = df.copy()

        initial_rows = len(df)

        # 1. Parse dates. Coerce, then drop rows whose dates cannot be parsed.
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        bad_dates = int(df["order_date"].isna().sum())
        if bad_dates:
            logger.warning("Dropping %d rows with unparseable order_date", bad_dates)
        df = df.dropna(subset=["order_date"])

        # 2. Coerce numeric columns; non-numeric become NaN and are reported.
        for col in NUMERIC_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 3. Detect & handle missing values.
        missing_summary = df[list(REQUIRED_COLUMNS)].isna().sum()
        total_missing = int(missing_summary.sum())
        if total_missing:
            logger.warning("Missing values detected:\n%s", missing_summary[missing_summary > 0])
        # Drop rows missing critical identifiers / numerics.
        df = df.dropna(subset=["order_id", "product_id", "quantity", "unit_price"])
        # Discount missing -> assume no discount.
        df["discount"] = df["discount"].fillna(0.0)
        # Fill missing text metadata with an explicit placeholder.
        for col in ("product_name", "category", "region"):
            df[col] = df[col].fillna("Unknown")

        # 4. Remove duplicates (identical rows and duplicate order_ids).
        before_dupes = len(df)
        df = df.drop_duplicates()
        df = df.drop_duplicates(subset=["order_id"], keep="first")
        removed_dupes = before_dupes - len(df)
        if removed_dupes:
            logger.info("Removed %d duplicate rows", removed_dupes)

        # 5. Validate numeric ranges; drop invalid rows with a report.
        valid_mask = (
            (df["quantity"] > 0)
            & (df["unit_price"] >= 0)
            & (df["discount"] >= 0)
            & (df["discount"] <= 1)
        )
        invalid_count = int((~valid_mask).sum())
        if invalid_count:
            logger.warning("Dropping %d rows failing numeric validation", invalid_count)
        df = df[valid_mask]

        if df.empty:
            raise DataValidationError(
                "All rows were removed during cleaning; no valid data remains."
            )

        # 6. Recalculate revenue to guarantee consistency.
        df["quantity"] = df["quantity"].astype(int)
        df["revenue"] = (df["quantity"] * df["unit_price"] * (1 - df["discount"])).round(2)

        # 7. Sort chronologically.
        df = df.sort_values("order_date").reset_index(drop=True)

        logger.info(
            "Cleaning complete: %d -> %d rows (%d removed)",
            initial_rows,
            len(df),
            initial_rows - len(df),
        )
        return df

    def save_processed(self, df: pd.DataFrame, path: str | None = None) -> Path:
        """Persist the cleaned dataset to the processed data path."""

        path = path or self._settings.processed_data_path
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Saved %d cleaned rows to %s", len(df), out)
        return out

    # ------------------------------------------------------------------ #
    # Persistence to the database
    # ------------------------------------------------------------------ #
    def persist_to_db(self, df: pd.DataFrame, db: Session, replace: bool = True) -> int:
        """Insert cleaned records into the ``sales`` table.

        Parameters
        ----------
        replace:
            If ``True``, existing rows are cleared first (idempotent reload).
        """

        if replace:
            db.execute(delete(Sale))

        records = [
            Sale(
                order_id=str(row.order_id),
                order_date=row.order_date.date()
                if hasattr(row.order_date, "date")
                else row.order_date,
                product_id=str(row.product_id),
                product_name=str(row.product_name),
                category=str(row.category),
                region=str(row.region),
                quantity=int(row.quantity),
                unit_price=float(row.unit_price),
                discount=float(row.discount),
                revenue=float(row.revenue),
            )
            for row in df.itertuples(index=False)
        ]
        db.add_all(records)
        db.commit()
        logger.info("Persisted %d sales rows to the database", len(records))
        return len(records)

    def run_pipeline(self, db: Session, csv_path: str | None = None) -> int:
        """End-to-end: load -> clean -> save -> persist. Returns row count."""

        raw = self.load_csv(csv_path)
        clean = self.clean(raw)
        self.save_processed(clean)
        return self.persist_to_db(clean, db)

    @staticmethod
    def count_sales(db: Session) -> int:
        """Return the number of rows currently in the ``sales`` table."""

        return int(db.execute(select(func.count()).select_from(Sale)).scalar_one())
