"""Exploratory Data Analysis (EDA) chart generation.

:class:`EDAService` produces the charts required by the spec from the cleaned
dataset and saves them under ``CHART_DIR``. All figures are computed from the
data; nothing is hard-coded. Matplotlib is used with the non-interactive
``Agg`` backend so it works headlessly (CI / servers).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from app.core.config import Settings, get_settings  # noqa: E402
from app.core.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)


class EDAService:
    """Generate and persist exploratory charts."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._chart_dir = Path(self._settings.chart_dir)
        self._chart_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig: plt.Figure, name: str) -> Path:
        path = self._chart_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        logger.info("Saved chart %s", path)
        return path

    def generate_all(self, df: pd.DataFrame) -> list[Path]:
        """Generate every required chart and return the saved file paths."""

        df = df.copy()
        df["order_date"] = pd.to_datetime(df["order_date"])
        paths: list[Path] = []

        # 1. Monthly sales trend
        monthly = df.groupby(df["order_date"].dt.to_period("M"))["revenue"].sum()
        fig, ax = plt.subplots(figsize=(10, 4))
        monthly.index = monthly.index.astype(str)
        ax.plot(monthly.index, monthly.values, marker="o")
        ax.set_title("Monthly Sales Trend")
        ax.set_xlabel("Month")
        ax.set_ylabel("Revenue")
        ax.tick_params(axis="x", rotation=45)
        paths.append(self._save(fig, "monthly_sales_trend.png"))

        # 2. Weekly sales trend
        weekly = df.groupby(df["order_date"].dt.to_period("W"))["revenue"].sum()
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(range(len(weekly)), weekly.values)
        ax.set_title("Weekly Sales Trend")
        ax.set_xlabel("Week index")
        ax.set_ylabel("Revenue")
        paths.append(self._save(fig, "weekly_sales_trend.png"))

        # 3. Product-wise sales
        product = df.groupby("product_name")["revenue"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(product.index[::-1], product.values[::-1])
        ax.set_title("Product-wise Sales")
        ax.set_xlabel("Revenue")
        paths.append(self._save(fig, "product_sales.png"))

        # 4. Category-wise sales
        category = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(category.index, category.values)
        ax.set_title("Category-wise Sales")
        ax.set_ylabel("Revenue")
        ax.tick_params(axis="x", rotation=30)
        paths.append(self._save(fig, "category_sales.png"))

        # 5. Region-wise sales
        region = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(region.index, region.values, color="tab:green")
        ax.set_title("Region-wise Sales")
        ax.set_ylabel("Revenue")
        paths.append(self._save(fig, "region_sales.png"))

        # 6. Quantity trend (monthly)
        qty = df.groupby(df["order_date"].dt.to_period("M"))["quantity"].sum()
        qty.index = qty.index.astype(str)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(qty.index, qty.values, marker="s", color="tab:orange")
        ax.set_title("Monthly Quantity Trend")
        ax.set_xlabel("Month")
        ax.set_ylabel("Units sold")
        ax.tick_params(axis="x", rotation=45)
        paths.append(self._save(fig, "quantity_trend.png"))

        # 7. Average order value (monthly)
        aov = df.groupby(df["order_date"].dt.to_period("M"))["revenue"].mean()
        aov.index = aov.index.astype(str)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(aov.index, aov.values, marker="^", color="tab:purple")
        ax.set_title("Average Order Value (Monthly)")
        ax.set_xlabel("Month")
        ax.set_ylabel("Average revenue per order")
        ax.tick_params(axis="x", rotation=45)
        paths.append(self._save(fig, "average_order_value.png"))

        logger.info("Generated %d EDA charts in %s", len(paths), self._chart_dir)
        return paths
