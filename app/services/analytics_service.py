"""Sales analytics service.

:class:`AnalyticsService` computes KPIs and aggregations directly against the
database using SQLAlchemy Core expressions (efficient, pushes work to SQL) and
returns plain dictionaries that map cleanly onto the Pydantic response schemas.

All figures are computed from the data — never hard-coded.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.exceptions import InsufficientDataError
from app.core.logging_config import get_logger
from app.database.models import Sale

logger = get_logger(__name__)


class AnalyticsService:
    """Read-only aggregations over the ``sales`` table."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def _ensure_data(self) -> None:
        count = self._db.execute(select(func.count()).select_from(Sale)).scalar_one()
        if not count:
            raise InsufficientDataError(
                "No sales data available. Load data before requesting analytics."
            )

    def summary(self) -> dict[str, Any]:
        """Return total revenue, order count, quantity and average order value."""

        self._ensure_data()
        row = self._db.execute(
            select(
                func.coalesce(func.sum(Sale.revenue), 0.0),
                func.count(Sale.id),
                func.coalesce(func.sum(Sale.quantity), 0),
            )
        ).one()
        total_revenue, total_orders, total_quantity = row
        avg_order_value = (total_revenue / total_orders) if total_orders else 0.0
        logger.info("Computed analytics summary over %d orders", total_orders)
        return {
            "total_revenue": round(float(total_revenue), 2),
            "total_orders": int(total_orders),
            "total_quantity": int(total_quantity),
            "average_order_value": round(float(avg_order_value), 2),
        }

    def monthly_sales(self) -> list[dict[str, Any]]:
        """Aggregate revenue/quantity/order-count by calendar month."""

        self._ensure_data()
        # strftime works for SQLite; for portability we format the date in SQL
        # via func.strftime on SQLite. For MySQL, DATE_FORMAT would be used;
        # to stay backend-agnostic we group in Python instead.
        rows = self._db.execute(select(Sale.order_date, Sale.revenue, Sale.quantity)).all()
        buckets: dict[str, dict[str, float]] = {}
        for order_date, revenue, quantity in rows:
            key = f"{order_date.year:04d}-{order_date.month:02d}"
            b = buckets.setdefault(
                key, {"total_revenue": 0.0, "total_quantity": 0, "order_count": 0}
            )
            b["total_revenue"] += float(revenue)
            b["total_quantity"] += int(quantity)
            b["order_count"] += 1
        result = [
            {
                "month": month,
                "total_revenue": round(vals["total_revenue"], 2),
                "total_quantity": int(vals["total_quantity"]),
                "order_count": int(vals["order_count"]),
            }
            for month, vals in sorted(buckets.items())
        ]
        return result

    def product_sales(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return product-wise revenue and quantity, highest revenue first."""

        self._ensure_data()
        stmt = (
            select(
                Sale.product_id,
                Sale.product_name,
                func.sum(Sale.revenue).label("total_revenue"),
                func.sum(Sale.quantity).label("total_quantity"),
            )
            .group_by(Sale.product_id, Sale.product_name)
            .order_by(func.sum(Sale.revenue).desc())
            .limit(limit)
        )
        return [
            {
                "product_id": pid,
                "product_name": pname,
                "total_revenue": round(float(rev), 2),
                "total_quantity": int(qty),
            }
            for pid, pname, rev, qty in self._db.execute(stmt).all()
        ]

    def region_sales(self) -> list[dict[str, Any]]:
        """Return region-wise revenue, quantity and order count."""

        self._ensure_data()
        stmt = (
            select(
                Sale.region,
                func.sum(Sale.revenue).label("total_revenue"),
                func.sum(Sale.quantity).label("total_quantity"),
                func.count(Sale.id).label("order_count"),
            )
            .group_by(Sale.region)
            .order_by(func.sum(Sale.revenue).desc())
        )
        return [
            {
                "region": region,
                "total_revenue": round(float(rev), 2),
                "total_quantity": int(qty),
                "order_count": int(count),
            }
            for region, rev, qty, count in self._db.execute(stmt).all()
        ]
