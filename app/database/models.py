"""SQLAlchemy ORM models.

Three tables model the domain:

* :class:`Sale`         - one row per historical order line.
* :class:`Forecast`     - one row per predicted future day.
* :class:`ModelMetric`  - one row per training run's evaluation metrics.

Primary keys, indexes and constraints are declared explicitly so the schema is
self-documenting and enforced at the database level (not just in Python).
"""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database.database import Base


class Sale(Base):
    """A single historical sales record (order line)."""

    __tablename__ = "sales"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    order_date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    product_id: Mapped[str] = mapped_column(String(50), index=True, nullable=False)
    product_name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    region: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_price: Mapped[float] = mapped_column(Float, nullable=False)
    discount: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    revenue: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        CheckConstraint("quantity > 0", name="ck_sales_quantity_positive"),
        CheckConstraint("unit_price >= 0", name="ck_sales_unit_price_non_negative"),
        CheckConstraint("discount >= 0 AND discount <= 1", name="ck_sales_discount_range"),
        CheckConstraint("revenue >= 0", name="ck_sales_revenue_non_negative"),
        Index("ix_sales_date_category", "order_date", "category"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Sale id={self.id} order_id={self.order_id!r} "
            f"date={self.order_date} revenue={self.revenue:.2f}>"
        )


class Forecast(Base):
    """A predicted sales value for a single future date."""

    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    forecast_date: Mapped[date] = mapped_column(Date, index=True, nullable=False)
    predicted_sales: Mapped[float] = mapped_column(Float, nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    __table_args__ = (CheckConstraint("predicted_sales >= 0", name="ck_forecast_non_negative"),)

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<Forecast id={self.id} date={self.forecast_date} "
            f"predicted={self.predicted_sales:.2f} model={self.model_name!r}>"
        )


class ModelMetric(Base):
    """Evaluation metrics recorded after each model training run."""

    __tablename__ = "model_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    rmse: Mapped[float] = mapped_column(Float, nullable=False)
    mae: Mapped[float] = mapped_column(Float, nullable=False)
    r2_score: Mapped[float] = mapped_column(Float, nullable=False)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<ModelMetric id={self.id} model={self.model_name!r} "
            f"rmse={self.rmse:.3f} r2={self.r2_score:.3f}>"
        )
