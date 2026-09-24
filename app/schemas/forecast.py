"""Pydantic schemas for forecasting, analytics and model endpoints."""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Forecasting
# --------------------------------------------------------------------------- #
class ForecastRequest(BaseModel):
    """Payload for ``POST /forecast``.

    ``days`` is bounded here (1..365) so obviously-invalid horizons are rejected
    with HTTP 422 before touching the model. The upper bound is also re-checked
    against configuration in the service layer.
    """

    model_config = ConfigDict(extra="forbid")

    days: Annotated[
        int,
        Field(gt=0, le=365, description="Number of future days to forecast", examples=[7]),
    ]


class ForecastPoint(BaseModel):
    """A single predicted day."""

    date: date
    predicted_sales: float


class ForecastResponse(BaseModel):
    """Response returned by ``POST /forecast``."""

    model: str
    forecast: list[ForecastPoint]


class ForecastHistoryItem(BaseModel):
    """A previously stored forecast row."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    forecast_date: date
    predicted_sales: float
    model_name: str
    created_at: datetime


# --------------------------------------------------------------------------- #
# Model training / metrics
# --------------------------------------------------------------------------- #
class TrainRequest(BaseModel):
    """Optional configuration for ``POST /model/train``."""

    # ``protected_namespaces=()`` disables Pydantic's warning about fields
    # beginning with ``model_`` — here ``model_type`` is a deliberate,
    # domain-meaningful name, not a clash with Pydantic internals.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_type: Annotated[
        str,
        Field(
            default="auto",
            description="'linear', 'random_forest', or 'auto' to pick the best",
            examples=["auto"],
        ),
    ]
    test_size: Annotated[
        float,
        Field(
            default=0.2,
            gt=0,
            lt=1,
            description="Fraction reserved for the chronological test split",
        ),
    ]


class ModelMetricRead(BaseModel):
    """A model-metrics row."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    model_name: str
    rmse: float
    mae: float
    r2_score: float
    trained_at: datetime


class TrainResponse(BaseModel):
    """Response returned by ``POST /model/train``."""

    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    rmse: float
    mae: float
    r2_score: float
    training_rows: int
    test_rows: int


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #
class AnalyticsSummary(BaseModel):
    """Headline KPIs for ``GET /analytics/summary``."""

    total_revenue: float
    total_orders: int
    total_quantity: int
    average_order_value: float


class MonthlySales(BaseModel):
    """One row of monthly aggregated sales."""

    month: str = Field(description="Year-month, e.g. '2025-01'")
    total_revenue: float
    total_quantity: int
    order_count: int


class ProductSales(BaseModel):
    """Product-wise aggregated sales."""

    product_id: str
    product_name: str
    total_revenue: float
    total_quantity: int


class RegionSales(BaseModel):
    """Region-wise aggregated sales."""

    region: str
    total_revenue: float
    total_quantity: int
    order_count: int


class HealthResponse(BaseModel):
    """Response for ``GET /health``."""

    status: str = "healthy"
