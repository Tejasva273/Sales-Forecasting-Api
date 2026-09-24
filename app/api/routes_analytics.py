"""Analytics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.schemas.forecast import (
    AnalyticsSummary,
    MonthlySales,
    ProductSales,
    RegionSales,
)
from app.services.analytics_service import AnalyticsService

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get(
    "/summary",
    response_model=AnalyticsSummary,
    summary="Sales KPI summary",
    description="Total revenue, total orders, total quantity and average order value.",
    responses={422: {"description": "No sales data available"}},
)
def summary(db: Session = Depends(get_db)) -> AnalyticsSummary:
    return AnalyticsSummary(**AnalyticsService(db).summary())


@router.get(
    "/monthly",
    response_model=list[MonthlySales],
    summary="Monthly sales",
    description="Revenue, quantity and order count aggregated by calendar month.",
)
def monthly(db: Session = Depends(get_db)) -> list[MonthlySales]:
    return [MonthlySales(**row) for row in AnalyticsService(db).monthly_sales()]


@router.get(
    "/products",
    response_model=list[ProductSales],
    summary="Product-wise sales",
    description="Revenue and quantity per product, highest revenue first.",
)
def products(db: Session = Depends(get_db)) -> list[ProductSales]:
    return [ProductSales(**row) for row in AnalyticsService(db).product_sales()]


@router.get(
    "/regions",
    response_model=list[RegionSales],
    summary="Region-wise sales",
    description="Revenue, quantity and order count per region.",
)
def regions(db: Session = Depends(get_db)) -> list[RegionSales]:
    return [RegionSales(**row) for row in AnalyticsService(db).region_sales()]
