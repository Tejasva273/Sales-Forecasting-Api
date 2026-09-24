"""Sales CRUD endpoints."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, Path, Query, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.database.database import get_db
from app.schemas.sales import (
    MessageResponse,
    SaleCreate,
    SaleFilter,
    SalePage,
    SaleRead,
)
from app.services.sales_service import SalesService

router = APIRouter(prefix="/sales", tags=["sales"])


@router.get(
    "",
    response_model=SalePage,
    summary="List sales (paginated)",
    description="Return sales records with pagination and optional filters "
    "(date range, category, region), most recent first.",
)
def list_sales(
    db: Session = Depends(get_db),
    page: int = Query(1, ge=1, description="1-based page number"),
    page_size: int = Query(None, ge=1, le=1000, description="Items per page"),
    start_date: date | None = Query(None, description="Filter: order_date >= start_date"),
    end_date: date | None = Query(None, description="Filter: order_date <= end_date"),
    category: str | None = Query(None, description="Filter by category"),
    region: str | None = Query(None, description="Filter by region"),
) -> SalePage:
    settings = get_settings()
    size = page_size or settings.default_page_size
    filters = SaleFilter(start_date=start_date, end_date=end_date, category=category, region=region)
    service = SalesService(db)
    items, total = service.list_sales(page=page, page_size=size, filters=filters)
    return SalePage(
        total=total,
        page=page,
        page_size=size,
        items=[SaleRead.model_validate(item) for item in items],
    )


@router.get(
    "/{order_id}",
    response_model=SaleRead,
    summary="Get one sale",
    description="Return a single sales record by its unique order_id.",
    responses={404: {"description": "Order not found"}},
)
def get_sale(
    order_id: str = Path(..., description="Unique order identifier"),
    db: Session = Depends(get_db),
) -> SaleRead:
    service = SalesService(db)
    return SaleRead.model_validate(service.get_by_order_id(order_id))


@router.post(
    "",
    response_model=SaleRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a sale",
    description="Create a new sales record. Revenue is computed server-side as "
    "`quantity * unit_price * (1 - discount)`.",
    responses={
        409: {"description": "Duplicate order_id"},
        422: {"description": "Validation error"},
    },
)
def create_sale(payload: SaleCreate, db: Session = Depends(get_db)) -> SaleRead:
    service = SalesService(db)
    return SaleRead.model_validate(service.create(payload))


@router.delete(
    "/{order_id}",
    response_model=MessageResponse,
    summary="Delete a sale",
    description="Delete a sales record by order_id.",
    responses={404: {"description": "Order not found"}},
)
def delete_sale(
    order_id: str = Path(..., description="Unique order identifier"),
    db: Session = Depends(get_db),
) -> MessageResponse:
    service = SalesService(db)
    service.delete(order_id)
    return MessageResponse(message=f"Sale '{order_id}' deleted successfully.")
