"""Sales CRUD service.

:class:`SalesService` encapsulates all database access for individual sales
records so the API routes remain thin. It translates low-level SQLAlchemy
results and integrity errors into domain exceptions defined in
``app.core.exceptions``.
"""

from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.exceptions import DuplicateResourceError, ResourceNotFoundError
from app.core.logging_config import get_logger
from app.database.models import Sale
from app.schemas.sales import SaleCreate, SaleFilter

logger = get_logger(__name__)


class SalesService:
    """CRUD + pagination for the ``sales`` table."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def list_sales(
        self, page: int, page_size: int, filters: SaleFilter | None = None
    ) -> tuple[list[Sale], int]:
        """Return a page of sales plus the total count of matching rows."""

        conditions = []
        if filters:
            if filters.start_date:
                conditions.append(Sale.order_date >= filters.start_date)
            if filters.end_date:
                conditions.append(Sale.order_date <= filters.end_date)
            if filters.category:
                conditions.append(Sale.category == filters.category)
            if filters.region:
                conditions.append(Sale.region == filters.region)

        count_stmt = select(func.count()).select_from(Sale)
        list_stmt = select(Sale).order_by(Sale.order_date.desc(), Sale.id.desc())
        for cond in conditions:
            count_stmt = count_stmt.where(cond)
            list_stmt = list_stmt.where(cond)

        total = int(self._db.execute(count_stmt).scalar_one())
        offset = (page - 1) * page_size
        items = list(self._db.execute(list_stmt.offset(offset).limit(page_size)).scalars())
        return items, total

    def get_by_order_id(self, order_id: str) -> Sale:
        """Return a single sale or raise :class:`ResourceNotFoundError`."""

        sale = self._db.execute(select(Sale).where(Sale.order_id == order_id)).scalar_one_or_none()
        if sale is None:
            raise ResourceNotFoundError(f"No sale found with order_id '{order_id}'.")
        return sale

    def create(self, payload: SaleCreate) -> Sale:
        """Insert a new sale, computing revenue server-side."""

        sale = Sale(
            order_id=payload.order_id,
            order_date=payload.order_date,
            product_id=payload.product_id,
            product_name=payload.product_name,
            category=payload.category,
            region=payload.region,
            quantity=payload.quantity,
            unit_price=payload.unit_price,
            discount=payload.discount,
            revenue=payload.revenue,
        )
        self._db.add(sale)
        try:
            self._db.commit()
        except IntegrityError as exc:
            self._db.rollback()
            raise DuplicateResourceError(
                f"A sale with order_id '{payload.order_id}' already exists."
            ) from exc
        self._db.refresh(sale)
        logger.info("Created sale %s", sale.order_id)
        return sale

    def delete(self, order_id: str) -> None:
        """Delete a sale by ``order_id`` or raise if it does not exist."""

        result = self._db.execute(delete(Sale).where(Sale.order_id == order_id))
        if result.rowcount == 0:
            self._db.rollback()
            raise ResourceNotFoundError(f"No sale found with order_id '{order_id}'.")
        self._db.commit()
        logger.info("Deleted sale %s", order_id)
