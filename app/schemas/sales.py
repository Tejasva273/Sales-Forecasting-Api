"""Pydantic schemas for the sales domain.

These models define the API contract and enforce input validation at the edge
of the system (returning HTTP 422 automatically on bad payloads). Business
rules encoded here mirror the database ``CheckConstraint``s so bad data is
rejected before it ever reaches the database.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class SaleBase(BaseModel):
    """Fields shared by create/read schemas."""

    order_id: Annotated[str, Field(min_length=1, max_length=50, examples=["ORD-100001"])]
    order_date: date
    product_id: Annotated[str, Field(min_length=1, max_length=50, examples=["P0007"])]
    product_name: Annotated[str, Field(min_length=1, max_length=255, examples=["Wireless Mouse"])]
    category: Annotated[str, Field(min_length=1, max_length=100, examples=["Electronics"])]
    region: Annotated[str, Field(min_length=1, max_length=100, examples=["North"])]
    quantity: Annotated[int, Field(gt=0, description="Units ordered; must be > 0", examples=[3])]
    unit_price: Annotated[
        float, Field(ge=0, description="Price per unit; must be >= 0", examples=[24.99])
    ]
    discount: Annotated[
        float, Field(ge=0, le=1, description="Fractional discount 0..1", examples=[0.1])
    ]


class SaleCreate(SaleBase):
    """Payload for ``POST /sales``.

    ``revenue`` is intentionally *not* accepted from the client — it is derived
    server-side to guarantee ``revenue = quantity * unit_price * (1 - discount)``
    and prevent tampering / inconsistency.
    """

    model_config = ConfigDict(extra="forbid")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def revenue(self) -> float:
        """Server-computed revenue."""

        return round(self.quantity * self.unit_price * (1 - self.discount), 2)


class SaleRead(SaleBase):
    """Representation returned by the API for a stored sale."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    revenue: float


class SalePage(BaseModel):
    """Paginated collection of sales."""

    total: int = Field(description="Total number of matching records")
    page: int = Field(description="Current 1-based page number")
    page_size: int = Field(description="Number of items per page")
    items: list[SaleRead]


class MessageResponse(BaseModel):
    """Generic success/acknowledgement response."""

    message: str


class SaleFilter(BaseModel):
    """Optional filters for listing sales (validated as a group)."""

    start_date: date | None = None
    end_date: date | None = None
    category: str | None = None
    region: str | None = None

    @model_validator(mode="after")
    def check_date_range(self) -> SaleFilter:
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        return self
