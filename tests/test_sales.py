"""Tests for sales CRUD endpoints, validation and the data pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from app.core.exceptions import DataValidationError
from app.services.data_service import DataService
from app.utils.validators import compute_revenue, validate_columns


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def test_compute_revenue_applies_discount() -> None:
    assert compute_revenue(2, 100.0, 0.1) == 180.0
    assert compute_revenue(1, 50.0, 0.0) == 50.0


def test_validate_columns_reports_missing() -> None:
    df = pd.DataFrame({"order_id": [1], "order_date": ["2023-01-01"]})
    with pytest.raises(DataValidationError) as exc:
        validate_columns(df)
    assert "missing required column" in str(exc.value).lower()


# --------------------------------------------------------------------------- #
# Data cleaning pipeline
# --------------------------------------------------------------------------- #
def test_clean_removes_duplicates_and_recomputes_revenue(sample_sales_df: pd.DataFrame) -> None:
    dirty = pd.concat([sample_sales_df, sample_sales_df.iloc[[0]]], ignore_index=True)
    cleaned = DataService().clean(dirty)
    # Duplicate order_id collapsed.
    assert cleaned["order_id"].is_unique
    # Revenue recomputed consistently.
    expected = (cleaned["quantity"] * cleaned["unit_price"] * (1 - cleaned["discount"])).round(2)
    assert (cleaned["revenue"] == expected).all()
    # Sorted chronologically.
    assert cleaned["order_date"].is_monotonic_increasing


def test_clean_drops_invalid_rows() -> None:
    df = pd.DataFrame(
        {
            "order_id": ["A", "B", "C"],
            "order_date": ["2023-01-01", "2023-01-02", "not-a-date"],
            "product_id": ["P1", "P2", "P3"],
            "product_name": ["x", "y", "z"],
            "category": ["c", "c", "c"],
            "region": ["N", "N", "N"],
            "quantity": [1, -5, 2],  # B has invalid quantity
            "unit_price": [10.0, 10.0, 10.0],
            "discount": [0.0, 0.0, 0.0],
        }
    )
    cleaned = DataService().clean(df)
    # Row B (bad quantity) and row C (bad date) removed -> only A remains.
    assert list(cleaned["order_id"]) == ["A"]


# --------------------------------------------------------------------------- #
# API: create / read / delete
# --------------------------------------------------------------------------- #
def _valid_payload() -> dict:
    return {
        "order_id": "ORD-TEST-1",
        "order_date": "2025-01-15",
        "product_id": "P001",
        "product_name": "Wireless Mouse",
        "category": "Electronics",
        "region": "North",
        "quantity": 3,
        "unit_price": 20.0,
        "discount": 0.1,
    }


def test_create_and_get_sale(client) -> None:
    resp = client.post("/sales", json=_valid_payload())
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["order_id"] == "ORD-TEST-1"
    assert body["revenue"] == pytest.approx(54.0)  # 3 * 20 * 0.9

    got = client.get("/sales/ORD-TEST-1")
    assert got.status_code == 200
    assert got.json()["order_id"] == "ORD-TEST-1"


def test_duplicate_sale_returns_409(client) -> None:
    client.post("/sales", json=_valid_payload())
    dup = client.post("/sales", json=_valid_payload())
    assert dup.status_code == 409


def test_get_missing_sale_returns_404(client) -> None:
    assert client.get("/sales/DOES-NOT-EXIST").status_code == 404


def test_delete_sale(client) -> None:
    client.post("/sales", json=_valid_payload())
    delete = client.delete("/sales/ORD-TEST-1")
    assert delete.status_code == 200
    assert client.get("/sales/ORD-TEST-1").status_code == 404


@pytest.mark.parametrize(
    "field,value",
    [
        ("quantity", 0),  # must be > 0
        ("quantity", -1),
        ("unit_price", -5.0),  # must be >= 0
        ("discount", 1.5),  # must be <= 1
        ("discount", -0.1),  # must be >= 0
    ],
)
def test_create_sale_validation_errors(client, field: str, value) -> None:
    payload = _valid_payload()
    payload[field] = value
    resp = client.post("/sales", json=payload)
    assert resp.status_code == 422


def test_list_sales_pagination(client, seeded_db) -> None:
    resp = client.get("/sales", params={"page": 1, "page_size": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["page"] == 1
    assert body["page_size"] == 10
    assert len(body["items"]) == 10
    assert body["total"] > 10


def test_health(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}
