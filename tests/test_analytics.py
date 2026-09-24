"""Tests for analytics calculations and endpoints."""

from __future__ import annotations

import pandas as pd
import pytest

from app.core.exceptions import InsufficientDataError
from app.services.analytics_service import AnalyticsService


def test_summary_matches_manual_totals(seeded_db, sample_sales_df: pd.DataFrame) -> None:
    from app.services.data_service import DataService

    cleaned = DataService().clean(sample_sales_df)
    service = AnalyticsService(seeded_db)
    summary = service.summary()

    assert summary["total_orders"] == len(cleaned)
    assert summary["total_quantity"] == int(cleaned["quantity"].sum())
    assert summary["total_revenue"] == pytest.approx(round(cleaned["revenue"].sum(), 2), rel=1e-6)
    expected_aov = round(cleaned["revenue"].sum() / len(cleaned), 2)
    assert summary["average_order_value"] == pytest.approx(expected_aov, rel=1e-6)


def test_summary_raises_without_data(db_session) -> None:
    with pytest.raises(InsufficientDataError):
        AnalyticsService(db_session).summary()


def test_monthly_sales_is_sorted_and_totals_reconcile(seeded_db, sample_sales_df) -> None:
    from app.services.data_service import DataService

    cleaned = DataService().clean(sample_sales_df)
    monthly = AnalyticsService(seeded_db).monthly_sales()
    months = [row["month"] for row in monthly]
    assert months == sorted(months)
    total = round(sum(row["total_revenue"] for row in monthly), 2)
    assert total == pytest.approx(round(cleaned["revenue"].sum(), 2), rel=1e-4)


def test_product_sales_ordered_desc(seeded_db) -> None:
    products = AnalyticsService(seeded_db).product_sales()
    revenues = [p["total_revenue"] for p in products]
    assert revenues == sorted(revenues, reverse=True)


def test_region_sales_endpoint(client, seeded_db) -> None:
    resp = client.get("/analytics/regions")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1
    assert {"region", "total_revenue", "total_quantity", "order_count"} <= data[0].keys()


def test_summary_endpoint(client, seeded_db) -> None:
    resp = client.get("/analytics/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_orders"] > 0
    assert body["average_order_value"] > 0
