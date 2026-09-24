"""Tests for feature engineering, model training and forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import InvalidForecastPeriodError, ModelNotTrainedError
from app.ml.feature_engineering import FEATURE_COLUMNS, build_features
from app.ml.preprocessing import aggregate_daily_sales
from app.ml.train import ModelTrainer
from app.services.forecasting_service import ForecastingService


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def test_aggregate_daily_sales_is_continuous(sample_sales_df) -> None:
    daily = aggregate_daily_sales(sample_sales_df)
    # No gaps: consecutive days differ by exactly 1 day.
    diffs = daily["date"].diff().dropna().dt.days.unique()
    assert list(diffs) == [1]


def test_build_features_no_leakage() -> None:
    # A strictly increasing series makes leakage detectable: the rolling mean
    # of PAST values must always be < the current value.
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    daily = pd.DataFrame({"date": dates, "sales": np.arange(1, 121, dtype=float)})
    feats = build_features(daily, dropna=True)
    # rolling_mean_7 uses days t-7..t-1, so it must be strictly less than sales[t].
    assert (feats["rolling_mean_7"] < feats["sales"]).all()
    # lag_1 must equal previous day's sales (which is sales - 1 here).
    assert np.allclose(feats["sales_lag_1"], feats["sales"] - 1)


def test_all_feature_columns_present(sample_sales_df) -> None:
    daily = aggregate_daily_sales(sample_sales_df)
    feats = build_features(daily)
    for col in FEATURE_COLUMNS:
        assert col in feats.columns


def test_chronological_split_preserves_order() -> None:
    df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=100), "sales": range(100)})
    train, test = ModelTrainer.chronological_split(df, test_size=0.2)
    assert len(train) == 80
    assert len(test) == 20
    # Train dates all come before test dates (no shuffling).
    assert train["date"].max() < test["date"].min()


# --------------------------------------------------------------------------- #
# Training + forecasting via the service
# --------------------------------------------------------------------------- #
def test_train_model_returns_metrics(seeded_db) -> None:
    service = ForecastingService(seeded_db)
    result = service.train_model(model_type="linear")
    assert result["model_name"] == "LinearRegression"
    assert result["training_rows"] > 0
    assert result["test_rows"] > 0
    assert "rmse" in result and "mae" in result and "r2_score" in result


def test_forecast_after_training(seeded_db) -> None:
    service = ForecastingService(seeded_db)
    service.train_model(model_type="linear")
    out = service.forecast(7)
    assert out["model"] == "LinearRegression"
    assert len(out["forecast"]) == 7
    # Dates are consecutive and predictions are non-negative.
    dates = [p["date"] for p in out["forecast"]]
    assert dates == sorted(dates)
    assert all(p["predicted_sales"] >= 0 for p in out["forecast"])


def test_forecast_before_training_raises(seeded_db) -> None:
    service = ForecastingService(seeded_db)
    with pytest.raises(ModelNotTrainedError):
        service.forecast(7)


@pytest.mark.parametrize("bad_days", [0, -1, 100_000])
def test_forecast_invalid_period(seeded_db, bad_days: int) -> None:
    service = ForecastingService(seeded_db)
    service.train_model(model_type="linear")
    with pytest.raises(InvalidForecastPeriodError):
        service.forecast(bad_days)


# --------------------------------------------------------------------------- #
# API-level forecast tests
# --------------------------------------------------------------------------- #
def test_forecast_endpoint_validation(client, seeded_db) -> None:
    # Pydantic rejects days <= 0 with 422 before hitting the service.
    assert client.post("/forecast", json={"days": 0}).status_code == 422
    assert client.post("/forecast", json={"days": -3}).status_code == 422
    assert client.post("/forecast", json={"days": 999}).status_code == 422  # > 365 bound


def test_forecast_endpoint_without_model_returns_409(client, seeded_db) -> None:
    resp = client.post("/forecast", json={"days": 5})
    assert resp.status_code == 409


def test_train_and_forecast_endpoints(client, seeded_db) -> None:
    train_resp = client.post("/model/train", json={"model_type": "linear"})
    assert train_resp.status_code == 201, train_resp.text

    forecast_resp = client.post("/forecast", json={"days": 5})
    assert forecast_resp.status_code == 200
    assert len(forecast_resp.json()["forecast"]) == 5

    history = client.get("/forecast/history")
    assert history.status_code == 200
    assert len(history.json()) == 5

    metrics = client.get("/model/metrics")
    assert metrics.status_code == 200
    assert len(metrics.json()) >= 1
