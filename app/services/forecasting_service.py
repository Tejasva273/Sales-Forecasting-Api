"""Forecasting orchestration service.

:class:`ForecastingService` is the business-logic layer between the API routes
and the ML modules. It:

* loads sales data from the database into a DataFrame,
* delegates training to :class:`~app.ml.train.ModelTrainer`,
* persists model metrics to ``model_metrics``,
* delegates prediction to :class:`~app.ml.predict.Forecaster`,
* persists forecast points to ``forecasts`` and reads history back.

Keeping this orchestration out of the routes preserves separation of concerns:
routes stay thin, and this logic is reusable + unit-testable without HTTP.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    InsufficientDataError,
    InvalidForecastPeriodError,
)
from app.core.logging_config import get_logger
from app.database.models import Forecast, ModelMetric, Sale
from app.ml.predict import Forecaster
from app.ml.train import ModelTrainer

logger = get_logger(__name__)


class ForecastingService:
    """Coordinates training, evaluation, persistence and prediction."""

    def __init__(self, db: Session, settings: Settings | None = None) -> None:
        self._db = db
        self._settings = settings or get_settings()
        self._trainer = ModelTrainer(self._settings)
        self._forecaster = Forecaster(self._settings)

    # ------------------------------------------------------------------ #
    def _load_sales_frame(self) -> pd.DataFrame:
        """Read every sale into a DataFrame ordered by date."""

        rows = self._db.execute(
            select(Sale.order_date, Sale.revenue).order_by(Sale.order_date)
        ).all()
        if not rows:
            raise InsufficientDataError("No sales data available to train on. Load data first.")
        return pd.DataFrame(rows, columns=["order_date", "revenue"])

    # ------------------------------------------------------------------ #
    def train_model(self, model_type: str = "auto", test_size: float = 0.2) -> dict[str, Any]:
        """Train, evaluate, persist metrics, and return a summary dict."""

        sales_df = self._load_sales_frame()
        result, _artifact = self._trainer.train(
            sales_df, model_type=model_type, test_size=test_size
        )

        metric = ModelMetric(
            model_name=result.model_name,
            rmse=result.rmse,
            mae=result.mae,
            r2_score=result.r2_score,
        )
        self._db.add(metric)
        self._db.commit()
        logger.info("Stored metrics for model %s", result.model_name)

        return {
            "model_name": result.model_name,
            "rmse": result.rmse,
            "mae": result.mae,
            "r2_score": result.r2_score,
            "training_rows": result.training_rows,
            "test_rows": result.test_rows,
        }

    def get_metrics(self, limit: int = 20) -> list[ModelMetric]:
        """Return recent model-metric rows (most recent first)."""

        return list(
            self._db.execute(
                select(ModelMetric).order_by(ModelMetric.trained_at.desc()).limit(limit)
            ).scalars()
        )

    # ------------------------------------------------------------------ #
    def forecast(self, days: int) -> dict[str, Any]:
        """Generate and persist a ``days``-day forecast.

        Validates the horizon against configuration (defence in depth on top of
        the Pydantic bound) and stores each predicted point in ``forecasts``.
        """

        if days <= 0:
            raise InvalidForecastPeriodError("`days` must be a positive integer.")
        if days > self._settings.max_forecast_days:
            raise InvalidForecastPeriodError(
                f"`days` ({days}) exceeds the maximum allowed ({self._settings.max_forecast_days})."
            )

        points = self._forecaster.forecast(days)
        model_name = self._forecaster.model_name

        forecast_rows = [
            Forecast(
                forecast_date=p["date"],
                predicted_sales=p["predicted_sales"],
                model_name=model_name,
            )
            for p in points
        ]
        self._db.add_all(forecast_rows)
        self._db.commit()
        logger.info("Persisted %d forecast rows (model=%s)", len(forecast_rows), model_name)

        return {"model": model_name, "forecast": points}

    def history(self, limit: int = 100) -> list[Forecast]:
        """Return previously stored forecast rows (most recent first)."""

        return list(
            self._db.execute(
                select(Forecast).order_by(Forecast.created_at.desc()).limit(limit)
            ).scalars()
        )
