"""Model loading and recursive multi-step forecasting.

Forecasting future daily revenue is inherently recursive: to predict day *t*
we need its lag/rolling features, which depend on days *t-1, t-7, ...* — some
of which are themselves predictions. :class:`Forecaster` therefore appends each
prediction back onto the working series before computing the next day's
features, exactly mirroring how the features were built during training.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.core.config import Settings, get_settings
from app.core.exceptions import ModelNotTrainedError
from app.core.logging_config import get_logger
from app.ml.feature_engineering import (
    FEATURE_COLUMNS,
    LAG_DAYS,
    ROLLING_WINDOWS,
    TARGET_COLUMN,
)

logger = get_logger(__name__)


class Forecaster:
    """Loads a trained artifact and produces future daily forecasts."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._artifact: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    def load(self) -> dict[str, Any]:
        """Load the persisted model artifact, raising if none exists."""

        if self._artifact is not None:
            return self._artifact
        path = Path(self._settings.model_path)
        if not path.exists():
            raise ModelNotTrainedError(
                "No trained model found. Call POST /model/train before forecasting."
            )
        self._artifact = joblib.load(path)
        logger.info("Loaded model artifact '%s' from %s", self._artifact.get("model_name"), path)
        return self._artifact

    @property
    def model_name(self) -> str:
        return str(self.load().get("model_name", "unknown"))

    # ------------------------------------------------------------------ #
    @staticmethod
    def _feature_row(series: pd.Series, target_date: pd.Timestamp) -> list[float]:
        """Build one feature vector for ``target_date`` from a daily series.

        ``series`` is indexed by date and holds historical + already-predicted
        values strictly *before* ``target_date``.
        """

        calendar = {
            "day": target_date.day,
            "week": int(target_date.isocalendar().week),
            "month": target_date.month,
            "quarter": (target_date.month - 1) // 3 + 1,
            "year": target_date.year,
            "day_of_week": target_date.dayofweek,
        }
        lags = {f"sales_lag_{k}": float(series.iloc[-k]) for k in LAG_DAYS}
        rolling = {f"rolling_mean_{w}": float(series.iloc[-w:].mean()) for w in ROLLING_WINDOWS}
        combined = {**calendar, **lags, **rolling}
        return [combined[col] for col in FEATURE_COLUMNS]

    def forecast(self, days: int) -> list[dict[str, Any]]:
        """Return a list of ``{date, predicted_sales}`` for the next ``days``."""

        artifact = self.load()
        model = artifact["model"]
        daily: pd.DataFrame = artifact["last_daily_series"]

        # Work on a date-indexed sales series we can extend in place.
        series = daily.set_index("date")[TARGET_COLUMN].copy()
        series.index = pd.to_datetime(series.index)

        last_date = series.index.max()
        predictions: list[dict[str, Any]] = []
        for step in range(1, days + 1):
            target_date = last_date + timedelta(days=step)
            features = self._feature_row(series, target_date)
            frame = pd.DataFrame([features], columns=list(FEATURE_COLUMNS))
            raw_pred = float(model.predict(frame)[0])
            predicted = max(0.0, round(raw_pred, 2))  # revenue cannot be negative
            predictions.append({"date": target_date.date(), "predicted_sales": predicted})
            # Feed the prediction back for subsequent steps (recursive forecast).
            series.loc[target_date] = predicted

        logger.info("Generated %d-day forecast using %s", days, artifact.get("model_name"))
        return predictions

    @staticmethod
    def next_dates(start: date, days: int) -> list[date]:
        """Utility: the list of future dates a forecast will cover."""

        return [start + timedelta(days=i) for i in range(1, days + 1)]
