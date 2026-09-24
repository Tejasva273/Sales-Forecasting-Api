"""Model training.

:class:`ModelTrainer` trains a daily-revenue forecasting model. It:

* builds features (leakage-safe, see ``feature_engineering``),
* performs a **chronological** train/test split (no shuffling — this is
  time-series data),
* fits a baseline ``LinearRegression`` and optionally a
  ``RandomForestRegressor``,
* evaluates with RMSE / MAE / R2,
* selects the better model (when ``model_type='auto'``),
* persists the fitted estimator + feature metadata with ``joblib``.

The persisted artifact is a dict so ``predict`` can re-create identical feature
vectors and know which model produced them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.core.config import Settings, get_settings
from app.core.exceptions import InsufficientDataError
from app.core.logging_config import get_logger
from app.ml.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN, build_features
from app.ml.preprocessing import aggregate_daily_sales

logger = get_logger(__name__)

# Need at least this many feature rows to train something meaningful.
MIN_TRAINING_ROWS = 60


@dataclass
class TrainingResult:
    """Outcome of a training run."""

    model_name: str
    rmse: float
    mae: float
    r2_score: float
    training_rows: int
    test_rows: int


class ModelTrainer:
    """Encapsulates the full model-training workflow."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @staticmethod
    def chronological_split(
        df: pd.DataFrame, test_size: float
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split an ordered DataFrame into (train, test) without shuffling."""

        n = len(df)
        split_idx = int(n * (1 - test_size))
        split_idx = max(1, min(split_idx, n - 1))
        return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    @staticmethod
    def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return round(rmse, 4), round(mae, 4), round(r2, 4)

    def train(
        self,
        sales_df: pd.DataFrame,
        model_type: str = "auto",
        test_size: float = 0.2,
    ) -> tuple[TrainingResult, dict[str, Any]]:
        """Train and evaluate a model, returning (result, saved_artifact)."""

        daily = aggregate_daily_sales(sales_df)
        features = build_features(daily, dropna=True)

        if len(features) < MIN_TRAINING_ROWS:
            raise InsufficientDataError(
                f"Only {len(features)} usable rows after feature engineering; "
                f"at least {MIN_TRAINING_ROWS} are required to train. Load more data."
            )

        train_df, test_df = self.chronological_split(features, test_size)
        x_train = train_df[list(FEATURE_COLUMNS)].to_numpy()
        y_train = train_df[TARGET_COLUMN].to_numpy()
        x_test = test_df[list(FEATURE_COLUMNS)].to_numpy()
        y_test = test_df[TARGET_COLUMN].to_numpy()

        candidates: dict[str, Any] = {}
        if model_type in ("linear", "auto"):
            candidates["LinearRegression"] = LinearRegression()
        if model_type in ("random_forest", "auto"):
            candidates["RandomForest"] = RandomForestRegressor(
                n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
            )
        if not candidates:
            raise InsufficientDataError(
                f"Unknown model_type '{model_type}'. Use 'linear', 'random_forest' or 'auto'."
            )

        best_name = ""
        best_metrics: tuple[float, float, float] | None = None
        best_estimator: Any = None
        for name, estimator in candidates.items():
            estimator.fit(x_train, y_train)
            preds = estimator.predict(x_test)
            rmse, mae, r2 = self._evaluate(y_test, preds)
            logger.info("Model %s -> RMSE=%.3f MAE=%.3f R2=%.3f", name, rmse, mae, r2)
            # Selection criterion: lowest RMSE.
            if best_metrics is None or rmse < best_metrics[0]:
                best_name, best_metrics, best_estimator = name, (rmse, mae, r2), estimator

        assert best_metrics is not None  # for type-checkers
        rmse, mae, r2 = best_metrics

        artifact: dict[str, Any] = {
            "model": best_estimator,
            "model_name": best_name,
            "feature_columns": list(FEATURE_COLUMNS),
            "target_column": TARGET_COLUMN,
            "last_daily_series": daily,  # needed to seed recursive forecasting
        }
        self._save_artifact(artifact)

        result = TrainingResult(
            model_name=best_name,
            rmse=rmse,
            mae=mae,
            r2_score=r2,
            training_rows=len(train_df),
            test_rows=len(test_df),
        )
        logger.info("Selected model %s and saved artifact", best_name)
        return result, artifact

    def _save_artifact(self, artifact: dict[str, Any]) -> Path:
        path = Path(self._settings.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, path)
        logger.info("Saved model artifact to %s", path)
        return path
