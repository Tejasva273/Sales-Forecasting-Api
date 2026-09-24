"""Forecasting and model-management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.database.database import get_db
from app.schemas.forecast import (
    ForecastHistoryItem,
    ForecastRequest,
    ForecastResponse,
    ModelMetricRead,
    TrainRequest,
    TrainResponse,
)
from app.services.forecasting_service import ForecastingService

router = APIRouter(tags=["forecast"])


@router.post(
    "/forecast",
    response_model=ForecastResponse,
    summary="Forecast future sales",
    description="Generate daily revenue predictions for the next `days` days "
    "using the trained model. Invalid horizons (<=0, too large) return 422; "
    "an untrained model returns 409.",
    responses={
        409: {"description": "Model not trained"},
        422: {"description": "Invalid forecast period or insufficient data"},
    },
)
def create_forecast(payload: ForecastRequest, db: Session = Depends(get_db)) -> ForecastResponse:
    service = ForecastingService(db)
    return ForecastResponse(**service.forecast(payload.days))


@router.get(
    "/forecast/history",
    response_model=list[ForecastHistoryItem],
    summary="Forecast history",
    description="Return previously generated forecast points, most recent first.",
)
def forecast_history(
    db: Session = Depends(get_db),
    limit: int = Query(100, ge=1, le=1000, description="Max rows to return"),
) -> list[ForecastHistoryItem]:
    service = ForecastingService(db)
    return [ForecastHistoryItem.model_validate(row) for row in service.history(limit)]


@router.post(
    "/model/train",
    response_model=TrainResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Train the forecasting model",
    description="Train (and evaluate) the model on stored sales data, persist "
    "metrics, and save the fitted estimator. Returns evaluation metrics.",
    responses={422: {"description": "Insufficient data to train"}},
)
def train_model(
    payload: TrainRequest | None = None, db: Session = Depends(get_db)
) -> TrainResponse:
    request = payload or TrainRequest()
    service = ForecastingService(db)
    result = service.train_model(model_type=request.model_type, test_size=request.test_size)
    return TrainResponse(**result)


@router.get(
    "/model/metrics",
    response_model=list[ModelMetricRead],
    summary="Model performance metrics",
    description="Return stored evaluation metrics (RMSE, MAE, R2) for past "
    "training runs, most recent first.",
)
def model_metrics(
    db: Session = Depends(get_db),
    limit: int = Query(20, ge=1, le=200),
) -> list[ModelMetricRead]:
    service = ForecastingService(db)
    return [ModelMetricRead.model_validate(m) for m in service.get_metrics(limit)]
