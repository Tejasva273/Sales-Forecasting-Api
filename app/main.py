"""FastAPI application factory and entry point.

Wires together configuration, logging, the database, routers, request-logging
middleware and centralized exception handlers so that:

* every request is logged (method, path, status, duration),
* domain errors map to correct HTTP status codes with safe messages,
* unexpected errors return a generic 500 without leaking stack traces.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api import routes_analytics, routes_forecast, routes_sales
from app.core.config import get_settings
from app.core.exceptions import AppError
from app.core.logging_config import configure_logging, get_logger
from app.database.database import init_db
from app.schemas.forecast import HealthResponse

configure_logging()
logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown hooks: ensure database tables exist."""

    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    init_db()
    yield
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """Application factory (used by ``run.py``, tests and uvicorn)."""

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Analyze historical sales data and forecast future sales. "
            "Provides CRUD, analytics, model training and forecasting endpoints."
        ),
        lifespan=lifespan,
    )

    # ---------------- Middleware: request logging ---------------- #
    @app.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = uuid.uuid4().hex[:8]
        start = time.perf_counter()
        logger.info("[%s] --> %s %s", request_id, request.method, request.url.path)
        try:
            response = await call_next(request)
        except Exception:  # noqa: BLE001 - handled below, logged here
            elapsed = (time.perf_counter() - start) * 1000
            logger.exception("[%s] !! unhandled error after %.1fms", request_id, elapsed)
            raise
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "[%s] <-- %s %s %d (%.1fms)",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response

    # ---------------- Centralized exception handlers ---------------- #
    @app.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        logger.warning("AppError (%d): %s", exc.status_code, exc.message)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning("Validation failure: %s", exc.errors())
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.exception_handler(Exception)
    async def handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
        # Log the full trace server-side, but never expose it to the client.
        logger.exception("Unhandled exception: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred. Please try again later."},
        )

    # ---------------- Routes ---------------- #
    @app.get("/health", response_model=HealthResponse, tags=["health"], summary="Health check")
    async def health() -> HealthResponse:
        """Liveness probe used by orchestrators and monitoring."""

        return HealthResponse(status="healthy")

    app.include_router(routes_sales.router)
    app.include_router(routes_analytics.router)
    app.include_router(routes_forecast.router)

    logger.info("FastAPI application initialised with all routers.")
    return app


app = create_app()
