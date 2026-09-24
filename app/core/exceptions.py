"""Domain-specific exceptions.

These exceptions are raised by the service / ML / data layers and are mapped
to appropriate HTTP responses by the centralized exception handlers in
``app.main``. Keeping them framework-agnostic means the same services can be
reused from CLI scripts or tests without importing FastAPI.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all expected application errors.

    Attributes
    ----------
    message:
        Safe, human-readable message intended to be shown to the client.
    status_code:
        HTTP status code the API layer should respond with.
    """

    status_code: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class DataValidationError(AppError):
    """Raised when input data fails validation (bad CSV, bad values)."""

    status_code = 400


class ResourceNotFoundError(AppError):
    """Raised when a requested entity does not exist."""

    status_code = 404


class DuplicateResourceError(AppError):
    """Raised when creating a resource that already exists."""

    status_code = 409


class ModelNotTrainedError(AppError):
    """Raised when a prediction is requested before a model is trained."""

    status_code = 409


class InsufficientDataError(AppError):
    """Raised when there is not enough data to train / forecast."""

    status_code = 422


class InvalidForecastPeriodError(AppError):
    """Raised for out-of-range forecast horizons."""

    status_code = 422
