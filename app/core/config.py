"""Application configuration.

All configuration is read from environment variables (optionally via a `.env`
file) so that the same code base runs unchanged across local, CI and
production environments. Nothing sensitive is ever hard-coded.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from the environment.

    Using ``pydantic_settings`` gives us validation and type coercion for free:
    an invalid ``MAX_FORECAST_DAYS`` (e.g. non-integer) fails fast at startup
    rather than deep inside a request handler.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Metadata ---
    app_name: str = "Sales Forecasting & Analytics API"
    app_version: str = "1.0.0"

    # --- Database ---
    database_url: str = Field(default="sqlite:///./sales.db")

    # --- Machine learning ---
    model_path: str = Field(default="models/sales_model.pkl")

    # --- Logging ---
    log_level: str = Field(default="INFO")
    log_dir: str = Field(default="logs")

    # --- Data paths ---
    raw_data_path: str = Field(default="data/raw/sales_data.csv")
    processed_data_path: str = Field(default="data/processed/sales_clean.csv")
    chart_dir: str = Field(default="data/processed/charts")

    # --- Behaviour tuning ---
    max_forecast_days: int = Field(default=365, ge=1)
    default_page_size: int = Field(default=50, ge=1, le=1000)


@lru_cache
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Caching means the environment is parsed exactly once per process, and the
    same object can be used as a FastAPI dependency without re-reading files.
    """

    return Settings()
