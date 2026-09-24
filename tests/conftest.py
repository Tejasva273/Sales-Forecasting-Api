"""Shared pytest fixtures.

The suite runs against an isolated, temporary SQLite database and a temporary
model/data directory so tests never touch real project artifacts. FastAPI's
``get_db`` dependency is overridden to use the test session.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool


@pytest.fixture()
def db_session(tmp_path, monkeypatch) -> Iterator[Session]:
    """Provide an isolated in-memory database session with all tables."""

    # Point settings at temp paths BEFORE importing modules that read them.
    # A temp-file DB keeps the app's own lifespan init_db() from creating a
    # stray sales.db in the repo; the request path still uses the in-memory
    # session via the get_db override below.
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.pkl"))
    monkeypatch.setenv("RAW_DATA_PATH", str(tmp_path / "raw.csv"))
    monkeypatch.setenv("PROCESSED_DATA_PATH", str(tmp_path / "clean.csv"))
    monkeypatch.setenv("CHART_DIR", str(tmp_path / "charts"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))

    # Reset the cached settings so the temp env vars take effect even if a
    # previous test already imported/instantiated Settings.
    from app.core.config import get_settings

    get_settings.cache_clear()

    import app.database.models  # noqa: F401  (register tables)
    from app.database.database import Base

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture()
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """A TestClient whose ``get_db`` dependency uses the test session."""

    from app.database.database import get_db
    from app.main import create_app

    app = create_app()

    def override_get_db() -> Iterator[Session]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture()
def sample_sales_df() -> pd.DataFrame:
    """A small, clean, deterministic dataset for unit tests."""

    rng = np.random.default_rng(0)
    start = date(2023, 1, 1)
    rows = []
    for i in range(400):
        d = start + timedelta(days=i % 200)
        qty = int(rng.integers(1, 6))
        price = round(float(rng.uniform(10, 100)), 2)
        discount = 0.1
        rows.append(
            {
                "order_id": f"ORD-{i:05d}",
                "order_date": d.isoformat(),
                "product_id": f"P{i % 5:03d}",
                "product_name": f"Product {i % 5}",
                "category": ["Electronics", "Furniture"][i % 2],
                "region": ["North", "South", "East"][i % 3],
                "quantity": qty,
                "unit_price": price,
                "discount": discount,
                "revenue": round(qty * price * (1 - discount), 2),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture()
def seeded_db(db_session: Session, sample_sales_df: pd.DataFrame) -> Session:
    """A database session pre-loaded with the sample dataset."""

    from app.services.data_service import DataService

    cleaned = DataService().clean(sample_sales_df)
    DataService().persist_to_db(cleaned, db_session)
    return db_session
