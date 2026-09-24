"""One-shot pipeline runner: generate data -> clean -> DB -> EDA -> train.

Run once after installing dependencies to populate the database, produce EDA
charts and train the initial model::

    python -m scripts.seed                 # full pipeline
    python -m scripts.seed --no-train      # skip model training
    python -m scripts.seed --records 6000  # custom dataset size

This mirrors the project's end-to-end workflow:
CSV -> validation -> cleaning -> SQLite -> analytics -> features -> ML -> model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.logging_config import configure_logging, get_logger
from app.database.database import SessionLocal, init_db
from app.services.data_service import DataService
from app.services.eda_service import EDAService
from app.services.forecasting_service import ForecastingService

logger = get_logger(__name__)


def run(records: int, train: bool, charts: bool) -> None:
    configure_logging()
    init_db()

    data_service = DataService()
    raw_path = Path(data_service._settings.raw_data_path)  # noqa: SLF001 (CLI convenience)
    if not raw_path.exists():
        logger.info("No raw dataset found; generating a synthetic one.")
        data_service.generate_sample_dataset(n_records=records)

    db = SessionLocal()
    try:
        rows = data_service.run_pipeline(db)
        logger.info("Seeded %d sales rows into the database.", rows)

        if charts:
            clean_df = data_service.load_csv(data_service._settings.processed_data_path)  # noqa: SLF001
            EDAService().generate_all(clean_df)

        if train:
            result = ForecastingService(db).train_model(model_type="auto")
            logger.info(
                "Trained model %s (RMSE=%.2f, MAE=%.2f, R2=%.3f)",
                result["model_name"],
                result["rmse"],
                result["mae"],
                result["r2_score"],
            )
    finally:
        db.close()

    logger.info("Seed pipeline complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed database and train model.")
    parser.add_argument("--records", type=int, default=8000, help="Sample dataset size")
    parser.add_argument("--no-train", action="store_true", help="Skip model training")
    parser.add_argument("--no-charts", action="store_true", help="Skip EDA chart generation")
    args = parser.parse_args()
    run(records=args.records, train=not args.no_train, charts=not args.no_charts)


if __name__ == "__main__":
    main()
