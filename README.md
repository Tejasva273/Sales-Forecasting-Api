# Sales Forecasting & Analytics API

A production-style REST API that ingests historical sales data, cleans and
stores it, exposes rich analytics, trains a machine-learning model, and
forecasts future daily sales — all behind a documented FastAPI service.

---

## 1. Problem Statement

Businesses accumulate large volumes of transactional sales data but often lack
an easy, programmatic way to (a) turn that raw data into trustworthy analytics
and (b) predict near-term sales for planning (inventory, staffing, cash flow).
Manual spreadsheets don't scale, aren't reproducible, and can't be consumed by
other applications.

## 2. Project Objectives

This project provides a single service that lets a user:

1. Load historical sales data (generated sample or an external CSV).
2. Validate and clean the data.
3. Store sales records in a relational database.
4. Perform sales analytics.
5. Generate machine-learning features.
6. Train a sales-forecasting model.
7. Evaluate the model.
8. Predict future sales.
9. Access analytics and predictions through REST APIs.
10. Store prediction results in the database.

It is designed so a frontend dashboard can later consume the same endpoints.

## 3. Features

- **Data pipeline** — CSV loading, column validation, date parsing, missing-value
  handling, de-duplication, numeric validation, revenue recomputation, chronological
  sorting, and persistence.
- **Analytics** — total revenue, orders, quantity, average order value, plus
  monthly / product / region breakdowns.
- **EDA charts** — monthly & weekly trends, product/category/region sales,
  quantity trend, average order value, saved as PNGs.
- **ML forecasting** — leakage-safe time-series features, chronological split,
  Linear Regression baseline + Random Forest, RMSE/MAE/R² evaluation, model
  auto-selection, and recursive multi-day forecasting.
- **REST API** — sales CRUD, analytics, model training/metrics, forecasting and
  forecast history, health check.
- **Engineering** — Pydantic validation, centralized error handling (no stack
  traces leaked), request + application logging, environment-based configuration,
  and a Pytest suite.

## 4. Architecture

```
                       ┌─────────────────────────────────────────────┐
                       │                 FastAPI app                  │
                       │  (routing, validation, error handling, logs) │
                       └───────────────┬─────────────────────────────┘
                                       │
        ┌──────────────┬───────────────┼───────────────┬──────────────┐
        ▼              ▼               ▼               ▼              ▼
   routes_sales   routes_analytics  routes_forecast  /health     middleware
        │              │               │                            (logging)
        ▼              ▼               ▼
  SalesService   AnalyticsService  ForecastingService
        │              │               │        │
        │              │               │        └────────────┐
        ▼              ▼               ▼                      ▼
   ┌──────────────────────────────┐        ┌──────────────────────────────┐
   │   SQLAlchemy ORM (models)     │        │            app.ml             │
   │  sales / forecasts / metrics  │        │  preprocessing → features →   │
   └───────────────┬──────────────┘        │  train (LR/RF) → predict      │
                   │                        └───────────────┬──────────────┘
                   ▼                                        ▼
        ┌────────────────────┐                   ┌────────────────────┐
        │  SQLite / MySQL     │                   │  joblib model .pkl  │
        └────────────────────┘                   └────────────────────┘
```

### End-to-end data flow

```
CSV Dataset → Data Validation → Data Cleaning → SQLite/MySQL → Analytics
   → Feature Engineering → ML Training → Model Evaluation → Saved Model
   → FastAPI → REST API → Prediction
```

### Directory layout

```
sales_forecasting_api/
├── app/
│   ├── main.py                 # app factory, middleware, error handlers, /health
│   ├── api/                    # thin HTTP routers (no business logic)
│   │   ├── routes_sales.py
│   │   ├── routes_forecast.py
│   │   └── routes_analytics.py
│   ├── core/
│   │   ├── config.py           # env-driven settings (pydantic-settings)
│   │   ├── logging_config.py   # console + rotating file logging
│   │   └── exceptions.py       # domain exceptions → HTTP status codes
│   ├── database/
│   │   ├── database.py         # engine/session (SQLite or MySQL via env)
│   │   └── models.py           # Sale / Forecast / ModelMetric ORM models
│   ├── schemas/                # Pydantic request/response models
│   │   ├── sales.py
│   │   └── forecast.py
│   ├── services/               # business logic (OOP service classes)
│   │   ├── data_service.py
│   │   ├── analytics_service.py
│   │   ├── forecasting_service.py
│   │   ├── sales_service.py
│   │   └── eda_service.py
│   ├── ml/                     # ML pipeline
│   │   ├── preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── train.py
│   │   └── predict.py
│   └── utils/
│       └── validators.py
├── data/{raw,processed/charts}/
├── models/                     # saved joblib model artifacts
├── notebooks/exploratory_analysis.ipynb
├── scripts/seed.py             # one-shot pipeline runner
├── tests/                      # pytest suite
├── requirements.txt
├── pyproject.toml
├── .env.example
├── .gitignore
└── run.py
```

## 5. Technology Stack

| Concern            | Technology                          |
|--------------------|-------------------------------------|
| Language           | Python 3.11+                        |
| Web framework      | FastAPI + Uvicorn                   |
| Data processing    | Pandas, NumPy                       |
| Machine learning   | scikit-learn (LinearRegression, RandomForest), joblib |
| Database / ORM     | SQLAlchemy 2.x, SQLite (MySQL-ready)|
| Validation         | Pydantic v2, pydantic-settings      |
| Charts             | Matplotlib                          |
| Testing            | Pytest, httpx (TestClient)          |
| Tooling            | Ruff, Mypy                          |

## 6. Database Schema

**`sales`** — one row per order line.

| Column       | Type    | Notes                                  |
|--------------|---------|----------------------------------------|
| id           | INT PK  | autoincrement                          |
| order_id     | STR     | unique, indexed                        |
| order_date   | DATE    | indexed                                |
| product_id   | STR     | indexed                                |
| product_name | STR     |                                        |
| category     | STR     | indexed                                |
| region       | STR     | indexed                                |
| quantity     | INT     | CHECK > 0                              |
| unit_price   | FLOAT   | CHECK >= 0                             |
| discount     | FLOAT   | CHECK 0..1                             |
| revenue      | FLOAT   | CHECK >= 0 (computed server-side)      |

Composite index `ix_sales_date_category` on `(order_date, category)`.

**`forecasts`** — one row per predicted future day.

| Column          | Type     | Notes                       |
|-----------------|----------|-----------------------------|
| id              | INT PK   |                             |
| forecast_date   | DATE     | indexed                     |
| predicted_sales | FLOAT    | CHECK >= 0                  |
| model_name      | STR      | indexed                     |
| created_at      | DATETIME | server default now()        |

**`model_metrics`** — one row per training run.

| Column     | Type     | Notes                |
|------------|----------|----------------------|
| id         | INT PK   |                      |
| model_name | STR      | indexed              |
| rmse       | FLOAT    |                      |
| mae        | FLOAT    |                      |
| r2_score   | FLOAT    |                      |
| trained_at | DATETIME | server default now() |

## 7. Machine-Learning Approach

**Target:** daily total revenue. Order-level rows are aggregated into a
**continuous daily series** (days with no orders are filled with 0 so windows
are regularly spaced).

**Features** (from past values only):

- Calendar: `day, week, month, quarter, year, day_of_week`
- Lags: `sales_lag_1, sales_lag_7, sales_lag_14, sales_lag_30`
- Rolling means: `rolling_mean_7, rolling_mean_14, rolling_mean_30`

**Avoiding data leakage.** Every feature is derived strictly from data available
*before* the day being predicted:

- Lag features use `sales.shift(k)` — literally the value `k` days earlier.
- Rolling means are computed on the **lag-1 series**
  (`sales.shift(1).rolling(w).mean()`), so the current day's own value is never
  part of its own feature. A naïve `sales.rolling(w).mean()` would include the
  target and leak it into the model, inflating scores and producing a model that
  cannot actually forecast. The warm-up rows without a full 30-day history are
  dropped.

**Split.** Because this is time-series data, the data is **never shuffled**. An
80/20 **chronological** split is used (earliest 80% train, most recent 20% test),
mirroring real forecasting.

**Models.** A `LinearRegression` baseline is trained first; a
`RandomForestRegressor` is trained for comparison. With `model_type="auto"` the
model with the **lowest test RMSE** is selected and persisted with `joblib`.

**Evaluation.** RMSE, MAE and R² are computed on the held-out test set and
stored in `model_metrics`.

**Forecasting.** Multi-day forecasts are **recursive**: each predicted day is
appended to the working series so the next day's lag/rolling features can be
computed exactly as during training. Predictions are clamped to be non-negative.

## 8. API Documentation

Interactive docs are auto-generated by FastAPI:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

| Method | Path                  | Description                              |
|--------|-----------------------|------------------------------------------|
| GET    | `/health`             | Liveness check                           |
| GET    | `/sales`              | List sales (paginated, filterable)       |
| GET    | `/sales/{order_id}`   | Get one sale                             |
| POST   | `/sales`              | Create a sale (revenue computed server-side) |
| DELETE | `/sales/{order_id}`   | Delete a sale                           |
| GET    | `/analytics/summary`  | Total revenue, orders, quantity, AOV     |
| GET    | `/analytics/monthly`  | Monthly sales                            |
| GET    | `/analytics/products` | Product-wise sales                       |
| GET    | `/analytics/regions`  | Region-wise sales                        |
| POST   | `/model/train`        | Train & evaluate the model               |
| GET    | `/model/metrics`      | Stored model metrics                     |
| POST   | `/forecast`           | Forecast future sales                    |
| GET    | `/forecast/history`   | Previously generated forecasts           |

## 9. Installation

Requires **Python 3.11+**.

```bash
# 1. Clone and enter the project
cd sales_forecasting_api

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your local config
cp .env.example .env               # edit values if desired
```

## 10. How to Run

```bash
# 1. Seed the database, generate EDA charts, and train the initial model.
#    (generates a synthetic ~8,000-row dataset if none exists)
python -m scripts.seed

#    Options:
#      python -m scripts.seed --records 6000   # dataset size
#      python -m scripts.seed --no-train        # skip training
#      python -m scripts.seed --no-charts       # skip EDA charts

# 2. Start the API
python run.py
#    or: uvicorn app.main:app --reload

# 3. Open the docs
#    http://127.0.0.1:8000/docs
```

To use your own dataset, place a CSV with the required columns at the path in
`RAW_DATA_PATH` (default `data/raw/sales_data.csv`) before seeding.

### Switching to MySQL

No code changes are needed — set the connection string in `.env` and install the
driver:

```bash
pip install pymysql
# .env
DATABASE_URL=mysql+pymysql://user:password@localhost:3306/sales
```

## 11. Example API Requests & Responses

**Create a sale**

```bash
curl -X POST http://127.0.0.1:8000/sales \
  -H "Content-Type: application/json" \
  -d '{
        "order_id": "ORD-999001",
        "order_date": "2026-01-15",
        "product_id": "P001",
        "product_name": "Wireless Mouse",
        "category": "Electronics",
        "region": "North",
        "quantity": 3,
        "unit_price": 20.0,
        "discount": 0.1
      }'
```

Response `201`:

```json
{
  "order_id": "ORD-999001",
  "order_date": "2026-01-15",
  "product_id": "P001",
  "product_name": "Wireless Mouse",
  "category": "Electronics",
  "region": "North",
  "quantity": 3,
  "unit_price": 20.0,
  "discount": 0.1,
  "id": 8001,
  "revenue": 54.0
}
```

**Analytics summary**

```bash
curl http://127.0.0.1:8000/analytics/summary
```

```json
{
  "total_revenue": 1543210.55,
  "total_orders": 8000,
  "total_quantity": 44012,
  "average_order_value": 192.90
}
```

**Train the model**

```bash
curl -X POST http://127.0.0.1:8000/model/train \
  -H "Content-Type: application/json" -d '{"model_type": "auto"}'
```

```json
{
  "model_name": "RandomForest",
  "rmse": 2451.30,
  "mae": 1890.12,
  "r2_score": 0.61,
  "training_rows": 664,
  "test_rows": 166
}
```

**Forecast**

```bash
curl -X POST http://127.0.0.1:8000/forecast \
  -H "Content-Type: application/json" -d '{"days": 7}'
```

```json
{
  "model": "RandomForest",
  "forecast": [
    {"date": "2026-10-01", "predicted_sales": 1723.40},
    {"date": "2026-10-02", "predicted_sales": 1688.10}
  ]
}
```

**Error examples**

```bash
# Invalid forecast horizon → 422
curl -X POST http://127.0.0.1:8000/forecast -H "Content-Type: application/json" -d '{"days": 0}'
# {"detail":[{"type":"greater_than", ...}]}

# Forecast before training → 409
# {"detail":"No trained model found. Call POST /model/train before forecasting."}
```

## 12. Testing

```bash
pytest            # run the full suite
pytest -v         # verbose
pytest --cov=app  # with coverage (requires pytest-cov)
```

The suite covers: sales creation & retrieval, input validation, the cleaning
pipeline, analytics reconciliation, feature engineering (including an explicit
**no-leakage** assertion), chronological splitting, model training, forecast
generation, and invalid-forecast handling. Tests run against an isolated
in-memory SQLite database.

Static quality gates:

```bash
ruff check .      # lint
ruff format .     # format
mypy app          # type-check
```

## 13. Future Improvements

- Add authentication (API keys / OAuth2) and rate limiting.
- Support additional models (Gradient Boosting, SARIMA/Prophet) and exogenous
  regressors (holidays, promotions).
- Add per-product / per-region forecasting instead of a single aggregate series.
- Backtesting with rolling-origin cross-validation and confidence intervals.
- Alembic migrations, Dockerfile + docker-compose, and CI (GitHub Actions).
- CSV upload endpoint and a React dashboard consuming these APIs.

---

## How This Project Demonstrates Software Engineering and AI/ML Skills

| Skill area          | Where it shows up in this project |
|---------------------|-----------------------------------|
| **Python**          | Type-hinted, modular Python 3.11 across `app/`; dataclasses, generators, context managers, comprehensions. |
| **OOP**             | Service classes with single responsibilities and encapsulated dependencies: `SalesService`, `AnalyticsService`, `ForecastingService`, `DataService`, `EDAService`, `ModelTrainer`, `Forecaster`. Constructors inject config; state is private. Inheritance is used only where it adds value (the `AppError` exception hierarchy that maps to HTTP status codes). |
| **Data Structures & Algorithms** | Chronological train/test split, recursive multi-step forecasting, dictionary-based monthly aggregation, and efficient pagination/offset logic. |
| **SQL & Databases** | SQLAlchemy ORM models with primary keys, unique keys, indexes, composite indexes and CHECK constraints; aggregate queries (`SUM`, `COUNT`, `GROUP BY`, `ORDER BY`) pushed to SQL; SQLite↔MySQL portability via configuration. |
| **REST APIs / FastAPI** | 13 documented endpoints with request/response schemas, correct status codes (201/400/404/409/422/500), pagination, filtering, and auto-generated Swagger/ReDoc. |
| **Data Processing** | Pandas/NumPy pipeline: validation, date parsing, missing-value handling, de-duplication, numeric validation, revenue recomputation, chronological sorting, persistence. |
| **Machine Learning** | Leakage-safe feature engineering, chronological split, baseline + ensemble models, RMSE/MAE/R² evaluation, model selection, joblib persistence, recursive forecasting. |
| **Automation**      | `scripts/seed.py` runs the whole pipeline end-to-end (generate → clean → DB → EDA → train) with one command; `run.py` launches the server. |
| **Testing**         | Pytest suite with fixtures, an isolated in-memory DB, TestClient integration tests, parametrized validation tests, and a targeted data-leakage test. |
| **Git**             | Incremental, meaningfully-scoped commits (setup → DB → pipeline → analytics → features → forecasting → API → validation/errors → tests → docs). |
| **SDLC**            | Clear separation of concerns (routes / services / ML / data / schemas), configuration management, centralized logging & error handling, and documentation — an application built stage-by-stage and verifiable at each step. |
| **Problem Solving** | Turning raw transactional data into trustworthy analytics and a genuinely forecastable model, while defending against the classic time-series pitfall (leakage) and invalid inputs. |
#   S a l e s - F o r e c a s t i n g - A p i  
 