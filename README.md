# 📊 Sales Forecasting & Analytics API

> Give it your sales history. Get back clean analytics and a forecast of what's coming next.

This is a small but complete backend service I built to answer a very practical
question every business eventually asks: *"How much are we going to sell next
week?"* You feed it a CSV of past orders, it tidies up the messy bits, stores
everything in a database, shows you what's been happening (revenue trends, best
products, strongest regions), and then trains a model to predict future daily
sales — all reachable through a clean, documented REST API.

Built with **FastAPI, pandas, scikit-learn, and SQLAlchemy**. Runs on SQLite out
of the box, switches to MySQL with a single config line.

---

## Why I built it

Most companies are sitting on a pile of sales transactions in spreadsheets. That
data is useful, but spreadsheets don't scale, they're painful to reproduce, and
you can't plug a dashboard or another app into them. I wanted one service that
takes raw sales data and turns it into two things people actually need:

1. **Analytics they can trust** — computed from the data, never hard-coded.
2. **A forecast they can plan around** — for inventory, staffing, and cash flow.

Everything is exposed as an API, so a frontend dashboard can sit on top of it later
without any changes to the backend.

## What it can do

Here's the full journey your data takes:

```
Your CSV  →  validate & clean  →  store in DB  →  analytics + charts
          →  build features  →  train model  →  evaluate  →  forecast  →  API
```

In plain terms:

- **📥 Loads & cleans your data** — checks the columns are right, parses dates,
  fills or drops missing values, removes duplicates, catches impossible numbers
  (negative quantities, discounts above 100%), recomputes revenue so it's always
  consistent, and sorts everything by date. If something's wrong with your file,
  it tells you *what* is wrong instead of silently swallowing bad rows.
- **📈 Answers the analytics questions** — total revenue, order count, quantity,
  average order value, plus breakdowns by month, product, and region.
- **🖼️ Draws charts** — monthly and weekly trends, product/category/region sales,
  quantity trends, and average order value, saved as PNGs you can drop into a report.
- **🤖 Forecasts future sales** — leakage-safe time-series features, a proper
  chronological train/test split, a Linear Regression baseline compared against a
  Random Forest, and honest RMSE/MAE/R² scores.
- **🔌 Serves it all over REST** — sales CRUD, analytics, model training, and
  forecasting, with automatic interactive docs at `/docs`.

## Getting started (5 minutes)

You'll need **Python 3.11 or newer**. Then:

```bash
# 1. Jump into the project
cd sales_forecasting_api

# 2. Set up an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install everything
pip install -r requirements.txt

# 4. Copy the example config (tweak it later if you want)
cp .env.example .env               # Windows: copy .env.example .env
```

Now run it:

```bash
# Seed the database, make the charts, and train the first model.
# No CSV yet? No problem — it generates a realistic ~8,000-row sample for you.
python -m scripts.seed

# Start the server
python run.py

# Open your browser 👉  http://127.0.0.1:8000/docs
```

That's it. The `/docs` page is a live playground — you can click any endpoint,
hit **"Try it out"**, and fire real requests without touching curl or Postman.

> **Want to use your own data?** Drop a CSV at `data/raw/sales_data.csv` (that's
> the default `RAW_DATA_PATH`) *before* running `python -m scripts.seed`. Already
> ran it once with the sample data? Just re-run the seed command — it clears the
> old rows and reloads yours cleanly. Your file needs these columns:
> `order_id, order_date, product_id, product_name, category, region, quantity,
> unit_price, discount` (revenue is optional — it gets recomputed).

## Taking it for a spin

Once the server is up, here are a few things worth trying (either in `/docs` or
with curl).

**Check it's alive:**

```bash
curl http://127.0.0.1:8000/health
# → {"status": "healthy"}
```

**Add a sale** — notice you don't send `revenue`; the server calculates it:

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

```json
{ "order_id": "ORD-999001", "id": 8001, "revenue": 54.0, "...": "..." }
```

**See the headline numbers:**

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

**Train a model, then forecast the next week:**

```bash
curl -X POST http://127.0.0.1:8000/model/train -H "Content-Type: application/json" -d '{"model_type": "auto"}'
curl -X POST http://127.0.0.1:8000/forecast    -H "Content-Type: application/json" -d '{"days": 7}'
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

**And when you do something wrong, it fails gracefully:**

```bash
# Ask for 0 days → 422, with a clear reason
curl -X POST http://127.0.0.1:8000/forecast -H "Content-Type: application/json" -d '{"days": 0}'

# Forecast before training a model → 409
# → {"detail": "No trained model found. Call POST /model/train before forecasting."}
```

## The endpoints at a glance

| Method | Path | What it does |
|--------|------|--------------|
| `GET` | `/health` | Is the service up? |
| `GET` | `/sales` | List sales (paginated, filterable by date/category/region) |
| `GET` | `/sales/{order_id}` | Fetch one order |
| `POST` | `/sales` | Add an order (revenue computed for you) |
| `DELETE` | `/sales/{order_id}` | Remove an order |
| `GET` | `/analytics/summary` | Revenue, orders, quantity, average order value |
| `GET` | `/analytics/monthly` | Sales by month |
| `GET` | `/analytics/products` | Sales by product |
| `GET` | `/analytics/regions` | Sales by region |
| `POST` | `/model/train` | Train & evaluate the forecasting model |
| `GET` | `/model/metrics` | See how past training runs scored |
| `POST` | `/forecast` | Predict the next *N* days |
| `GET` | `/forecast/history` | Look back at past forecasts |

Full interactive docs live at **`/docs`** (Swagger) and **`/redoc`** (ReDoc) —
both generated automatically from the code.

## How the forecasting actually works

This is the part I'm most careful about, because time-series forecasting is easy
to get *subtly* wrong.

**The target is daily total revenue.** Since orders arrive at irregular times, I
first roll everything up into one row per day (days with no sales become 0, so the
series is evenly spaced).

**The features** are all built from the *past* only:

- Calendar signals: `day, week, month, quarter, year, day_of_week`
- Lags: yesterday, last week, two weeks ago, a month ago (`sales_lag_1/7/14/30`)
- Rolling averages over the last 7, 14, and 30 days

**The important detail — no data leakage.** The rolling averages are computed on
the *shifted* series (`sales.shift(1).rolling(w)`), so a day's own value never
sneaks into its own features. A naïve `rolling().mean()` would include the number
you're trying to predict, giving you a model that looks brilliant in testing and
falls apart in real life. I also drop the early "warm-up" rows that don't have a
full 30-day history yet.

**The split is chronological, never shuffled** — train on the earliest 80%, test
on the most recent 20%, exactly like forecasting the future from the past.

**Two models compete.** A `LinearRegression` baseline and a `RandomForestRegressor`.
With `"model_type": "auto"`, whichever gets the lower RMSE on the test set wins,
gets saved with joblib, and its scores are recorded in the database.

**Forecasting is recursive** — to predict tomorrow it needs yesterday, so each
prediction is fed back into the series before predicting the next day. Predictions
can't go negative (you can't sell negative revenue).

## Under the hood

A quick tour of how the code is organized — each layer has one job, so routes stay
thin and the logic stays testable.

```
sales_forecasting_api/
├── app/
│   ├── main.py                 # app setup, logging middleware, error handlers, /health
│   ├── api/                    # HTTP routes only — no business logic here
│   │   ├── routes_sales.py
│   │   ├── routes_analytics.py
│   │   └── routes_forecast.py
│   ├── core/                   # config, logging, and domain exceptions
│   ├── database/               # SQLAlchemy engine + the three ORM tables
│   ├── schemas/                # Pydantic request/response models (validation)
│   ├── services/               # the real work: data, analytics, forecasting, sales, EDA
│   ├── ml/                     # preprocessing → features → train → predict
│   └── utils/                  # small reusable validators
├── data/{raw,processed/charts}/
├── models/                     # saved model artifacts (.pkl)
├── notebooks/                  # exploratory analysis notebook
├── scripts/seed.py             # one command to run the whole pipeline
├── tests/                      # pytest suite
└── run.py                      # start the server
```

**The three database tables:**

- **`sales`** — one row per order line, with proper constraints (`quantity > 0`,
  `discount` between 0 and 1, etc.) enforced *at the database level*, plus indexes
  on the columns you'll actually query by.
- **`forecasts`** — every predicted day the API has ever produced.
- **`model_metrics`** — the RMSE / MAE / R² of each training run, so you can track
  whether the model is getting better over time.

**The toolbox:** Python 3.11+, FastAPI + Uvicorn, pandas + NumPy, scikit-learn +
joblib, SQLAlchemy 2.x, Pydantic v2, Matplotlib, and Pytest — with Ruff and Mypy
keeping the code tidy.

## Switching to MySQL

No code changes needed. Install the driver and point the config at your database:

```bash
pip install pymysql
```

```dotenv
# in your .env
DATABASE_URL=mysql+pymysql://user:password@localhost:3306/sales
```

The engine reads that string and adapts (connection pooling, the right connect
args) automatically.

## Running the tests

```bash
pytest              # run everything
pytest -v           # ...with names
pytest --cov=app    # ...with a coverage report (needs pytest-cov)
```

The suite runs against a throwaway in-memory database and covers the stuff that
actually matters: creating and reading sales, validation rules, the cleaning
pipeline, whether the analytics numbers reconcile, the chronological split, model
training, forecasting — and there's a dedicated test that *proves the features
don't leak the target*, because that's the one bug that would quietly ruin the
whole model.

And to keep the code healthy:

```bash
ruff check .        # lint
ruff format .       # format
mypy app            # type-check
```

## Where I'd take it next

- 🔐 Authentication (API keys / OAuth2) and rate limiting
- 🧠 More models (Gradient Boosting, SARIMA/Prophet) and extra signals like
  holidays and promotions
- 🎯 Per-product / per-region forecasts instead of one aggregate series
- 📏 Proper backtesting (rolling-origin CV) with confidence intervals
- 📦 Alembic migrations, a Dockerfile, and CI on GitHub Actions
- ⬆️ A CSV-upload endpoint and a React dashboard sitting on top of the API

---

## A note for reviewers: what this project demonstrates

I built this as a portfolio piece for an AI/ML role, so here's an honest map of
the skills it exercises and where to look for them in the code:

- **Python & OOP** — clean, type-hinted Python throughout. The real logic lives in
  focused service classes (`DataService`, `AnalyticsService`, `ForecastingService`,
  `ModelTrainer`, `Forecaster`) that each do one thing and hide their internals.
  Inheritance shows up only where it earns its keep — the `AppError` exception
  family that maps neatly onto HTTP status codes.
- **Data structures & algorithms** — the chronological split, recursive multi-step
  forecasting, dictionary-based aggregation, and offset pagination.
- **SQL & databases** — ORM models with primary/unique keys, indexes, a composite
  index, and CHECK constraints; aggregation (`SUM`, `COUNT`, `GROUP BY`) pushed
  down into SQL; and SQLite↔MySQL portability through config alone.
- **Data processing** — the full pandas/NumPy cleaning pipeline that turns a messy
  CSV into trustworthy rows.
- **Machine learning** — leakage-safe feature engineering, honest evaluation,
  model comparison and selection, persistence, and recursive forecasting.
- **REST APIs / FastAPI** — 13 documented endpoints with request/response schemas,
  the *right* status codes (201/400/404/409/422/500), pagination and filtering.
- **Engineering discipline** — centralized error handling that never leaks stack
  traces, structured logging with request tracing, environment-based config, a
  meaningful test suite, and a one-command pipeline (`scripts/seed.py`).
- **Problem solving** — turning raw transactions into something both *readable* and
  *predictable*, while defending against the classic time-series trap (leakage) and
  bad user input.

If you want the shortest possible tour: read `app/ml/feature_engineering.py` (the
no-leakage bit), then `app/services/forecasting_service.py` (how it all ties
together), then open `/docs` and click around.
