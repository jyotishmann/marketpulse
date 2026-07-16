# MarketPulse

> A production-grade stock market intelligence pipeline.
> Free data. Zero API keys. Pure Python.

[![CI](https://github.com/jyotishmann/marketpulse/actions/workflows/ci.yml/badge.svg)](https://github.com/jyotishmann/marketpulse/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/YOUR_USERNAME/marketpulse/branch/main/graph/badge.svg)](https://codecov.io/gh/YOUR_USERNAME/marketpulse)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What It Is

MarketPulse is a **scheduled data intelligence system** for stock markets. It ingests
OHLCV price bars from [yfinance](https://github.com/ranaroussi/yfinance) (free, no API key)
and financial headlines from public RSS feeds every 15–30 minutes, computes 10 technical
indicators, runs a machine learning pipeline that emits BUY/HOLD/SELL signals, and
renders everything on a live Streamlit dashboard — all orchestrated by Docker Compose
and guarded by a GitHub Actions CI/CD pipeline.

**Stack:** FastAPI · Streamlit · PostgreSQL · Redis · scikit-learn · APScheduler ·
Docker Compose · GitHub Actions · pytest

## Quick Start


```bash
git clone https://github.com/YOUR_USERNAME/marketpulse.git
cd marketpulse
cp .env.example .env
docker compose up
```

### Open:

- **Dashboard:** http://localhost:8501
- **API (Swagger UI):** http://localhost:8000/docs


The first ingestion cycle runs on startup. Price data appears within ~60 seconds.
ML signals appear after the first hourly pipeline run.

## What It Demonstrates

### Data Engineering

- **ETL pipeline** — yfinance → pydantic validation → pandas transformation → PostgreSQL
- **Scheduled batch jobs** — APScheduler with cron-style intervals per job type
- **Deduplication & upsert** — `ON CONFLICT DO NOTHING` prevents double-inserting on overlap
- **Connection pooling** — SQLAlchemy engine with `pool_size=5, max_overflow=10`

### Machine Learning

- **End-to-end ML pipeline** — ingest → feature engineering → train → persist → serve
- **Supervised classification** — `sklearn.Pipeline(StandardScaler + RandomForestClassifier)`
- **Unsupervised anomaly detection** — `IsolationForest` flags unusual price patterns
- **Model versioning** — `ModelRegistry` table tracks all training runs; rollback via flag
- **Lazy model loading** — `PredictionService` caches models in memory per ticker

### API & Caching

- **FastAPI** with async-native routing, OpenAPI auto-documentation, pydantic schemas
- **Two-layer caching** — Streamlit `@st.cache_data` (30s) → Redis (5 min) → PostgreSQL
- **Cache invalidation** — pattern-based Redis bust (`SCAN_ITER`) after each write
- **Dependency injection** — `Depends()` for DB session, Redis client, ML service

### Infrastructure & DevOps

- **Docker Compose** — six-service stack (api, dashboard, db, redis, and test variants)
- **GitHub Actions** — lint → type-check + test (parallel), Docker image build on merge
- **Pip caching** — `actions/cache` reduces per-job pip install from 2 min to 15 sec
- **Health-driven startup** — `docker compose --wait` replaces brittle `sleep N`
- **Coverage gate** — `--cov-fail-under=80` blocks merges that reduce test coverage

### Code Quality

- **96 pytest tests** — unit tests with mocked externals, API integration tests
- **Conventional Commits** — structured commit messages (`feat/fix/chore/test/ci`)
- **Type annotations** — Python 3.11 syntax throughout, checked with mypy
- **ruff** — single tool replacing flake8 + black + isortThen open:

## High-Level Data Flow

```
yfinance ──────────────────────────────────┐
                                           ▼
RSS Feeds → feedparser → VADER ──►  ETL Layer  ──►  PostgreSQL
                                           │               │
                                    APScheduler            │
                                           │               ▼
                                    ML Pipeline ──►  ml_signals
                                                          │
                                                     FastAPI  ◄── Redis cache
                                                          │
                                                    Streamlit Dashboard
```

## Development

```bash
# Install dev + test dependencies
pip install -e ".[dev,test]"

# Run linting + formatting check
ruff check . && ruff format --check .

# Type checking
mypy marketpulse/ --ignore-missing-imports

# Run tests (requires docker compose test services)
docker compose -f docker-compose.test.yml up -d --wait
alembic upgrade head
pytest tests/ -v --cov=marketpulse
docker compose -f docker-compose.test.yml down

# Run one layer of tests at a time
pytest tests/unit/test_ingestion.py -v     # schema + connector tests
pytest tests/unit/test_processing.py -v    # ETL + indicator tests
pytest tests/unit/test_ml.py -v            # ML pipeline tests
pytest tests/integration/test_api.py -v    # FastAPI endpoint tests
```

### Git Workflow

```bash
# Start a feature branch
git checkout main && git pull origin main
git checkout -b feat/your-feature-name

# Make changes, commit with Conventional Commits
git add path/to/file
git commit -m "feat(scope): description"

# Push and open PR (CI runs automatically)
git push -u origin feat/your-feature-name
```

## Project Structure

```bash
marketpulse/                          ← repo root (also the GitHub repo name)
│
├── .github/
│   └── workflows/
│       ├── ci.yml                    ← lint + type-check + test + coverage gate
│       └── docker.yml                ← build and tag Docker images on merge to main
│
├── marketpulse/                      ← the main Python package (importable)
│   ├── __init__.py
│   ├── config.py                     ← Pydantic Settings: reads all env vars
│   │
│   ├── db/
│   │   ├── __init__.py
│   │   ├── models.py                 ← SQLAlchemy ORM table definitions
│   │   ├── session.py                ← engine creation + session factory
│   │   └── migrations/               ← Alembic migration scripts
│   │       ├── env.py
│   │       ├── script.py.mako
│   │       └── versions/
│   │           └── 0001_initial.py   ← first migration (creates all tables)
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── stock.py                  ← yfinance connector
│   │   ├── news.py                   ← feedparser + RSS connector
│   │   └── schemas.py                ← Pydantic models for raw data validation
│   │
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── etl.py                    ← clean, normalise, deduplicate
│   │   └── indicators.py             ← SMA, EMA, RSI, MACD, Bollinger Bands
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── features.py               ← build feature matrix from DB data
│   │   ├── classifier.py             ← MA crossover BUY/HOLD/SELL classifier
│   │   ├── anomaly.py                ← Isolation Forest anomaly detector
│   │   └── service.py                ← load model, run prediction, return signal
│   │
│   ├── cache/
│   │   ├── __init__.py
│   │   └── redis_client.py           ← Redis wrapper + cache-aside decorator
│   │
│   ├── scheduler/
│   │   ├── __init__.py
│   │   └── jobs.py                   ← APScheduler job definitions + registry
│   │
│   └── api/
│       ├── __init__.py
│       ├── main.py                   ← FastAPI app factory + startup events
│       ├── dependencies.py           ← shared DI: db session, redis client
│       ├── middleware.py             ← CORS, request timing, logging
│       └── routers/
│           ├── __init__.py
│           ├── stocks.py             ← GET /api/v1/stocks/{ticker}/prices
│           ├── signals.py            ← GET /api/v1/stocks/{ticker}/signals
│           ├── news.py               ← GET /api/v1/news
│           └── health.py             ← GET /api/v1/health
│
├── dashboard/
│   └── app.py                        ← entire Streamlit UI (single file)
│
├── tests/
│   ├── conftest.py                   ← shared pytest fixtures
│   ├── unit/
│   │   ├── test_ingestion.py
│   │   ├── test_processing.py
│   │   ├── test_ml.py
│   │   └── test_cache.py
│   └── integration/
│       └── test_api.py
│
├── models/                           ← persisted .pkl model files (gitignored)
│
├── docker-compose.yml                ← development stack
├── docker-compose.test.yml           ← isolated test stack (used by CI)
├── Dockerfile.api                    ← image for FastAPI + scheduler
├── Dockerfile.dashboard              ← image for Streamlit
├── pyproject.toml                    ← dependencies + ruff + mypy + pytest config
├── alembic.ini                       ← Alembic config (points to migrations/)
├── .env.example                      ← template: copy to .env and fill in
├── .gitignore
└── README.md
```

## Environment Variables

Copy `.env.example` to `.env` before running. All variables are documented
in `.env.example` with explanations and working defaults for Docker Compose.

Key variables:
| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | (required) | PostgreSQL connection string |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection string |
| `TICKERS` | `AAPL,GOOGL,MSFT,TSLA` | Comma-separated stock symbols |
| `RSS_FEED_URLS` | (empty) | Comma-separated RSS feed URLs |
| `SCHEDULE_STOCK_MINUTES` | `15` | Price ingestion frequency |
| `SCHEDULE_ML_MINUTES` | `60` | ML pipeline frequency |

## License

MIT