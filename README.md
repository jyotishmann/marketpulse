# MarketPulse

> A stock market intelligence pipeline and dashboard.
> Free data, zero API keys, pure Python.

[![CI](https://github.com/YOUR_USERNAME/marketpulse/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/marketpulse/actions/workflows/ci.yml)

## What It Does

MarketPulse pulls stock price data and financial news on a schedule, computes
technical indicators, runs ML-based signal detection, and displays live insights
on a Streamlit dashboard — all running locally in Docker with no paid services.

**Stack:** FastAPI · Streamlit · PostgreSQL · Redis · scikit-learn · Docker Compose · GitHub Actions

## Quick Start

# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/marketpulse.git
cd marketpulse

# 2. Create your environment file
cp .env.example .env

# 3. Start the full stack
docker compose up

Then open:

- **Dashboard:** http://localhost:8501
- **API docs (Swagger):** http://localhost:8000/docs

## Architecture

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

See [MASTER_DOC.md](./MASTER_DOC.md) for the full architecture reference,
including every component, connector, and design decision.

## Development

# Install dev + test dependencies
pip install -e ".[dev,test]"

# Run linting
ruff check . && ruff format .

# Run tests (requires test Docker services)
docker compose -f docker-compose.test.yml up -d
pytest tests/ -v --cov=marketpulse
docker compose -f docker-compose.test.yml down

## Project Structure

marketpulse/     ← Python package (API, ingestion, ML, cache, scheduler)
dashboard/       ← Streamlit dashboard
tests/           ← pytest test suite
.github/         ← GitHub Actions CI/CD workflows
docker-compose.yml          ← development stack
docker-compose.test.yml     ← isolated test stack

## License

MIT