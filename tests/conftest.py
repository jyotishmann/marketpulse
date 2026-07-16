# tests/conftest.py
# Shared pytest fixtures available to all test files in tests/ and subdirectories.
# Import is automatic — pytest discovers conftest.py files without explicit imports.

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone  # noqa: F401
from decimal import Decimal  # noqa: F401
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ── Synthetic data factories (used by fixtures below) ─────────────────────────


def _make_ohlcv_df(n: int = 50, ticker: str = "AAPL") -> pd.DataFrame:
    """
    Generate a synthetic OHLCV DataFrame with n rows.

    Uses a random walk for prices so the series has realistic structure.
    np.random.seed(42) makes output deterministic across runs.
    """
    np.random.seed(42)
    closes = 180.0 + np.cumsum(np.random.normal(0, 0.5, n))
    opens = closes * (1 + np.random.uniform(-0.002, 0.002, n))
    highs = np.maximum(opens, closes) * (1 + np.random.uniform(0.001, 0.005, n))
    lows = np.minimum(opens, closes) * (1 - np.random.uniform(0.001, 0.005, n))

    base_ts = datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    timestamps = [base_ts + timedelta(minutes=15 * i) for i in range(n)]

    return pd.DataFrame(
        {
            "ticker": [ticker] * n,
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.random.randint(100_000, 2_000_000, n).tolist(),
        }
    )


def _make_feature_df(n: int = 100) -> pd.DataFrame:
    """
    Generate a synthetic feature matrix with n rows.

    Column values are scaled to realistic ranges for each feature.
    """
    from marketpulse.ml.features import FEATURE_COLS

    np.random.seed(42)
    df = pd.DataFrame(
        np.random.randn(n, len(FEATURE_COLS)),
        columns=FEATURE_COLS,
    )
    # Scale features to realistic ranges
    df["rsi_14"] = 50.0 + df["rsi_14"] * 15.0  # roughly 20–80
    df["bb_position"] = df["bb_position"].abs() % 1.0  # 0–1
    return df


# ── Data fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    """50-row synthetic OHLCV DataFrame. Sufficient for SMA-50 but not SMA-200."""
    return _make_ohlcv_df(n=50)


@pytest.fixture
def large_ohlcv_df() -> pd.DataFrame:
    """250-row synthetic OHLCV DataFrame. Sufficient for all indicator computations."""
    return _make_ohlcv_df(n=250)


@pytest.fixture
def feature_df() -> pd.DataFrame:
    """100-row synthetic feature matrix for ML tests."""
    return _make_feature_df(n=100)


@pytest.fixture
def feature_labels(feature_df: pd.DataFrame) -> pd.Series:
    """Synthetic labels aligned to feature_df."""
    np.random.seed(99)
    return pd.Series(
        np.random.choice([-1, 0, 1], size=len(feature_df), p=[0.3, 0.4, 0.3]),
    )


@pytest.fixture
def raw_ohlcv_rows():
    """Two validated RawOHLCVRow objects for ingestion/ETL tests."""
    from marketpulse.ingestion.schemas import RawOHLCVRow

    base = datetime(2024, 1, 15, 14, 0, tzinfo=UTC)
    return [
        RawOHLCVRow(
            ticker="AAPL",
            timestamp=base,
            open=182.10,
            high=183.45,
            low=181.90,
            close=183.00,
            volume=1_234_567,
        ),
        RawOHLCVRow(
            ticker="AAPL",
            timestamp=base + timedelta(minutes=15),
            open=183.00,
            high=183.80,
            low=182.50,
            close=183.50,
            volume=987_654,
        ),
    ]


# ── Mock service fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def mock_db():
    """
    MagicMock database session.

    Configured for 'no data' by default:
    - .all() returns [] (empty list)
    - .first() returns None

    Tests that need data configure their own query chains.
    """
    session = MagicMock()
    # Common query chain: .query(Model).filter(...).order_by(...).limit(...).all()
    (
        session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value
    ) = []
    # .query(Model).filter(...).order_by(...).first()
    (
        session.query.return_value.filter.return_value.order_by.return_value.first.return_value
    ) = None
    return session


@pytest.fixture
def mock_redis():
    """
    MagicMock RedisClient.

    Configured for 'cache miss' by default (get_json returns None).
    Tests that want a cache hit set mock_redis.get_json.return_value themselves.
    """
    client = MagicMock()
    client.ping.return_value = True  # Redis is reachable
    client.get_json.return_value = None  # cache miss by default
    client.set_json.return_value = True  # writes succeed
    client.get.return_value = None
    client.set.return_value = True
    client.delete.return_value = 1
    client.delete_pattern.return_value = 2
    return client


# ── FastAPI TestClient fixture ─────────────────────────────────────────────────


@pytest.fixture
def api_client(mock_db, mock_redis):
    """
    FastAPI TestClient with mocked DB, Redis, and ML service dependencies.

    Creates a minimal FastAPI app (no scheduler lifespan) with all four routers.
    Uses dependency_overrides to inject mock_db and mock_redis into routes.

    Usage:
        def test_health(api_client):
            response = api_client.get("/api/v1/health")
            assert response.status_code == 200
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from marketpulse.api.dependencies import get_db, get_ml_service, get_redis
    from marketpulse.api.middleware import setup_middleware
    from marketpulse.api.routers import health, news, signals, stocks

    # Minimal app — no scheduler lifespan (prevents background jobs in tests)
    test_app = FastAPI(title="MarketPulse Test API")
    setup_middleware(test_app)
    test_app.include_router(health.router, prefix="/api/v1")
    test_app.include_router(stocks.router, prefix="/api/v1")
    test_app.include_router(signals.router, prefix="/api/v1")
    test_app.include_router(news.router, prefix="/api/v1")

    # Inject mock dependencies — replaces real DB/Redis/ML with mocks
    captured_db = mock_db
    captured_redis = mock_redis

    test_app.dependency_overrides = {
        get_db: lambda: captured_db,
        get_redis: lambda: captured_redis,
        get_ml_service: lambda: MagicMock(),
    }

    with TestClient(test_app, raise_server_exceptions=False) as client:
        yield client
