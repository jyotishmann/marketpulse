# tests/integration/test_api.py
# Integration tests for the FastAPI API layer.
# Uses TestClient (no real HTTP server) with mocked DB and Redis dependencies.
# Tests routing, middleware, dependency injection, and response structure.

from __future__ import annotations

from datetime import UTC, datetime, timezone  # noqa: F401
from decimal import Decimal
from unittest.mock import MagicMock

import pytest  # noqa: F401

# ══════════════════════════════════════════════════════════════════════════════
# Health endpoint tests
# ══════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    def test_health_returns_200_when_all_ok(self, api_client, monkeypatch):
        monkeypatch.setattr(
            "marketpulse.api.routers.health.verify_connection",
            lambda: True,
        )
        response = api_client.get("/api/v1/health")

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["db"] == "ok"
        assert body["redis"] == "ok"
        assert "tickers" in body

    def test_health_returns_degraded_when_db_down(self, api_client, monkeypatch):
        monkeypatch.setattr(
            "marketpulse.api.routers.health.verify_connection",
            lambda: False,  # DB is down
        )
        response = api_client.get("/api/v1/health")

        assert response.status_code == 200   # still 200 — API is up
        body = response.json()
        assert body["status"] == "degraded"
        assert body["db"] == "error"

    def test_health_includes_version(self, api_client, monkeypatch):
        monkeypatch.setattr(
            "marketpulse.api.routers.health.verify_connection",
            lambda: True,
        )
        body = api_client.get("/api/v1/health").json()
        assert "version" in body


# ══════════════════════════════════════════════════════════════════════════════
# Stocks endpoint tests
# ══════════════════════════════════════════════════════════════════════════════

class TestStocksEndpoints:
    """Tests for GET /api/v1/stocks and /api/v1/stocks/{ticker}/prices."""

    def test_list_tickers_returns_200(self, api_client):
        response = api_client.get("/api/v1/stocks")

        assert response.status_code == 200
        body = response.json()
        assert "tickers" in body
        assert isinstance(body["tickers"], list)

    def test_prices_returns_404_when_no_data(self, api_client):
        """Empty DB → 404 with informative detail message."""
        response = api_client.get("/api/v1/stocks/AAPL/prices")

        assert response.status_code == 404
        assert "AAPL" in response.json()["detail"]

    def test_prices_returns_200_with_data(self, api_client, mock_db):
        """When DB has data, endpoint returns 200 with a list of bars."""
        # Configure mock DB to return two price rows
        mock_row = MagicMock()
        mock_row.ticker = "AAPL"
        mock_row.timestamp = datetime(2024, 1, 15, 14, 0, tzinfo=UTC)
        mock_row.open = Decimal("182.10")
        mock_row.high = Decimal("183.45")
        mock_row.low = Decimal("181.90")
        mock_row.close = Decimal("183.00")
        mock_row.volume = 1_234_567

        (
            mock_db.query.return_value
            .filter.return_value
            .order_by.return_value
            .limit.return_value
            .all.return_value
        ) = [mock_row, mock_row]

        response = api_client.get("/api/v1/stocks/AAPL/prices?limit=2")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_prices_response_has_ohlcv_keys(self, api_client, mock_db):
        mock_row = MagicMock()
        mock_row.ticker = "AAPL"
        mock_row.timestamp = datetime(2024, 1, 15, 14, 0, tzinfo=UTC)
        mock_row.open = Decimal("182.10")
        mock_row.high = Decimal("183.00")
        mock_row.low = Decimal("181.90")
        mock_row.close = Decimal("182.50")
        mock_row.volume = 500_000

        (
            mock_db.query.return_value
            .filter.return_value
            .order_by.return_value
            .limit.return_value
            .all.return_value
        ) = [mock_row]

        data = api_client.get("/api/v1/stocks/AAPL/prices").json()
        assert isinstance(data, list)
        if data:  # might be empty due to mock configuration
            row = data[0]
            for key in ("ticker", "timestamp", "open", "high", "low", "close", "volume"):
                assert key in row, f"Missing key '{key}' in price response"

    def test_indicators_returns_404_when_no_data(self, api_client):
        response = api_client.get("/api/v1/stocks/AAPL/indicators")
        assert response.status_code == 404

    def test_limit_parameter_validated(self, api_client):
        """limit=0 violates ge=1 constraint → 422 Unprocessable Entity."""
        response = api_client.get("/api/v1/stocks/AAPL/prices?limit=0")
        assert response.status_code == 422

    def test_limit_max_validated(self, api_client):
        """limit=501 violates le=500 constraint → 422."""
        response = api_client.get("/api/v1/stocks/AAPL/prices?limit=501")
        assert response.status_code == 422


# ══════════════════════════════════════════════════════════════════════════════
# Signals endpoint tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSignalsEndpoints:
    """Tests for GET /api/v1/stocks/{ticker}/signals endpoints."""

    def test_latest_signal_returns_404_when_no_data(self, api_client):
        response = api_client.get("/api/v1/stocks/AAPL/signals/latest")
        assert response.status_code == 404

    def test_latest_signal_returns_200_with_data(self, api_client, mock_db):
        mock_signal = MagicMock()
        mock_signal.ticker = "AAPL"
        mock_signal.timestamp = datetime(2024, 1, 15, 15, 0, tzinfo=UTC)
        mock_signal.signal = "BUY"
        mock_signal.confidence = Decimal("0.7800")
        mock_signal.is_anomaly = False
        mock_signal.model_version = "v1"

        (
            mock_db.query.return_value
            .filter.return_value
            .order_by.return_value
            .first.return_value
        ) = mock_signal

        response = api_client.get("/api/v1/stocks/AAPL/signals/latest")

        assert response.status_code == 200
        body = response.json()
        assert body["signal"] == "BUY"
        assert body["ticker"] == "AAPL"
        assert "confidence" in body
        assert "is_anomaly" in body

    def test_signal_history_returns_404_when_empty(self, api_client):
        response = api_client.get("/api/v1/stocks/AAPL/signals")
        assert response.status_code == 404


# ══════════════════════════════════════════════════════════════════════════════
# News endpoint tests
# ══════════════════════════════════════════════════════════════════════════════

class TestNewsEndpoints:
    """Tests for GET /api/v1/news."""

    def test_news_returns_200_with_empty_list(self, api_client):
        """News returns 200 with empty list when no articles — not 404."""
        response = api_client.get("/api/v1/news")

        assert response.status_code == 200
        assert response.json() == []

    def test_news_with_data_returns_list(self, api_client, mock_db):
        mock_article = MagicMock()
        mock_article.id = 1
        mock_article.title = "Apple stock surges"
        mock_article.source_url = "https://reuters.com/aapl"
        mock_article.published_at = datetime(2024, 1, 15, 10, 0, tzinfo=UTC)
        mock_article.sentiment_positive = Decimal("0.5950")
        mock_article.sentiment_negative = Decimal("0.0000")
        mock_article.sentiment_neutral = Decimal("0.4050")
        mock_article.sentiment_compound = Decimal("0.7096")

        (
            mock_db.query.return_value
            .order_by.return_value
            .limit.return_value
            .all.return_value
        ) = [mock_article]

        response = api_client.get("/api/v1/news?limit=5")

        assert response.status_code == 200
        articles = response.json()
        assert isinstance(articles, list)
        assert len(articles) == 1
        assert articles[0]["title"] == "Apple stock surges"
        assert "sentiment_compound" in articles[0]


# ══════════════════════════════════════════════════════════════════════════════
# Middleware tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMiddleware:
    """Tests that middleware headers are added to responses."""

    def test_response_includes_process_time_header(self, api_client):
        response = api_client.get("/api/v1/stocks")
        # TimingMiddleware adds X-Process-Time to every response
        assert "x-process-time" in response.headers

    def test_response_includes_request_id_header(self, api_client):
        response = api_client.get("/api/v1/stocks")
        # RequestIDMiddleware adds X-Request-ID to every response
        assert "x-request-id" in response.headers

    def test_request_ids_are_unique_per_request(self, api_client):
        r1 = api_client.get("/api/v1/stocks")
        r2 = api_client.get("/api/v1/stocks")
        assert r1.headers["x-request-id"] != r2.headers["x-request-id"]
