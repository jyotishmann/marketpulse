# tests/unit/test_scheduler.py
from unittest.mock import MagicMock, patch  # noqa: F401

import pytest  # noqa: F401


def test_create_scheduler_has_four_jobs():
    from marketpulse.scheduler.jobs import create_scheduler

    scheduler = create_scheduler()
    job_ids = {job.id for job in scheduler.get_jobs()}
    assert {
        "warmup_cache",
        "ingest_stock_data",
        "ingest_news",
        "run_ml_pipeline",
    } == job_ids


def test_get_prediction_service_returns_singleton():
    from marketpulse.scheduler.jobs import get_prediction_service

    s1 = get_prediction_service()
    s2 = get_prediction_service()
    assert s1 is s2


def test_ingest_news_skips_when_no_urls():
    with (
        patch("marketpulse.scheduler.jobs.settings") as mock_settings,
        patch("marketpulse.scheduler.jobs.fetch_all_feeds") as mock_fetch,
    ):
        mock_settings.rss_url_list = []  # ← patch directly
        from marketpulse.scheduler.jobs import ingest_news

        ingest_news()
        mock_fetch.assert_not_called()


def test_ingest_stock_data_handles_empty_response():
    with (
        patch("marketpulse.scheduler.jobs.settings") as mock_settings,
        patch("marketpulse.scheduler.jobs.fetch_ohlcv", return_value=[]) as mock_fetch,
        patch("marketpulse.scheduler.jobs.SessionLocal"),
    ):
        mock_settings.ticker_list = ["AAPL"]  # ← only 1 ticker
        from marketpulse.scheduler.jobs import ingest_stock_data

        ingest_stock_data()
        mock_fetch.assert_called_once()


def test_warmup_cache_skips_when_db_unavailable(monkeypatch):
    with (
        patch("marketpulse.scheduler.jobs.verify_connection", return_value=False),
        patch("marketpulse.scheduler.jobs.ingest_stock_data") as mock_ingest,
    ):
        from marketpulse.scheduler.jobs import warmup_cache

        warmup_cache()
        mock_ingest.assert_not_called()
