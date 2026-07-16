# tests/unit/test_ingestion.py
# Unit tests for the ingestion layer:
# - RawOHLCVRow schema validation
# - RawNewsItem schema validation
# - fetch_ohlcv() with mocked yfinance
# - fetch_all_feeds() deduplication with mocked feedparser

from __future__ import annotations

import math
from datetime import UTC, datetime, timezone  # noqa: F401
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import ValidationError

from marketpulse.ingestion.schemas import RawNewsItem, RawOHLCVRow

# ══════════════════════════════════════════════════════════════════════════════
# RawOHLCVRow schema tests
# ══════════════════════════════════════════════════════════════════════════════


class TestRawOHLCVRow:
    """Tests for the OHLCV price bar pydantic schema."""

    BASE_VALID = {
        "ticker": "AAPL",
        "timestamp": datetime(2024, 1, 15, 14, 0, tzinfo=UTC),
        "open": 182.10,
        "high": 183.45,
        "low": 181.90,
        "close": 183.00,
        "volume": 1_234_567,
    }

    def test_valid_row_accepted(self):
        row = RawOHLCVRow(**self.BASE_VALID)
        assert row.ticker == "AAPL"
        assert row.close == pytest.approx(183.00, abs=0.01)
        assert row.volume == 1_234_567

    def test_ticker_normalised_to_uppercase(self):
        """Ticker should be stripped and uppercased regardless of input case."""
        row = RawOHLCVRow(**{**self.BASE_VALID, "ticker": "  aapl  "})
        assert row.ticker == "AAPL"

    def test_prices_rounded_to_four_decimal_places(self):
        row = RawOHLCVRow(**{**self.BASE_VALID, "close": 182.300000001})
        assert row.close == pytest.approx(182.3, abs=0.001)

    def test_extra_fields_ignored(self):
        """extra='ignore' — unknown fields like 'dividends' are silently dropped."""
        row = RawOHLCVRow(**self.BASE_VALID, dividends=0.0, stock_splits=0.0)
        assert not hasattr(row, "dividends")

    def test_negative_close_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            RawOHLCVRow(**{**self.BASE_VALID, "close": -1.0})
        errors = exc_info.value.errors()
        assert any("close" in str(e["loc"]) for e in errors)

    def test_nan_price_raises(self):
        with pytest.raises(ValidationError):
            RawOHLCVRow(**{**self.BASE_VALID, "open": math.nan})

    def test_high_less_than_low_raises(self):
        """Candlestick constraint: high must be >= low."""
        with pytest.raises(ValidationError) as exc_info:
            RawOHLCVRow(
                **{
                    **self.BASE_VALID,
                    "high": 180.00,  # lower than low=181.90
                    "low": 181.90,
                }
            )
        # Should be a model validator error
        assert exc_info.value.errors()

    def test_zero_volume_accepted(self):
        """Volume=0 is valid (market halt or off-hours bar)."""
        row = RawOHLCVRow(**{**self.BASE_VALID, "volume": 0})
        assert row.volume == 0

    def test_negative_volume_raises(self):
        with pytest.raises(ValidationError):
            RawOHLCVRow(**{**self.BASE_VALID, "volume": -100})

    def test_empty_ticker_raises(self):
        with pytest.raises(ValidationError):
            RawOHLCVRow(**{**self.BASE_VALID, "ticker": "   "})


# ══════════════════════════════════════════════════════════════════════════════
# RawNewsItem schema tests
# ══════════════════════════════════════════════════════════════════════════════


class TestRawNewsItem:
    """Tests for the news article pydantic schema."""

    BASE_VALID = {
        "title": "Apple reports record earnings",
        "source_url": "https://reuters.com/article/aapl-earnings",
        "published_at": datetime(2024, 1, 15, 14, 30, tzinfo=UTC),
    }

    def test_valid_item_accepted(self):
        item = RawNewsItem(**self.BASE_VALID)
        assert item.title == "Apple reports record earnings"
        assert item.source_url.startswith("https://")

    def test_title_stripped_on_creation(self):
        item = RawNewsItem(**{**self.BASE_VALID, "title": "  Apple earnings up  "})
        assert item.title == "Apple earnings up"

    def test_title_too_short_raises(self):
        """Titles shorter than 5 characters are likely broken feed entries."""
        with pytest.raises(ValidationError):
            RawNewsItem(**{**self.BASE_VALID, "title": "Hi"})

    def test_non_http_url_raises(self):
        """source_url must be an absolute HTTP/HTTPS URL."""
        with pytest.raises(ValidationError):
            RawNewsItem(**{**self.BASE_VALID, "source_url": "ftp://invalid"})

    def test_relative_url_raises(self):
        with pytest.raises(ValidationError):
            RawNewsItem(**{**self.BASE_VALID, "source_url": "/article/123"})

    def test_http_url_accepted(self):
        """http:// (not just https://) should be accepted."""
        item = RawNewsItem(
            **{**self.BASE_VALID, "source_url": "http://example.com/news"}
        )
        assert item.source_url.startswith("http://")


# ══════════════════════════════════════════════════════════════════════════════
# fetch_ohlcv() connector tests (mocked yfinance)
# ══════════════════════════════════════════════════════════════════════════════


class TestFetchOHLCV:
    """Tests for the yfinance OHLCV connector with mocked network calls."""

    def _make_yfinance_df(self, n: int = 5) -> pd.DataFrame:
        """Synthetic DataFrame that mimics yfinance.download() output."""
        import numpy as np  # noqa: F401

        base = datetime(2024, 1, 15, 14, 0, tzinfo=UTC)
        closes = [183.0 + i * 0.1 for i in range(n)]
        return pd.DataFrame(
            {
                "Datetime": [base + pd.Timedelta(minutes=15 * i) for i in range(n)],
                "Open": [c - 0.1 for c in closes],
                "High": [c + 0.2 for c in closes],
                "Low": [c - 0.2 for c in closes],
                "Close": closes,
                "Volume": [1_000_000] * n,
            }
        )

    def test_returns_validated_rows(self):
        """fetch_ohlcv should return a list of RawOHLCVRow objects."""
        from marketpulse.ingestion.stock import fetch_ohlcv

        with patch("yfinance.download") as mock_yf:
            mock_yf.return_value = self._make_yfinance_df(n=5)
            rows = fetch_ohlcv("AAPL", period="1d", interval="15m")

        assert isinstance(rows, list)
        assert len(rows) == 5
        assert all(isinstance(r, RawOHLCVRow) for r in rows)
        assert rows[0].ticker == "AAPL"

    def test_empty_dataframe_returns_empty_list(self):
        """fetch_ohlcv should return [] when yfinance has no data."""
        from marketpulse.ingestion.stock import fetch_ohlcv

        with patch("yfinance.download") as mock_yf:
            mock_yf.return_value = pd.DataFrame()
            rows = fetch_ohlcv("AAPL")

        assert rows == []

    def test_network_error_returns_empty_list(self):
        """fetch_ohlcv should return [] on any exception (fail-safe)."""
        from marketpulse.ingestion.stock import fetch_ohlcv

        with patch("yfinance.download", side_effect=ConnectionError("timeout")):
            rows = fetch_ohlcv("AAPL")

        assert rows == []

    def test_fetch_multiple_tickers_returns_dict(self):
        """fetch_multiple_tickers should return a dict keyed by ticker."""
        from marketpulse.ingestion.stock import fetch_multiple_tickers

        with patch("yfinance.download") as mock_yf:
            mock_yf.return_value = self._make_yfinance_df(n=3)
            result = fetch_multiple_tickers(["AAPL", "GOOGL"])

        assert set(result.keys()) == {"AAPL", "GOOGL"}
        assert len(result["AAPL"]) == 3


# ══════════════════════════════════════════════════════════════════════════════
# fetch_all_feeds() deduplication tests (mocked feedparser)
# ══════════════════════════════════════════════════════════════════════════════


class TestFetchAllFeeds:
    """Tests for the RSS news connector with mocked feedparser."""

    def _make_feed(
        self, n_entries: int = 2, url_prefix: str = "https://news.com"
    ) -> MagicMock:
        """Create a mock feedparser feed with n entries."""
        feed = MagicMock()
        feed.bozo = False
        import time as _time

        feed.entries = [
            MagicMock(
                title=f"Headline {i}",
                link=f"{url_prefix}/article/{i}",
                published_parsed=_time.gmtime(),
            )
            for i in range(n_entries)
        ]
        return feed

    def test_empty_url_list_returns_empty(self):
        """No configured feeds → return [] without making network calls."""
        from marketpulse.ingestion.news import fetch_all_feeds

        with patch("feedparser.parse") as mock_fp:
            result = fetch_all_feeds([])

        assert result == []
        mock_fp.assert_not_called()

    def test_deduplicates_same_url_across_feeds(self):
        from marketpulse.ingestion.news import fetch_all_feeds

        def make_feed():
            feed = MagicMock()
            feed.bozo = False
            entries = []
            for i in range(3):
                entry = MagicMock()
                entry.title = f"Headline {i}"
                entry.link = f"https://news.com/article/{i}"
                entry.get = lambda k, d="", i=i: {
                    "title": f"Headline {i}",
                    "link": f"https://news.com/article/{i}",
                }.get(k, d)
                entry.published_parsed = None
                entry.published = None
                entries.append(entry)
            feed.entries = entries
            return feed

        with patch("feedparser.parse", return_value=make_feed()):
            result = fetch_all_feeds(["https://feed1.com", "https://feed1.com"])

        assert len(result) == 3

    def test_sorted_newest_first(self):
        import time as _time

        from marketpulse.ingestion.news import fetch_all_feeds

        older_time = _time.gmtime(0)  # epoch — Jan 1 1970
        newer_time = _time.gmtime()  # now

        feed = MagicMock()
        feed.bozo = False

        old_entry = MagicMock()
        old_entry.published_parsed = older_time
        old_entry.get = lambda k, d="": {
            "title": "Old article here",
            "link": "https://news.com/old",
            "published_parsed": older_time,  # ← ADD THIS
        }.get(k, d)

        new_entry = MagicMock()
        new_entry.published_parsed = newer_time
        new_entry.get = lambda k, d="": {
            "title": "New article here",
            "link": "https://news.com/new",
            "published_parsed": newer_time,  # ← ADD THIS
        }.get(k, d)

        feed.entries = [old_entry, new_entry]

        with patch("feedparser.parse", return_value=feed):
            result = fetch_all_feeds(["https://news.com/rss"])

        assert len(result) == 2
        assert result[0].title == "New article here"


def test_fetch_feed_returns_empty_on_bozo_no_entries():
    from unittest.mock import MagicMock, patch

    from marketpulse.ingestion.news import fetch_feed

    mock_feed = MagicMock()
    mock_feed.bozo = True
    mock_feed.entries = []
    mock_feed.bozo_exception = Exception("bad xml")
    with patch("feedparser.parse", return_value=mock_feed):
        result = fetch_feed("https://bad-feed.com/rss")
    assert result == []


def test_fetch_feed_continues_on_bozo_with_entries():
    import time as _time
    from unittest.mock import MagicMock, patch

    from marketpulse.ingestion.news import fetch_feed

    mock_feed = MagicMock()
    mock_feed.bozo = True
    mock_feed.bozo_exception = Exception("minor xml issue")
    entry = MagicMock()
    entry.get = lambda k, d="": {
        "title": "Valid headline here",
        "link": "https://news.com/1",
    }.get(k, d)
    entry.published_parsed = _time.gmtime()
    mock_feed.entries = [entry]
    with patch("feedparser.parse", return_value=mock_feed):
        result = fetch_feed("https://slightly-bad-feed.com/rss")
    assert len(result) == 1
