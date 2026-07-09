# marketpulse/ingestion/__init__.py
# Public API of the ingestion package.
# Import from marketpulse.ingestion, not from the internal submodules.

from marketpulse.ingestion.news import fetch_all_feeds, fetch_feed
from marketpulse.ingestion.schemas import RawNewsItem, RawOHLCVRow
from marketpulse.ingestion.stock import fetch_multiple_tickers, fetch_ohlcv

__all__ = [
    # Schemas
    "RawOHLCVRow",
    "RawNewsItem",
    # Stock connector
    "fetch_ohlcv",
    "fetch_multiple_tickers",
    # News connector
    "fetch_feed",
    "fetch_all_feeds",
]
