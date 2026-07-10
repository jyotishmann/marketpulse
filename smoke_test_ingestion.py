# Run in an interactive Python session: python -i
# Or paste into a script and run: python smoke_test_ingestion.py

# Load .env file for local smoke testing
from dotenv import load_dotenv

load_dotenv()

# ────────────────────────────────────────────────────────────────────────────
# SMOKE TEST 1: Config layer is healthy
# ────────────────────────────────────────────────────────────────────────────
from marketpulse.config import settings

print("=== Config ===")
print("Ticker list:   ", settings.ticker_list)
print("Database URL:  ", settings.database_url[:40] + "...")
print("Redis URL:     ", settings.redis_url)
print("RSS URLs:      ", settings.rss_url_list[:1], "... (first)")

# ────────────────────────────────────────────────────────────────────────────
# SMOKE TEST 2: yfinance stock connector
# ────────────────────────────────────────────────────────────────────────────
from marketpulse.ingestion import fetch_ohlcv, RawOHLCVRow

print("\n=== Stock Connector ===")
bars = fetch_ohlcv("AAPL", period="1d", interval="1h")
print(f"Fetched {len(bars)} bars for AAPL")
assert isinstance(bars, list), "Expected list"
if bars:
    b = bars[0]
    print(f"First bar: {b.ticker} @ {b.timestamp} → close={b.close}")
    print(f"  open={b.open} high={b.high} low={b.low} vol={b.volume}")
    assert b.ticker == "AAPL", "ticker should be AAPL"
    assert b.high >= b.low, "high must be >= low"
    assert b.close > 0, "close must be positive"
    print("  ✓ All assertions pass")

# ────────────────────────────────────────────────────────────────────────────
# SMOKE TEST 3: Pydantic validation — valid row
# ────────────────────────────────────────────────────────────────────────────
from datetime import UTC, datetime

print("\n=== Schema Validation (valid row) ===")
valid_row = RawOHLCVRow(
    ticker="  msft  ",          # intentionally mixed case + spaces
    timestamp=datetime.now(tz=UTC),
    open=390.10,
    high=392.45,
    low=389.80,
    close=391.75,
    volume=1_234_567,
    extra_field="ignored",     # extra="ignore" should silently drop this
)
print(f"Ticker normalised to: {valid_row.ticker!r}")  # should be "MSFT"
assert valid_row.ticker == "MSFT", "ticker should be uppercased and stripped"
assert valid_row.open == round(390.10, 4)
print("  ✓ Normalisation and validation pass")

# ────────────────────────────────────────────────────────────────────────────
# SMOKE TEST 4: Pydantic validation — invalid row (high < low)
# ────────────────────────────────────────────────────────────────────────────
from pydantic import ValidationError

print("\n=== Schema Validation (invalid row — high < low) ===")
try:
    bad_row = RawOHLCVRow(
        ticker="AAPL",
        timestamp=datetime.now(tz=UTC),
        open=180.00,
        high=178.00,    # ← impossible: high less than low
        low=181.00,
        close=179.00,
        volume=100,
    )
    print("  ✗ ERROR: ValidationError was not raised!")
except ValidationError as exc:
    errors = exc.errors()
    print(f"  ✓ ValidationError raised with {len(errors)} error(s)")
    print(f"  Error: {errors[0]['msg']}")

# ────────────────────────────────────────────────────────────────────────────
# SMOKE TEST 5: Pydantic validation — NaN price
# ────────────────────────────────────────────────────────────────────────────
import math

print("\n=== Schema Validation (NaN price) ===")
try:
    nan_row = RawOHLCVRow(
        ticker="AAPL",
        timestamp=datetime.now(tz=UTC),
        open=math.nan,  # ← NaN should be rejected
        high=182.00,
        low=180.00,
        close=181.00,
        volume=500,
    )
    print("  ✗ ERROR: ValidationError was not raised!")
except ValidationError as exc:
    print(f"  ✓ ValidationError raised: {exc.errors()[0]['msg']}")

# ────────────────────────────────────────────────────────────────────────────
# SMOKE TEST 6: RSS news connector (requires internet)
# ────────────────────────────────────────────────────────────────────────────
from marketpulse.ingestion import fetch_feed, fetch_all_feeds, RawNewsItem

print("\n=== News Connector ===")
yahoo_aapl_rss = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline"
    "?s=AAPL&region=US&lang=en-US"
)
items = fetch_feed(yahoo_aapl_rss)
print(f"Fetched {len(items)} items from Yahoo Finance AAPL feed")

if items:
    item = items[0]
    print(f"Latest item: {item.title[:60]!r}")
    print(f"  URL:      {item.source_url[:60]}...")
    print(f"  Published: {item.published_at}")
    assert isinstance(item, RawNewsItem)
    assert item.source_url.startswith("http")
    assert item.published_at.tzinfo is not None, "datetime must be timezone-aware"
    print("  ✓ All assertions pass")

# ────────────────────────────────────────────────────────────────────────────
# SMOKE TEST 7: Deduplication in fetch_all_feeds
# ────────────────────────────────────────────────────────────────────────────
print("\n=== Deduplication ===")
# Feed the same URL twice — we should get N items, not 2×N
deduped = fetch_all_feeds([yahoo_aapl_rss, yahoo_aapl_rss])
single = fetch_feed(yahoo_aapl_rss)
assert len(deduped) == len(single), (
    f"Deduplication failed: {len(deduped)} != {len(single)}"
)
print(f"  ✓ Deduplication works: {len(deduped)} unique items "
      f"(same as single feed call: {len(single)})")

print("\n=== All smoke tests passed ✓ ===")
