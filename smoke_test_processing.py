# Run with: python -i  (or paste into smoke_test_processing.py and run it)
# Load .env file for local smoke testing
from dotenv import load_dotenv

load_dotenv()

import nltk  # noqa: E402

nltk.download('vader_lexicon', quiet=True)  # no-op if already downloaded

from datetime import UTC, datetime, timedelta, timezone  # noqa: E402, F401

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ────────────────────────────────────────────────────────────────────────────
# SECTION A: Test compute_all() with synthetic data (no database needed)
# ────────────────────────────────────────────────────────────────────────────
print("=== SECTION A: compute_all() with synthetic data ===\n")

from marketpulse.processing import INDICATOR_COLS, compute_all  # noqa: E402

# Generate 250 synthetic OHLCV bars (random walk)
np.random.seed(42)
n = 250
base_price = 180.0
changes = np.random.normal(0, 0.5, n)       # daily returns ~N(0, 0.5)
closes = base_price + np.cumsum(changes)    # random walk

# Build DataFrame with realistic OHLCV structure
timestamps = [
    datetime(2024, 1, 2, 9, 30, tzinfo=UTC) + timedelta(hours=i)
    for i in range(n)
]
synthetic_df = pd.DataFrame({
    "ticker": ["AAPL"] * n,
    "timestamp": timestamps,
    "open": closes * (1 + np.random.uniform(-0.001, 0.001, n)),
    "high": closes * (1 + np.random.uniform(0.001, 0.003, n)),
    "low": closes * (1 - np.random.uniform(0.001, 0.003, n)),
    "close": closes,
    "volume": np.random.randint(100_000, 2_000_000, n),
})
print(f"Input: {len(synthetic_df)} rows of synthetic AAPL data")

# Run compute_all()
result_df = compute_all(synthetic_df)
print(f"Output: {len(result_df)} rows with columns: {list(result_df.columns)}\n")

# ── Check 1: All indicator columns present ────────────────────────────────────
for col in INDICATOR_COLS:
    assert col in result_df.columns, f"Missing column: {col}"
print("✓ All 10 indicator columns present")

# ── Check 2: NaN pattern is correct ──────────────────────────────────────────
first_sma20_idx = result_df["sma_20"].first_valid_index()
first_sma200_idx = result_df["sma_200"].first_valid_index()
print(f"✓ sma_20 first valid index: {first_sma20_idx} (expected 19)")
print(f"✓ sma_200 first valid index: {first_sma200_idx} (expected 199)")
assert first_sma20_idx == 19, f"Expected 19, got {first_sma20_idx}"
assert first_sma200_idx == 199, f"Expected 199, got {first_sma200_idx}"

# ── Check 3: RSI is always within [0, 100] ────────────────────────────────────
rsi_values = result_df["rsi_14"].dropna()
assert (rsi_values >= 0).all() and (rsi_values <= 100).all(), "RSI out of [0,100]"
print(f"✓ RSI range: [{rsi_values.min():.2f}, {rsi_values.max():.2f}] — within [0, 100]")

# ── Check 4: Bollinger Bands straddle SMA-20 ─────────────────────────────────
valid = result_df.dropna(subset=["sma_20", "bb_upper", "bb_lower"])
assert (valid["bb_upper"] >= valid["sma_20"]).all(), "bb_upper < sma_20!"
assert (valid["bb_lower"] <= valid["sma_20"]).all(), "bb_lower > sma_20!"
print("✓ Bollinger Bands: bb_upper >= sma_20 >= bb_lower for all valid rows")

# ── Check 5: Input DataFrame is NOT mutated ───────────────────────────────────
assert "sma_20" not in synthetic_df.columns, "compute_all() mutated the input!"
print("✓ Input DataFrame not mutated (compute_all returns a copy)")

# ── Print last 5 rows of indicators ──────────────────────────────────────────
print("\nLast 5 indicator rows:")
print(result_df[["timestamp", "close", "sma_20", "rsi_14", "macd", "bb_upper", "bb_lower"]].tail())

# ────────────────────────────────────────────────────────────────────────────
# SECTION B: Test score_sentiment()
# ────────────────────────────────────────────────────────────────────────────
print("\n=== SECTION B: score_sentiment() ===\n")

from marketpulse.processing import score_sentiment  # noqa: E402

headlines = [
    ("Apple reports record earnings, stock surges!", "expect positive"),
    ("Market crash fears grow as recession looms", "expect negative"),
    ("The stock closed at 182.30 on Tuesday",       "expect neutral"),
]

for text, expectation in headlines:
    scores = score_sentiment(text)
    print(f"Text: {text[:55]!r}")
    print(f"  compound={scores['compound']:+.4f}  pos={scores['pos']:.4f}  "
          f"neg={scores['neg']:.4f}  → {expectation}")
    assert "compound" in scores
    assert "pos" in scores and "neg" in scores and "neu" in scores
    assert -1.0 <= scores["compound"] <= 1.0

print("\n✓ All sentiment scores in [-1.0, 1.0]")

# ────────────────────────────────────────────────────────────────────────────
# SECTION C: Full pipeline (requires docker compose up -d)
# Comment out if not running Docker
# ────────────────────────────────────────────────────────────────────────────
print("\n=== SECTION C: Full pipeline with database (requires Docker) ===\n")

import os  # noqa: E402

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql://marketpulse:marketpulse@localhost:5432/marketpulse"
)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

from marketpulse.db import SessionLocal, verify_connection  # noqa: E402

if not verify_connection():
    print("⚠  Database not reachable — skipping Section C")
    print("   Run: docker compose up -d && sleep 5 && alembic upgrade head")
else:
    print("✓ Database connection OK")

    from marketpulse.ingestion import fetch_ohlcv
    from marketpulse.processing import (
        compute_all,
        rows_to_ohlcv_df,
        upsert_indicators,
        upsert_prices,
    )

    # 1. Fetch fresh data
    bars = fetch_ohlcv("AAPL", period="5d", interval="1h")
    print(f"Fetched {len(bars)} bars from yfinance")

    # 2. Convert to DataFrame
    df = rows_to_ohlcv_df(bars)
    print(f"Cleaned DataFrame: {len(df)} rows")

    # 3. Write prices to DB
    with SessionLocal() as session:
        inserted = upsert_prices(bars, session)
        print(f"Upserted {inserted} price rows")

    # 4. Compute indicators (limited by only 5 days — sma_200 will be NaN)
    enriched = compute_all(df)
    sma20_count = enriched["sma_20"].notna().sum()
    print(f"Indicators computed: {sma20_count}/{len(enriched)} rows have sma_20")

    # 5. Write indicators to DB
    with SessionLocal() as session:
        ind_count = upsert_indicators(enriched, session)
        print(f"Upserted {ind_count} indicator rows")

    print("\n✓ Full pipeline completed successfully")

print("\n=== All smoke tests passed ✓ ===")
