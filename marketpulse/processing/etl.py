# marketpulse/processing/etl.py
# ETL layer: convert, clean, score, and persist ingested data.
# This module is the only layer that writes to the database.

from __future__ import annotations

import logging
from decimal import Decimal

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from marketpulse.db import NewsArticle, StockPrice, TechnicalIndicator  # noqa: F401
from marketpulse.ingestion.schemas import RawNewsItem, RawOHLCVRow

logger = logging.getLogger(__name__)

# ── VADER sentiment analyser — module-level singleton ─────────────────────────
# Reads ~1 MB lexicon from disk once on import. Thread-safe for read-only use.
# polarity_scores() is O(n_words) and takes < 1ms per headline.
_vader = SentimentIntensityAnalyzer()


# ══════════════════════════════════════════════════════════════════════════════
# Section 1: OHLCV DataFrame utilities
# ══════════════════════════════════════════════════════════════════════════════

def rows_to_ohlcv_df(rows: list[RawOHLCVRow]) -> pd.DataFrame:
    """
    Convert a list of validated RawOHLCVRow objects to a cleaned pandas DataFrame.

    Performs:
    - model_dump() conversion from pydantic → dict → DataFrame
    - Timestamp normalisation to UTC-aware pandas Timestamps
    - Sort ascending by timestamp (required for rolling-window indicators)
    - Deduplication by (ticker, timestamp) — keep first occurrence
    - Forward-fill of isolated NaN values in price columns (max 2 consecutive)

    Args:
        rows: Validated OHLCV rows from the ingestion layer.

    Returns:
        Cleaned DataFrame with columns: ticker, timestamp, open, high,
        low, close, volume. Empty DataFrame if rows is empty.
    """
    if not rows:
        return pd.DataFrame()

    # pydantic v2: .model_dump() → plain Python dict (replaces .dict() from v1)
    df = pd.DataFrame([row.model_dump() for row in rows])

    # Normalise timestamp to UTC-aware — pandas may strip tz on DataFrame construction
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Sort ascending — rolling windows require chronological order
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Deduplicate: the pipeline may fetch overlapping periods on consecutive runs
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "timestamp"], keep="first")
    dropped = before - len(df)
    if dropped:
        logger.debug("Removed %d duplicate rows from OHLCV DataFrame", dropped)

    # Forward-fill isolated NaN values (brief market halts, API glitches)
    # limit=2: fill at most 2 consecutive missing bars — do not fill weekends
    price_cols = ["open", "high", "low", "close"]
    df[price_cols] = df[price_cols].ffill(limit=2)

    logger.debug("rows_to_ohlcv_df: %d rows after cleaning", len(df))
    return df

# ══════════════════════════════════════════════════════════════════════════════
# Section 2: Database read/write for OHLCV data
# ══════════════════════════════════════════════════════════════════════════════

def load_recent_prices(
    ticker: str,
    session: Session,
    lookback: int = 220,
) -> pd.DataFrame:
    """
    Load the most recent N price rows for a ticker from PostgreSQL.

    Used by the scheduler to provide enough historical context for
    SMA-200 computation (requires at least 200 bars).

    Args:
        ticker:   Stock symbol.
        session:  Active SQLAlchemy session.
        lookback: Maximum number of rows to load. Default 220 gives
                  comfortable buffer above SMA-200's 200-bar requirement.

    Returns:
        DataFrame sorted ascending by timestamp. Empty DataFrame if no
        rows exist yet (first run for this ticker).
    """
    rows = (
        session.query(StockPrice)
        .filter(StockPrice.ticker == ticker)
        .order_by(StockPrice.timestamp.desc())
        .limit(lookback)
        .all()
    )

    if not rows:
        logger.info("No existing price rows for %s (first ingest run?)", ticker)
        return pd.DataFrame()

    records = [
        {
            "ticker": r.ticker,
            "timestamp": r.timestamp,
            "open": float(r.open),
            "high": float(r.high),
            "low": float(r.low),
            "close": float(r.close),
            "volume": int(r.volume),
        }
        for r in rows
    ]

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # DB query returned DESC order (newest first) — reverse to ASC for indicators
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.debug("Loaded %d historical rows for %s from DB", len(df), ticker)
    return df


def upsert_prices(rows: list[RawOHLCVRow], session: Session) -> int:
    """
    Upsert validated OHLCV rows into the stock_prices table.

    Converts float prices to Decimal (via str) for exact representation.
    ON CONFLICT DO NOTHING: rows with the same (ticker, timestamp) are skipped.

    Args:
        rows:    Validated rows from fetch_ohlcv().
        session: Active SQLAlchemy session.

    Returns:
        Number of rows passed to the upsert statement (includes both
        inserted and skipped — PostgreSQL doesn't report skipped count).
    """
    if not rows:
        return 0

    records = [
        {
            "ticker": row.ticker,
            "timestamp": row.timestamp,
            # Decimal(str(float)) avoids floating-point representation errors
            # e.g. str(182.3) = "182.3" → Decimal("182.3") = exact 182.3
            "open": Decimal(str(row.open)),
            "high": Decimal(str(row.high)),
            "low": Decimal(str(row.low)),
            "close": Decimal(str(row.close)),
            "volume": row.volume,
        }
        for row in rows
    ]

    # pg_insert: PostgreSQL-specific INSERT supporting ON CONFLICT clauses
    stmt = pg_insert(StockPrice).values(records)
    stmt = stmt.on_conflict_do_nothing(
        constraint="uq_stock_prices_ticker_ts",   # defined in 0001_initial migration
    )

    session.execute(stmt)
    session.commit()

    logger.info("upsert_prices: %d rows submitted for %s", len(records), rows[0].ticker)
    return len(records)

# ══════════════════════════════════════════════════════════════════════════════
# Section 3: Sentiment scoring and news persistence
# ══════════════════════════════════════════════════════════════════════════════

def score_sentiment(text: str) -> dict[str, float]:
    """
    Compute VADER sentiment scores for a text string.

    Uses the module-level _vader singleton (instantiated once on import).

    Args:
        text: The text to analyse. For news articles, use the headline only.

    Returns:
        Dict with keys: pos, neg, neu (each 0.0–1.0), compound (-1.0 to +1.0).
        Values are rounded to 4 decimal places.

    Example:
        score_sentiment("Apple stock surges to all-time high!")
        # → {"pos": 0.595, "neg": 0.0, "neu": 0.405, "compound": 0.7096}
    """
    raw = _vader.polarity_scores(text)
    return {
        "pos": round(raw["pos"], 4),
        "neg": round(raw["neg"], 4),
        "neu": round(raw["neu"], 4),
        "compound": round(raw["compound"], 4),
    }


def upsert_news(items: list[RawNewsItem], session: Session) -> int:
    """
    Score sentiment and upsert news articles into the news_articles table.

    For each RawNewsItem:
    1. Runs the title through VADER to get sentiment scores
    2. Inserts into news_articles with ON CONFLICT DO NOTHING (dedup by URL)

    Args:
        items:   Validated news items from fetch_all_feeds().
        session: Active SQLAlchemy session.

    Returns:
        Number of rows submitted (includes both inserted and skipped).
        Returns 0 if items list is empty.
    """
    if not items:
        return 0

    records = []
    for item in items:
        scores = score_sentiment(item.title)
        records.append(
            {
                "title": item.title,
                "source_url": item.source_url,
                "published_at": item.published_at,
                # Decimal(str(...)) for exact representation in Numeric(5,4) columns
                "sentiment_positive": Decimal(str(scores["pos"])),
                "sentiment_negative": Decimal(str(scores["neg"])),
                "sentiment_neutral": Decimal(str(scores["neu"])),
                "sentiment_compound": Decimal(str(scores["compound"])),
            }
        )

    try:
        stmt = pg_insert(NewsArticle).values(records)
        stmt = stmt.on_conflict_do_nothing(
            constraint="uq_news_articles_source_url",
        )
        session.execute(stmt)
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("upsert_news: database error, transaction rolled back")
        raise

    logger.info(
        "upsert_news: %d articles submitted (%d unique URLs)",
        len(records),
        len({r["source_url"] for r in records}),
    )
    return len(records)
