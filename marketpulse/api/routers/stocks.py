# marketpulse/api/routers/stocks.py
# GET /api/v1/stocks                         — list tracked tickers
# GET /api/v1/stocks/{ticker}/prices          — OHLCV bars (newest N, reversed to asc)
# GET /api/v1/stocks/{ticker}/indicators      — technical indicator rows

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from marketpulse.api.dependencies import DbDep, RedisDep
from marketpulse.config import settings
from marketpulse.db import StockPrice, TechnicalIndicator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/stocks", tags=["Stocks"])

# ── Helper: convert Decimal/None to float/None for JSON serialisation ─────────

def _f(value: object) -> float | None:
    """Convert SQLAlchemy Decimal to float; return None as None."""
    return float(value) if value is not None else None  # type: ignore[arg-type]


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/v1/stocks — list all tracked tickers
# ══════════════════════════════════════════════════════════════════════════════

@router.get("")
async def list_tickers() -> dict:
    """
    List all tickers configured in the pipeline.

    Returns the TICKERS environment variable as a list.
    No database or cache call needed — this is static configuration.
    """
    return {
        "tickers": settings.ticker_list,
        "count": len(settings.ticker_list),
    }


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/v1/stocks/{ticker}/prices
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/{ticker}/prices")
async def get_prices(
    ticker: str,
    db: DbDep,
    redis: RedisDep,
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
        description="Number of price bars to return (1–500)",
    ),
) -> list[dict]:
    """
    Return the last N OHLCV price bars for a ticker, oldest-first.

    Response is cached for CACHE_TTL_PRICES seconds (default 300 = 5 minutes).
    Cache is invalidated by the scheduler after each ingestion cycle.

    Returns 404 if no price data exists for this ticker yet.
    """
    ticker = ticker.upper()
    cache_key = f"marketpulse:stocks:{ticker}:prices:{limit}"

    # ── Cache-aside: check Redis first ────────────────────────────────────────
    cached = redis.get_json(cache_key)
    if cached is not None:
        logger.debug("Cache HIT for %s", cache_key)
        return cached  # type: ignore[return-value, no-any-return]

    # ── Cache miss: query PostgreSQL ──────────────────────────────────────────
    logger.debug("Cache MISS for %s — querying DB", cache_key)
    rows = (
        db.query(StockPrice)
        .filter(StockPrice.ticker == ticker)
        .order_by(StockPrice.timestamp.desc())  # newest first (efficient with desc index)
        .limit(limit)
        .all()
    )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No price data for '{ticker}'. "
                "Check that the ticker is in TICKERS and that the ingestion "
                "job has run at least once (see GET /api/v1/health)."
            ),
        )

    # Build response — reverse so oldest-first (chronological for chart rendering)
    result = [
        {
            "ticker": r.ticker,
            "timestamp": r.timestamp.isoformat(),
            "open": float(r.open),
            "high": float(r.high),
            "low": float(r.low),
            "close": float(r.close),
            "volume": r.volume,
        }
        for r in reversed(rows)
    ]

    # ── Cache the result ──────────────────────────────────────────────────────
    redis.set_json(cache_key, result, settings.cache_ttl_prices)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/v1/stocks/{ticker}/indicators
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/{ticker}/indicators")
async def get_indicators(
    ticker: str,
    db: DbDep,
    redis: RedisDep,
    limit: int = Query(
        default=100,
        ge=1,
        le=500,
        description="Number of indicator rows to return (1–500)",
    ),
) -> list[dict]:
    """
    Return the last N technical indicator rows for a ticker, oldest-first.

    Includes all 10 computed indicators: SMA-20/50/200, EMA-12/26, RSI-14,
    MACD, MACD Signal, Bollinger Upper/Lower. Values are null for early bars
    with insufficient history (e.g. SMA-200 requires 200 bars).

    Response is cached for CACHE_TTL_INDICATORS seconds (default 300).
    """
    ticker = ticker.upper()
    cache_key = f"marketpulse:stocks:{ticker}:indicators:{limit}"

    cached = redis.get_json(cache_key)
    if cached is not None:
        logger.debug("Cache HIT for %s", cache_key)
        return cached  # type: ignore[return-value, no-any-return]

    rows = (
        db.query(TechnicalIndicator)
        .filter(TechnicalIndicator.ticker == ticker)
        .order_by(TechnicalIndicator.timestamp.desc())
        .limit(limit)
        .all()
    )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No indicator data for '{ticker}'. "
                "Indicators are computed during ingestion — ensure the ingestion "
                "job has run and that enough price history exists for SMA-20 (20 bars)."
            ),
        )

    result = [
        {
            "ticker": r.ticker,
            "timestamp": r.timestamp.isoformat(),
            # _f() converts Decimal→float and None→None for JSON safety
            "sma_20": _f(r.sma_20),
            "sma_50": _f(r.sma_50),
            "sma_200": _f(r.sma_200),
            "ema_12": _f(r.ema_12),
            "ema_26": _f(r.ema_26),
            "rsi_14": _f(r.rsi_14),
            "macd": _f(r.macd),
            "macd_signal": _f(r.macd_signal),
            "bb_upper": _f(r.bb_upper),
            "bb_lower": _f(r.bb_lower),
        }
        for r in reversed(rows)
    ]

    redis.set_json(cache_key, result, settings.cache_ttl_indicators)
    return result
