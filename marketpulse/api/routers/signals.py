# marketpulse/api/routers/signals.py
# GET /api/v1/stocks/{ticker}/signals         — ML signal history
# GET /api/v1/stocks/{ticker}/signals/latest  — most recent single signal

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from marketpulse.api.dependencies import DbDep, RedisDep
from marketpulse.config import settings
from marketpulse.db import MLSignal

logger = logging.getLogger(__name__)

# prefix="/stocks" matches the path structure: /api/v1/stocks/{ticker}/signals
router = APIRouter(prefix="/stocks", tags=["Signals"])


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/v1/stocks/{ticker}/signals/latest — single most recent signal
# (defined BEFORE /{ticker}/signals to avoid path ambiguity)
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/{ticker}/signals/latest")
async def get_latest_signal(
    ticker: str,
    db: DbDep,
    redis: RedisDep,
) -> dict:
    """
    Return the single most recent ML signal for a ticker.

    The dashboard signal badge polls this endpoint every 30 seconds.
    Cached for CACHE_TTL_SIGNALS seconds (default 3600 = 1 hour, matching
    the ML pipeline's run frequency).

    Returns 404 if the ML pipeline has not yet run for this ticker.
    """
    ticker = ticker.upper()
    cache_key = f"marketpulse:stocks:{ticker}:signals:latest"

    cached = redis.get_json(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value, no-any-return]

    row = (
        db.query(MLSignal)
        .filter(MLSignal.ticker == ticker)
        .order_by(MLSignal.timestamp.desc())
        .first()   # .first() → LIMIT 1 SQL, returns None if no rows
    )

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No ML signal for '{ticker}'. "
                "The ML pipeline must run at least once. Check "
                "GET /api/v1/health and ensure the ingestion job has collected "
                "at least 50 rows of feature data."
            ),
        )

    result = {
        "ticker": row.ticker,
        "timestamp": row.timestamp.isoformat(),
        "signal": row.signal,              # "BUY", "HOLD", or "SELL"
        "confidence": float(row.confidence),
        "is_anomaly": row.is_anomaly,
        "model_version": row.model_version,
    }

    redis.set_json(cache_key, result, settings.cache_ttl_signals)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/v1/stocks/{ticker}/signals — signal history timeline
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/{ticker}/signals")
async def get_signals(
    ticker: str,
    db: DbDep,
    redis: RedisDep,
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Number of recent signals to return (1–100)",
    ),
) -> list[dict]:
    """
    Return the last N ML signals for a ticker, oldest-first.

    20 rows ≈ 20 hours of history (pipeline runs every 60 minutes).
    Used by the dashboard for a signal history chart.
    Cached for CACHE_TTL_SIGNALS seconds.
    """
    ticker = ticker.upper()
    cache_key = f"marketpulse:stocks:{ticker}:signals:{limit}"

    cached = redis.get_json(cache_key)
    if cached is not None:
        return cached  # type: ignore[return-value, no-any-return]

    rows = (
        db.query(MLSignal)
        .filter(MLSignal.ticker == ticker)
        .order_by(MLSignal.timestamp.desc())
        .limit(limit)
        .all()
    )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No ML signals for '{ticker}'. Run the ML pipeline first.",
        )

    result = [
        {
            "ticker": r.ticker,
            "timestamp": r.timestamp.isoformat(),
            "signal": r.signal,
            "confidence": float(r.confidence),
            "is_anomaly": r.is_anomaly,
            "model_version": r.model_version,
        }
        for r in reversed(rows)  # oldest-first for timeline charts
    ]

    redis.set_json(cache_key, result, settings.cache_ttl_signals)
    return result
