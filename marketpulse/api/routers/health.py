# marketpulse/api/routers/health.py
# GET /api/v1/health — checks database and Redis connectivity.

from __future__ import annotations

import logging

from fastapi import APIRouter

from marketpulse.api.dependencies import RedisDep
from marketpulse.config import settings
from marketpulse.db.session import verify_connection

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check(redis: RedisDep) -> dict:
    """
    Service health check.

    Verifies:
    - This API process is running (if you get a response, this is true)
    - PostgreSQL is reachable (runs SELECT 1)
    - Redis is reachable (sends PING)

    Returns:
        status: "ok" if all services healthy, "degraded" if any are down.
        db:     "ok" or "error"
        redis:  "ok" or "error"
        tickers: list of configured ticker symbols

    Used by:
    - Docker Compose healthcheck
    - Streamlit dashboard connection indicator
    - Manual sanity checks during deployment
    """
    db_ok = verify_connection()
    redis_ok = redis.ping()

    overall = "ok" if (db_ok and redis_ok) else "degraded"

    if overall == "degraded":
        logger.warning(
            "Health check: DEGRADED (db=%s, redis=%s)",
            "ok" if db_ok else "error",
            "ok" if redis_ok else "error",
        )

    return {
        "status": overall,
        "db": "ok" if db_ok else "error",
        "redis": "ok" if redis_ok else "error",
        "tickers": settings.ticker_list,
        "version": "0.1.0",
    }
