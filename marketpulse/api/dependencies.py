# marketpulse/api/dependencies.py
# Shared FastAPI dependency functions — injected into route handlers via Depends().
# These are the "wiring" between the HTTP layer and the service/data layers.

from __future__ import annotations

from collections.abc import Generator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from marketpulse.cache import RedisClient, get_redis_client
from marketpulse.db.session import get_db as _get_db_generator
from marketpulse.ml.service import PredictionService
from marketpulse.scheduler.jobs import get_prediction_service

# ── Database session ──────────────────────────────────────────────────────────


def get_db() -> Generator[Session, None, None]:
    """
    Provide a database session for the duration of one HTTP request.

    FastAPI calls this before the route function and resumes it (running
    finally: db.close()) after the response is sent — even on exceptions.

    The generator pattern guarantees the session is always closed.
    Connection is returned to the pool immediately after db.close().
    """
    yield from _get_db_generator()


# ── Redis client ──────────────────────────────────────────────────────────────


def get_redis() -> RedisClient:
    """
    Provide the shared RedisClient singleton.

    Returns the same module-level instance on every call (lru_cache).
    All routes share the same underlying connection pool.
    """
    return get_redis_client()


# ── ML prediction service ─────────────────────────────────────────────────────


def get_ml_service() -> PredictionService:
    """
    Provide the shared PredictionService singleton.

    This is the SAME instance the scheduler uses for run_ml_pipeline().
    When the scheduler calls service.invalidate_cache(ticker) after retraining,
    it evicts the cached model from the very object that API routes use —
    so the next route call reloads the fresh model from disk.
    """
    return get_prediction_service()


# ── Annotated type aliases (modern FastAPI DI syntax) ─────────────────────────
# Declare once here; import into routers to avoid repeating Depends(fn) everywhere.

DbDep = Annotated[Session, Depends(get_db)]
RedisDep = Annotated[RedisClient, Depends(get_redis)]
MLServiceDep = Annotated[PredictionService, Depends(get_ml_service)]
