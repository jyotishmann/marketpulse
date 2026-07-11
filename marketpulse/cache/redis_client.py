# marketpulse/cache/redis_client.py
# Redis wrapper with fail-open error handling and JSON serialisation.
# Provides: RedisClient class, cached() decorator, invalidate_ticker() helper.

from __future__ import annotations

import json
import logging
from typing import Any

import redis

from marketpulse.config import settings

logger = logging.getLogger(__name__)

# Key namespace prefix — all MarketPulse cache entries start with this
_NAMESPACE = "marketpulse"


class RedisClient:
    """
    Thin wrapper around redis.Redis with:
    - JSON auto-serialisation (get_json / set_json)
    - Fail-open error handling (Redis unavailable → log warning, return safe default)
    - Pattern-based key deletion via scan_iter (non-blocking)
    - Connection pool via redis.Redis (shared across all RedisClient instances)
    """

    def __init__(self, redis_url: str | None = None) -> None:
        url = redis_url or settings.redis_url
        # decode_responses=True: Redis returns str instead of bytes
        # max_connections=20: pool size (default 2^31 which is too many)
        self._r: redis.Redis = redis.Redis.from_url(
            url,
            decode_responses=True,
            max_connections=20,
        )

    # ── Core key-value operations ──────────────────────────────────────────────

    def get(self, key: str) -> str | None:
        """
        Get a raw string value by key.

        Returns None on cache miss OR on any Redis error (fail-open).
        Callers should treat None as "not cached — fetch from DB".
        """
        try:
            return self._r.get(key)  # type: ignore[return-value]
        except redis.RedisError as exc:
            logger.warning("Redis GET failed for key=%r: %s", key, exc)
            return None

    def set(self, key: str, value: str, ttl: int) -> bool:
        """
        Store a string value with a TTL (time-to-live) in seconds.

        Returns True on success, False on error (fail-open).
        After TTL seconds, Redis automatically deletes the key.
        """
        try:
            self._r.setex(name=key, time=ttl, value=value)
            return True
        except redis.RedisError as exc:
            logger.warning("Redis SET failed for key=%r: %s", key, exc)
            return False

    def delete(self, *keys: str) -> int:
        """
        Delete one or more keys by name.

        Returns the number of keys actually deleted (0 if none existed).
        """
        if not keys:
            return 0
        try:
            return int(self._r.delete(*keys))
        except redis.RedisError as exc:
            logger.warning("Redis DELETE failed: %s", exc)
            return 0

    def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a Redis glob pattern.

        Uses SCAN_ITER (non-blocking, cursor-based) rather than KEYS
        (blocking full-keyspace scan). Safe for production use.

        Example patterns:
            "marketpulse:stocks:AAPL:*"  → all cached AAPL data
            "marketpulse:news:*"          → all cached news responses

        Returns:
            Number of keys deleted (0 if none matched).
        """
        try:
            matching = list(self._r.scan_iter(match=pattern))
            if not matching:
                logger.debug("delete_pattern: no keys matched %r", pattern)
                return 0
            deleted = int(self._r.delete(*matching))
            logger.debug(
                "delete_pattern: deleted %d keys matching %r",
                deleted, pattern,
            )
            return deleted
        except redis.RedisError as exc:
            logger.warning(
                "Redis delete_pattern failed for pattern=%r: %s", pattern, exc
            )
            return 0

    # ── JSON helpers ───────────────────────────────────────────────────────────

    def get_json(self, key: str) -> Any | None:
        """
        Get a cached value and deserialise it from JSON.

        Returns None on cache miss, Redis error, or JSON decode failure.
        A corrupt cache entry is treated as a miss (the old value is overwritten
        on the next cache-miss + set cycle).
        """
        raw = self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Corrupt JSON in cache for key=%r (%s) — treating as miss",
                key, exc,
            )
            return None

    def set_json(self, key: str, value: Any, ttl: int) -> bool:
        """
        Serialise a value to JSON and cache it with a TTL.

        `default=str` converts non-JSON-serialisable types (datetime, Decimal)
        to their string representation automatically.

        Returns True on success, False on error.
        """
        try:
            serialised = json.dumps(value, default=str)
        except (TypeError, ValueError) as exc:
            logger.warning("Cannot serialise value to JSON for key=%r: %s", key, exc)
            return False
        return self.set(key, serialised, ttl)

    # ── Health check ───────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """
        Check if the Redis server is reachable.

        Used by GET /api/v1/health and Docker Compose healthchecks.
        """
        try:
            return bool(self._r.ping())
        except redis.RedisError:
            return False

# ── cache-aside decorator ─────────────────────────────────────────────────────

import functools  # noqa: E402
from collections.abc import Callable  # noqa: E402
from typing import TypeVar  # noqa: E402

F = TypeVar("F", bound=Callable[..., Any])


def cached(
    key_fn: Callable[..., str],
    ttl: int,
) -> Callable[[F], F]:
    """
    Cache-aside decorator for pure (synchronous) functions.

    The function is called with the same arguments as the decorated function.
    key_fn returns the cache key suffix (the namespace prefix is added automatically).

    Behaviour:
    - Cache HIT:  Returns deserialised JSON immediately (~2ms).
    - Cache MISS: Calls the function, stores result, returns it (~40ms first time).
    - Redis down: Calls the function directly — fail-open, no caching.

    Args:
        key_fn: Callable that takes the same args as the decorated function
                and returns a string suffix for the cache key.
        ttl:    Seconds before the cached value expires automatically.

    Example:
        @cached(
            key_fn=lambda ticker, limit: f"stocks:{ticker}:prices:{limit}",
            ttl=300,
        )
        def fetch_prices(ticker: str, limit: int = 100) -> list[dict]:
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            client = get_redis_client()
            cache_key = f"{_NAMESPACE}:{key_fn(*args, **kwargs)}"

            # ── Check cache first ─────────────────────────────────────────────
            cached_value = client.get_json(cache_key)
            if cached_value is not None:
                logger.debug("Cache HIT for %s", cache_key)
                return cached_value

            # ── Cache miss: execute the function ──────────────────────────────
            logger.debug("Cache MISS for %s — calling function", cache_key)
            result = func(*args, **kwargs)

            # ── Store in cache (fail-open: errors are logged, not raised) ─────
            stored = client.set_json(cache_key, result, ttl)
            if not stored:
                logger.warning("Failed to cache result for %s", cache_key)

            return result

        return wrapper  # type: ignore[return-value]
    return decorator


# ── Convenience helpers ────────────────────────────────────────────────────────

def invalidate_ticker(ticker: str) -> int:
    """
    Delete all cache entries for a specific ticker.

    Called by the scheduler after writing new prices or indicators to the
    database. Forces the next API request for this ticker to hit the DB.

    Deletes all keys matching: marketpulse:stocks:{ticker}:*

    Args:
        ticker: Stock symbol (e.g., "AAPL").

    Returns:
        Number of cache entries deleted.
    """
    pattern = f"{_NAMESPACE}:stocks:{ticker}:*"
    count = get_redis_client().delete_pattern(pattern)
    if count:
        logger.info("Cache busted for %s: deleted %d key(s)", ticker, count)
    else:
        logger.debug("Cache bust for %s: no keys matched (already empty)", ticker)
    return count


def invalidate_news() -> int:
    """Delete all cached news responses. Called after news ingestion."""
    pattern = f"{_NAMESPACE}:news:*"
    count = get_redis_client().delete_pattern(pattern)
    logger.debug("Cache bust: deleted %d news cache key(s)", count)
    return count


# ── Module-level singleton (created once per process) ─────────────────────────

@functools.lru_cache(maxsize=None)  # noqa: UP033
def _create_client() -> RedisClient:
    """
    Create the shared RedisClient instance.

    lru_cache ensures this runs exactly once — subsequent calls return the
    same object. The underlying redis.Redis uses a connection pool.
    """
    client = RedisClient()
    logger.info("RedisClient initialised (url=%s)", settings.redis_url[:30] + "...")
    return client


def get_redis_client() -> RedisClient:
    """
    Return the module-level RedisClient singleton.

    Use as a FastAPI dependency:
        from marketpulse.cache import get_redis_client

        @router.get("/prices")
        async def prices(redis: RedisClient = Depends(get_redis_client)):
            ...

    Or import directly:
        from marketpulse.cache import get_redis_client
        client = get_redis_client()
    """
    return _create_client()
