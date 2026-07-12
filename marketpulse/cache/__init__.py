# marketpulse/cache/__init__.py
# Public API of the cache package.

from marketpulse.cache.redis_client import (
    RedisClient,
    cached,
    get_redis_client,
    invalidate_news,
    invalidate_ticker,
)

__all__ = [
    "RedisClient",
    "cached",
    "get_redis_client",
    "invalidate_ticker",
    "invalidate_news",
]
