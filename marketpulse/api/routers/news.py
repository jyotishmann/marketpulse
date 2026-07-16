# marketpulse/api/routers/news.py
# GET /api/v1/news — latest news articles with VADER sentiment scores

from __future__ import annotations

import logging

from fastapi import APIRouter, Query

from marketpulse.api.dependencies import DbDep, RedisDep
from marketpulse.config import settings
from marketpulse.db import NewsArticle

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/news", tags=["News"])


@router.get("")
async def get_news(
    db: DbDep,
    redis: RedisDep,
    limit: int = Query(
        default=30,
        ge=1,
        le=200,
        description="Number of articles to return, newest-first (1–200)",
    ),
) -> list[dict]:
    """
    Return the latest N news articles with VADER sentiment scores.

    Articles are ordered by publication date, newest first.
    This is NOT reversed (unlike prices/indicators) — the dashboard news feed
    naturally shows newest at the top.

    Cache key: marketpulse:news:{limit}
    Invalidated by ingest_news() every SCHEDULE_NEWS_MINUTES (default 30).

    Each article includes:
    - title, source_url, published_at
    - sentiment_compound: -1.0 (most negative) to +1.0 (most positive)
    - sentiment_positive/negative/neutral: component ratios (sum to ~1.0)
    """
    cache_key = f"marketpulse:news:{limit}"

    cached = redis.get_json(cache_key)
    if cached is not None:
        logger.debug("Cache HIT for %s", cache_key)
        return cached  # type: ignore[return-value, no-any-return]

    logger.debug("Cache MISS for %s — querying DB", cache_key)

    rows = (
        db.query(NewsArticle)
        # Order by publication date (not insertion date) for correct timeline ordering
        .order_by(NewsArticle.published_at.desc())
        .limit(limit)
        .all()
    )

    # Empty result is OK (no news yet) — return empty list, not 404
    # The dashboard handles an empty news feed gracefully
    result = [
        {
            "id": r.id,
            "title": r.title,
            "source_url": r.source_url,
            "published_at": r.published_at.isoformat(),
            # Decimal → float for JSON serialisation
            "sentiment_positive": float(r.sentiment_positive),
            "sentiment_negative": float(r.sentiment_negative),
            "sentiment_neutral": float(r.sentiment_neutral),
            "sentiment_compound": float(r.sentiment_compound),
        }
        for r in rows  # already newest-first — do NOT reverse
    ]

    redis.set_json(cache_key, result, settings.cache_ttl_news)
    return result
