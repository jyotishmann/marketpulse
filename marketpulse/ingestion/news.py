# marketpulse/ingestion/news.py
# RSS news connector: fetches headlines and validates them.
# Sentiment scoring (VADER) happens in the processing layer, not here.

from __future__ import annotations

import calendar
import logging
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import feedparser

from marketpulse.ingestion.schemas import RawNewsItem

logger = logging.getLogger(__name__)


# ── Internal helper ────────────────────────────────────────────────────────────


def _parse_entry_date(entry: feedparser.util.FeedParserDict) -> datetime:
    """
    Extract a timezone-aware UTC datetime from a feedparser entry.

    Handles three fallback levels:
    1. entry.published_parsed (time.struct_time) — most common, most reliable
    2. entry.published (raw string) — RFC 2822 / ISO 8601
    3. Current UTC time — last resort, logs a debug message

    Args:
        entry: A single entry from feed.entries

    Returns:
        A timezone-aware datetime in UTC.
    """
    # Attempt 1: feedparser's pre-parsed struct_time
    if entry.get("published_parsed") and entry.published_parsed:
        try:
            # calendar.timegm() treats struct_time as UTC (unlike time.mktime())
            utc_ts = calendar.timegm(entry.published_parsed)
            return datetime.fromtimestamp(utc_ts, tz=UTC)
        except (TypeError, OverflowError, OSError):
            logger.debug(
                "published_parsed conversion failed for: %s", entry.get("link")
            )

    # Attempt 2: parse the raw published string (RFC 2822 format from RSS 2.0)
    if entry.get("published"):
        try:
            dt = parsedate_to_datetime(entry.published)
            # Ensure timezone-aware (parsedate_to_datetime always returns tz-aware)
            return dt.astimezone(UTC)
        except Exception:
            logger.debug("parsedate_to_datetime failed for: %r", entry.get("published"))

    # Attempt 3: fallback to current UTC time
    logger.debug(
        "No parseable date for entry: %r — using now()", entry.get("link", "?")
    )
    return datetime.now(tz=UTC)


# ── Public API ─────────────────────────────────────────────────────────────────


def fetch_feed(url: str) -> list[RawNewsItem]:
    """
    Fetch and parse one RSS feed, returning validated news items.

    Args:
        url: Full URL of the RSS feed endpoint.
             Example: "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL"

    Returns:
        List of RawNewsItem objects (may be empty on connection failure
        or if the feed contains no valid entries).
    """
    logger.info("Parsing RSS feed: %s", url)

    # ── Step 1: Fetch and parse the RSS XML ───────────────────────────────────
    try:
        feed = feedparser.parse(url)
    except Exception:
        logger.exception("feedparser.parse() raised for URL: %s", url)
        return []

    # bozo=True means the XML had errors. Distinguish broken vs parseable.
    if getattr(feed, "bozo", False):
        if not feed.entries:
            # Truly broken — no entries could be extracted
            logger.error(
                "Unparseable RSS feed at %s (bozo_exception: %s)",
                url,
                getattr(feed, "bozo_exception", "unknown"),
            )
            return []
        # Bozo but has entries — log and continue (common with real-world feeds)
        logger.warning(
            "Malformed XML in feed %s (bozo_exception: %s) — parsing anyway",
            url,
            getattr(feed, "bozo_exception", "unknown"),
        )

    # ── Step 2: Validate each entry with pydantic ─────────────────────────────
    items: list[RawNewsItem] = []
    for entry in feed.entries:
        try:
            item = RawNewsItem(
                title=entry.get("title", ""),
                source_url=entry.get("link", ""),
                published_at=_parse_entry_date(entry),
            )
            items.append(item)
        except Exception as exc:
            # Skip entries that fail validation — log at DEBUG to avoid log spam
            logger.debug(
                "Skipping invalid RSS entry from %s: %s",
                url,
                exc,
            )

    logger.info("Parsed %d valid items from %s", len(items), url)
    return items


def fetch_all_feeds(urls: list[str]) -> list[RawNewsItem]:
    """
    Fetch and deduplicate news from multiple RSS feeds.

    Calls fetch_feed() for each URL. Deduplicates by source_url
    (first occurrence wins). Sorts results newest-first.

    Args:
        urls: List of RSS feed URLs. Empty list returns [] immediately.

    Returns:
        Deduplicated list of RawNewsItem objects sorted by published_at
        descending (newest first).

    Example:
        items = fetch_all_feeds([
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL",
            "https://feeds.finance.yahoo.com/rss/2.0/headline?s=MSFT",
        ])
        print(f"Got {len(items)} unique articles")
    """
    if not urls:
        logger.warning(
            "No RSS feed URLs provided — news ingestion skipped. "
            "Set RSS_FEED_URLS in your .env file to enable news."
        )
        return []

    # set tracks URLs already added — O(1) membership tests
    seen_urls: set[str] = set()
    all_items: list[RawNewsItem] = []

    for url in urls:
        feed_items = fetch_feed(url)
        added = 0
        for item in feed_items:
            if item.source_url not in seen_urls:
                seen_urls.add(item.source_url)
                all_items.append(item)
                added += 1
        dupes = len(feed_items) - added
        if dupes:
            logger.debug("Deduplicated %d items from %s", dupes, url)

    # Sort newest-first so callers receive articles in display order
    all_items.sort(key=lambda x: x.published_at, reverse=True)

    logger.info(
        "Fetched %d unique articles from %d feed(s) (%d total before deduplication)",
        len(all_items),
        len(urls),
        sum(1 for _ in seen_urls),  # == len(all_items), but explicit
    )
    return all_items
