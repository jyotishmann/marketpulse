# marketpulse/scheduler/jobs.py
# APScheduler job definitions — the orchestration layer.
# Wires together ingestion (FILE_03), ETL (FILE_04), ML (FILE_05),
# and cache invalidation (this file) into automated timed jobs.

from __future__ import annotations

import logging

import pandas as pd
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from marketpulse.cache import get_redis_client, invalidate_news, invalidate_ticker
from marketpulse.config import settings
from marketpulse.db import SessionLocal, verify_connection
from marketpulse.ingestion import fetch_all_feeds, fetch_ohlcv
from marketpulse.ml import (
    PredictionService,
    build_feature_matrix,
    save_anomaly_detector,
    save_classifier,
    train_anomaly_detector,
    train_classifier,
)
from marketpulse.processing import (
    compute_all,
    load_recent_prices,
    rows_to_ohlcv_df,
    upsert_indicators,
    upsert_news,
    upsert_prices,
)

logger = logging.getLogger(__name__)

# ── PredictionService singleton ────────────────────────────────────────────────
# Shared between the scheduler (writes predictions) and the API (reads predictions).
# Lazy-loads trained models on first predict() call per ticker.
_prediction_service: PredictionService | None = None


def get_prediction_service() -> PredictionService:
    """
    Return the module-level PredictionService singleton.

    This service instance is shared between:
    - run_ml_pipeline(): calls service.invalidate_cache() after retraining
    - FastAPI routes (FILE_07): calls service.predict() for inference

    Import and use this function wherever a PredictionService is needed.
    """
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
        logger.info("PredictionService singleton created")
    return _prediction_service


# ══════════════════════════════════════════════════════════════════════════════
# Job 1: Stock data ingestion + indicator computation (every N minutes)
# ══════════════════════════════════════════════════════════════════════════════

def ingest_stock_data() -> None:
    """
    Scheduled job: fetch OHLCV bars, compute indicators, write to DB, bust cache.

    Full cycle per ticker:
    1. fetch_ohlcv()          — yfinance → list[RawOHLCVRow]
    2. upsert_prices()        — write new price rows to PostgreSQL
    3. load_recent_prices()   — load 220 historical rows for SMA-200 context
    4. combine + dedup        — merge fresh + historical DataFrames
    5. compute_all()          — add 10 indicator columns to combined DataFrame
    6. upsert_indicators()    — write new indicator rows to PostgreSQL
    7. invalidate_ticker()    — delete all AAPL:* keys from Redis

    Scheduled every SCHEDULE_STOCK_MINUTES (default: 15).
    """
    logger.info("▶ ingest_stock_data() started for tickers: %s", settings.ticker_list)

    session = SessionLocal()
    try:
        for ticker in settings.ticker_list:
            logger.info("  Processing %s", ticker)

            # ── Step 1: Fetch fresh data from Yahoo Finance ────────────────────
            # period="5d": last 5 trading days → catches any missed bars
            fresh_rows = fetch_ohlcv(ticker, period="5d", interval="15m")
            if not fresh_rows:
                logger.warning("  No data returned for %s — skipping", ticker)
                continue

            # ── Step 2: Upsert prices (ON CONFLICT DO NOTHING) ─────────────────
            upsert_prices(fresh_rows, session)

            # ── Step 3: Load historical context for SMA-200 ───────────────────
            # SMA-200 needs at least 200 bars. We load 220 for buffer.
            hist_df = load_recent_prices(ticker, session, lookback=220)
            fresh_df = rows_to_ohlcv_df(fresh_rows)

            # ── Step 4: Merge fresh + historical (deduplicate on timestamp) ────
            if not hist_df.empty:
                combined = pd.concat(
                    [hist_df, fresh_df],
                    ignore_index=True,
                )
                # keep="last": fresh data overrides history for same timestamp
                # (handles price adjustments from splits/dividends)
                combined = (
                    combined
                    .drop_duplicates(subset=["timestamp"], keep="last")
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )
            else:
                # First run: no history in DB yet
                combined = fresh_df

            logger.debug(
                "  %s combined DataFrame: %d rows (hist=%d, fresh=%d)",
                ticker, len(combined), len(hist_df), len(fresh_df),
            )

            # ── Step 5: Compute all 10 technical indicators ────────────────────
            enriched = compute_all(combined)

            # ── Step 6: Upsert indicators (only new rows) ──────────────────────
            upsert_indicators(enriched, session)

            # ── Step 7: Bust the Redis cache for this ticker ───────────────────
            invalidate_ticker(ticker)

            logger.info("  ✓ %s ingest cycle complete", ticker)

    except Exception:
        logger.exception("ingest_stock_data() raised an unexpected exception")
    finally:
        session.close()  # always return connection to pool

    logger.info("◀ ingest_stock_data() completed")

# ══════════════════════════════════════════════════════════════════════════════
# Job 2: News ingestion + sentiment scoring (every N minutes)
# ══════════════════════════════════════════════════════════════════════════════

def ingest_news() -> None:
    """
    Scheduled job: fetch RSS headlines, score sentiment, write to DB, bust cache.

    Fetches from all configured RSS_FEED_URLS. Deduplicates by source_url
    (same article from multiple feeds is stored once). Scores each headline
    with VADER. ON CONFLICT DO NOTHING on source_url prevents re-inserting
    articles from previous runs.

    Scheduled every SCHEDULE_NEWS_MINUTES (default: 30).
    """
    urls = settings.rss_url_list
    if not urls:
        logger.info("ingest_news(): no RSS_FEED_URLS configured — skipping")
        return

    logger.info("▶ ingest_news() started (%d feed URLs)", len(urls))

    session = SessionLocal()
    try:
        # Fetch + deduplicate items across all feeds
        items = fetch_all_feeds(urls)
        if not items:
            logger.info("ingest_news(): no new items from any feed")
            return

        # Score sentiment + upsert (VADER runs inside upsert_news)
        inserted = upsert_news(items, session)  # noqa: F841

        # Bust the news cache so the dashboard shows fresh articles
        invalidate_news()

        logger.info("◀ ingest_news(): %d articles processed", len(items))

    except Exception:
        logger.exception("ingest_news() raised an unexpected exception")
    finally:
        session.close()


# ══════════════════════════════════════════════════════════════════════════════
# Job 3: ML pipeline — train models + generate signals (every N minutes)
# ══════════════════════════════════════════════════════════════════════════════

def run_ml_pipeline() -> None:
    """
    Scheduled job: retrain models for all tickers, generate and save fresh signals.

    For each ticker: builds feature matrix → trains RandomForest + IsolationForest
    → saves models to disk → registers in ModelRegistry → generates prediction
    for the most recent bar → writes signal to ml_signals table.

    Scheduled every SCHEDULE_ML_MINUTES (default: 60).
    Requires at least 50 rows of feature data per ticker (skips if less).
    """
    logger.info("▶ run_ml_pipeline() started for tickers: %s", settings.ticker_list)

    service = get_prediction_service()
    session = SessionLocal()

    try:
        for ticker in settings.ticker_list:
            logger.info("  ML pipeline: %s", ticker)

            # ── Step 1: Build feature matrix from DB ──────────────────────────
            X, y, feature_names = build_feature_matrix(
                ticker,
                session,
                lookback_days=settings.ml_lookback_days,
            )

            if X.empty or len(X) < 50:
                logger.warning(
                    "  %s: %d usable rows (minimum 50) — skipping ML pipeline. "
                    "Run more ingestion cycles to accumulate data.",
                    ticker, len(X) if not X.empty else 0,
                )
                continue

            logger.info("  %s: %d feature rows, %d features", ticker, len(X), len(feature_names))

            # ── Step 2: Train BUY/HOLD/SELL classifier ────────────────────────
            try:
                clf_pipeline, accuracy = train_classifier(X, y, ticker)
                save_classifier(clf_pipeline, ticker, accuracy, session)
                logger.info("  %s classifier: accuracy=%.4f", ticker, accuracy)
            except Exception:
                logger.exception("  Classifier training failed for %s", ticker)
                continue

            # ── Step 3: Train anomaly detector ────────────────────────────────
            try:
                iso = train_anomaly_detector(X, ticker)
                save_anomaly_detector(iso, ticker, session)
            except Exception:
                logger.exception("  Anomaly detector training failed for %s", ticker)
                # Continue — anomaly detection is optional; classifier is primary

            # ── Step 4: Invalidate cache → force reload of new model ──────────
            service.invalidate_cache(ticker)

            # ── Step 5: Predict for the most recent bar ───────────────────────
            # Use the last row of X (most recent feature values)
            result = service.predict(ticker, X)
            if result:
                service.save_signal(result, session)
                logger.info(
                    "  %s signal: %s (conf=%.2f, anomaly=%s)",
                    ticker, result["signal"], result["confidence"], result["is_anomaly"],
                )
            else:
                logger.warning("  %s: predict() returned None after retraining", ticker)

    except Exception:
        logger.exception("run_ml_pipeline() raised an unexpected exception")
    finally:
        session.close()

    logger.info("◀ run_ml_pipeline() completed")

# ══════════════════════════════════════════════════════════════════════════════
# Job 4: Startup warmup (fires once immediately when scheduler starts)
# ══════════════════════════════════════════════════════════════════════════════

def warmup_cache() -> None:
    """
    Startup job: verify connections and trigger the first ingest cycle.

    Fires once immediately when the scheduler starts (DateTrigger with no date).
    Ensures the dashboard has live data within seconds of startup rather than
    waiting for the first interval trigger (15 minutes for stock data).

    Sequence:
    1. Verify PostgreSQL connection
    2. Verify Redis connection
    3. Run ingest_stock_data() once (loads initial price + indicator data)
    4. Run ingest_news() once (loads initial news headlines)
    """
    logger.info("▶ warmup_cache(): startup sequence beginning")

    # ── Verify database ────────────────────────────────────────────────────────
    if not verify_connection():
        logger.error(
            "warmup_cache(): PostgreSQL not reachable! "
            "Ensure 'docker compose up -d' has been run and migrations applied."
        )
        return

    logger.info("  ✓ PostgreSQL connection OK")

    # ── Verify Redis ───────────────────────────────────────────────────────────
    if not get_redis_client().ping():
        logger.warning(
            "warmup_cache(): Redis not reachable — API will run without cache "
            "(slower responses). Check docker compose logs redis."
        )
        # Not fatal — application works without cache, just slower
    else:
        logger.info("  ✓ Redis connection OK")

    # ── Fire first ingestion cycle ─────────────────────────────────────────────
    logger.info("  Starting first stock data ingestion cycle...")
    ingest_stock_data()

    logger.info("  Starting first news ingestion cycle...")
    ingest_news()

    logger.info("◀ warmup_cache(): startup sequence complete")


# ══════════════════════════════════════════════════════════════════════════════
# Scheduler factory
# ══════════════════════════════════════════════════════════════════════════════

def create_scheduler() -> AsyncIOScheduler:
    """
    Create and configure the APScheduler AsyncIOScheduler.

    Called once in api/main.py during the FastAPI startup event.
    The scheduler runs in the same event loop as FastAPI.

    Returns:
        Configured but not yet started AsyncIOScheduler.
        Caller must call scheduler.start() to begin execution.

    Job summary:
    ┌────────────────────┬──────────────────────────────────┬────────────────┐
    │ Job ID             │ Function                         │ Trigger        │
    ├────────────────────┼──────────────────────────────────┼────────────────┤
    │ warmup_cache       │ warmup_cache()                   │ Once at start  │
    │ ingest_stock_data  │ ingest_stock_data()              │ Every 15 min   │
    │ ingest_news        │ ingest_news()                    │ Every 30 min   │
    │ run_ml_pipeline    │ run_ml_pipeline()                │ Every 60 min   │
    └────────────────────┴──────────────────────────────────┴────────────────┘
    """
    scheduler = AsyncIOScheduler(
        # Timezone for all cron/interval calculations
        timezone="UTC",
        # Job store: in-memory (sufficient for our use case)
        # For persistence across restarts, use SQLAlchemyJobStore
        jobstores={"default": {"type": "memory"}},
    )

    # Shared job defaults: prevent overlapping and catch-up storms
    job_defaults = {
        "max_instances": 1,      # never run two instances of the same job
        "coalesce": True,        # if missed runs: fire once, not N times
        "misfire_grace_time": 120,  # allow 2 minutes of lateness before skipping
    }

    # ── Job 4 (runs first): startup warmup ────────────────────────────────────
    # DateTrigger without run_date → fires immediately when scheduler.start() is called
    scheduler.add_job(
        func=warmup_cache,
        trigger=DateTrigger(),   # run_date defaults to "now"
        id="warmup_cache",
        replace_existing=True,
        **job_defaults,
    )

    # ── Job 1: stock data + indicators ────────────────────────────────────────
    scheduler.add_job(
        func=ingest_stock_data,
        trigger=IntervalTrigger(minutes=settings.schedule_stock_minutes),
        id="ingest_stock_data",
        replace_existing=True,
        **job_defaults,
    )

    # ── Job 2: news + sentiment ───────────────────────────────────────────────
    scheduler.add_job(
        func=ingest_news,
        trigger=IntervalTrigger(minutes=settings.schedule_news_minutes),
        id="ingest_news",
        replace_existing=True,
        **job_defaults,
    )

    # ── Job 3: ML retraining + signal generation ──────────────────────────────
    scheduler.add_job(
        func=run_ml_pipeline,
        trigger=IntervalTrigger(minutes=settings.schedule_ml_minutes),
        id="run_ml_pipeline",
        replace_existing=True,
        **job_defaults,
    )

    logger.info(
        "Scheduler configured: "
        "stock=every %dmin | news=every %dmin | ml=every %dmin",
        settings.schedule_stock_minutes,
        settings.schedule_news_minutes,
        settings.schedule_ml_minutes,
    )

    return scheduler
