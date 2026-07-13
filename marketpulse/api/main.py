# marketpulse/api/main.py
# FastAPI application factory with lifespan startup/shutdown management.
# Entry point for uvicorn: marketpulse.api.main:create_app (--factory flag)

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from marketpulse.api.middleware import setup_middleware
from marketpulse.api.routers import health, news, signals, stocks
from marketpulse.config import settings
from marketpulse.scheduler.jobs import create_scheduler

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    Code before 'yield' runs on startup.
    Code after 'yield' runs on shutdown.
    The 'yield' expression is where the application handles HTTP requests.

    This replaces the deprecated @app.on_event("startup") / ("shutdown")
    pattern from FastAPI < 0.93.
    """
    # ── STARTUP ───────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)-40s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("=" * 60)
    logger.info("MarketPulse API starting up")
    logger.info("Log level: %s", settings.log_level)
    logger.info("Tickers: %s", settings.ticker_list)
    logger.info("=" * 60)

    # Create and start the APScheduler instance.
    # Stored on app.state so the shutdown block can stop it.
    # AsyncIOScheduler runs in the same event loop as FastAPI — no thread needed.
    scheduler = create_scheduler()
    app.state.scheduler = scheduler
    scheduler.start()

    logger.info(
        "Scheduler started — warmup_cache() fires immediately, then: "
        "stock data every %dmin | news every %dmin | ML every %dmin",
        settings.schedule_stock_minutes,
        settings.schedule_news_minutes,
        settings.schedule_ml_minutes,
    )
    logger.info("API docs: http://%s:%d/docs", settings.api_host, settings.api_port)

    # ── APPLICATION RUNS HERE ─────────────────────────────────────────────────
    yield

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    logger.info("MarketPulse API shutting down")

    if hasattr(app.state, "scheduler") and app.state.scheduler.running:
        # wait=False: signal jobs to stop but don't wait for completion.
        # Prevents the server hanging if the ML job is mid-training at shutdown.
        app.state.scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")

    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """
    FastAPI application factory.

    Creates a new FastAPI application instance with all middleware and routers
    registered. Called by uvicorn with the --factory flag.

    Rationale for factory pattern:
    - Tests call create_app() to get a fresh isolated instance per test
    - uvicorn's --factory calls it exactly once at startup
    - Prevents module-level side effects from app = FastAPI() at import time

    Returns:
        Configured FastAPI application, ready to handle HTTP requests.
        Lifespan (scheduler start/stop) is managed automatically.
    """
    app = FastAPI(
        title="MarketPulse API",
        description=(
            "Real-time stock market intelligence pipeline. "
            "Provides OHLCV prices, technical indicators (SMA/RSI/MACD/BB), "
            "ML-generated BUY/HOLD/SELL signals, and news sentiment scores."
        ),
        version="0.1.0",
        lifespan=_lifespan,
        # Interactive docs auto-generated from route type annotations
        docs_url="/docs",       # Swagger UI — try every endpoint in the browser
        redoc_url="/redoc",     # ReDoc — cleaner read-only documentation
        openapi_url="/openapi.json",
    )

    # Register middleware (CORS outermost, Timing innermost — see middleware.py)
    setup_middleware(app)

    # Register all API routers under the /api/v1 prefix
    api_prefix = "/api/v1"
    app.include_router(health.router, prefix=api_prefix)
    app.include_router(stocks.router, prefix=api_prefix)
    app.include_router(signals.router, prefix=api_prefix)
    app.include_router(news.router, prefix=api_prefix)

    logger.debug(
        "Routes registered: %s",
        sorted({r.path for r in app.routes if hasattr(r, "path")}),
    )

    return app
