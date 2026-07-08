# marketpulse/config.py
# Single source of truth for all application configuration.
# Every environment variable the application reads is declared here.

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    pydantic-settings validates every field's type on instantiation.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",          # don't crash on extra env vars (e.g. CI injections)
    )

    # ── Database ───────────────────────────────────────────────────────────────
    database_url: str            # required — no default (app must fail if missing)

    # ── Redis ──────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Tickers & Data Sources ─────────────────────────────────────────────────
    tickers: str = "AAPL,GOOGL,MSFT,TSLA"
    rss_feed_urls: str = ""      # empty default = no news ingestion until configured

    # ── Schedule (in minutes) ──────────────────────────────────────────────────
    schedule_stock_minutes: int = 15
    schedule_news_minutes: int = 30
    schedule_ml_minutes: int = 60

    # ── Cache TTLs (in seconds) ────────────────────────────────────────────────
    cache_ttl_prices: int = 300       # 5 minutes
    cache_ttl_indicators: int = 300   # 5 minutes
    cache_ttl_signals: int = 3600     # 1 hour
    cache_ttl_news: int = 600         # 10 minutes

    # ── Machine Learning ───────────────────────────────────────────────────────
    model_dir: str = "./models"
    ml_lookback_days: int = 90
    ml_anomaly_contamination: float = 0.05

    # ── API Server ─────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # ── Dashboard ──────────────────────────────────────────────────────────────
    dashboard_api_url: str = "http://localhost:8000/api/v1"

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is one of the standard Python logging levels."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            msg = f"log_level must be one of {allowed}, got {v!r}"
            raise ValueError(msg)
        return upper

    @field_validator("ml_anomaly_contamination")
    @classmethod
    def validate_contamination(cls, v: float) -> float:
        """IsolationForest contamination must be between 0 and 0.5."""
        if not 0 < v < 0.5:
            msg = f"ml_anomaly_contamination must be between 0 and 0.5, got {v}"
            raise ValueError(msg)
        return v

    # ── Computed properties (string → list conversions) ───────────────────────
    @property
    def ticker_list(self) -> list[str]:
        """Return TICKERS as a cleaned list of uppercase symbols."""
        return [t.strip().upper() for t in self.tickers.split(",") if t.strip()]

    @property
    def rss_url_list(self) -> list[str]:
        """Return RSS_FEED_URLS as a list of URL strings."""
        if not self.rss_feed_urls:
            return []
        return [u.strip() for u in self.rss_feed_urls.split(",") if u.strip()]


@lru_cache
def get_settings() -> Settings:
    """
    Return the application Settings singleton.

    Uses @lru_cache so Settings() is only instantiated once per process.
    In tests, call get_settings.cache_clear() then patch env vars to reset.

    Usage:
        from marketpulse.config import settings  # most common
        from marketpulse.config import get_settings  # for FastAPI Depends()
    """
    return Settings()


# Module-level singleton — import this for direct access anywhere in the codebase.
# FastAPI routes that need testable settings should use Depends(get_settings) instead.
settings: Settings = get_settings()
