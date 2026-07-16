# marketpulse/db/models.py
# SQLAlchemy ORM model definitions for all five database tables.
# Uses SQLAlchemy 2.0 Mapped[] annotation style throughout.

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy ORM models.

    All models inherit from Base. Base.metadata holds the schema registry
    that Alembic reads to detect changes and generate migrations.
    """


# ══════════════════════════════════════════════════════════════════════════════
# Table 1: stock_prices
# Stores raw OHLCV (Open, High, Low, Close, Volume) bars from yfinance.
# One row = one 15-minute price bar for one ticker.
# ══════════════════════════════════════════════════════════════════════════════
class StockPrice(Base):
    __tablename__ = "stock_prices"
    __table_args__ = (
        # Composite unique constraint: one bar per ticker per timestamp
        UniqueConstraint("ticker", "timestamp", name="uq_stock_prices_ticker_ts"),
        # Composite index: fast lookup of "all bars for AAPL, newest first"
        Index("ix_stock_prices_ticker_ts", "ticker", "timestamp"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,  # TIMESTAMPTZ in PostgreSQL
    )
    # Exact decimals — never use Float for financial prices
    open: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # created_at uses server-side default (PostgreSQL's clock, not Python's)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<StockPrice ticker={self.ticker!r} "
            f"timestamp={self.timestamp} close={self.close}>"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Table 2: technical_indicators
# Computed indicator values for each price bar.
# All indicator columns are Optional — early bars lack enough history to compute.
# ══════════════════════════════════════════════════════════════════════════════
class TechnicalIndicator(Base):
    __tablename__ = "technical_indicators"
    __table_args__ = (
        UniqueConstraint("ticker", "timestamp", name="uq_indicators_ticker_ts"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Simple Moving Averages — require N previous closes to compute
    sma_20: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    sma_50: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    sma_200: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))

    # Exponential Moving Averages — same period names as MACD inputs
    ema_12: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    ema_26: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))

    # RSI: 0.0 to 100.0, 4 decimal precision
    rsi_14: Mapped[Decimal | None] = mapped_column(Numeric(6, 4))

    # MACD = EMA-12 minus EMA-26; macd_signal = 9-period EMA of MACD
    macd: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))
    macd_signal: Mapped[Decimal | None] = mapped_column(Numeric(10, 6))

    # Bollinger Bands = SMA-20 ± 2 standard deviations
    bb_upper: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))
    bb_lower: Mapped[Decimal | None] = mapped_column(Numeric(12, 4))

    def __repr__(self) -> str:
        return (
            f"<TechnicalIndicator ticker={self.ticker!r} "
            f"timestamp={self.timestamp} rsi={self.rsi_14}>"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Table 3: news_articles
# Financial news headlines from RSS feeds with VADER sentiment scores.
# Deduplicated by source_url — each article stored exactly once.
# ══════════════════════════════════════════════════════════════════════════════
class NewsArticle(Base):
    __tablename__ = "news_articles"
    __table_args__ = (
        # Index published_at descending — dashboard queries "latest N articles"
        Index("ix_news_published", "published_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    # unique=True prevents storing the same article twice across polling cycles
    source_url: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # VADER sentiment scores — all values between -1.0 and +1.0
    sentiment_positive: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    sentiment_negative: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    sentiment_neutral: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)
    # compound is the single most useful score: -1=most negative, +1=most positive
    sentiment_compound: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<NewsArticle id={self.id} "
            f"compound={self.sentiment_compound} title={self.title[:40]!r}>"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Table 4: ml_signals
# BUY / HOLD / SELL predictions from the scikit-learn classifier,
# plus anomaly flags from the Isolation Forest.
# ══════════════════════════════════════════════════════════════════════════════
class MLSignal(Base):
    __tablename__ = "ml_signals"
    __table_args__ = (
        UniqueConstraint("ticker", "timestamp", name="uq_ml_signals_ticker_ts"),
        # Database-level check: signal must be one of exactly three values
        CheckConstraint(
            "signal IN ('BUY', 'HOLD', 'SELL')",
            name="ck_ml_signals_valid_signal",
        ),
        Index("ix_ml_signals_ticker_ts", "ticker", "timestamp"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # "BUY", "HOLD", or "SELL" — enforced by CheckConstraint above
    signal: Mapped[str] = mapped_column(String(4), nullable=False)

    # Classifier's confidence: predict_proba() result for the winning class (0.0–1.0)
    confidence: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)

    # True if IsolationForest flagged this bar as anomalous
    is_anomaly: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Version tag for the model that made this prediction (for audit / debugging)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<MLSignal ticker={self.ticker!r} signal={self.signal!r} "
            f"confidence={self.confidence} anomaly={self.is_anomaly}>"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Table 5: model_registry
# Tracks all trained model files, one row per training run.
# is_active=True marks the current model for each (ticker, model_type) pair.
# ══════════════════════════════════════════════════════════════════════════════
class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(10), nullable=False)

    # "classifier" (RandomForest BUY/HOLD/SELL) or "anomaly" (IsolationForest)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)

    # Absolute or relative path to the saved .pkl file
    file_path: Mapped[str] = mapped_column(Text, nullable=False)

    # Test-set accuracy for the classifier; NULL for anomaly detector (unsupervised)
    accuracy: Mapped[Decimal | None] = mapped_column(Numeric(5, 4))

    # When this model was trained (server-side timestamp)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Only one model per (ticker, model_type) should be active at a time
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    def __repr__(self) -> str:
        return (
            f"<ModelRegistry ticker={self.ticker!r} type={self.model_type!r} "
            f"active={self.is_active} accuracy={self.accuracy}>"
        )
