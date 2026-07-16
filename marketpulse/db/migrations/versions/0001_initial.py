# marketpulse/db/migrations/versions/0001_initial.py
# First migration: creates all five tables and their indexes.
# Generated manually (not via --autogenerate) to teach Alembic op calls.
"""Initial database schema — create all five tables.

Revision ID: 31a8d4f6b2c9
Revises:
Create Date: 2024-01-15 10:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# Revision chain identifiers — Alembic uses these to order migrations
revision: str = "31a8d4f6b2c9"
down_revision: str | None = None  # first migration: nothing before it
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create all five tables and their indexes."""

    # ── Table 1: stock_prices ──────────────────────────────────────────────────
    op.create_table(
        "stock_prices",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(12, 4), nullable=False),
        sa.Column("high", sa.Numeric(12, 4), nullable=False),
        sa.Column("low", sa.Numeric(12, 4), nullable=False),
        sa.Column("close", sa.Numeric(12, 4), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ticker", "timestamp", name="uq_stock_prices_ticker_ts"),
    )
    op.create_index(
        "ix_stock_prices_ticker_ts", "stock_prices", ["ticker", "timestamp"]
    )

    # ── Table 2: technical_indicators ─────────────────────────────────────────
    op.create_table(
        "technical_indicators",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sma_20", sa.Numeric(12, 4), nullable=True),
        sa.Column("sma_50", sa.Numeric(12, 4), nullable=True),
        sa.Column("sma_200", sa.Numeric(12, 4), nullable=True),
        sa.Column("ema_12", sa.Numeric(12, 4), nullable=True),
        sa.Column("ema_26", sa.Numeric(12, 4), nullable=True),
        sa.Column("rsi_14", sa.Numeric(6, 4), nullable=True),
        sa.Column("macd", sa.Numeric(10, 6), nullable=True),
        sa.Column("macd_signal", sa.Numeric(10, 6), nullable=True),
        sa.Column("bb_upper", sa.Numeric(12, 4), nullable=True),
        sa.Column("bb_lower", sa.Numeric(12, 4), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ticker", "timestamp", name="uq_indicators_ticker_ts"),
    )

    # ── Table 3: news_articles ────────────────────────────────────────────────
    op.create_table(
        "news_articles",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sentiment_positive", sa.Numeric(5, 4), nullable=False),
        sa.Column("sentiment_negative", sa.Numeric(5, 4), nullable=False),
        sa.Column("sentiment_neutral", sa.Numeric(5, 4), nullable=False),
        sa.Column("sentiment_compound", sa.Numeric(5, 4), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_url", name="uq_news_articles_source_url"),
    )
    op.create_index("ix_news_published", "news_articles", ["published_at"])

    # ── Table 4: ml_signals ───────────────────────────────────────────────────
    op.create_table(
        "ml_signals",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("signal", sa.String(length=4), nullable=False),
        sa.Column("confidence", sa.Numeric(5, 4), nullable=False),
        sa.Column("is_anomaly", sa.Boolean(), nullable=False),
        sa.Column("model_version", sa.String(length=50), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "signal IN ('BUY', 'HOLD', 'SELL')",
            name="ck_ml_signals_valid_signal",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("ticker", "timestamp", name="uq_ml_signals_ticker_ts"),
    )
    op.create_index("ix_ml_signals_ticker_ts", "ml_signals", ["ticker", "timestamp"])

    # ── Table 5: model_registry ───────────────────────────────────────────────
    op.create_table(
        "model_registry",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("ticker", sa.String(length=10), nullable=False),
        sa.Column("model_type", sa.String(length=50), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("accuracy", sa.Numeric(5, 4), nullable=True),
        sa.Column(
            "trained_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    """Drop all five tables in reverse order of creation."""
    # Drop indexes explicitly before tables (Alembic does this automatically on
    # most DBs, but being explicit makes downgrade() self-documenting)
    op.drop_index("ix_ml_signals_ticker_ts", table_name="ml_signals")
    op.drop_table("ml_signals")

    op.drop_index("ix_news_published", table_name="news_articles")
    op.drop_table("news_articles")

    op.drop_table("technical_indicators")

    op.drop_index("ix_stock_prices_ticker_ts", table_name="stock_prices")
    op.drop_table("stock_prices")

    op.drop_table("model_registry")
