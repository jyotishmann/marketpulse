# marketpulse/ml/features.py
# Feature engineering: convert DB rows into model-ready X (features) and y (labels).
# All features are ratio-based and scale-invariant across price regimes.

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta, timezone  # noqa: F401

import pandas as pd
from sqlalchemy.orm import Session

from marketpulse.db import StockPrice, TechnicalIndicator

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Canonical feature list — single source of truth used by train and predict
FEATURE_COLS: list[str] = [
    "rsi_14",  # RSI-14 momentum oscillator (range: 0–100)
    "macd",  # MACD line (EMA-12 minus EMA-26)
    "macd_signal",  # MACD signal line (9-bar EMA of MACD)
    "close_to_sma20",  # (close / sma_20) - 1: price vs short-term average
    "close_to_sma50",  # (close / sma_50) - 1: price vs medium-term average
    "ema_cross",  # (ema_12 / ema_26) - 1: EMA crossover ratio
    "bb_position",  # (close - bb_lower) / (bb_upper - bb_lower): 0=lower, 1=upper
    "volume_change",  # volume / rolling_20_mean(volume) - 1: relative volume
]

LOOKAHEAD: int = 3  # bars ahead for label (3 × 15min = 45min horizon)
THRESHOLD: float = 0.005  # 0.5% minimum move to classify as BUY or SELL


# ── Internal helpers ───────────────────────────────────────────────────────────


def _make_labels(
    close: pd.Series,
    lookahead: int = LOOKAHEAD,
    threshold: float = THRESHOLD,
) -> pd.Series:
    """
    Generate BUY / HOLD / SELL labels from future price returns.

    For each bar at index i:
    - Compute future_return = (close[i+lookahead] / close[i]) - 1
    - If future_return > +threshold  → BUY  (+1)
    - If future_return < -threshold  → SELL (-1)
    - Otherwise                      → HOLD ( 0)

    The last `lookahead` rows have no future data → returned as NaN.
    Callers should use dropna() to remove them.

    Args:
        close:     Closing price Series, sorted ascending by time.
        lookahead: How many bars ahead to look for the return.
        threshold: Minimum absolute return to classify as BUY/SELL.

    Returns:
        Series of float: 1.0 (BUY), 0.0 (HOLD), -1.0 (SELL), NaN (no future data).
    """
    # future_return[i] = how much the price changed lookahead bars after bar i
    future_return = close.shift(-lookahead) / close - 1

    # Start with NaN — only assign labels where future_return is defined
    labels = pd.Series(float("nan"), index=close.index, dtype=float)

    mask = future_return.notna()  # last `lookahead` rows have NaN future_return
    labels[mask & (future_return > threshold)] = 1.0  # BUY
    labels[mask & (future_return < -threshold)] = -1.0  # SELL
    # HOLD: return is defined but within threshold band
    is_hold = mask & ~(future_return > threshold) & ~(future_return < -threshold)
    labels[is_hold] = 0.0

    buy_count = (labels == 1.0).sum()
    sell_count = (labels == -1.0).sum()
    hold_count = (labels == 0.0).sum()
    logger.debug(
        "Label distribution: BUY=%d HOLD=%d SELL=%d (threshold=%.3f)",
        buy_count,
        hold_count,
        sell_count,
        threshold,
    )
    return labels


# ── Public API ─────────────────────────────────────────────────────────────────


def build_feature_matrix(
    ticker: str,
    session: Session,
    lookback_days: int = 90,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Build the ML feature matrix (X) and label vector (y) from the database.

    Loads the last `lookback_days` of prices and indicators, engineers
    ratio features, generates forward-looking labels, and drops rows with
    any NaN in features or labels.

    Args:
        ticker:        Stock symbol.
        session:       Active SQLAlchemy session.
        lookback_days: How many calendar days of history to use.
                       90 days ≈ 1,800 rows at 15-min interval (3 bars/hr × 6 hrs × 90 days × 0.5 efficiency).

    Returns:
        Tuple of (X, y, feature_names):
        - X: DataFrame with FEATURE_COLS columns. Empty if insufficient data.
        - y: Integer Series of labels (-1, 0, 1). Empty if insufficient data.
        - feature_names: List of column names (FEATURE_COLS constant).

    Minimum data: 50 rows required (enforced by classifier.py). Returns
    empty X, y if fewer rows available after cleaning.
    """
    cutoff = datetime.now(tz=UTC) - timedelta(days=lookback_days)

    # ── Query prices ──────────────────────────────────────────────────────────
    price_rows = (
        session.query(StockPrice)
        .filter(StockPrice.ticker == ticker, StockPrice.timestamp >= cutoff)
        .order_by(StockPrice.timestamp.asc())
        .all()
    )

    if not price_rows:
        logger.warning("No price rows for %s (lookback=%d days)", ticker, lookback_days)
        return pd.DataFrame(), pd.Series(dtype=int), FEATURE_COLS

    # ── Query indicators ──────────────────────────────────────────────────────
    indicator_rows = (
        session.query(TechnicalIndicator)
        .filter(
            TechnicalIndicator.ticker == ticker, TechnicalIndicator.timestamp >= cutoff
        )
        .order_by(TechnicalIndicator.timestamp.asc())
        .all()
    )

    if not indicator_rows:
        logger.warning("No indicator rows for %s — run the ingestion job first", ticker)
        return pd.DataFrame(), pd.Series(dtype=int), FEATURE_COLS

    # ── Build price DataFrame ─────────────────────────────────────────────────
    price_df = pd.DataFrame(
        [
            {
                "timestamp": r.timestamp,
                "close": float(r.close),
                "volume": int(r.volume),
            }
            for r in price_rows
        ]
    )

    # ── Build indicator DataFrame ─────────────────────────────────────────────
    def _to_float(v: object) -> float | None:
        """Convert Decimal/None to float, preserving None."""
        return float(v) if v is not None else None  # type: ignore[arg-type]

    ind_df = pd.DataFrame(
        [
            {
                "timestamp": r.timestamp,
                "sma_20": _to_float(r.sma_20),
                "sma_50": _to_float(r.sma_50),
                "ema_12": _to_float(r.ema_12),
                "ema_26": _to_float(r.ema_26),
                "rsi_14": _to_float(r.rsi_14),
                "macd": _to_float(r.macd),
                "macd_signal": _to_float(r.macd_signal),
                "bb_upper": _to_float(r.bb_upper),
                "bb_lower": _to_float(r.bb_lower),
            }
            for r in indicator_rows
        ]
    )

    # ── Merge prices and indicators on timestamp ───────────────────────────────
    df = pd.merge(price_df, ind_df, on="timestamp", how="inner")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        "build_feature_matrix: %s — %d rows after merge (prices=%d, indicators=%d)",
        ticker,
        len(df),
        len(price_rows),
        len(indicator_rows),
    )

    # ── Engineer ratio features (scale-invariant) ─────────────────────────────
    # close_to_sma20: positive = price above average (bullish), negative = below
    df["close_to_sma20"] = df["close"] / df["sma_20"] - 1

    # close_to_sma50: same but medium-term average
    df["close_to_sma50"] = df["close"] / df["sma_50"] - 1

    # ema_cross: positive = fast EMA above slow EMA (bullish momentum)
    df["ema_cross"] = df["ema_12"] / df["ema_26"] - 1

    # bb_position: 0 = at lower band (oversold), 1 = at upper band (overbought)
    bb_width = (df["bb_upper"] - df["bb_lower"]).replace(0.0, float("nan"))
    df["bb_position"] = (df["close"] - df["bb_lower"]) / bb_width

    # volume_change: relative volume vs 20-bar rolling average
    # positive = above-average volume (confirms moves), negative = thin volume
    rolling_vol = df["volume"].rolling(20, min_periods=20).mean()
    df["volume_change"] = df["volume"] / rolling_vol - 1

    # ── Generate labels ────────────────────────────────────────────────────────
    df["y"] = _make_labels(df["close"])

    # ── Drop rows with NaN in any feature or label ────────────────────────────
    # This removes: early bars without enough history, last LOOKAHEAD bars (no future),
    # and any rows where sma_50 / bb_upper etc. are still NaN.
    before_drop = len(df)
    df = df.dropna(subset=FEATURE_COLS + ["y"])
    logger.info(
        "Dropped %d rows with NaN features/labels, %d rows remain",
        before_drop - len(df),
        len(df),
    )

    if len(df) < 50:
        logger.warning(
            "%s: only %d usable rows after cleaning (minimum 50 required)",
            ticker,
            len(df),
        )
        return pd.DataFrame(), pd.Series(dtype=int), FEATURE_COLS

    X = df[FEATURE_COLS].copy()
    y = df["y"].astype(int)

    return X, y, FEATURE_COLS
