# marketpulse/processing/indicators.py
# Computes all ten technical indicators using pandas rolling-window operations.
# Persists computed indicators to the technical_indicators table.

from __future__ import annotations

import logging
from decimal import Decimal

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from marketpulse.db import TechnicalIndicator

logger = logging.getLogger(__name__)

# All indicator column names — used for DataFrame column selection and DB write
INDICATOR_COLS = [
    "sma_20",
    "sma_50",
    "sma_200",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd",
    "macd_signal",
    "bb_upper",
    "bb_lower",
]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _float_or_none(value: object) -> Decimal | None:
    """
    Convert a float/numpy scalar to Decimal for database insertion.

    Treats NaN, None, and non-numeric values as NULL (Python None),
    which SQLAlchemy maps to SQL NULL for Nullable columns.

    Uses str() conversion to avoid floating-point representation errors
    (same pattern as etl.py upsert functions).
    """
    if value is None:
        return None
    try:
        f = float(value)  # type: ignore[arg-type]
        if f != f:  # NaN check: NaN is the only float not equal to itself
            return None
        return Decimal(str(round(f, 6)))
    except (TypeError, ValueError):
        return None


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI using Wilder's smoothing method.

    RSI interpretation:
    - RSI > 70: overbought (potential reversal / pullback signal)
    - RSI < 30: oversold (potential bounce / buy signal)
    - RSI = 50: neutral momentum

    Algorithm:
    1. delta  = price change vs previous bar
    2. gain   = delta when positive, else 0
    3. loss   = abs(delta) when negative, else 0
    4. avg_gain, avg_loss = Wilder EMA with alpha = 1/period
    5. RS  = avg_gain / avg_loss
    6. RSI = 100 - 100 / (1 + RS)

    Args:
        close:  Series of closing prices, sorted ascending by time.
        period: RSI window length. Standard is 14.

    Returns:
        Series of RSI values (float, 0.0–100.0). NaN for first period-1 bars.
    """
    # Step 1: price change vs previous bar (NaN for first row)
    delta = close.diff()

    # Step 2 & 3: separate gains and losses
    gain = delta.where(delta > 0, 0.0)   # positive changes only, else 0
    loss = (-delta).where(delta < 0, 0.0) # absolute value of negative changes

    # Step 4: Wilder's smoothing — EWM with alpha = 1/period
    # alpha=1/14 ≈ 0.0714 gives more weight to older values vs standard EMA
    # min_periods=period: don't emit values until we have a full window
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    # Step 5: RS ratio — replace 0 avg_loss with NaN to avoid ZeroDivisionError
    # (if all bars were up, avg_loss=0, RS=inf, RSI=100 — handle via fillna)
    rs = avg_gain / avg_loss.replace(0.0, float("nan"))

    # Step 6: RSI formula
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Edge case: avg_loss was 0 (all-gains window) → RSI should be 100
    rsi = rsi.fillna(100.0)

    return rsi.round(4)

# ══════════════════════════════════════════════════════════════════════════════
# Public API: compute all indicators on an OHLCV DataFrame
# ══════════════════════════════════════════════════════════════════════════════

def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all ten technical indicators and add them as columns.

    The input DataFrame must be sorted ascending by timestamp and have
    a 'close' column (at minimum). Enough rows are needed for the
    longest indicator: SMA-200 requires at least 200 rows.

    Indicators computed:
    ┌─────────────┬────────────────────────────────────────────────────┐
    │ Column      │ Description                                        │
    ├─────────────┼────────────────────────────────────────────────────┤
    │ sma_20      │ Simple Moving Avg, 20 bars (short-term trend)      │
    │ sma_50      │ Simple Moving Avg, 50 bars (medium-term trend)     │
    │ sma_200     │ Simple Moving Avg, 200 bars (long-term trend)      │
    │ ema_12      │ Exponential MA, 12 bars (fast; used in MACD)       │
    │ ema_26      │ Exponential MA, 26 bars (slow; used in MACD)       │
    │ rsi_14      │ Relative Strength Index, 14 bars (0–100)           │
    │ macd        │ MACD line = EMA-12 minus EMA-26                    │
    │ macd_signal │ Signal = 9-bar EMA of MACD                         │
    │ bb_upper    │ Bollinger Upper = SMA-20 + 2 × StdDev(20)         │
    │ bb_lower    │ Bollinger Lower = SMA-20 − 2 × StdDev(20)         │
    └─────────────┴────────────────────────────────────────────────────┘

    Args:
        df: OHLCV DataFrame sorted ascending. Must have 'close' column.
            Should also have 'ticker' and 'timestamp' for logging.

    Returns:
        New DataFrame (copy of input) with 10 additional indicator columns.
        Early rows where not enough history exists will have NaN values.
    """
    if df.empty:
        logger.warning("compute_all called with empty DataFrame — returning as-is")
        return df.copy()

    result = df.copy()   # never mutate the caller's DataFrame
    close = result["close"]

    ticker = result["ticker"].iloc[0] if "ticker" in result.columns else "?"
    n_rows = len(result)
    logger.info("Computing indicators for %s (%d rows)", ticker, n_rows)

    if n_rows < 20:
        logger.warning(
            "%s: only %d rows — most indicators will be NaN (need 200 for SMA-200)",
            ticker, n_rows,
        )

    # ── Simple Moving Averages ─────────────────────────────────────────────────
    # rolling(window=N, min_periods=N): mean of last N bars; NaN until N bars exist
    result["sma_20"] = close.rolling(window=20, min_periods=20).mean().round(4)
    result["sma_50"] = close.rolling(window=50, min_periods=50).mean().round(4)
    result["sma_200"] = close.rolling(window=200, min_periods=200).mean().round(4)

    # ── Exponential Moving Averages ────────────────────────────────────────────
    # adjust=False: recursive formula EMA_t = α×P_t + (1−α)×EMA_{t-1}
    # (industry standard — adjust=True uses a different formula, not what traders use)
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean().round(4)
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean().round(4)
    result["ema_12"] = ema_12
    result["ema_26"] = ema_26

    # ── RSI — uses Wilder's smoothing (see _compute_rsi for full explanation) ──
    result["rsi_14"] = _compute_rsi(close, period=14)

    # ── MACD (Moving Average Convergence/Divergence) ───────────────────────────
    # MACD line = fast EMA (12) minus slow EMA (26)
    # Positive MACD: short-term momentum > long-term (bullish)
    # Negative MACD: short-term momentum < long-term (bearish)
    macd_line = (ema_12 - ema_26).round(6)
    result["macd"] = macd_line

    # Signal line = 9-bar EMA of the MACD line
    # Crossover of MACD above signal = buy signal; below = sell signal
    result["macd_signal"] = (
        macd_line.ewm(span=9, adjust=False, min_periods=9).mean().round(6)
    )

    # ── Bollinger Bands ────────────────────────────────────────────────────────
    # Bands expand during high volatility, contract during low volatility
    # Price above upper band: potentially overbought
    # Price below lower band: potentially oversold
    sma_20 = result["sma_20"]  # already computed above
    # std() defaults to ddof=1 (sample std) — matches TradingView, Bloomberg
    std_20 = close.rolling(window=20, min_periods=20).std()
    result["bb_upper"] = (sma_20 + 2.0 * std_20).round(4)
    result["bb_lower"] = (sma_20 - 2.0 * std_20).round(4)

    # ── Summary log ───────────────────────────────────────────────────────────
    # Count how many rows have a non-NaN sma_20 (proxy for "indicator-ready" rows)
    ready_rows = result["sma_20"].notna().sum()
    logger.info(
        "%s: %d/%d rows have indicators (sma_20 non-null) "
        "[sma_200 non-null: %d]",
        ticker,
        ready_rows,
        n_rows,
        result["sma_200"].notna().sum(),
    )

    return result

# ══════════════════════════════════════════════════════════════════════════════
# Public API: persist computed indicators to the database
# ══════════════════════════════════════════════════════════════════════════════

def upsert_indicators(df: pd.DataFrame, session: Session) -> int:
    """
    Write indicator rows from compute_all() output to technical_indicators table.

    Filters to rows where sma_20 is non-null (skips early bars with
    insufficient history). Uses ON CONFLICT DO NOTHING to skip rows
    already in the database.

    Args:
        df:      DataFrame returned by compute_all() — must have 'ticker',
                 'timestamp', and all INDICATOR_COLS columns.
        session: Active SQLAlchemy session.

    Returns:
        Number of rows submitted to the upsert.
    """
    if df.empty:
        return 0

    # Only write rows that have at least sma_20 computed
    # (first 19 bars have no indicators at all — writing NULL-only rows is useless)
    has_indicators = df["sma_20"].notna()
    df_to_write = df[has_indicators].copy()

    if df_to_write.empty:
        ticker = df["ticker"].iloc[0] if "ticker" in df.columns else "?"
        logger.warning(
            "upsert_indicators: no indicator rows for %s "
            "(all sma_20 are NaN — need >= 20 rows of data)",
            ticker,
        )
        return 0

    # Build list of dicts for bulk INSERT
    # _float_or_none() converts NaN → None and float → Decimal(str(...))
    records = [
        {
            "ticker": str(row["ticker"]),
            "timestamp": row["timestamp"],
            **{col: _float_or_none(row.get(col)) for col in INDICATOR_COLS},
        }
        for _, row in df_to_write.iterrows()
    ]

    try:
        stmt = pg_insert(TechnicalIndicator).values(records)
        stmt = stmt.on_conflict_do_nothing(
            constraint="uq_indicators_ticker_ts",
        )
        session.execute(stmt)
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("upsert_indicators: database error, rolling back")
        raise

    ticker = df_to_write["ticker"].iloc[0]
    logger.info(
        "upsert_indicators: %d rows submitted for %s "
        "(sma_200 available for %d rows)",
        len(records),
        ticker,
        df_to_write["sma_200"].notna().sum(),
    )
    return len(records)
