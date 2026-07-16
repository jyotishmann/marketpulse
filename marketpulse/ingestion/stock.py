# marketpulse/ingestion/stock.py
# yfinance connector: fetches OHLCV price bars and validates them.
# Does NOT write to the database — that is the ETL/scheduler's job.

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf
from pydantic import ValidationError

from marketpulse.ingestion.schemas import RawOHLCVRow

logger = logging.getLogger(__name__)


# ── Internal helper ────────────────────────────────────────────────────────────


def _to_flat_df(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalise a yfinance DataFrame to a flat structure we can iterate over.

    yfinance >= 0.2.48 returns MultiIndex columns even for single-ticker
    downloads. Older versions return flat string columns. This function
    handles both and produces a consistent flat DataFrame.

    Args:
        raw:    The DataFrame returned by yf.download()
        ticker: The ticker symbol (added as a column)

    Returns:
        DataFrame with lowercase columns: timestamp, open, high, low,
        close, volume, ticker.
    """
    df = raw.copy()

    # Reset the DatetimeIndex → regular column named "Datetime" or "Date"
    df = df.reset_index()

    # Flatten MultiIndex columns: ("Open", "AAPL") → "Open"
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # Rename yfinance column names to our internal snake_case names
    rename_map = {
        "Datetime": "timestamp",
        "Date": "timestamp",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=rename_map)

    # Add the ticker as a column so each row is self-describing
    df["ticker"] = ticker

    return df


# ── Public API ─────────────────────────────────────────────────────────────────


def fetch_ohlcv(
    ticker: str,
    period: str = "2d",
    interval: str = "15m",
) -> list[RawOHLCVRow]:
    """
    Fetch OHLCV bars for one ticker from Yahoo Finance.

    Args:
        ticker:   Stock symbol, e.g. "AAPL"
        period:   How far back to fetch. Valid values: "1d", "5d", "1mo",
                  "3mo", "6mo", "1y", "2y", "5y", "ytd", "max".
                  Use "2d" for pipeline updates (catches any missed bars).
                  Use "90d" for initial data load (enough for SMA-200... wait
                  SMA-200 needs 200 bars × 15 min. For daily bars use "1y").
        interval: Bar size. Valid values: "1m", "5m", "15m", "30m", "60m",
                  "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo".
                  Note: intraday intervals (< 1d) only available for last 60 days.

    Returns:
        List of validated RawOHLCVRow objects. Returns [] on any error
        (connection failure, empty response, all rows invalid).
    """
    logger.info(
        "yfinance.download('%s', period='%s', interval='%s')", ticker, period, interval
    )

    # ── Step 1: Fetch raw data from Yahoo Finance ─────────────────────────────
    try:
        raw: pd.DataFrame = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,  # suppress tqdm progress bar in container logs
            auto_adjust=True,  # adjust for splits and dividends (continuous chart)
        )
    except Exception:
        # logger.exception includes the full traceback — crucial for debugging
        logger.exception("yf.download() raised an exception for %s", ticker)
        return []

    if raw.empty:
        logger.warning(
            "yfinance returned empty DataFrame for %s (market closed?)", ticker
        )
        return []

    # ── Step 2: Normalise the DataFrame structure ─────────────────────────────
    df = _to_flat_df(raw, ticker)

    # Drop rows missing any required column (NaN in core fields)
    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    before_drop = len(df)
    df = df.dropna(subset=required_cols)
    if len(df) < before_drop:
        logger.warning(
            "Dropped %d rows with NaN values for %s",
            before_drop - len(df),
            ticker,
        )

    # ── Step 3: Validate each row with pydantic ───────────────────────────────
    validated: list[RawOHLCVRow] = []
    skipped = 0

    for _, row in df.iterrows():
        try:
            bar = RawOHLCVRow(
                ticker=ticker,
                timestamp=row["timestamp"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
            )
            validated.append(bar)
        except ValidationError as exc:
            # Log at DEBUG level — individual bad rows are expected occasionally
            logger.debug(
                "Skipping invalid bar for %s at %s: %s",
                ticker,
                row.get("timestamp"),
                exc.errors()[0]["msg"],
            )
            skipped += 1

    if skipped:
        logger.warning("Skipped %d invalid bars for %s", skipped, ticker)

    logger.info("Fetched and validated %d bars for %s", len(validated), ticker)
    return validated


def fetch_multiple_tickers(
    tickers: list[str],
    period: str = "2d",
    interval: str = "15m",
) -> dict[str, list[RawOHLCVRow]]:
    """
    Fetch OHLCV bars for multiple tickers, one by one.

    Calls fetch_ohlcv() for each ticker independently. A failure for one
    ticker (e.g. invalid symbol, connection timeout) does not stop the
    others — it logs a warning and stores an empty list for that ticker.

    Args:
        tickers:  List of ticker symbols, e.g. ["AAPL", "GOOGL", "MSFT"]
        period:   Passed through to yf.download() for each ticker
        interval: Passed through to yf.download() for each ticker

    Returns:
        Dict mapping ticker → list[RawOHLCVRow]. Tickers that failed
        or returned no data have an empty list value.

    Example:
        results = fetch_multiple_tickers(["AAPL", "GOOGL"])
        for ticker, bars in results.items():
            print(f"{ticker}: {len(bars)} bars")
    """
    results: dict[str, list[RawOHLCVRow]] = {}

    for ticker in tickers:
        results[ticker] = fetch_ohlcv(ticker, period=period, interval=interval)

    total_bars = sum(len(v) for v in results.values())
    successful = [t for t, v in results.items() if v]
    failed = [t for t, v in results.items() if not v]

    logger.info(
        "fetch_multiple_tickers: %d total bars across %d tickers "
        "(successful: %s, no data: %s)",
        total_bars,
        len(tickers),
        successful,
        failed,
    )
    return results
