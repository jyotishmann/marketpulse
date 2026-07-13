# tests/unit/test_processing.py
# Unit tests for the processing layer (ETL and technical indicators).
# All tests use synthetic DataFrames — no database or network calls.

from __future__ import annotations

from datetime import datetime, timezone  # noqa: F401

import numpy as np  # noqa: F401
import pandas as pd
import pytest

from marketpulse.processing.etl import rows_to_ohlcv_df, score_sentiment
from marketpulse.processing.indicators import (
    INDICATOR_COLS,
    _compute_rsi,
    compute_all,
)

# ══════════════════════════════════════════════════════════════════════════════
# ETL function tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRowsToOhlcvDf:
    """Tests for rows_to_ohlcv_df() conversion and cleaning."""

    def test_converts_rows_to_dataframe(self, raw_ohlcv_rows):
        df = rows_to_ohlcv_df(raw_ohlcv_rows)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert set(df.columns) >= {"ticker", "timestamp", "open", "high", "low", "close", "volume"}

    def test_empty_input_returns_empty_df(self):
        df = rows_to_ohlcv_df([])
        assert df.empty

    def test_timestamps_are_utc_aware(self, raw_ohlcv_rows):
        df = rows_to_ohlcv_df(raw_ohlcv_rows)
        assert df["timestamp"].dt.tz is not None

    def test_sorted_ascending_by_timestamp(self, raw_ohlcv_rows):
        # Reverse the input order to verify sorting
        reversed_rows = list(reversed(raw_ohlcv_rows))
        df = rows_to_ohlcv_df(reversed_rows)
        ts = df["timestamp"].tolist()
        assert ts == sorted(ts)

    def test_deduplicates_by_ticker_timestamp(self, raw_ohlcv_rows):
        """Duplicate rows should be dropped — only one row per (ticker, timestamp)."""
        doubled = raw_ohlcv_rows + raw_ohlcv_rows  # 4 rows, but 2 unique
        df = rows_to_ohlcv_df(doubled)
        assert len(df) == 2  # deduped back to 2


# ══════════════════════════════════════════════════════════════════════════════
# Sentiment scoring tests
# ══════════════════════════════════════════════════════════════════════════════

class TestScoreSentiment:
    """Tests for VADER sentiment scoring."""

    def test_returns_four_keys(self):
        scores = score_sentiment("Apple stock hits all-time high")
        assert set(scores.keys()) == {"pos", "neg", "neu", "compound"}

    def test_compound_within_range(self):
        scores = score_sentiment("market crash recession fear")
        assert -1.0 <= scores["compound"] <= 1.0

    def test_positive_headline_has_positive_compound(self):
        scores = score_sentiment("record profits soar earnings beat")
        assert scores["compound"] > 0

    def test_negative_headline_has_negative_compound(self):
        scores = score_sentiment("crash collapse bankruptcy disaster loss")
        assert scores["compound"] < 0

    def test_components_sum_to_one(self):
        scores = score_sentiment("Apple released a new product today")
        total = scores["pos"] + scores["neg"] + scores["neu"]
        assert abs(total - 1.0) < 0.01  # allow rounding tolerance


# ══════════════════════════════════════════════════════════════════════════════
# Technical indicator tests
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeAll:
    """Tests for the full indicator computation pipeline."""

    def test_returns_all_indicator_columns(self, large_ohlcv_df):
        result = compute_all(large_ohlcv_df)
        for col in INDICATOR_COLS:
            assert col in result.columns, f"Missing indicator column: {col}"

    def test_sma20_nan_for_first_19_rows(self, large_ohlcv_df):
        result = compute_all(large_ohlcv_df)
        # Rows 0–18 (first 19) have < 20 data points → SMA-20 must be NaN
        assert result["sma_20"].iloc[:19].isna().all()

    def test_sma20_non_nan_from_row_19(self, large_ohlcv_df):
        result = compute_all(large_ohlcv_df)
        # Row index 19 has exactly 20 data points → first non-NaN SMA-20
        assert pd.notna(result["sma_20"].iloc[19])

    def test_sma200_non_nan_from_row_199(self, large_ohlcv_df):
        result = compute_all(large_ohlcv_df)
        # Row 199 is the first with 200 data points for SMA-200
        assert result["sma_200"].iloc[:199].isna().all()
        assert pd.notna(result["sma_200"].iloc[199])

    def test_rsi_within_0_and_100(self, large_ohlcv_df):
        result = compute_all(large_ohlcv_df)
        valid_rsi = result["rsi_14"].dropna()
        assert len(valid_rsi) > 0
        assert (valid_rsi >= 0).all(), "RSI below 0"
        assert (valid_rsi <= 100).all(), "RSI above 100"

    def test_bollinger_upper_greater_than_lower(self, large_ohlcv_df):
        result = compute_all(large_ohlcv_df)
        valid = result.dropna(subset=["bb_upper", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_macd_is_ema12_minus_ema26(self, large_ohlcv_df):
        result = compute_all(large_ohlcv_df)
        valid = result.dropna(subset=["macd", "ema_12", "ema_26"])
        expected_macd = valid["ema_12"] - valid["ema_26"]
        pd.testing.assert_series_equal(
            valid["macd"].round(4),
            expected_macd.round(4),
            check_names=False,
        )

    def test_does_not_mutate_input_dataframe(self, large_ohlcv_df):
        """compute_all() must return a copy — never modify the caller's DataFrame."""
        original_cols = set(large_ohlcv_df.columns.tolist())
        _ = compute_all(large_ohlcv_df)
        # Input should NOT have gained any indicator columns
        assert set(large_ohlcv_df.columns.tolist()) == original_cols

    def test_empty_dataframe_returns_empty(self):
        empty = pd.DataFrame(columns=["ticker", "timestamp", "close"])
        result = compute_all(empty)
        assert result.empty

    def test_returns_more_rows_than_sma200_requires(self, large_ohlcv_df):
        """250-row input should produce at least 51 non-NaN SMA-200 values."""
        result = compute_all(large_ohlcv_df)
        non_null = result["sma_200"].notna().sum()
        assert non_null >= 50  # 250 - 200 = 50 valid rows


class TestComputeRSI:
    """Tests for the RSI helper function directly."""

    def test_all_gains_gives_100(self):
        """If every bar went up, RSI should be 100 (or very close)."""
        close = pd.Series([100.0 + i for i in range(30)])  # strictly increasing
        rsi = _compute_rsi(close, period=14)
        assert rsi.dropna().iloc[-1] == pytest.approx(100.0, abs=1.0)

    def test_all_losses_gives_low_rsi(self):
        """If every bar went down, RSI should be very low."""
        close = pd.Series([100.0 - i * 0.5 for i in range(30)])
        rsi = _compute_rsi(close, period=14)
        assert rsi.dropna().iloc[-1] < 10
