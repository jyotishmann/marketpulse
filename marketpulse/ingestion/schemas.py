# marketpulse/ingestion/schemas.py


from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# Schema 1: Raw OHLCV price bar from yfinance


class RawOHLCVRow(BaseModel):
    """
    One 15-minute price bar returned by yfinance.download().

    Field validators reject impossible prices (negative, NaN) before
    they reach the database. The model_validator checks cross-field
    constraints (high >= low, close within range).
    """

    model_config = ConfigDict(
        extra="ignore",  # silently drop Dividends, Stock Splits, etc.
        str_strip_whitespace=True,
    )

    ticker: str
    timestamp: datetime  # timezone-aware; yfinance returns tz-aware values
    open: float
    high: float
    low: float
    close: float
    volume: int

    # Field-level validators

    @field_validator("ticker")
    @classmethod
    def normalise_ticker(cls, v: str) -> str:
        """Strip whitespace and uppercase the ticker symbol."""
        cleaned = v.strip().upper()
        if not cleaned:
            raise ValueError("ticker cannot be empty")
        return cleaned

    @field_validator("open", "high", "low", "close")
    @classmethod
    def price_must_be_positive(cls, v: float) -> float:
        """Reject zero, negative, or NaN prices."""
        if v != v:  # NaN check: NaN is the only float not equal to itself
            raise ValueError("price is NaN")
        if v <= 0:
            raise ValueError(f"price must be positive, got {v}")
        return round(v, 4)  # normalise floating-point noise (182.30000001 → 182.3)

    @field_validator("volume")
    @classmethod
    def volume_non_negative(cls, v: int) -> int:
        """Volume of 0 is valid (halted trading); negative volume is not."""
        if v < 0:
            raise ValueError(f"volume must be >= 0, got {v}")
        return v

    # Cross-field validator (runs after all field validators pass)

    @model_validator(mode="after")
    def validate_ohlcv_relationships(self) -> RawOHLCVRow:
        """
        Validate relationships between price fields.

        These constraints define what a well-formed candlestick bar looks like:
        - high is the highest price traded: must be >= low, open, and close
        - low is the lowest price traded: must be <= high, open, and close
        - close is the last traded price: must be between low and high
        """
        tol = 0.01  # 1 cent tolerance for float arithmetic noise

        if self.high < self.low - tol:
            raise ValueError(f"high ({self.high}) must be >= low ({self.low})")
        if not (self.low - tol <= self.close <= self.high + tol):
            raise ValueError(
                f"close ({self.close}) outside range [low={self.low}, high={self.high}]"
            )
        return self


# Schema 2: Raw news article from an RSS feed entry


class RawNewsItem(BaseModel):
    """
    One headline from a parsed RSS feed entry.

    The connector (news.py) handles all feedparser-specific date parsing
    before constructing this schema. By the time this schema is instantiated,
    all values should already be Python-native types.
    """

    model_config = ConfigDict(
        extra="ignore",  # RSS entries carry many extra fields we ignore
        str_strip_whitespace=True,
    )

    title: str
    source_url: str
    published_at: datetime  # must be timezone-aware

    # Field-level validators

    @field_validator("title")
    @classmethod
    def title_not_empty(cls, v: str) -> str:
        """Reject empty or near-empty titles from broken RSS generators."""
        if len(v) < 5:
            raise ValueError(
                f"title too short ({len(v)} chars) — likely a broken feed entry: {v!r}"
            )
        return v

    @field_validator("source_url")
    @classmethod
    def url_must_be_absolute(cls, v: str) -> str:
        """Require an absolute HTTP/HTTPS URL for reliable deduplication."""
        if not v.startswith(("http://", "https://")):
            raise ValueError(
                f"source_url must be an absolute HTTP/HTTPS URL, got {v!r}"
            )
        return v
