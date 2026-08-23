"""Immutable Decimal-safe USD-M derivatives source and context models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum


class DerivativesSource(str, Enum):
    FUNDING_RATE = "fundingRate"
    METRICS = "metrics"
    MARK_PRICE = "markPriceKlines"
    INDEX_PRICE = "indexPriceKlines"
    PREMIUM_INDEX = "premiumIndexKlines"


def decimal_value(name: str, value: object, *, non_negative: bool = False) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a valid Decimal.") from exc
    if not parsed.is_finite() or (non_negative and parsed < 0):
        raise ValueError(f"{name} is outside its valid range.")
    return parsed


def utc_timestamp(value: object, *, unit: str) -> datetime:
    """Normalize one explicitly identified source timestamp to UTC."""

    if unit == "milliseconds":
        if isinstance(value, bool) or not str(value).lstrip("-").isdigit():
            raise ValueError("Millisecond timestamp must be an integer.")
        raw = int(str(value))
        if raw < 946_684_800_000 or raw > 4_102_444_800_000:
            raise ValueError("Millisecond timestamp is impossible.")
        result = datetime.fromtimestamp(raw / 1000, tz=timezone.utc)
    elif unit == "iso8601":
        text = str(value).strip()
        if len(text) == 10:
            text += "T00:00:00+00:00"
        try:
            result = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError("ISO timestamp is invalid.") from exc
        if result.tzinfo is None or result.utcoffset() is None:
            # Binance's archived textual create_time/calc_time fields are UTC
            # even when the CSV omits an offset. The explicit iso8601 unit selects
            # this source convention; arbitrary unit guessing remains forbidden.
            result = result.replace(tzinfo=timezone.utc)
        result = result.astimezone(timezone.utc)
    else:
        raise ValueError("Timestamp unit must be explicit: milliseconds or iso8601.")
    return result


@dataclass(frozen=True, slots=True)
class FundingRateRecord:
    timestamp: datetime
    funding_rate: Decimal
    symbol: str = "BTCUSDT"
    funding_interval_hours: Decimal | None = None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() != timedelta(0):
            raise ValueError("Funding timestamp must be timezone-aware UTC.")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "funding_rate", decimal_value("funding_rate", self.funding_rate))
        if self.symbol != "BTCUSDT":
            raise ValueError("V8 derivatives context supports BTCUSDT only.")
        if self.funding_interval_hours is not None:
            interval = decimal_value(
                "funding_interval_hours", self.funding_interval_hours, non_negative=True
            )
            if interval == 0:
                raise ValueError("Funding interval must be positive when present.")
            object.__setattr__(self, "funding_interval_hours", interval)


@dataclass(frozen=True, slots=True)
class FuturesMetricsRecord:
    timestamp: datetime
    symbol: str
    sum_open_interest: Decimal
    sum_open_interest_value: Decimal
    count_toptrader_long_short_ratio: Decimal | None
    sum_toptrader_long_short_ratio: Decimal | None
    count_long_short_ratio: Decimal | None
    sum_taker_long_short_vol_ratio: Decimal | None

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() != timedelta(0):
            raise ValueError("Metrics timestamp must be timezone-aware UTC.")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        if self.symbol != "BTCUSDT":
            raise ValueError("V8 derivatives context supports BTCUSDT only.")
        for field in ("sum_open_interest", "sum_open_interest_value"):
            object.__setattr__(self, field, decimal_value(field, getattr(self, field), non_negative=True))
        for field in (
            "count_toptrader_long_short_ratio",
            "sum_toptrader_long_short_ratio",
            "count_long_short_ratio",
            "sum_taker_long_short_vol_ratio",
        ):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, decimal_value(field, value, non_negative=True))


@dataclass(frozen=True, slots=True)
class DerivativesPriceKline:
    source: DerivativesSource
    open_time: datetime
    close_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal

    def __post_init__(self) -> None:
        if self.source not in {
            DerivativesSource.MARK_PRICE,
            DerivativesSource.INDEX_PRICE,
            DerivativesSource.PREMIUM_INDEX,
        }:
            raise ValueError("Price kline source is invalid.")
        if any(
            timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0)
            for timestamp in (self.open_time, self.close_time)
        ) or self.close_time <= self.open_time:
            raise ValueError("Price kline timestamps are invalid.")
        for field in ("open", "high", "low", "close"):
            object.__setattr__(self, field, decimal_value(field, getattr(self, field)))
        if self.high < max(self.open, self.low, self.close) or self.low > min(
            self.open, self.high, self.close
        ):
            raise ValueError("Price kline OHLC values are inconsistent.")
        if self.source is not DerivativesSource.PREMIUM_INDEX and self.low <= 0:
            raise ValueError("Mark/index prices must be positive.")


@dataclass(frozen=True, slots=True)
class DerivativesContext15m:
    bucket_open_time: datetime
    bucket_close_time: datetime
    latest_known_funding_rate: Decimal | None
    funding_timestamp: datetime | None
    open_interest: Decimal | None
    open_interest_value: Decimal | None
    open_interest_timestamp: datetime | None
    mark_price: Decimal | None
    mark_price_timestamp: datetime | None
    index_price: Decimal | None
    index_price_timestamp: datetime | None
    premium_index: Decimal | None
    premium_index_timestamp: datetime | None
    premium_fraction: Decimal | None

    def __post_init__(self) -> None:
        if self.bucket_close_time != self.bucket_open_time + timedelta(minutes=15):
            raise ValueError("Derivatives context must span exactly 15 minutes.")
        timestamp_fields = (
            "funding_timestamp",
            "open_interest_timestamp",
            "mark_price_timestamp",
            "index_price_timestamp",
            "premium_index_timestamp",
        )
        if any(
            timestamp is not None and timestamp > self.bucket_close_time
            for timestamp in (getattr(self, field) for field in timestamp_fields)
        ):
            raise ValueError("Derivatives context contains a future observation.")

    @staticmethod
    def derive_premium(mark_price: Decimal, index_price: Decimal) -> Decimal:
        if index_price <= 0:
            raise ValueError("Index price must be positive for derived premium.")
        return (mark_price - index_price) / index_price
