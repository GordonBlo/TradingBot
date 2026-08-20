"""OHLCV candle domain model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation


def _as_decimal(name: str, value: Decimal) -> Decimal:
    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a valid decimal value.") from exc
    if not decimal_value.is_finite():
        raise ValueError(f"{name} must be finite.")
    return decimal_value


@dataclass(frozen=True, slots=True)
class Candle:
    """Immutable, precision-safe representation of a market candle."""

    timestamp: datetime
    symbol: str
    interval: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    is_closed: bool

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware.")

        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "interval", self.interval.strip())
        for name in ("open", "high", "low", "close", "volume"):
            object.__setattr__(self, name, _as_decimal(name, getattr(self, name)))

        if not self.symbol:
            raise ValueError("symbol must not be empty.")
        if not self.interval:
            raise ValueError("interval must not be empty.")
        if self.volume < 0:
            raise ValueError("volume must not be negative.")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must be greater than or equal to all OHLC values.")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must be less than or equal to all OHLC values.")

