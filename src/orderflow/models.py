"""Immutable Binance Spot aggregate-trade domain model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum


class AggressorSide(str, Enum):
    TAKER_BUY = "TAKER_BUY"
    TAKER_SELL = "TAKER_SELL"


def _positive_decimal(name: str, value: Decimal) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a valid Decimal.") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{name} must be finite and greater than zero.")
    return result


@dataclass(frozen=True, slots=True)
class AggregateTrade:
    """One official Binance aggTrade row without loss of numeric precision."""

    aggregate_trade_id: int
    price: Decimal
    quantity: Decimal
    first_trade_id: int
    last_trade_id: int
    timestamp: datetime
    buyer_is_maker: bool
    best_price_match: bool | None = None

    def __post_init__(self) -> None:
        if self.aggregate_trade_id < 0:
            raise ValueError("aggregate_trade_id must not be negative.")
        if self.first_trade_id < 0 or self.last_trade_id < self.first_trade_id:
            raise ValueError("first_trade_id must not exceed last_trade_id.")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() != timedelta(0):
            raise ValueError("timestamp must be timezone-aware UTC.")
        if not isinstance(self.buyer_is_maker, bool):
            raise ValueError("buyer_is_maker must be boolean.")
        if self.best_price_match is not None and not isinstance(
            self.best_price_match, bool
        ):
            raise ValueError("best_price_match must be boolean when provided.")
        object.__setattr__(self, "price", _positive_decimal("price", self.price))
        object.__setattr__(
            self, "quantity", _positive_decimal("quantity", self.quantity)
        )
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))

    @property
    def quote_quantity(self) -> Decimal:
        return self.price * self.quantity

    @property
    def aggressor_side(self) -> AggressorSide:
        # buyer_is_maker=False means the buyer crossed the book (taker buy).
        return (
            AggressorSide.TAKER_SELL
            if self.buyer_is_maker
            else AggressorSide.TAKER_BUY
        )

    @property
    def underlying_trade_count(self) -> int:
        return self.last_trade_id - self.first_trade_id + 1
