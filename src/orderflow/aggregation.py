"""Causal UTC-aligned 15-minute aggregate-trade bucketing."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Iterator
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from src.market.intervals import is_interval_aligned, require_utc
from src.orderflow.integrity import validate_trade_segment
from src.orderflow.models import AggregateTrade, AggressorSide


_BUCKET_SIZE = timedelta(minutes=15)
_BUCKET_MICROSECONDS = 15 * 60 * 1_000_000
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class OrderFlowBucket:
    bucket_open_time: datetime
    bucket_close_time: datetime
    aggregate_trade_count: int
    underlying_trade_count: int
    total_base_volume: Decimal
    total_quote_volume: Decimal
    taker_buy_base_volume: Decimal
    taker_buy_quote_volume: Decimal
    taker_buy_aggtrade_count: int
    taker_sell_base_volume: Decimal
    taker_sell_quote_volume: Decimal
    taker_sell_aggtrade_count: int

    def __post_init__(self) -> None:
        require_utc(self.bucket_open_time, name="bucket_open_time")
        require_utc(self.bucket_close_time, name="bucket_close_time")
        if not is_interval_aligned(self.bucket_open_time, "15m"):
            raise ValueError("Order-flow bucket open must be UTC-aligned to 15m.")
        if self.bucket_close_time != self.bucket_open_time + _BUCKET_SIZE:
            raise ValueError("Order-flow bucket must span exactly 15 minutes.")
        if self.aggregate_trade_count <= 0 or self.underlying_trade_count <= 0:
            raise ValueError("Order-flow bucket counts must be positive.")
        if self.underlying_trade_count < self.aggregate_trade_count:
            raise ValueError("Underlying trade count cannot be below aggTrade count.")
        if self.taker_buy_aggtrade_count < 0 or self.taker_sell_aggtrade_count < 0:
            raise ValueError("Taker-side aggTrade counts must not be negative.")
        if self.taker_buy_aggtrade_count + self.taker_sell_aggtrade_count != (
            self.aggregate_trade_count
        ):
            raise ValueError("Taker-side aggTrade counts do not reconcile.")
        if self.taker_buy_base_volume + self.taker_sell_base_volume != (
            self.total_base_volume
        ):
            raise ValueError("Taker-side base volume does not reconcile.")
        if self.taker_buy_quote_volume + self.taker_sell_quote_volume != (
            self.total_quote_volume
        ):
            raise ValueError("Taker-side quote volume does not reconcile.")
        if any(
            value < 0
            for value in (
                self.taker_buy_base_volume,
                self.taker_buy_quote_volume,
                self.taker_sell_base_volume,
                self.taker_sell_quote_volume,
            )
        ):
            raise ValueError("Taker-side volumes must not be negative.")
        if self.total_base_volume <= 0 or self.total_quote_volume <= 0:
            raise ValueError("Order-flow bucket volumes must be positive.")

    @property
    def signed_base_volume(self) -> Decimal:
        return self.taker_buy_base_volume - self.taker_sell_base_volume

    @property
    def signed_quote_volume(self) -> Decimal:
        return self.taker_buy_quote_volume - self.taker_sell_quote_volume

    @property
    def taker_buy_base_ratio(self) -> Decimal:
        return self.taker_buy_base_volume / self.total_base_volume

    @property
    def taker_buy_quote_ratio(self) -> Decimal:
        return self.taker_buy_quote_volume / self.total_quote_volume

    @property
    def base_volume_imbalance(self) -> Decimal:
        return self.signed_base_volume / self.total_base_volume

    @property
    def quote_volume_imbalance(self) -> Decimal:
        return self.signed_quote_volume / self.total_quote_volume

    @property
    def average_aggtrade_base_size(self) -> Decimal:
        return self.total_base_volume / self.aggregate_trade_count

    @property
    def average_aggtrade_quote_size(self) -> Decimal:
        return self.total_quote_volume / self.aggregate_trade_count


@dataclass(slots=True)
class _Accumulator:
    aggregate_trade_count: int = 0
    underlying_trade_count: int = 0
    total_base_volume: Decimal = Decimal("0")
    total_quote_volume: Decimal = Decimal("0")
    taker_buy_base_volume: Decimal = Decimal("0")
    taker_buy_quote_volume: Decimal = Decimal("0")
    taker_buy_aggtrade_count: int = 0
    taker_sell_base_volume: Decimal = Decimal("0")
    taker_sell_quote_volume: Decimal = Decimal("0")
    taker_sell_aggtrade_count: int = 0

    def add(self, trade: AggregateTrade) -> None:
        self.aggregate_trade_count += 1
        self.underlying_trade_count += trade.underlying_trade_count
        self.total_base_volume += trade.quantity
        self.total_quote_volume += trade.quote_quantity
        if trade.aggressor_side is AggressorSide.TAKER_BUY:
            self.taker_buy_base_volume += trade.quantity
            self.taker_buy_quote_volume += trade.quote_quantity
            self.taker_buy_aggtrade_count += 1
        else:
            self.taker_sell_base_volume += trade.quantity
            self.taker_sell_quote_volume += trade.quote_quantity
            self.taker_sell_aggtrade_count += 1


def bucket_open_15m(timestamp: datetime) -> datetime:
    delta = timestamp - _EPOCH
    total_microseconds = (
        (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds
    )
    return _EPOCH + timedelta(
        microseconds=(total_microseconds // _BUCKET_MICROSECONDS)
        * _BUCKET_MICROSECONDS
    )


def aggregate_15m(
    trades: list[AggregateTrade] | tuple[AggregateTrade, ...],
) -> tuple[OrderFlowBucket, ...]:
    validate_trade_segment(trades)
    return tuple(iter_aggregate_15m(trades))


def _finish_bucket(open_time: datetime, values: _Accumulator) -> OrderFlowBucket:
    return OrderFlowBucket(
        bucket_open_time=open_time,
        bucket_close_time=open_time + _BUCKET_SIZE,
        aggregate_trade_count=values.aggregate_trade_count,
        underlying_trade_count=values.underlying_trade_count,
        total_base_volume=values.total_base_volume,
        total_quote_volume=values.total_quote_volume,
        taker_buy_base_volume=values.taker_buy_base_volume,
        taker_buy_quote_volume=values.taker_buy_quote_volume,
        taker_buy_aggtrade_count=values.taker_buy_aggtrade_count,
        taker_sell_base_volume=values.taker_sell_base_volume,
        taker_sell_quote_volume=values.taker_sell_quote_volume,
        taker_sell_aggtrade_count=values.taker_sell_aggtrade_count,
    )


def iter_aggregate_15m(trades: Iterable[AggregateTrade]) -> Iterator[OrderFlowBucket]:
    """Aggregate a chronological stream while retaining only one open bucket."""

    current_open: datetime | None = None
    values = _Accumulator()
    for trade in trades:
        open_time = bucket_open_15m(trade.timestamp)
        if current_open is not None and open_time < current_open:
            raise ValueError("Aggregate-trade stream is not chronological.")
        if current_open is not None and open_time != current_open:
            yield _finish_bucket(current_open, values)
            values = _Accumulator()
        current_open = open_time
        values.add(trade)
    if current_open is not None:
        yield _finish_bucket(current_open, values)


def completed_bucket_at(
    buckets: list[OrderFlowBucket] | tuple[OrderFlowBucket, ...],
    signal_timestamp: datetime,
) -> OrderFlowBucket | None:
    """Return only the bucket closing exactly at a UTC signal timestamp.

    Exact-close lookup intentionally neither exposes an incomplete next bucket nor
    forward-fills an older value into a signal timestamp with missing flow data.
    """

    signal_timestamp = require_utc(signal_timestamp, name="signal_timestamp")
    matches = [
        bucket for bucket in buckets if bucket.bucket_close_time == signal_timestamp
    ]
    if len(matches) > 1:
        raise ValueError("Completed order-flow bucket is duplicated.")
    return matches[0] if matches else None
