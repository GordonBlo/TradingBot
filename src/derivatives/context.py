"""Causal alignment of derivatives observations to closed Spot 15m buckets."""

from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timedelta
from typing import TypeVar

from src.derivatives.models import (
    DerivativesContext15m,
    DerivativesPriceKline,
    FundingRateRecord,
    FuturesMetricsRecord,
)
from src.models.candle import Candle


T = TypeVar("T")


def _latest(
    records: tuple[T, ...],
    timestamps: tuple[datetime, ...],
    cutoff: datetime,
) -> T | None:
    """Return the latest record whose timestamp is <= cutoff.

    `timestamps` must correspond one-to-one with `records` and be
    chronologically sorted.

    Lookup is O(log N) instead of rescanning the full source history.
    """

    index = bisect_right(timestamps, cutoff) - 1
    return records[index] if index >= 0 else None


def build_context_15m(
    *,
    spot_candles: tuple[Candle, ...],
    funding: tuple[FundingRateRecord, ...] = (),
    metrics: tuple[FuturesMetricsRecord, ...] = (),
    mark: tuple[DerivativesPriceKline, ...] = (),
    index: tuple[DerivativesPriceKline, ...] = (),
    premium: tuple[DerivativesPriceKline, ...] = (),
) -> tuple[DerivativesContext15m, ...]:
    # Build searchable timestamp indexes exactly once per source.
    funding_timestamps = tuple(row.timestamp for row in funding)
    metrics_timestamps = tuple(row.timestamp for row in metrics)
    def exact_price_rows(records: tuple[DerivativesPriceKline, ...]):
        output = {}
        for row in records:
            if row.open_time in output:
                raise ValueError("Derivatives price source has duplicate intervals.")
            output[row.open_time] = row
        return output

    mark_by_open = exact_price_rows(mark)
    index_by_open = exact_price_rows(index)
    premium_by_open = exact_price_rows(premium)

    output = []

    for candle in spot_candles:
        if candle.interval != "15m" or not candle.is_closed:
            raise ValueError(
                "Context alignment requires closed Spot 15m candles."
            )

        bucket_close = candle.timestamp + timedelta(minutes=15)

        funding_row = _latest(
            funding,
            funding_timestamps,
            bucket_close,
        )
        metrics_row = _latest(
            metrics,
            metrics_timestamps,
            bucket_close,
        )
        if (
            metrics_row is not None
            and bucket_close - metrics_row.timestamp > timedelta(minutes=5)
        ):
            metrics_row = None
        mark_row = mark_by_open.get(candle.timestamp)
        index_row = index_by_open.get(candle.timestamp)
        premium_row = premium_by_open.get(candle.timestamp)
        mark_row = mark_row if mark_row is not None and mark_row.close_time <= bucket_close else None
        index_row = index_row if index_row is not None and index_row.close_time <= bucket_close else None
        premium_row = premium_row if premium_row is not None and premium_row.close_time <= bucket_close else None

        derived = (
            DerivativesContext15m.derive_premium(
                mark_row.close,
                index_row.close,
            )
            if mark_row is not None and index_row is not None
            else None
        )

        output.append(
            DerivativesContext15m(
                bucket_open_time=candle.timestamp,
                bucket_close_time=bucket_close,
                latest_known_funding_rate=(
                    funding_row.funding_rate
                    if funding_row
                    else None
                ),
                funding_timestamp=(
                    funding_row.timestamp
                    if funding_row
                    else None
                ),
                open_interest=(
                    metrics_row.sum_open_interest
                    if metrics_row
                    else None
                ),
                open_interest_value=(
                    metrics_row.sum_open_interest_value
                    if metrics_row
                    else None
                ),
                open_interest_timestamp=(
                    metrics_row.timestamp
                    if metrics_row
                    else None
                ),
                mark_price=(
                    mark_row.close
                    if mark_row
                    else None
                ),
                mark_price_timestamp=(
                    mark_row.close_time
                    if mark_row
                    else None
                ),
                index_price=(
                    index_row.close
                    if index_row
                    else None
                ),
                index_price_timestamp=(
                    index_row.close_time
                    if index_row
                    else None
                ),
                premium_index=(
                    premium_row.close
                    if premium_row
                    else None
                ),
                premium_index_timestamp=(
                    premium_row.close_time
                    if premium_row
                    else None
                ),
                premium_fraction=derived,
            )
        )

    return tuple(output)
