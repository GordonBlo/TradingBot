"""Causal alignment of derivatives observations to closed Spot 15m buckets."""

from __future__ import annotations

from datetime import timedelta

from src.derivatives.models import (
    DerivativesContext15m,
    DerivativesPriceKline,
    FundingRateRecord,
    FuturesMetricsRecord,
)
from src.models.candle import Candle


def _latest(records: tuple, timestamp_field: str, cutoff):
    eligible = tuple(
        record for record in records if getattr(record, timestamp_field) <= cutoff
    )
    return eligible[-1] if eligible else None


def build_context_15m(
    *,
    spot_candles: tuple[Candle, ...],
    funding: tuple[FundingRateRecord, ...] = (),
    metrics: tuple[FuturesMetricsRecord, ...] = (),
    mark: tuple[DerivativesPriceKline, ...] = (),
    index: tuple[DerivativesPriceKline, ...] = (),
    premium: tuple[DerivativesPriceKline, ...] = (),
) -> tuple[DerivativesContext15m, ...]:
    output = []
    for candle in spot_candles:
        if candle.interval != "15m" or not candle.is_closed:
            raise ValueError("Context alignment requires closed Spot 15m candles.")
        bucket_close = candle.timestamp + timedelta(minutes=15)
        funding_row = _latest(funding, "timestamp", bucket_close)
        metrics_row = _latest(metrics, "timestamp", bucket_close)
        mark_row = _latest(mark, "close_time", bucket_close)
        index_row = _latest(index, "close_time", bucket_close)
        premium_row = _latest(premium, "close_time", bucket_close)
        derived = (
            DerivativesContext15m.derive_premium(mark_row.close, index_row.close)
            if mark_row is not None and index_row is not None
            else None
        )
        output.append(
            DerivativesContext15m(
                bucket_open_time=candle.timestamp,
                bucket_close_time=bucket_close,
                latest_known_funding_rate=(funding_row.funding_rate if funding_row else None),
                funding_timestamp=(funding_row.timestamp if funding_row else None),
                open_interest=(metrics_row.sum_open_interest if metrics_row else None),
                open_interest_value=(metrics_row.sum_open_interest_value if metrics_row else None),
                open_interest_timestamp=(metrics_row.timestamp if metrics_row else None),
                mark_price=(mark_row.close if mark_row else None),
                mark_price_timestamp=(mark_row.close_time if mark_row else None),
                index_price=(index_row.close if index_row else None),
                index_price_timestamp=(index_row.close_time if index_row else None),
                premium_index=(premium_row.close if premium_row else None),
                premium_index_timestamp=(premium_row.close_time if premium_row else None),
                premium_fraction=derived,
            )
        )
    return tuple(output)
