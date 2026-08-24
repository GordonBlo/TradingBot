from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.cli.build_v7_orderflow_dataset import authoritative_context
from src.cli.build_v8_derivatives_dataset import (
    _acquire_one,
    exclude_ambiguous_timestamps,
    normalize_source_order,
)
from src.derivatives.context import build_context_15m
from src.derivatives.dataset import (
    HOLDOUT_START,
    build_derivatives_acquisition_plan,
)
from src.derivatives.integrity import CoverageClassification, analyze_funding_coverage
from src.derivatives.models import (
    DerivativesPriceKline,
    DerivativesSource,
    FundingRateRecord,
    FuturesMetricsRecord,
)
from src.derivatives.sources import build_archive_location
from src.models.candle import Candle
from src.orderflow.dataset import ArchivePlanItem


BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def candle(timestamp=BASE):
    return Candle(
        timestamp=timestamp,
        symbol="BTCUSDC",
        interval="15m",
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100"),
        volume=Decimal("1"),
        is_closed=True,
    )


def price(timestamp, close="100"):
    value = Decimal(close)
    return DerivativesPriceKline(
        DerivativesSource.MARK_PRICE,
        timestamp,
        timestamp + timedelta(minutes=15) - timedelta(milliseconds=1),
        value,
        value,
        value,
        value,
    )


def test_authoritative_plan_is_exact_and_deterministic() -> None:
    _, _, coverage = authoritative_context()
    first = build_derivatives_acquisition_plan(coverage)
    second = build_derivatives_acquisition_plan(coverage)
    assert first == second
    assert first.dataset_id == "eec764735d270f9d"
    assert len(first.coverage.windows) == 11
    assert len(first.coverage.required_timestamps) == 95_040
    assert first.required_unique_days == 991
    assert first.required_complete_months == 31
    assert len(first.archives) == 1_265
    assert all(
        not (item.archive_start < datetime(2026, 2, 1, tzinfo=timezone.utc) and HOLDOUT_START < item.archive_end)
        for item in first.archives
    )


def test_plan_archive_source_counts() -> None:
    _, _, coverage = authoritative_context()
    plan = build_derivatives_acquisition_plan(coverage)
    counts = {
        source.value: sum(item.source_dataset == source.value for item in plan.archives)
        for source in DerivativesSource
    }
    assert counts == {
        "fundingRate": 34,
        "metrics": 991,
        "markPriceKlines": 80,
        "indexPriceKlines": 80,
        "premiumIndexKlines": 80,
    }


def test_plan_rejects_holdout_timestamp() -> None:
    _, _, coverage = authoritative_context()
    contaminated = replace(
        coverage,
        required_timestamps=(*coverage.required_timestamps, HOLDOUT_START),
    )
    with pytest.raises(ValueError, match="holdout"):
        build_derivatives_acquisition_plan(contaminated)


def test_official_archive_urls_and_unsupported_cadence() -> None:
    location = build_archive_location(
        DerivativesSource.METRICS,
        BASE.date(),
        cadence="daily",
    )
    assert location.url == (
        "https://data.binance.vision/data/futures/um/daily/metrics/"
        "BTCUSDT/BTCUSDT-metrics-2024-01-01.zip"
    )
    assert location.checksum_url.endswith(".zip.CHECKSUM")
    with pytest.raises(ValueError, match="no official monthly"):
        build_archive_location(
            DerivativesSource.METRICS,
            BASE.date(),
            cadence="monthly",
        )


def test_cached_only_acquisition_never_fetches_missing_archive(tmp_path) -> None:
    item = ArchivePlanItem(
        source_dataset="metrics",
        granularity="daily",
        period="2024-01-01",
        archive_start=BASE,
        archive_end=BASE + timedelta(days=1),
        url="https://data.binance.vision/never-requested.zip",
        checksum_url="https://data.binance.vision/never-requested.zip.CHECKSUM",
        local_path=str(tmp_path / "missing.zip"),
        checksum_local_path=str(tmp_path / "missing.zip.CHECKSUM"),
        required_by_windows=("W002",),
    )
    result = _acquire_one(item, cached_only=True)
    assert result.verified is False
    assert "local cache" in result.error


def test_price_context_requires_exact_interval_without_forward_fill() -> None:
    old = price(BASE - timedelta(minutes=15))
    context = build_context_15m(spot_candles=(candle(),), mark=(old,))[0]
    assert context.mark_price is None
    exact = build_context_15m(spot_candles=(candle(),), mark=(old, price(BASE),))[0]
    assert exact.mark_price == Decimal("100")


def test_metrics_context_does_not_carry_beyond_source_cadence() -> None:
    old = FuturesMetricsRecord(
        BASE,
        "BTCUSDT",
        Decimal("1"),
        Decimal("2"),
        None,
        None,
        None,
        None,
    )
    context = build_context_15m(spot_candles=(candle(),), metrics=(old,))[0]
    assert context.open_interest is None


def test_duplicate_price_timestamp_is_rejected_and_excludable() -> None:
    rows = (price(BASE, "100"), price(BASE, "101"))
    assert exclude_ambiguous_timestamps(rows, "open_time") == ()
    with pytest.raises(ValueError, match="duplicate"):
        build_context_15m(spot_candles=(candle(),), mark=rows)


def test_non_monotonic_source_order_is_counted_and_timestamp_normalized() -> None:
    rows = (price(BASE + timedelta(minutes=15)), price(BASE))
    ordered, decreases = normalize_source_order(rows, "open_time")
    assert decreases == 1
    assert tuple(row.open_time for row in ordered) == (
        BASE,
        BASE + timedelta(minutes=15),
    )


def test_funding_gap_uses_archived_interval_not_assumed_cadence() -> None:
    records = (
        FundingRateRecord(BASE, Decimal("0.001"), funding_interval_hours=Decimal("4")),
        FundingRateRecord(
            BASE + timedelta(hours=8),
            Decimal("0.001"),
            funding_interval_hours=Decimal("8"),
        ),
    )
    result = analyze_funding_coverage(records)
    assert result.classification is CoverageClassification.GAPPED
    assert result.missing_intervals == 1
