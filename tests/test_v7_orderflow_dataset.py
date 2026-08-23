from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.cli.build_v7_orderflow_dataset import (
    _present_verified,
    authoritative_context,
    inventory_plan,
)
from src.models.candle import Candle
from src.orderflow.aggregation import iter_aggregate_15m
from src.orderflow.dataset import (
    AcquisitionPlan,
    ArchivePlanItem,
    build_acquisition_plan,
    deterministic_manifest_sha256,
    derive_window_coverage,
)
from src.orderflow.integrity import AggregateTradeIntegrityError, verify_sha256
from src.orderflow.models import AggregateTrade


UTC = timezone.utc
BASE = datetime(2024, 3, 1, tzinfo=UTC)


def candle(timestamp: datetime) -> Candle:
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


def fake_context(timestamps: tuple[datetime, ...]) -> tuple[tuple, tuple]:
    kind = object()
    start = min(timestamps)
    end = max(timestamps) + timedelta(minutes=15)
    region = SimpleNamespace(
        metadata=SimpleNamespace(
            kind=kind,
            start=start,
            end=end,
            status="CONSUMED_RESEARCH_DATA",
            source="SYNTHETIC_SAFE_CONSUMED",
        )
    )
    windows = tuple(
        SimpleNamespace(
            window_id=f"W{index:03d}",
            data_status="CONSUMED_RESEARCH_DATA",
            partition_kind=kind,
            start=start,
            end=end,
            replay_dataset=SimpleNamespace(
                candles=tuple(candle(timestamp) for timestamp in timestamps)
            ),
        )
        for index in range(1, 12)
    )
    return windows, (region,)


def aggtrade(aggregate_id: int, timestamp: datetime) -> AggregateTrade:
    return AggregateTrade(
        aggregate_trade_id=aggregate_id,
        price=Decimal("100"),
        quantity=Decimal("1"),
        first_trade_id=aggregate_id,
        last_trade_id=aggregate_id,
        timestamp=timestamp,
        buyer_is_maker=False,
        best_price_match=True,
    )


def test_authoritative_v6_coverage_is_exact_and_holdout_free() -> None:
    _, windows, coverage = authoritative_context()

    assert tuple(item.window_id for item in coverage.windows) == (
        "W002",
        "W003",
        "W004",
        "W005",
        "W006",
        "W007",
        "W008",
        "W009",
        "W010",
        "W012",
        "W013",
    )
    assert len(windows) == 11
    assert len(coverage.required_timestamps) == 95_040
    assert coverage.safe_ranges == (
        (
            datetime(2023, 3, 24, 14, tzinfo=UTC),
            datetime(2025, 6, 11, 14, tzinfo=UTC),
        ),
        (
            datetime(2026, 2, 1, tzinfo=UTC),
            datetime(2026, 7, 31, tzinfo=UTC),
        ),
    )


def test_overlapping_window_union_is_deduplicated() -> None:
    timestamps = tuple(BASE + timedelta(minutes=15 * index) for index in range(4))
    windows, regions = fake_context(timestamps)

    coverage = derive_window_coverage(windows, regions)

    assert coverage.required_timestamps == timestamps
    assert coverage.timestamp_windows[timestamps[0]] == tuple(
        f"W{index:03d}" for index in range(1, 12)
    )


def test_holdout_intersection_is_rejected_before_archive_planning() -> None:
    timestamp = datetime(2025, 8, 1, tzinfo=UTC)
    windows, regions = fake_context((timestamp,))
    with pytest.raises(ValueError, match="holdout"):
        derive_window_coverage(windows, regions)


def test_acquisition_plan_uses_monthly_only_for_complete_safe_months() -> None:
    timestamps = tuple(
        BASE + timedelta(minutes=15 * index)
        for index in range(31 * 96)
    )
    windows, regions = fake_context(timestamps)
    plan = build_acquisition_plan(derive_window_coverage(windows, regions))

    agg = tuple(item for item in plan.archives if item.source_dataset == "aggTrades")
    assert len(agg) == 1
    assert agg[0].granularity == "monthly"
    assert agg[0].period == "2024-03"
    assert agg[0].holdout_status == "SAFE_NO_INTERSECTION"


def test_partial_month_uses_daily_fallback_and_plan_is_deterministic() -> None:
    timestamps = tuple(BASE + timedelta(minutes=15 * index) for index in range(2 * 96))
    windows, regions = fake_context(timestamps)
    coverage = derive_window_coverage(windows, regions)

    first = build_acquisition_plan(coverage)
    second = build_acquisition_plan(coverage)
    agg = tuple(item for item in first.archives if item.source_dataset == "aggTrades")

    assert [item.granularity for item in agg] == ["daily", "daily"]
    assert first.dataset_id == second.dataset_id
    assert first.archives == second.archives


def test_authoritative_archive_plan_counts_and_stable_id() -> None:
    _, _, coverage = authoritative_context()
    plan = build_acquisition_plan(coverage)

    assert plan.required_unique_days == 991
    assert plan.required_complete_months == 31
    assert plan.required_daily_archives == 49
    assert sum(item.source_dataset == "aggTrades" for item in plan.archives) == 80
    assert sum(
        item.source_dataset == "klines_15m_reference" for item in plan.archives
    ) == 80
    assert len(plan.dataset_id) == 16


def test_verified_archive_reuse_and_checksum_mismatch_rejection(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    checksum = tmp_path / "archive.zip.CHECKSUM"
    archive.write_bytes(b"verified")
    digest = hashlib.sha256(b"verified").hexdigest()
    checksum.write_text(f"{digest}  archive.zip\n", encoding="utf-8")
    item = ArchivePlanItem(
        source_dataset="aggTrades",
        granularity="daily",
        period="2024-03-01",
        archive_start=BASE,
        archive_end=BASE + timedelta(days=1),
        url="https://data.binance.vision/safe.zip",
        checksum_url="https://data.binance.vision/safe.zip.CHECKSUM",
        local_path=str(archive),
        checksum_local_path=str(checksum),
        required_by_windows=("W001",),
    )
    plan = AcquisitionPlan(
        dataset_id="0123456789abcdef",
        coverage=SimpleNamespace(required_timestamps=()),
        archives=(item,),
        required_unique_days=1,
        required_complete_months=0,
        required_daily_archives=1,
    )

    inventoried = inventory_plan(plan, query_sizes=False)
    assert inventoried.archives[0].already_present_verified

    archive.write_bytes(b"corrupt")
    assert not _present_verified(item)
    with pytest.raises(AggregateTradeIntegrityError, match="does not match"):
        verify_sha256(archive, checksum)


def test_streaming_aggregation_handles_archive_boundary_without_leakage() -> None:
    first_archive = iter((aggtrade(1, BASE + timedelta(minutes=14)),))
    second_archive = iter((aggtrade(2, BASE + timedelta(minutes=15)),))
    stream = (trade for archive in (first_archive, second_archive) for trade in archive)

    buckets = tuple(iter_aggregate_15m(stream))

    assert len(buckets) == 2
    assert buckets[0].bucket_open_time == BASE
    assert buckets[1].bucket_open_time == BASE + timedelta(minutes=15)
    assert buckets[0].aggregate_trade_count == buckets[1].aggregate_trade_count == 1


def test_streaming_timestamp_transition_spans_ms_and_us_policy_safely() -> None:
    before = aggtrade(1, datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC))
    after = aggtrade(2, datetime(2025, 1, 1, tzinfo=UTC))

    buckets = tuple(iter_aggregate_15m(iter((before, after))))

    assert [item.bucket_open_time for item in buckets] == [
        datetime(2024, 12, 31, 23, 45, tzinfo=UTC),
        datetime(2025, 1, 1, tzinfo=UTC),
    ]


def test_no_extra_required_bucket_is_introduced_by_union() -> None:
    timestamps = (
        BASE,
        BASE + timedelta(minutes=15),
        BASE + timedelta(hours=1),
    )
    windows, regions = fake_context(timestamps)
    coverage = derive_window_coverage(windows, regions)

    assert coverage.required_timestamps == timestamps
    assert coverage.safe_ranges == (
        (BASE, BASE + timedelta(minutes=30)),
        (BASE + timedelta(hours=1), BASE + timedelta(hours=1, minutes=15)),
    )


def test_dataset_manifest_hash_is_deterministic_and_excludes_generated_time() -> None:
    first = {
        "dataset_version": "V7_ORDERFLOW_CONSUMED_1",
        "archives": [{"period": "2024-03", "sha256": "abc"}],
        "generated_at": "2026-08-23T00:00:00+00:00",
    }
    second = {**first, "generated_at": "2027-01-01T00:00:00+00:00"}

    assert deterministic_manifest_sha256(first) == deterministic_manifest_sha256(second)
    assert len(deterministic_manifest_sha256(first)) == 64
