"""Deterministic planning for the V7 consumed-data order-flow dataset."""

from __future__ import annotations

import hashlib
import json
from calendar import monthrange
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from src.orderflow.archive import (
    ArchiveLocation,
    build_archive_location,
    build_kline_archive_location,
)
from src.orderflow.reconciliation import HOLDOUT_END, HOLDOUT_START


@dataclass(frozen=True, slots=True)
class WindowCoverage:
    window_id: str
    source_region: str
    replay_start: datetime
    evaluation_start: datetime
    evaluation_end: datetime
    warmup_start: datetime | None
    warmup_end: datetime | None
    evaluated_start: datetime
    evaluated_end: datetime


@dataclass(frozen=True, slots=True)
class CoverageDefinition:
    windows: tuple[WindowCoverage, ...]
    required_timestamps: tuple[datetime, ...]
    safe_ranges: tuple[tuple[datetime, datetime], ...]
    timestamp_windows: dict[datetime, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class ArchivePlanItem:
    source_dataset: str
    granularity: str
    period: str
    archive_start: datetime
    archive_end: datetime
    url: str
    checksum_url: str
    local_path: str
    checksum_local_path: str
    required_by_windows: tuple[str, ...]
    holdout_status: str = "SAFE_NO_INTERSECTION"
    content_length_bytes: int | None = None
    already_present_verified: bool = False


@dataclass(frozen=True, slots=True)
class AcquisitionPlan:
    dataset_id: str
    coverage: CoverageDefinition
    archives: tuple[ArchivePlanItem, ...]
    required_unique_days: int
    required_complete_months: int
    required_daily_archives: int


def _status_value(value: object) -> str:
    return str(getattr(value, "value", value))


def derive_window_coverage(windows: tuple, regions: tuple) -> CoverageDefinition:
    if len(windows) != 11:
        raise ValueError(f"V7 requires the exact 11 eligible windows, got {len(windows)}.")
    records: list[WindowCoverage] = []
    timestamp_windows: dict[datetime, set[str]] = {}
    for window in windows:
        if _status_value(window.data_status) != "CONSUMED_RESEARCH_DATA":
            raise ValueError("V7 window is not CONSUMED_RESEARCH_DATA.")
        owner = next(
            (
                region
                for region in regions
                if region.metadata.kind is window.partition_kind
                and region.metadata.start <= window.start
                and window.end <= region.metadata.end
            ),
            None,
        )
        if owner is None or _status_value(owner.metadata.status) != "CONSUMED_RESEARCH_DATA":
            raise ValueError("V7 window lacks a safe consumed-data owner region.")
        candles = window.replay_dataset.candles
        if not candles:
            raise ValueError("V7 replay coverage is empty.")
        replay_start = candles[0].timestamp
        records.append(
            WindowCoverage(
                window_id=window.window_id,
                source_region=owner.metadata.source,
                replay_start=replay_start,
                evaluation_start=window.start,
                evaluation_end=window.end,
                warmup_start=replay_start if replay_start < window.start else None,
                warmup_end=window.start if replay_start < window.start else None,
                evaluated_start=window.start,
                evaluated_end=window.end,
            )
        )
        for candle in candles:
            timestamp_windows.setdefault(candle.timestamp, set()).add(window.window_id)
    required = tuple(sorted(timestamp_windows))
    if any(timestamp < HOLDOUT_END and HOLDOUT_START <= timestamp for timestamp in required):
        raise ValueError("Planned V7 coverage intersects the locked blind holdout.")
    ranges: list[tuple[datetime, datetime]] = []
    start = previous = required[0]
    interval = timedelta(minutes=15)
    for timestamp in required[1:]:
        if timestamp != previous + interval:
            ranges.append((start, previous + interval))
            start = timestamp
        previous = timestamp
    ranges.append((start, previous + interval))
    return CoverageDefinition(
        windows=tuple(records),
        required_timestamps=required,
        safe_ranges=tuple(ranges),
        timestamp_windows={
            timestamp: tuple(sorted(ids)) for timestamp, ids in timestamp_windows.items()
        },
    )


def _archive_bounds(period: date, granularity: str) -> tuple[datetime, datetime]:
    start = datetime.combine(period, time.min, tzinfo=timezone.utc)
    if granularity == "daily":
        return start, start + timedelta(days=1)
    year = period.year + int(period.month == 12)
    month = 1 if period.month == 12 else period.month + 1
    return start.replace(day=1), start.replace(year=year, month=month, day=1)


def _plan_item(
    location: ArchiveLocation,
    *,
    dataset: str,
    granularity: str,
    period: date,
    window_ids: tuple[str, ...],
) -> ArchivePlanItem:
    start, end = _archive_bounds(period, granularity)
    if start < HOLDOUT_END and HOLDOUT_START < end:
        raise ValueError("Archive plan intersects the locked blind holdout.")
    period_text = period.isoformat() if granularity == "daily" else f"{period.year:04d}-{period.month:02d}"
    return ArchivePlanItem(
        source_dataset=dataset,
        granularity=granularity,
        period=period_text,
        archive_start=start,
        archive_end=end,
        url=location.url,
        checksum_url=location.checksum_url,
        local_path=str(location.destination),
        checksum_local_path=str(location.checksum_destination),
        required_by_windows=window_ids,
    )


def _stable_dataset_id(coverage: CoverageDefinition, archives: tuple[ArchivePlanItem, ...]) -> str:
    stable = {
        "dataset_version": "V7_ORDERFLOW_CONSUMED_1",
        "symbol": "BTCUSDC",
        "interval": "15m",
        "windows": [asdict(item) for item in coverage.windows],
        "safe_ranges": coverage.safe_ranges,
        "archives": [
            {
                "source_dataset": item.source_dataset,
                "granularity": item.granularity,
                "period": item.period,
                "url": item.url,
                "checksum_url": item.checksum_url,
                "required_by_windows": item.required_by_windows,
                "holdout_status": item.holdout_status,
            }
            for item in archives
        ],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def deterministic_manifest_sha256(definition: dict) -> str:
    """Hash stable dataset content while explicitly excluding generation time."""

    stable = {key: value for key, value in definition.items() if key != "generated_at"}
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_acquisition_plan(
    coverage: CoverageDefinition,
    *,
    raw_root: str | Path = "data/orderflow/raw",
) -> AcquisitionPlan:
    days: dict[date, list[datetime]] = {}
    for timestamp in coverage.required_timestamps:
        days.setdefault(timestamp.date(), []).append(timestamp)
    months: dict[tuple[int, int], list[date]] = {}
    for day in days:
        months.setdefault((day.year, day.month), []).append(day)
    primary: list[ArchivePlanItem] = []
    references: list[ArchivePlanItem] = []
    complete_months = 0
    daily_count = 0
    for (year, month), month_days in sorted(months.items()):
        full = (
            len(month_days) == monthrange(year, month)[1]
            and all(len(days[day]) == 96 for day in month_days)
        )
        periods = (date(year, month, 1),) if full else tuple(sorted(month_days))
        granularity = "monthly" if full else "daily"
        complete_months += int(full)
        daily_count += 0 if full else len(periods)
        for period in periods:
            start, end = _archive_bounds(period, granularity)
            ids = tuple(
                sorted(
                    {
                        window_id
                        for timestamp, window_ids in coverage.timestamp_windows.items()
                        if start <= timestamp < end
                        for window_id in window_ids
                    }
                )
            )
            primary.append(
                _plan_item(
                    build_archive_location(
                        period, cadence=granularity, root=raw_root
                    ),
                    dataset="aggTrades",
                    granularity=granularity,
                    period=period,
                    window_ids=ids,
                )
            )
            references.append(
                _plan_item(
                    build_kline_archive_location(
                        period, cadence=granularity, root=raw_root
                    ),
                    dataset="klines_15m_reference",
                    granularity=granularity,
                    period=period,
                    window_ids=ids,
                )
            )
    archives = tuple(
        sorted(
            (*primary, *references),
            key=lambda item: (
                item.archive_start,
                item.source_dataset,
                item.granularity,
            ),
        )
    )
    return AcquisitionPlan(
        dataset_id=_stable_dataset_id(coverage, archives),
        coverage=coverage,
        archives=archives,
        required_unique_days=len(days),
        required_complete_months=complete_months,
        required_daily_archives=daily_count,
    )
