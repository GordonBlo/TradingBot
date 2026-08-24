"""Deterministic planning for the V8 consumed-window derivatives dataset."""

from __future__ import annotations

import hashlib
import json
from calendar import monthrange
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from src.derivatives.models import DerivativesSource
from src.derivatives.sources import build_archive_location
from src.orderflow.dataset import ArchivePlanItem, CoverageDefinition


HOLDOUT_START = datetime(2025, 8, 1, tzinfo=timezone.utc)
HOLDOUT_END = datetime(2026, 2, 1, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class DerivativesAcquisitionPlan:
    dataset_id: str
    coverage: CoverageDefinition
    archives: tuple[ArchivePlanItem, ...]
    required_unique_days: int
    required_complete_months: int


def _archive_bounds(period: date, cadence: str) -> tuple[datetime, datetime]:
    start = datetime.combine(period, time.min, tzinfo=timezone.utc)
    if cadence == "daily":
        return start, start + timedelta(days=1)
    year = period.year + int(period.month == 12)
    month = 1 if period.month == 12 else period.month + 1
    return start.replace(day=1), start.replace(year=year, month=month, day=1)


def _window_ids(
    coverage: CoverageDefinition, start: datetime, end: datetime
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                window_id
                for timestamp, window_ids in coverage.timestamp_windows.items()
                if start <= timestamp < end
                for window_id in window_ids
            }
        )
    )


def _item(
    coverage: CoverageDefinition,
    source: DerivativesSource,
    period: date,
    cadence: str,
    raw_root: str | Path,
) -> ArchivePlanItem:
    start, end = _archive_bounds(period, cadence)
    if start < HOLDOUT_END and HOLDOUT_START < end:
        raise ValueError("V8 archive plan intersects the locked blind holdout.")
    location = build_archive_location(source, period, cadence=cadence, root=raw_root)
    return ArchivePlanItem(
        source_dataset=source.value,
        granularity=cadence,
        period=(period.isoformat() if cadence == "daily" else period.strftime("%Y-%m")),
        archive_start=start,
        archive_end=end,
        url=location.url,
        checksum_url=location.checksum_url,
        local_path=str(location.destination),
        checksum_local_path=str(location.checksum_destination),
        required_by_windows=_window_ids(coverage, start, end),
    )


def _stable_id(
    coverage: CoverageDefinition, archives: tuple[ArchivePlanItem, ...]
) -> str:
    definition = {
        "dataset_version": "V8_DERIVATIVES_CONTEXT_CONSUMED_1",
        "traded_market": "BTCUSDC_SPOT_15M",
        "context_market": "BTCUSDT_USDM_PERPETUAL",
        "windows": [asdict(window) for window in coverage.windows],
        "safe_ranges": coverage.safe_ranges,
        "sources": [source.value for source in DerivativesSource],
        "archives": [
            {
                "source": item.source_dataset,
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
    encoded = json.dumps(definition, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def build_derivatives_acquisition_plan(
    coverage: CoverageDefinition,
    *,
    raw_root: str | Path = "data/derivatives/raw",
) -> DerivativesAcquisitionPlan:
    if len(coverage.windows) != 11:
        raise ValueError("V8 requires exactly 11 consumed research windows.")
    if any(
        timestamp < HOLDOUT_END and HOLDOUT_START <= timestamp
        for timestamp in coverage.required_timestamps
    ):
        raise ValueError("V8 required coverage intersects the blind holdout.")
    days: dict[date, tuple[datetime, ...]] = {}
    for timestamp in coverage.required_timestamps:
        days.setdefault(timestamp.date(), ())
        days[timestamp.date()] += (timestamp,)
    months: dict[tuple[int, int], tuple[date, ...]] = {}
    for day in days:
        months.setdefault((day.year, day.month), ())
        months[(day.year, day.month)] += (day,)

    items = []
    for year, month in sorted(months):
        period = date(year, month, 1)
        items.append(
            _item(
                coverage,
                DerivativesSource.FUNDING_RATE,
                period,
                "monthly",
                raw_root,
            )
        )
    for day in sorted(days):
        items.append(
            _item(
                coverage,
                DerivativesSource.METRICS,
                day,
                "daily",
                raw_root,
            )
        )

    complete_months = 0
    for (year, month), month_days in sorted(months.items()):
        full = (
            len(month_days) == monthrange(year, month)[1]
            and all(len(days[day]) == 96 for day in month_days)
        )
        periods = (date(year, month, 1),) if full else tuple(sorted(month_days))
        cadence = "monthly" if full else "daily"
        complete_months += int(full)
        for source in (
            DerivativesSource.MARK_PRICE,
            DerivativesSource.INDEX_PRICE,
            DerivativesSource.PREMIUM_INDEX,
        ):
            items.extend(
                _item(coverage, source, period, cadence, raw_root)
                for period in periods
            )
    archives = tuple(
        sorted(
            items,
            key=lambda item: (
                item.archive_start,
                item.source_dataset,
                item.granularity,
            ),
        )
    )
    return DerivativesAcquisitionPlan(
        dataset_id=_stable_id(coverage, archives),
        coverage=coverage,
        archives=archives,
        required_unique_days=len(days),
        required_complete_months=complete_months,
    )
