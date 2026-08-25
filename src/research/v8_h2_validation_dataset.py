"""Independent, outcome-unseen data preparation for frozen V8-H2.

This module deliberately prepares only causal market context and V6 signal
timestamps.  It has no backtest, trade, return, or outcome dependency.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterable

from src.derivatives.context import build_context_15m
from src.derivatives.integrity import CoverageClassification, analyze_coverage
from src.derivatives.models import DerivativesPriceKline, DerivativesSource, utc_timestamp
from src.derivatives.parser import parse_price_kline_archive
from src.derivatives.sources import KLINE_SCHEMA, build_archive_location
from src.hypotheses.manifest import HoldoutStatus, ResearchManifest
from src.models.candle import Candle
from src.orderflow.archive import (
    AggregateTradeArchiveDownloader,
    AggregateTradeArchiveError,
    ArchiveDownloadResult,
    ArchiveLocation,
    build_kline_archive_location,
)
from src.orderflow.integrity import verify_sha256
from src.strategy.context import StrategyContext
from src.strategy.models import StrategyAction, TrendMomentumConfig
from src.strategy.v6_mtf_continuation import V6MTFContinuationStrategy, prepare_v6_context


UTC = timezone.utc
INTERVAL = timedelta(minutes=15)
DERIVATIVE_SOURCES = (DerivativesSource.MARK_PRICE, DerivativesSource.INDEX_PRICE)
DATASET_VERSION = "v8_h2_independent_validation_v1"


@dataclass(frozen=True, slots=True, order=True)
class TimeRange:
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None or self.start >= self.end:
            raise ValueError("Validation ranges must be non-empty UTC ranges.")

    def overlaps(self, other: "TimeRange") -> bool:
        return self.start < other.end and other.start < self.end

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

    def as_dict(self) -> dict[str, str]:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}


@dataclass(frozen=True, slots=True)
class ArchiveItem:
    source: str
    location: ArchiveLocation


def merge_ranges(ranges: Iterable[TimeRange]) -> tuple[TimeRange, ...]:
    output: list[TimeRange] = []
    for item in sorted(ranges):
        if output and item.start <= output[-1].end:
            output[-1] = TimeRange(output[-1].start, max(output[-1].end, item.end))
        else:
            output.append(item)
    return tuple(output)


def subtract_ranges(available: Iterable[TimeRange], excluded: Iterable[TimeRange]) -> tuple[TimeRange, ...]:
    output: list[TimeRange] = []
    blocked = merge_ranges(excluded)
    for source in available:
        cursor = source.start
        for item in blocked:
            if item.end <= cursor or source.end <= item.start:
                continue
            if cursor < item.start:
                output.append(TimeRange(cursor, min(item.start, source.end)))
            cursor = max(cursor, item.end)
            if cursor >= source.end:
                break
        if cursor < source.end:
            output.append(TimeRange(cursor, source.end))
    return tuple(output)


def select_validation_interval(intervals: Iterable[TimeRange]) -> TimeRange:
    candidates = tuple(intervals)
    if not candidates:
        raise ValueError("No independent archive-verifiable validation interval exists.")
    return min(candidates, key=lambda item: (-item.duration.total_seconds(), item.start))


def consumed_and_holdout_exclusions(manifest: ResearchManifest) -> tuple[tuple[TimeRange, ...], TimeRange]:
    if manifest.holdout_status is not HoldoutStatus.LOCKED_BLIND_HOLDOUT:
        raise ValueError("Blind holdout must remain LOCKED.")
    holdout = manifest.blind_holdout
    if holdout is None or holdout.reveal_timestamp is not None or holdout.consumed_timestamp is not None:
        raise ValueError("Blind holdout contamination detected.")
    consumed = tuple(TimeRange(item.start, item.end) for item in manifest.consumed_dataset_ranges)
    return merge_ranges(consumed), TimeRange(holdout.start, holdout.end)


def discovered_availability() -> tuple[TimeRange, ...]:
    """Official-source availability established by checksum/API planning probes.

    The BTCUSDC source has a real gap from 2022-09-29 03:00 until its
    2023-03-12 restart.  The final current day is excluded until both official
    USD-M checksum files exist, so this cannot select a partial live day.
    """

    return (
        TimeRange(datetime(2020, 1, 1, tzinfo=UTC), datetime(2022, 9, 29, 3, tzinfo=UTC)),
        TimeRange(datetime(2026, 8, 18, tzinfo=UTC), datetime(2026, 8, 24, tzinfo=UTC)),
    )


def validation_plan(manifest: ResearchManifest) -> tuple[tuple[TimeRange, ...], TimeRange, tuple[TimeRange, ...], TimeRange]:
    consumed, holdout = consumed_and_holdout_exclusions(manifest)
    eligible = subtract_ranges(discovered_availability(), (*consumed, holdout))
    selected = select_validation_interval(eligible)
    if selected.overlaps(holdout) or any(selected.overlaps(item) for item in consumed):
        raise ValueError("Independent validation selection overlaps excluded research history.")
    return consumed, holdout, eligible, selected


def _month_start(value: datetime) -> datetime:
    return value.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def archive_plan(selected: TimeRange, *, raw_root: str | Path) -> tuple[ArchiveItem, ...]:
    """Use monthly archives only; every planned month is outcome-unseen."""

    months: list[datetime] = []
    cursor = _month_start(selected.start)
    while cursor < selected.end:
        months.append(cursor)
        cursor = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
    output: list[ArchiveItem] = []
    for month in months:
        period = month.date()
        output.append(
            ArchiveItem(
                "spot_klines",
                build_kline_archive_location(
                    period, cadence="monthly", root=Path(raw_root) / "spot"
                ),
            )
        )
        for source in DERIVATIVE_SOURCES:
            output.append(
                ArchiveItem(
                    source.value,
                    build_archive_location(
                        source, period, cadence="monthly", root=Path(raw_root) / "derivatives"
                    ),
                )
            )
    return tuple(output)


def availability_archive_plan(*, raw_root: str | Path) -> tuple[ArchiveItem, ...]:
    """Bounded archive plan needed to audit every remaining availability branch."""

    early, late = discovered_availability()
    output = list(archive_plan(early, raw_root=raw_root))
    cursor = late.start
    while cursor < late.end:
        period = cursor.date()
        output.append(ArchiveItem("spot_klines", build_kline_archive_location(
            period, cadence="daily", root=Path(raw_root) / "spot"
        )))
        for source in DERIVATIVE_SOURCES:
            output.append(ArchiveItem(source.value, build_archive_location(
                source, period, cadence="daily", root=Path(raw_root) / "derivatives"
            )))
        cursor += timedelta(days=1)
    return tuple(output)


def _parse_spot_archive(path: Path) -> tuple[Candle, ...]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = tuple(name for name in archive.namelist() if name.lower().endswith(".csv"))
            if len(members) != 1:
                raise ValueError("Spot archive must contain exactly one CSV.")
            rows = tuple(csv.reader(io.TextIOWrapper(archive.open(members[0]), encoding="utf-8-sig")))
    except (OSError, UnicodeError, zipfile.BadZipFile) as exc:
        raise ValueError("Spot archive cannot be read safely.") from exc
    if rows and tuple(cell.strip().lower().replace(" ", "_") for cell in rows[0]) == KLINE_SCHEMA:
        rows = rows[1:]
    output: list[Candle] = []
    for row in rows:
        if len(row) != len(KLINE_SCHEMA):
            raise ValueError("Spot kline archive schema changed.")
        unit = "microseconds" if len(row[0].lstrip("-")) > 13 else "milliseconds"
        output.append(Candle(
            timestamp=utc_timestamp(row[0], unit=unit), symbol="BTCUSDC", interval="15m",
            open=Decimal(row[1]), high=Decimal(row[2]), low=Decimal(row[3]), close=Decimal(row[4]),
            volume=Decimal(row[5]), is_closed=True,
        ))
    return tuple(output)


def _in_range(records: Iterable, selected: TimeRange, timestamp_field: str) -> tuple:
    return tuple(item for item in records if selected.start <= getattr(item, timestamp_field) < selected.end)


def _coverage(records: tuple, source: str, timestamp_field: str, expected: int) -> dict[str, object]:
    result = analyze_coverage(records, source=source, timestamp_field=timestamp_field, expected_step=INTERVAL, expected_count=expected)
    return {
        "classification": result.classification.value, "rows": result.rows,
        "duplicate_timestamps": result.duplicate_timestamps,
        "duplicate_exact_records": result.duplicate_exact_records,
        "missing_intervals": result.missing_intervals,
        "first_timestamp": result.first_timestamp.isoformat() if result.first_timestamp else None,
        "last_timestamp": result.last_timestamp.isoformat() if result.last_timestamp else None,
    }


def common_contiguous_intervals(
    spot: tuple[Candle, ...],
    mark: tuple[DerivativesPriceKline, ...],
    index: tuple[DerivativesPriceKline, ...],
    *,
    available: TimeRange,
) -> tuple[TimeRange, ...]:
    """Find exact common 15m coverage; missing source data remains a hard split."""

    common = set(candle.timestamp for candle in spot if available.start <= candle.timestamp < available.end)
    common &= set(row.open_time for row in mark if available.start <= row.open_time < available.end)
    common &= set(row.open_time for row in index if available.start <= row.open_time < available.end)
    timestamps = sorted(common)
    if not timestamps:
        return ()
    output: list[TimeRange] = []
    start = previous = timestamps[0]
    for timestamp in timestamps[1:]:
        if timestamp - previous != INTERVAL:
            output.append(TimeRange(start, previous + INTERVAL))
            start = timestamp
        previous = timestamp
    output.append(TimeRange(start, previous + INTERVAL))
    return tuple(output)


def _signal_count(candles: tuple[Candle, ...]) -> int:
    """Count immutable V6 signals only; no engine, fills, exits, or outcomes."""

    prepared = prepare_v6_context(candles)
    strategy = V6MTFContinuationStrategy(TrendMomentumConfig(), prepared_context=prepared)
    count = 0
    for index, candle in enumerate(candles):
        if index < strategy.required_history_bars:
            continue
        current = prepared.indicators_15m_by_timestamp[candle.timestamp]
        previous = prepared.indicators_15m_by_timestamp.get(candles[index - 1].timestamp)
        decision = strategy.evaluate(StrategyContext(
            timestamp=candle.timestamp, current_candle=candle,
            recent_history=candles[max(0, index - 20) : index + 1],
            indicators=current, previous_indicators=previous, has_position=False, bars_in_position=0,
            equity=Decimal("100"), cash_usdc=Decimal("100"), completed_trade_count=0,
            bars_since_exit=None, entry_fee_rate=Decimal("0"),
        ))
        count += int(decision.action is StrategyAction.ENTER_LONG)
    return count


def _jsonable(value):
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _dataset_id(payload: dict) -> str:
    stable = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()[:16]


def build_validation_dataset(
    *, manifest: ResearchManifest, raw_root: str | Path, output_root: str | Path
) -> tuple[dict, Path]:
    consumed, holdout, _, _ = validation_plan(manifest)
    planned = availability_archive_plan(raw_root=raw_root)
    downloader = AggregateTradeArchiveDownloader(timeout_seconds=60)
    inventory = []
    spot_segments: list[Candle] = []
    derivative_segments: dict[DerivativesSource, list[DerivativesPriceKline]] = {
        source: [] for source in DERIVATIVE_SOURCES
    }
    for item in planned:
        if item.location.destination.is_file() and item.location.checksum_destination.is_file():
            verify_sha256(item.location.destination, item.location.checksum_destination)
            result = ArchiveDownloadResult(
                item.location.destination, checksum_verified=True, skipped_existing=True
            )
        else:
            try:
                result = downloader.download(item.location, verify_checksum=True)
            except AggregateTradeArchiveError:
                # Windows can report a transient replace lock after the archive has
                # already landed.  Accept only a separately re-verified final file.
                if not (item.location.destination.is_file() and item.location.checksum_destination.is_file()):
                    raise
                verify_sha256(item.location.destination, item.location.checksum_destination)
                result = ArchiveDownloadResult(
                    item.location.destination, checksum_verified=True, skipped_existing=True
                )
        digest = verify_sha256(item.location.destination, item.location.checksum_destination)
        inventory.append({
            "source": item.source, "archive_url": item.location.url,
            "checksum_url": item.location.checksum_url, "sha256": digest,
            "bytes": item.location.destination.stat().st_size, "checksum_verified": result.checksum_verified,
        })
        if item.source == "spot_klines":
            spot_segments.extend(_parse_spot_archive(item.location.destination))
        else:
            source = DerivativesSource(item.source)
            derivative_segments[source].extend(parse_price_kline_archive(item.location.destination, source=source))
    all_spot = tuple(spot_segments)
    all_mark = tuple(derivative_segments[DerivativesSource.MARK_PRICE])
    all_index = tuple(derivative_segments[DerivativesSource.INDEX_PRICE])
    eligible = tuple(
        interval
        for available in discovered_availability()
        for interval in common_contiguous_intervals(all_spot, all_mark, all_index, available=available)
        if not interval.overlaps(holdout) and not any(interval.overlaps(item) for item in consumed)
    )
    selected = select_validation_interval(eligible)
    spot = _in_range(all_spot, selected, "timestamp")
    mark = _in_range(all_mark, selected, "open_time")
    index = _in_range(all_index, selected, "open_time")
    expected = int(selected.duration / INTERVAL)
    coverage = {
        "spot": _coverage(spot, "spot", "timestamp", expected),
        "mark": _coverage(mark, "markPriceKlines", "open_time", expected),
        "index": _coverage(index, "indexPriceKlines", "open_time", expected),
    }
    if any(item["classification"] != CoverageClassification.COMPLETE.value for item in coverage.values()):
        raise ValueError("Validation data has missing or duplicate source records.")
    contexts = build_context_15m(spot_candles=spot, mark=mark, index=index)
    future_violations = sum(
        int(
            context.mark_price_timestamp is not None and context.mark_price_timestamp > context.bucket_close_time
            or context.index_price_timestamp is not None and context.index_price_timestamp > context.bucket_close_time
        )
        for context in contexts
    )
    derived_available = sum(context.premium_fraction is not None for context in contexts)
    if future_violations or derived_available != expected:
        raise ValueError("Causal mark/index alignment is incomplete or future-contaminated.")
    candidate_count = _signal_count(spot)
    core = {
        "dataset_version": DATASET_VERSION, "selection_rule": "longest_contiguous_available_interval; ties=earliest; data_availability_only",
        "selected_interval": selected.as_dict(), "eligible_intervals": [item.as_dict() for item in eligible],
        "excluded_outcome_consumed_ranges": [item.as_dict() for item in consumed],
        "blind_holdout_exclusion": holdout.as_dict(), "source_inventory": inventory,
        "coverage": coverage, "future_data_violations": future_violations,
        "derived_mark_index_premium_available": derived_available, "candidate_count": candidate_count,
        "discovery_overlap": 0, "blind_holdout_overlap": 0,
    }
    dataset_id = _dataset_id(core)
    root = Path(output_root) / dataset_id
    root.mkdir(parents=True, exist_ok=True)
    context_path = root / "context.csv"
    with context_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("bucket_open_time", "bucket_close_time", "mark_price", "mark_price_timestamp", "index_price", "index_price_timestamp", "derived_mark_index_premium"))
        writer.writeheader()
        for item in contexts:
            writer.writerow({
                "bucket_open_time": item.bucket_open_time.isoformat(), "bucket_close_time": item.bucket_close_time.isoformat(),
                "mark_price": item.mark_price, "mark_price_timestamp": item.mark_price_timestamp.isoformat() if item.mark_price_timestamp else None,
                "index_price": item.index_price, "index_price_timestamp": item.index_price_timestamp.isoformat() if item.index_price_timestamp else None,
                "derived_mark_index_premium": item.premium_fraction,
            })
    result = {**core, "dataset_id": dataset_id, "classification": "VALIDATION_DATA_READY", "context_path": str(context_path)}
    manifest_path = root / "manifest.json"
    encoded = json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n"
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != encoded:
        raise FileExistsError("Existing validation manifest differs; refusing overwrite.")
    manifest_path.write_text(encoded, encoding="utf-8")
    return result, manifest_path
