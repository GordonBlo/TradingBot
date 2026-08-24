"""Plan and build the V8 consumed-window derivatives context dataset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path

from src.cli.build_v7_orderflow_dataset import authoritative_context
from src.derivatives.context import build_context_15m
from src.derivatives.dataset import DerivativesAcquisitionPlan, build_derivatives_acquisition_plan
from src.derivatives.integrity import (
    ContinuityIssue,
    CoverageClassification,
    analyze_coverage,
    analyze_funding_coverage,
)
from src.derivatives.models import DerivativesSource
from src.derivatives.parser import (
    parse_funding_archive,
    parse_metrics_archive,
    parse_price_kline_archive,
)
from src.orderflow.archive import AggregateTradeArchiveDownloader, ArchiveLocation
from src.orderflow.dataset import ArchivePlanItem, deterministic_manifest_sha256
from src.orderflow.integrity import parse_checksum, verify_sha256


REPORT_ROOT = Path("reports/derivatives_dataset/v8_consumed")
PROCESSED_ROOT = Path("data/derivatives/processed/15m/BTCUSDT")
HOLDOUT_START = datetime(2025, 8, 1, tzinfo=timezone.utc)
HOLDOUT_END = datetime(2026, 2, 1, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class ArchiveAcquisition:
    item: ArchivePlanItem
    verified: bool
    downloaded: bool
    error: str | None = None


def _plain(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, Path)):
        return str(value) if isinstance(value, Path) else value.isoformat()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: tuple[dict, ...], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_plain(rows))
    temporary.replace(path)


def _present_verified(item: ArchivePlanItem) -> bool:
    archive = Path(item.local_path)
    checksum = Path(item.checksum_local_path)
    if not archive.is_file() or not checksum.is_file():
        return False
    try:
        verify_sha256(archive, checksum)
    except ValueError:
        return False
    return True


def inventory_plan(plan: DerivativesAcquisitionPlan) -> DerivativesAcquisitionPlan:
    return replace(
        plan,
        archives=tuple(
            replace(item, already_present_verified=_present_verified(item))
            for item in plan.archives
        ),
    )


def plan_summary(plan: DerivativesAcquisitionPlan) -> dict:
    counts = Counter((item.source_dataset, item.granularity) for item in plan.archives)
    free = shutil.disk_usage(Path.cwd()).free
    return {
        "dataset_id": plan.dataset_id,
        "eligible_windows": len(plan.coverage.windows),
        "required_15m_context_buckets": len(plan.coverage.required_timestamps),
        "safe_ranges": plan.coverage.safe_ranges,
        "required_unique_days": plan.required_unique_days,
        "required_complete_months": plan.required_complete_months,
        "planned_archives": len(plan.archives),
        "archive_counts": {
            f"{source}:{cadence}": count
            for (source, cadence), count in sorted(counts.items())
        },
        "already_present_verified": sum(item.already_present_verified for item in plan.archives),
        "archives_remaining": sum(not item.already_present_verified for item in plan.archives),
        "free_disk_bytes": free,
        "disk_space_sufficient": free > 1_000_000_000,
        "holdout_intersection": "NONE",
    }


def write_plan(plan: DerivativesAcquisitionPlan) -> Path:
    output = REPORT_ROOT / plan.dataset_id
    _write_json(
        output / "acquisition_plan.json",
        {
            "summary": plan_summary(plan),
            "windows": [asdict(window) for window in plan.coverage.windows],
            "archives": [asdict(item) for item in plan.archives],
        },
    )
    rows = tuple(asdict(item) for item in plan.archives)
    _write_csv(output / "archive_inventory.csv", rows, tuple(rows[0]))
    return output


def _location(item: ArchivePlanItem) -> ArchiveLocation:
    return ArchiveLocation(
        item.url,
        item.checksum_url,
        Path(item.local_path),
        Path(item.checksum_local_path),
    )


def _acquire_one(
    item: ArchivePlanItem, *, cached_only: bool = False
) -> ArchiveAcquisition:
    if _present_verified(item):
        return ArchiveAcquisition(item, True, False)
    if cached_only:
        return ArchiveAcquisition(
            item,
            False,
            False,
            "Archive unavailable in checksum-verified local cache.",
        )
    last_error = None
    for _ in range(2):
        try:
            result = AggregateTradeArchiveDownloader(timeout_seconds=60).download(
                _location(item), verify_checksum=True
            )
            verify_sha256(result.path, Path(item.checksum_local_path))
            return ArchiveAcquisition(item, True, not result.skipped_existing)
        except Exception as exc:
            last_error = exc
    return ArchiveAcquisition(item, False, False, str(last_error))


def acquire_archives(
    plan: DerivativesAcquisitionPlan,
    *,
    workers: int = 8,
    cached_only: bool = False,
) -> tuple[ArchiveAcquisition, ...]:
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_acquire_one, item, cached_only=cached_only): item
            for item in plan.archives
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 50 == 0 or completed == len(futures):
                print(f"ARCHIVES {completed}/{len(futures)}", flush=True)
    by_key = {
        (result.item.source_dataset, result.item.granularity, result.item.period): result
        for result in results
    }
    return tuple(
        by_key[(item.source_dataset, item.granularity, item.period)]
        for item in plan.archives
    )


def _parse(item: ArchivePlanItem) -> tuple:
    source = DerivativesSource(item.source_dataset)
    if source is DerivativesSource.FUNDING_RATE:
        return parse_funding_archive(item.local_path)
    if source is DerivativesSource.METRICS:
        return parse_metrics_archive(item.local_path)
    return parse_price_kline_archive(item.local_path, source=source)


def _timestamp_field(source: DerivativesSource) -> str:
    return "timestamp" if source in {DerivativesSource.FUNDING_RATE, DerivativesSource.METRICS} else "open_time"


def _expected_count(item: ArchivePlanItem, source: DerivativesSource) -> int | None:
    days = int((item.archive_end - item.archive_start).total_seconds() // 86_400)
    if source is DerivativesSource.METRICS:
        return days * 288
    if source in {
        DerivativesSource.MARK_PRICE,
        DerivativesSource.INDEX_PRICE,
        DerivativesSource.PREMIUM_INDEX,
    }:
        return days * 96
    return None


def _step(source: DerivativesSource) -> timedelta | None:
    if source is DerivativesSource.METRICS:
        return timedelta(minutes=5)
    if source in {
        DerivativesSource.MARK_PRICE,
        DerivativesSource.INDEX_PRICE,
        DerivativesSource.PREMIUM_INDEX,
    }:
        return timedelta(minutes=15)
    return None


def exclude_ambiguous_timestamps(records: tuple, timestamp_field: str) -> tuple:
    counts = Counter(getattr(record, timestamp_field) for record in records)
    return tuple(record for record in records if counts[getattr(record, timestamp_field)] == 1)


def normalize_source_order(records: tuple, timestamp_field: str) -> tuple[tuple, int]:
    decreases = sum(
        getattr(later, timestamp_field) < getattr(earlier, timestamp_field)
        for earlier, later in zip(records, records[1:])
    )
    return (
        tuple(sorted(records, key=lambda record: getattr(record, timestamp_field))),
        decreases,
    )


def parse_and_audit(
    acquisitions: tuple[ArchiveAcquisition, ...],
) -> tuple[dict[DerivativesSource, tuple], tuple[dict, ...], tuple[dict, ...], tuple[dict, ...]]:
    records_by_source: dict[DerivativesSource, list] = defaultdict(list)
    issues = []
    checksums = []
    counters = {
        source: {
            "source": source.value,
            "planned_archives": 0,
            "verified_archives": 0,
            "unavailable_archives": 0,
            "rows": 0,
            "missing_intervals": 0,
            "duplicate_timestamps": 0,
            "duplicate_exact_records": 0,
            "non_monotonic_archives": 0,
        }
        for source in DerivativesSource
    }
    for acquisition in acquisitions:
        source = DerivativesSource(acquisition.item.source_dataset)
        row = counters[source]
        row["planned_archives"] += 1
        if not acquisition.verified:
            row["unavailable_archives"] += 1
            missing = _expected_count(acquisition.item, source) or 0
            row["missing_intervals"] += missing
            issues.append(
                {
                    "source": source.value,
                    "issue_type": "ARCHIVE_UNAVAILABLE",
                    "timestamp": acquisition.item.archive_start,
                    "detail": acquisition.error,
                }
            )
            continue
        try:
            records = _parse(acquisition.item)
            records, decreases = normalize_source_order(
                records, _timestamp_field(source)
            )
            if decreases:
                row["non_monotonic_archives"] += 1
                issues.append(
                    {
                        "source": source.value,
                        "issue_type": "NON_MONOTONIC_ARCHIVE_ORDER",
                        "timestamp": acquisition.item.archive_start,
                        "detail": f"{decreases} decreases; records indexed by authoritative timestamp.",
                    }
                )
            audit = (
                analyze_funding_coverage(records, source=source.value)
                if source is DerivativesSource.FUNDING_RATE
                else analyze_coverage(
                    records,
                    source=source.value,
                    timestamp_field=_timestamp_field(source),
                    expected_step=_step(source),
                    expected_count=_expected_count(acquisition.item, source),
                )
            )
        except Exception as exc:
            row["unavailable_archives"] += 1
            issues.append(
                {
                    "source": source.value,
                    "issue_type": "ARCHIVE_PARSE_FAILED",
                    "timestamp": acquisition.item.archive_start,
                    "detail": str(exc),
                }
            )
            continue
        row["verified_archives"] += 1
        row["rows"] += len(records)
        row["missing_intervals"] += audit.missing_intervals
        records_by_source[source].extend(records)
        issues.extend(asdict(issue) for issue in audit.issues)
        checksums.append(
            {
                "source": source.value,
                "granularity": acquisition.item.granularity,
                "period": acquisition.item.period,
                "archive_url": acquisition.item.url,
                "sha256": parse_checksum(
                    Path(acquisition.item.checksum_local_path).read_text(encoding="utf-8"),
                    expected_filename=Path(acquisition.item.local_path).name,
                ),
            }
        )
    final_records = {}
    for source in DerivativesSource:
        timestamp_field = _timestamp_field(source)
        records = tuple(
            sorted(records_by_source[source], key=lambda record: getattr(record, timestamp_field))
        )
        timestamps = tuple(getattr(record, timestamp_field) for record in records)
        duplicate_timestamps = len(timestamps) - len(set(timestamps))
        duplicate_records = len(records) - len(set(records))
        counters[source]["duplicate_timestamps"] += duplicate_timestamps
        counters[source]["duplicate_exact_records"] += duplicate_records
        final_records[source] = exclude_ambiguous_timestamps(records, timestamp_field)
        row = counters[source]
        if row["verified_archives"] == 0:
            classification = CoverageClassification.UNAVAILABLE
        elif row["duplicate_timestamps"]:
            classification = CoverageClassification.DUPLICATED
        elif row["unavailable_archives"]:
            classification = CoverageClassification.PARTIAL
        elif row["non_monotonic_archives"]:
            classification = CoverageClassification.PARTIAL
        elif row["missing_intervals"]:
            classification = CoverageClassification.GAPPED
        else:
            classification = CoverageClassification.COMPLETE
        row["classification"] = classification.value
        row["usable_rows"] = len(final_records[source])
    return (
        {source: tuple(final_records[source]) for source in DerivativesSource},
        tuple(counters[source] for source in DerivativesSource),
        tuple(issues),
        tuple(checksums),
    )


def _in_safe_ranges(timestamp: datetime, safe_ranges: tuple) -> bool:
    return any(start <= timestamp < end for start, end in safe_ranges)


def _local_candles(windows: tuple) -> tuple:
    by_time = {}
    for window in windows:
        for candle in window.replay_dataset.candles:
            previous = by_time.get(candle.timestamp)
            if previous is not None and previous != candle:
                raise ValueError("Conflicting Spot candles in V8 coverage.")
            by_time[candle.timestamp] = candle
    return tuple(by_time[timestamp] for timestamp in sorted(by_time))


def build_dataset(
    plan: DerivativesAcquisitionPlan,
    windows: tuple,
    acquisitions: tuple[ArchiveAcquisition, ...],
) -> dict:
    output = PROCESSED_ROOT / plan.dataset_id
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError("V8 derivatives dataset manifest exists; refusing overwrite.")
    records, source_audit, issues, checksums = parse_and_audit(acquisitions)
    for source, values in records.items():
        timestamp_field = _timestamp_field(source)
        if any(
            HOLDOUT_START <= getattr(record, timestamp_field) < HOLDOUT_END
            for record in values
        ):
            raise ValueError("Loaded V8 derivatives data intersects the blind holdout.")
    required = frozenset(plan.coverage.required_timestamps)
    spot_candles = _local_candles(windows)
    if len(spot_candles) != len(required) or {candle.timestamp for candle in spot_candles} != required:
        raise ValueError("V8 Spot context coverage differs from the frozen windows.")
    price_sources = {
        source: tuple(row for row in records[source] if row.open_time in required)
        for source in (
            DerivativesSource.MARK_PRICE,
            DerivativesSource.INDEX_PRICE,
            DerivativesSource.PREMIUM_INDEX,
        )
    }
    metrics = tuple(
        row
        for row in records[DerivativesSource.METRICS]
        if _in_safe_ranges(row.timestamp, plan.coverage.safe_ranges)
    )
    contexts = build_context_15m(
        spot_candles=spot_candles,
        funding=records[DerivativesSource.FUNDING_RATE],
        metrics=metrics,
        mark=price_sources[DerivativesSource.MARK_PRICE],
        index=price_sources[DerivativesSource.INDEX_PRICE],
        premium=price_sources[DerivativesSource.PREMIUM_INDEX],
    )
    future_violations = sum(
        timestamp is not None and timestamp > context.bucket_close_time
        for context in contexts
        for timestamp in (
            context.funding_timestamp,
            context.open_interest_timestamp,
            context.mark_price_timestamp,
            context.index_price_timestamp,
            context.premium_index_timestamp,
        )
    )
    missing = {
        "funding_rate": sum(context.latest_known_funding_rate is None for context in contexts),
        "open_interest": sum(context.open_interest is None for context in contexts),
        "mark_price": sum(context.mark_price is None for context in contexts),
        "index_price": sum(context.index_price is None for context in contexts),
        "premium_index": sum(context.premium_index is None for context in contexts),
        "derived_premium": sum(context.premium_fraction is None for context in contexts),
    }
    complete_sources = sum(row["classification"] == "COMPLETE" for row in source_audit)
    limitations = any(row["classification"] != "COMPLETE" for row in source_audit) or any(missing.values())
    if len(contexts) != len(required) or future_violations or complete_sources == 0:
        classification = "INCOMPLETE"
    elif limitations:
        classification = "READY_WITH_SOURCE_LIMITATIONS"
    else:
        classification = "READY"

    partitions: dict[tuple[int, int], list] = defaultdict(list)
    for context in contexts:
        partitions[(context.bucket_open_time.year, context.bucket_open_time.month)].append(
            asdict(context)
        )
    partition_paths = []
    fields = tuple(asdict(contexts[0]))
    for (year, month), rows in sorted(partitions.items()):
        path = output / "partitions" / f"{year:04d}" / f"{month:02d}" / "context_15m.csv"
        _write_csv(path, tuple(rows), fields)
        partition_paths.append(path)

    stable = {
        "dataset_version": "V8_DERIVATIVES_CONTEXT_CONSUMED_1",
        "dataset_id": plan.dataset_id,
        "traded_market": "BTCUSDC_SPOT_15M_LONG_ONLY",
        "context_market": "BTCUSDT_BINANCE_PUBLIC_USDM_PERPETUAL",
        "window_ids": [window.window_id for window in plan.coverage.windows],
        "safe_ranges": plan.coverage.safe_ranges,
        "required_context_buckets": len(required),
        "context_buckets": len(contexts),
        "archive_checksums": checksums,
        "source_audit": source_audit,
        "missing_context_values": missing,
        "future_data_violations": future_violations,
        "partition_paths": partition_paths,
        "classification": classification,
    }
    manifest = {
        **stable,
        "dataset_definition_sha256": deterministic_manifest_sha256(_plain(stable)),
        "generated_at": datetime.now(timezone.utc),
        "acquisition": {
            "planned": len(acquisitions),
            "verified": sum(item.verified for item in acquisitions),
            "downloaded": sum(item.downloaded for item in acquisitions),
            "reused": sum(item.verified and not item.downloaded for item in acquisitions),
            "unavailable": sum(not item.verified for item in acquisitions),
        },
        "blind_holdout": {
            "status": "LOCKED_BLIND_HOLDOUT",
            "intersection": "NONE",
            "downloaded": False,
            "loaded": False,
            "revealed": False,
            "consumed": False,
            "evaluated": False,
        },
        "integrity": {
            "interpolation": False,
            "future_fill": False,
            "strategy_or_backtest": False,
            "profitability_or_threshold_analysis": False,
        },
    }
    _write_json(manifest_path, manifest)
    report = REPORT_ROOT / plan.dataset_id
    _write_json(report / "summary.json", manifest)
    _write_csv(report / "source_audit.csv", source_audit, tuple(source_audit[0]))
    issue_fields = ("source", "issue_type", "timestamp", "detail")
    _write_csv(report / "continuity_issues.csv", issues, issue_fields)
    return manifest


def _print_plan(summary: dict) -> None:
    print(f"Dataset ID: {summary['dataset_id']}")
    print(f"Windows: {summary['eligible_windows']} | 15m buckets: {summary['required_15m_context_buckets']}")
    print(f"Planned archives: {summary['planned_archives']} | remaining: {summary['archives_remaining']}")
    for name, count in summary["archive_counts"].items():
        print(f"  {name}: {count}")
    print(f"Free disk bytes: {summary['free_disk_bytes']}")
    print("Holdout intersection: NONE")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cached-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 16:
        raise ValueError("V8 acquisition workers must be between 1 and 16.")
    _, windows, coverage = authoritative_context()
    plan = inventory_plan(build_derivatives_acquisition_plan(coverage))
    report = write_plan(plan)
    summary = plan_summary(plan)
    _print_plan(summary)
    print(f"Plan: {(report / 'acquisition_plan.json').resolve()}")
    if args.plan:
        print("PLAN COMPLETE — NO DOWNLOAD EXECUTED")
        return 0
    if not summary["disk_space_sufficient"]:
        print("INCOMPLETE — insufficient disk before acquisition")
        return 1
    if (PROCESSED_ROOT / plan.dataset_id / "manifest.json").exists():
        print("INCOMPLETE — final manifest already exists")
        return 1
    acquisitions = acquire_archives(
        plan,
        workers=args.workers,
        cached_only=args.cached_only,
    )
    manifest = build_dataset(plan, windows, acquisitions)
    print(f"Context buckets: {manifest['context_buckets']}")
    print(f"Future-data violations: {manifest['future_data_violations']}")
    print(f"Dataset manifest: {(PROCESSED_ROOT / plan.dataset_id / 'manifest.json').resolve()}")
    print(manifest["classification"])
    return 0 if manifest["classification"] != "INCOMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
