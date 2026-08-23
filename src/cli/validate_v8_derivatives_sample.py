"""Acquire and validate one bounded official Binance USD-M sample day."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from statistics import median

from src.cli.run_multiregime import _require_locked_holdout
from src.derivatives.context import build_context_15m
from src.derivatives.integrity import CoverageClassification, analyze_coverage
from src.derivatives.models import DerivativesSource
from src.derivatives.parser import (
    parse_funding_archive,
    parse_metrics_archive,
    parse_price_kline_archive,
)
from src.derivatives.sources import OFFICIAL_SOURCES, build_sample_locations
from src.historical.storage import HistoricalDatasetStore
from src.hypotheses.manifest import HoldoutStatus, ResearchManifestStore
from src.orderflow.archive import (
    AggregateTradeArchiveDownloader,
    AggregateTradeArchiveError,
    ArchiveDownloadResult,
)
from src.orderflow.integrity import verify_sha256


HOLDOUT_START = datetime(2025, 8, 1, tzinfo=timezone.utc)
HOLDOUT_END = datetime(2026, 2, 1, tzinfo=timezone.utc)
EXPECTED_SPOT_BUCKETS = 96


def _plain(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def reject_holdout_day(sample_date: date) -> None:
    start = datetime.combine(sample_date, time.min, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    if start < HOLDOUT_END and HOLDOUT_START < end:
        raise ValueError("V8 sample day overlaps the blind holdout.")


def _range_value(item, name: str):
    return item[name] if isinstance(item, dict) else getattr(item, name)


def select_safe_sample_date(candles: tuple, consumed_ranges: tuple) -> date:
    """Select the earliest complete UTC day by data policy, never market behavior."""

    allowed = tuple(
        (
            (
                datetime.fromisoformat(_range_value(item, "start"))
                if isinstance(_range_value(item, "start"), str)
                else _range_value(item, "start")
            ).astimezone(timezone.utc),
            (
                datetime.fromisoformat(_range_value(item, "end"))
                if isinstance(_range_value(item, "end"), str)
                else _range_value(item, "end")
            ).astimezone(timezone.utc),
        )
        for item in consumed_ranges
        if _range_value(item, "data_status") == "CONSUMED_RESEARCH_DATA"
        and _range_value(item, "symbol") == "BTCUSDC"
        and _range_value(item, "interval") == "15m"
    )
    by_day: dict[date, list] = {}
    for candle in candles:
        by_day.setdefault(candle.timestamp.date(), []).append(candle)
    for candidate in sorted(by_day):
        start = datetime.combine(candidate, time.min, tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        if HOLDOUT_START < end and start < HOLDOUT_END:
            continue
        if not any(range_start <= start and end <= range_end for range_start, range_end in allowed):
            continue
        selected = sorted(by_day[candidate], key=lambda candle: candle.timestamp)
        expected = tuple(start + index * timedelta(minutes=15) for index in range(96))
        if len(selected) == 96 and tuple(candle.timestamp for candle in selected) == expected:
            return candidate
    raise ValueError("No complete consumed-data Spot UTC day is available.")


def _observed_cadence(records: tuple, timestamp_field: str) -> str | None:
    timestamps = tuple(getattr(record, timestamp_field) for record in records)
    deltas = tuple(later - earlier for earlier, later in zip(timestamps, timestamps[1:]) if later > earlier)
    if not deltas:
        return None
    return str(median(deltas))


def _write_csv(path: Path, rows: tuple[dict, ...], fieldnames: tuple[str, ...]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_plain(rows))
    temporary.replace(path)


def _source_inventory_base() -> dict[DerivativesSource, dict]:
    return {
        item.source: {
            "source": item.source.value,
            "official_path_name": item.official_path_name,
            "daily_available": item.daily_available,
            "monthly_available": item.monthly_available,
            "official_schema": "|".join(item.schema),
            "timestamp_unit": item.timestamp_unit,
            "earliest_safe_date_if_determinable": item.earliest_safe_date,
            "checksum_available": item.checksum_available,
            "availability": "SUPPORTED",
        }
        for item in OFFICIAL_SOURCES
    }


SOURCE_INVENTORY_FIELDS = (
    "source",
    "official_path_name",
    "daily_available",
    "monthly_available",
    "official_schema",
    "timestamp_unit",
    "earliest_safe_date_if_determinable",
    "checksum_available",
    "availability",
    "archive_url",
    "archive_path",
    "checksum_url",
    "archive_bytes",
    "checksum_verified",
    "sha256",
    "row_count",
    "first_timestamp",
    "last_timestamp",
    "observed_cadence",
    "coverage_classification",
    "duplicate_timestamps",
    "duplicate_exact_records",
    "missing_intervals",
    "error",
)


def _coverage_for(source: DerivativesSource, records: tuple):
    if source is DerivativesSource.FUNDING_RATE:
        return analyze_coverage(records, source=source.value, timestamp_field="timestamp")
    if source is DerivativesSource.METRICS:
        return analyze_coverage(
            records,
            source=source.value,
            timestamp_field="timestamp",
            expected_step=timedelta(minutes=5),
            expected_count=288,
        )
    return analyze_coverage(
        records,
        source=source.value,
        timestamp_field="open_time",
        expected_step=timedelta(minutes=15),
        expected_count=96,
    )


def _context_rows(contexts) -> tuple[dict, ...]:
    return tuple(asdict(context) for context in contexts)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate one safe V8 derivatives sample day.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root-manifest", default="research/hypothesis_manifest.json")
    parser.add_argument("--spot-root", default="data/historical/multiregime_expansion")
    parser.add_argument("--raw-root", default="data/derivatives/raw")
    parser.add_argument("--processed-root", default="data/derivatives/processed/15m")
    parser.add_argument("--output-root", default="reports/derivatives_validation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        print("V8 SAFE-SAMPLE VALIDATION NOT EXECUTED. Explicit --execute is required.")
        return 1

    root_manifest = ResearchManifestStore(args.root_manifest).load()
    _require_locked_holdout(root_manifest)
    if (
        root_manifest.holdout_status is not HoldoutStatus.LOCKED_BLIND_HOLDOUT
        or root_manifest.blind_holdout.reveal_timestamp is not None
        or root_manifest.blind_holdout.consumed_timestamp is not None
    ):
        raise ValueError("Blind holdout contamination detected.")
    spot = HistoricalDatasetStore(args.spot_root, allow_source_gaps=True).load("BTCUSDC", "15m")
    if spot is None:
        raise ValueError("Consumed Spot reference data is missing.")
    sample_date = select_safe_sample_date(
        spot.candles, root_manifest.consumed_dataset_ranges
    )
    reject_holdout_day(sample_date)
    day_start = datetime.combine(sample_date, time.min, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)
    spot_day = tuple(candle for candle in spot.candles if day_start <= candle.timestamp < day_end)
    if len(spot_day) != EXPECTED_SPOT_BUCKETS:
        raise ValueError("Safe sample does not contain exactly 96 Spot buckets.")

    inventory = _source_inventory_base()
    parsed: dict[DerivativesSource, tuple] = {}
    coverage = {}
    downloader = AggregateTradeArchiveDownloader(timeout_seconds=60)
    downloaded_paths = []
    for source, location in build_sample_locations(sample_date, root=args.raw_root):
        item = inventory[source]
        item.update(
            {
                "archive_url": location.url,
                "archive_path": str(location.destination),
                "checksum_url": location.checksum_url,
            }
        )
        try:
            if location.destination.is_file() and location.checksum_destination.is_file():
                verify_sha256(location.destination, location.checksum_destination)
                result = ArchiveDownloadResult(
                    location.destination, checksum_verified=True, skipped_existing=True
                )
            else:
                result = downloader.download(location, verify_checksum=True)
            digest = verify_sha256(location.destination, location.checksum_destination)
            if source is DerivativesSource.FUNDING_RATE:
                records = parse_funding_archive(result.path)
            elif source is DerivativesSource.METRICS:
                records = parse_metrics_archive(result.path)
            else:
                records = parse_price_kline_archive(result.path, source=source)
            parsed[source] = records
            result_coverage = _coverage_for(source, records)
            coverage[source] = result_coverage
            item.update(
                {
                    "archive_bytes": result.path.stat().st_size,
                    "checksum_verified": result.checksum_verified,
                    "sha256": digest,
                    "row_count": len(records),
                    "first_timestamp": result_coverage.first_timestamp,
                    "last_timestamp": result_coverage.last_timestamp,
                    "observed_cadence": _observed_cadence(
                        records,
                        "timestamp" if source in {DerivativesSource.FUNDING_RATE, DerivativesSource.METRICS} else "open_time",
                    ),
                    "coverage_classification": result_coverage.classification.value,
                    "duplicate_timestamps": result_coverage.duplicate_timestamps,
                    "duplicate_exact_records": result_coverage.duplicate_exact_records,
                    "missing_intervals": result_coverage.missing_intervals,
                }
            )
            downloaded_paths.append(str(result.path))
        except (AggregateTradeArchiveError, OSError, ValueError) as exc:
            parsed[source] = ()
            coverage[source] = analyze_coverage(
                (), source=source.value, timestamp_field="timestamp"
            )
            item.update(
                {
                    "availability": "UNSUPPORTED_OR_UNAVAILABLE_FOR_SAMPLE",
                    "coverage_classification": "UNAVAILABLE",
                    "checksum_verified": False,
                    "row_count": 0,
                    "duplicate_timestamps": 0,
                    "duplicate_exact_records": 0,
                    "missing_intervals": 0,
                    "error": str(exc),
                }
            )

    funding = tuple(
        row for row in parsed[DerivativesSource.FUNDING_RATE] if row.timestamp <= day_end
    )
    metrics = tuple(
        row for row in parsed[DerivativesSource.METRICS] if day_start <= row.timestamp < day_end
    )
    contexts = build_context_15m(
        spot_candles=spot_day,
        funding=funding,
        metrics=metrics,
        mark=parsed[DerivativesSource.MARK_PRICE],
        index=parsed[DerivativesSource.INDEX_PRICE],
        premium=parsed[DerivativesSource.PREMIUM_INDEX],
    )
    if len(contexts) != 96:
        raise ValueError("V8 causal context alignment did not produce 96 buckets.")
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

    comparisons = tuple(
        (context.premium_fraction, context.premium_index)
        for context in contexts
        if context.premium_fraction is not None and context.premium_index is not None
    )
    differences = tuple(abs(derived - official) for derived, official in comparisons)
    usable = tuple(source for source, records in parsed.items() if records)
    incomplete = any(
        result.classification is not CoverageClassification.COMPLETE
        for result in coverage.values()
    )
    if not usable or future_violations or len(contexts) != 96:
        classification = "NOT_FEASIBLE"
    elif incomplete:
        classification = "PARTIALLY_VALIDATED"
    else:
        classification = "VALIDATED"

    definition = {
        "version": "v8-derivatives-foundation-sample-1",
        "sample_date": sample_date.isoformat(),
        "sources": [source.value for source in DerivativesSource],
        "symbol": "BTCUSDT",
        "context_interval": "15m",
    }
    deterministic_id = hashlib.sha256(
        json.dumps(definition, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    output = Path(args.output_root) / deterministic_id
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise ValueError("V8 derivatives validation exists; refusing overwrite.")
    output.mkdir(parents=True, exist_ok=True)
    context_rows = _context_rows(contexts)
    inventory_rows = tuple(inventory[source] for source in DerivativesSource)
    issues = tuple(
        asdict(issue) for result in coverage.values() for issue in result.issues
    )
    issue_fields = ("source", "issue_type", "timestamp", "detail")
    _write_csv(output / "source_inventory.csv", inventory_rows, SOURCE_INVENTORY_FIELDS)
    _write_csv(output / "context_15m.csv", context_rows, tuple(context_rows[0]))
    _write_csv(output / "continuity_issues.csv", issues, issue_fields)
    processed = Path(args.processed_root) / "BTCUSDT" / sample_date.isoformat()
    processed.mkdir(parents=True, exist_ok=True)
    _write_csv(processed / "context_15m.csv", context_rows, tuple(context_rows[0]))

    payload = {
        "validation_id": deterministic_id,
        "definition": definition,
        "classification": classification,
        "selected_safe_utc_date": sample_date,
        "source_inventory": inventory_rows,
        "archives": downloaded_paths,
        "spot_context_buckets": len(contexts),
        "future_data_violations": future_violations,
        "premium_consistency": {
            "comparable_buckets": len(comparisons),
            "mean_absolute_difference": (
                sum(differences, Decimal("0")) / Decimal(len(differences))
                if differences
                else None
            ),
            "maximum_absolute_difference": max(differences) if differences else None,
            "semantic_note": (
                "Official premium-index kline close is preserved as its own source value; "
                "it is not forced equal to (mark-index)/index because Binance premium-index "
                "and mark-price construction semantics differ."
            ),
        },
        "integrity": {
            "bulk_download": False,
            "strategy_or_backtest_executed": False,
            "threshold_or_profitability_analysis": False,
            "blind_holdout": {
                "status": "LOCKED_BLIND_HOLDOUT",
                "loaded": False,
                "revealed": False,
                "consumed": False,
                "evaluated": False,
            },
        },
    }
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    print(f"V8 DERIVATIVES SAMPLE: {sample_date.isoformat()} | {classification}")
    print(f"Context buckets: {len(contexts)} | future violations: {future_violations}")
    print(f"Report: {summary_path.resolve()}")
    print("Blind holdout: LOCKED | NOT LOADED | NOT REVEALED | NOT CONSUMED | NOT EVALUATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
