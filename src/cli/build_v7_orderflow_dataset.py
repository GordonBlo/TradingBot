"""Plan and build the holdout-safe V7 consumed-data order-flow dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.request import Request, urlopen

from src.cli.run_multiregime import _require_locked_holdout
from src.cli.run_v33_h5 import _build_regions, _read_reference_windows
from src.cli.run_v6_h0_replay import (
    expand_complete_region_history,
    validate_eligible_windows,
)
from src.config.settings import load_settings
from src.historical.dataset import HistoricalDataset
from src.hypotheses.manifest import HoldoutStatus, ResearchManifestStore
from src.orderflow.aggregation import bucket_open_15m, iter_aggregate_15m
from src.orderflow.archive import (
    AggregateTradeArchiveDownloader,
    ArchiveLocation,
)
from src.orderflow.dataset import (
    AcquisitionPlan,
    ArchivePlanItem,
    build_acquisition_plan,
    deterministic_manifest_sha256,
    derive_window_coverage,
)
from src.orderflow.integrity import parse_checksum, verify_sha256
from src.orderflow.parser import iter_aggtrade_archive
from src.orderflow.reconciliation import (
    parse_kline_archive,
    reconcile_orderflow_dataset,
)
from src.orderflow.storage import OrderFlowBucketStore
from src.research.multiregime.windows import construct_windows


REPORT_ROOT = Path("reports/orderflow_dataset/v7_consumed")
AGGREGATED_ROOT = Path("data/orderflow/aggregated/15m")
EXPANSION_RATIO = Decimal("5.07628993267491")


def _json_value(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def authoritative_context() -> tuple[tuple, tuple, object]:
    root = ResearchManifestStore("research/hypothesis_manifest.json").load()
    _require_locked_holdout(root)
    if (
        root.holdout_status is not HoldoutStatus.LOCKED_BLIND_HOLDOUT
        or root.blind_holdout is None
        or root.blind_holdout.reveal_timestamp is not None
        or root.blind_holdout.consumed_timestamp is not None
    ):
        raise ValueError("Blind holdout integrity failed.")
    regions = _build_regions(
        root_manifest=root,
        expansion_data_root="data/historical/multiregime_expansion",
        consumed_data_root="data/historical",
    )
    windows = construct_windows(regions, load_settings().multiregime_config())
    references = _read_reference_windows(Path("reports/mechanisms/bc2496aed05555b5"))
    eligible = expand_complete_region_history(
        tuple(window for window in windows if window.window_id in references), regions
    )
    validate_eligible_windows(eligible)
    coverage = derive_window_coverage(eligible, regions)
    return regions, eligible, coverage


def _head_size(item: ArchivePlanItem) -> ArchivePlanItem:
    try:
        with urlopen(Request(item.url, method="HEAD"), timeout=15) as response:
            raw = response.headers.get("Content-Length")
            size = int(raw) if raw is not None else None
    except Exception:
        size = None
    return replace(item, content_length_bytes=size)


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


def inventory_plan(plan: AcquisitionPlan, *, query_sizes: bool) -> AcquisitionPlan:
    archives = tuple(
        replace(item, already_present_verified=_present_verified(item))
        for item in plan.archives
    )
    if query_sizes:
        with ThreadPoolExecutor(max_workers=8) as pool:
            archives = tuple(pool.map(_head_size, archives))
    return replace(plan, archives=archives)


def plan_summary(plan: AcquisitionPlan) -> dict:
    primary = tuple(item for item in plan.archives if item.source_dataset == "aggTrades")
    remaining = tuple(item for item in plan.archives if not item.already_present_verified)
    known_remaining = tuple(
        item.content_length_bytes
        for item in remaining
        if item.content_length_bytes is not None
    )
    primary_bytes = sum(
        item.content_length_bytes or 0
        for item in primary
        if not item.already_present_verified
    )
    free = shutil.disk_usage(Path.cwd()).free
    return {
        "dataset_id": plan.dataset_id,
        "required_15m_timestamps": len(plan.coverage.required_timestamps),
        "required_unique_utc_days": plan.required_unique_days,
        "required_complete_months": plan.required_complete_months,
        "required_daily_archives": plan.required_daily_archives,
        "aggtrades_archive_count": len(primary),
        "kline_reference_archive_count": len(plan.archives) - len(primary),
        "total_archive_count": len(plan.archives),
        "already_present_verified": sum(
            item.already_present_verified for item in plan.archives
        ),
        "archives_still_required": len(remaining),
        "known_content_lengths": len(known_remaining),
        "estimated_remaining_download_bytes": (
            sum(known_remaining) if len(known_remaining) == len(remaining) else None
        ),
        "aggtrades_remaining_compressed_bytes": primary_bytes,
        "estimated_aggtrades_streamed_csv_bytes": int(
            Decimal(primary_bytes) * EXPANSION_RATIO
        ),
        "free_disk_bytes": free,
        "disk_space_sufficient": free > sum(known_remaining) * 2 + 1_000_000_000,
        "holdout_intersection": "NONE",
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_csv(path: Path, rows: tuple[dict, ...], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(_json_value(row))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_plan_reports(plan: AcquisitionPlan) -> Path:
    output = REPORT_ROOT / plan.dataset_id
    summary = plan_summary(plan)
    _write_json(
        output / "acquisition_plan.json",
        {
            "generated_at": datetime.now(timezone.utc),
            "summary": summary,
            "windows": [asdict(item) for item in plan.coverage.windows],
            "safe_ranges": plan.coverage.safe_ranges,
            "archives": [asdict(item) for item in plan.archives],
        },
    )
    archive_rows = tuple(asdict(item) for item in plan.archives)
    _write_csv(
        output / "archive_inventory.csv",
        archive_rows,
        tuple(archive_rows[0]),
    )
    coverage_rows = tuple(asdict(item) for item in plan.coverage.windows)
    _write_csv(output / "coverage.csv", coverage_rows, tuple(coverage_rows[0]))
    return output


def _location(item: ArchivePlanItem) -> ArchiveLocation:
    return ArchiveLocation(
        url=item.url,
        checksum_url=item.checksum_url,
        destination=Path(item.local_path),
        checksum_destination=Path(item.checksum_local_path),
    )


def acquire_archives(plan: AcquisitionPlan) -> tuple[int, int]:
    downloader = AggregateTradeArchiveDownloader(timeout_seconds=60)
    downloaded = reused = 0
    for index, item in enumerate(plan.archives, start=1):
        print(
            f"ARCHIVE {index}/{len(plan.archives)} "
            f"{item.source_dataset} {item.granularity} {item.period}",
            flush=True,
        )
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                result = downloader.download(_location(item), verify_checksum=True)
                reused += int(result.skipped_existing)
                downloaded += int(not result.skipped_existing)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    print("Bounded retry 1/1 after archive failure", flush=True)
        if last_error is not None:
            raise RuntimeError(f"Archive acquisition failed: {item.url}") from last_error
    return downloaded, reused


def _local_candles(windows: tuple) -> tuple:
    by_time = {}
    for window in windows:
        for candle in window.replay_dataset.candles:
            previous = by_time.get(candle.timestamp)
            if previous is not None and previous != candle:
                raise ValueError("Conflicting local candles in V7 coverage.")
            by_time[candle.timestamp] = candle
    return tuple(by_time[timestamp] for timestamp in sorted(by_time))


def build_dataset(plan: AcquisitionPlan, windows: tuple, *, downloaded: int, reused: int) -> dict:
    required = frozenset(plan.coverage.required_timestamps)
    agg_items = tuple(
        item for item in plan.archives if item.source_dataset == "aggTrades"
    )
    counters = {
        "archive_raw_rows": 0,
        "required_raw_rows": 0,
        "required_underlying_trades": 0,
    }

    def required_trades():
        previous_id = None
        previous_timestamp = None
        for item in agg_items:
            for trade in iter_aggtrade_archive(item.local_path):
                counters["archive_raw_rows"] += 1
                if previous_timestamp is not None and trade.timestamp < previous_timestamp:
                    raise ValueError("Cross-archive timestamp ordering failed.")
                if previous_id is not None and trade.aggregate_trade_id <= previous_id:
                    raise ValueError("Cross-archive duplicate/non-increasing aggTrade ID.")
                previous_id = trade.aggregate_trade_id
                previous_timestamp = trade.timestamp
                if bucket_open_15m(trade.timestamp) in required:
                    counters["required_raw_rows"] += 1
                    counters["required_underlying_trades"] += trade.underlying_trade_count
                    yield trade

    buckets = tuple(iter_aggregate_15m(required_trades()))
    bucket_times = tuple(item.bucket_open_time for item in buckets)
    missing = tuple(sorted(required.difference(bucket_times)))
    extra = tuple(sorted(set(bucket_times).difference(required)))
    duplicates = len(bucket_times) - len(set(bucket_times))
    if missing or extra or duplicates:
        raise ValueError(
            f"Order-flow coverage failed: missing={len(missing)}, "
            f"extra={len(extra)}, duplicates={duplicates}."
        )

    kline_by_time = {}
    for item in plan.archives:
        if item.source_dataset != "klines_15m_reference":
            continue
        for kline in parse_kline_archive(item.local_path):
            if kline.open_time not in required:
                continue
            if kline.open_time in kline_by_time:
                raise ValueError("Duplicate kline reference timestamp.")
            kline_by_time[kline.open_time] = kline
    klines = tuple(kline_by_time[timestamp] for timestamp in sorted(kline_by_time))
    local_candles = _local_candles(windows)
    result = reconcile_orderflow_dataset(
        buckets, klines, local_candles, include_bucket_rows=False
    )
    if result.classification != "VALIDATED":
        raise ValueError(
            "Complete kline reconciliation failed: "
            + ", ".join(result.discrepancy_categories)
        )

    partitions: dict[tuple[int, int], list] = defaultdict(list)
    for bucket in buckets:
        partitions[(bucket.bucket_open_time.year, bucket.bucket_open_time.month)].append(
            bucket
        )
    store = OrderFlowBucketStore(AGGREGATED_ROOT)
    partition_paths = tuple(
        store.save_partition(values, year=year, month=month)
        for (year, month), values in sorted(partitions.items())
    )
    checksum_inventory = tuple(
        {
            "source_dataset": item.source_dataset,
            "period": item.period,
            "granularity": item.granularity,
            "archive_url": item.url,
            "sha256": parse_checksum(
                Path(item.checksum_local_path).read_text(encoding="utf-8"),
                expected_filename=Path(item.local_path).name,
            ),
        }
        for item in plan.archives
    )
    stable = {
        "dataset_version": "V7_ORDERFLOW_CONSUMED_1",
        "dataset_id": plan.dataset_id,
        "symbol": "BTCUSDC",
        "interval": "15m",
        "source": "BINANCE_PUBLIC_SPOT_AGGTRADES",
        "window_ids": [item.window_id for item in plan.coverage.windows],
        "safe_ranges": plan.coverage.safe_ranges,
        "archive_checksums": checksum_inventory,
        "required_raw_rows": counters["required_raw_rows"],
        "aggregated_buckets": len(buckets),
        "reconciliation": {
            key: value
            for key, value in asdict(result).items()
            if key != "bucket_rows"
        },
        "coverage": {"missing": 0, "extra": 0, "duplicates": 0},
    }
    dataset_sha256 = deterministic_manifest_sha256(_json_value(stable))
    manifest = {
        **stable,
        "generated_at": datetime.now(timezone.utc),
        "dataset_definition_sha256": dataset_sha256,
        "archive_raw_rows": counters["archive_raw_rows"],
        "required_underlying_trade_count": counters["required_underlying_trades"],
        "partition_paths": partition_paths,
        "acquisition": {"downloaded_archives": downloaded, "reused_archives": reused},
        "blind_holdout": {
            "status": "LOCKED_BLIND_HOLDOUT",
            "downloaded": False,
            "loaded": False,
            "revealed": False,
            "consumed": False,
            "evaluated": False,
            "intersection": "NONE",
        },
        "classification": "DATASET_READY",
    }
    manifest_path = AGGREGATED_ROOT / "BTCUSDC" / "dataset_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("V7 final dataset manifest exists; refusing overwrite.")
    _write_json(manifest_path, manifest)
    output = REPORT_ROOT / plan.dataset_id
    _write_json(output / "reconciliation_summary.json", stable["reconciliation"])
    _write_csv(
        output / "reconciliation_mismatches.csv",
        (),
        ("bucket_open_time", "field", "actual", "reference", "difference"),
    )
    _write_json(output / "summary.json", manifest)
    return manifest


def _print_plan(summary: dict) -> None:
    print(f"Dataset ID: {summary['dataset_id']}")
    print(f"Coverage timestamps: {summary['required_15m_timestamps']}")
    print(
        f"AggTrades plan: {summary['required_complete_months']} monthly + "
        f"{summary['required_daily_archives']} daily = "
        f"{summary['aggtrades_archive_count']} archives"
    )
    print(f"Kline reference archives: {summary['kline_reference_archive_count']}")
    print(f"Remaining download bytes: {summary['estimated_remaining_download_bytes']}")
    print(f"Estimated streamed aggTrade CSV bytes: {summary['estimated_aggtrades_streamed_csv_bytes']}")
    print(f"Free disk bytes: {summary['free_disk_bytes']}")
    print("Holdout intersection: NONE")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    _, windows, coverage = authoritative_context()
    plan = inventory_plan(build_acquisition_plan(coverage), query_sizes=True)
    output = write_plan_reports(plan)
    summary = plan_summary(plan)
    _print_plan(summary)
    print(f"Plan report: {output / 'acquisition_plan.json'}")
    if args.plan:
        print("PLAN COMPLETE — NO DATASET BUILD EXECUTED")
        return 0
    if not summary["disk_space_sufficient"]:
        print("DATASET_INCOMPLETE — insufficient disk before acquisition")
        return 1
    manifest_path = AGGREGATED_ROOT / "BTCUSDC" / "dataset_manifest.json"
    if manifest_path.exists():
        print("DATASET_INCOMPLETE — final manifest already exists")
        return 1
    downloaded, reused = acquire_archives(plan)
    manifest = build_dataset(plan, windows, downloaded=downloaded, reused=reused)
    print(f"Raw required aggTrades: {manifest['required_raw_rows']}")
    print(f"15m buckets: {manifest['aggregated_buckets']}")
    print(f"Dataset manifest: {manifest_path}")
    print("DATASET_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
