"""Validate one deterministic consumed-data day of public Binance order flow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from src.historical.storage import HistoricalDatasetStore
from src.orderflow.aggregation import aggregate_15m
from src.orderflow.archive import (
    AggregateTradeArchiveDownloader,
    build_archive_location,
    build_kline_archive_location,
)
from src.orderflow.models import AggressorSide
from src.orderflow.parser import parse_aggtrade_archive
from src.orderflow.reconciliation import (
    HOLDOUT_END,
    HOLDOUT_START,
    OrderFlowValidationError,
    parse_kline_archive,
    reconcile_orderflow_day,
    require_safe_day,
    select_earliest_complete_consumed_day,
)


SYMBOL = "BTCUSDC"
INTERVAL = "15m"
MANIFEST_PATH = Path("research/hypothesis_manifest.json")
EXPANSION_ROOT = Path("data/historical/multiregime_expansion")
REPORT_ROOT = Path("reports/orderflow_validation")


def _json_value(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _load_consumed_ranges(path: Path = MANIFEST_PATH) -> tuple[tuple[datetime, datetime], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    holdout = payload["blind_holdout"]
    if (
        payload["holdout_status"] != "LOCKED_BLIND_HOLDOUT"
        or datetime.fromisoformat(holdout["start"]) != HOLDOUT_START
        or datetime.fromisoformat(holdout["end"]) != HOLDOUT_END
        or holdout["status"] != "LOCKED_BLIND_HOLDOUT"
        or holdout["reveal_timestamp"] is not None
        or holdout["consumed_timestamp"] is not None
    ):
        raise OrderFlowValidationError("Blind holdout integrity failed.")
    ranges: list[tuple[datetime, datetime]] = []
    for item in payload["consumed_dataset_ranges"]:
        if (
            item["data_status"] == "CONSUMED_RESEARCH_DATA"
            and item["symbol"] == SYMBOL
            and item["interval"] == INTERVAL
        ):
            start = datetime.fromisoformat(item["start"])
            end = datetime.fromisoformat(item["end"])
            if start < HOLDOUT_END and HOLDOUT_START < end:
                raise OrderFlowValidationError("Consumed range overlaps blind holdout.")
            ranges.append((start, end))
    if not ranges:
        raise OrderFlowValidationError("No consumed research ranges are registered.")
    return tuple(ranges)


def _deterministic_id(day: date) -> str:
    definition = f"BINANCE_PUBLIC_SPOT|{SYMBOL}|{INTERVAL}|{day.isoformat()}|aggTrades-kline-v1"
    return hashlib.sha256(definition.encode("utf-8")).hexdigest()[:16]


def _write_reports(summary: dict, rows: tuple[dict[str, object], ...], run_id: str) -> tuple[Path, Path]:
    output = REPORT_ROOT / run_id
    summary_path = output / "summary.json"
    csv_path = output / "bucket_comparison.csv"
    if summary_path.exists() or csv_path.exists():
        raise FileExistsError("V7 validation report already exists; refusing overwrite.")
    output.mkdir(parents=True, exist_ok=True)
    summary_temp = summary_path.with_suffix(".json.tmp")
    csv_temp = csv_path.with_suffix(".csv.tmp")
    try:
        summary_temp.write_text(
            json.dumps(_json_value(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with csv_temp.open("w", encoding="utf-8", newline="") as stream:
            fieldnames = tuple(rows[0]) if rows else ()
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(_json_value(row))
        summary_temp.replace(summary_path)
        csv_temp.replace(csv_path)
    finally:
        for temporary in (summary_temp, csv_temp):
            if temporary.exists():
                temporary.unlink()
    return summary_path, csv_path


def execute_validation() -> tuple[dict, Path, Path]:
    consumed_ranges = _load_consumed_ranges()
    dataset = HistoricalDatasetStore(
        EXPANSION_ROOT, allow_source_gaps=True
    ).load(SYMBOL, INTERVAL)
    if dataset is None:
        raise OrderFlowValidationError("Consumed expansion candle cache is missing.")
    day = select_earliest_complete_consumed_day(dataset.candles, consumed_ranges)
    require_safe_day(day)
    print(f"Selected safe UTC date: {day.isoformat()}", flush=True)

    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    local_candles = tuple(
        candle for candle in dataset.candles if start <= candle.timestamp < end
    )
    agg_location = build_archive_location(day, cadence="daily")
    kline_location = build_kline_archive_location(day)
    downloader = AggregateTradeArchiveDownloader()
    agg_download = downloader.download(agg_location, verify_checksum=True)
    kline_download = downloader.download(kline_location, verify_checksum=True)

    trades = parse_aggtrade_archive(agg_download.path)
    if not trades or any(not start <= trade.timestamp < end for trade in trades):
        raise OrderFlowValidationError("aggTrades escape the selected UTC day.")
    buckets = aggregate_15m(trades)
    klines = parse_kline_archive(kline_download.path)
    if any(not start <= item.open_time < end for item in klines):
        raise OrderFlowValidationError("Klines escape the selected UTC day.")
    result = reconcile_orderflow_day(buckets, klines, local_candles)

    summary = {
        "classification": result.classification,
        "run_id": _deterministic_id(day),
        "market": {
            "source": "BINANCE_PUBLIC_SPOT",
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "selected_utc_date": day.isoformat(),
            "selection_rule": "EARLIEST_COMPLETE_UTC_DAY_IN_CONSUMED_RESEARCH_DATA",
        },
        "archives": {
            "aggtrades": {
                "url": agg_location.url,
                "path": str(agg_download.path),
                "checksum_verified": agg_download.checksum_verified,
                "skipped_existing_verified": agg_download.skipped_existing,
            },
            "kline_validation_reference": {
                "url": kline_location.url,
                "path": str(kline_download.path),
                "checksum_verified": kline_download.checksum_verified,
                "skipped_existing_verified": kline_download.skipped_existing,
            },
        },
        "raw_aggtrades": {
            "rows": len(trades),
            "first_timestamp": trades[0].timestamp,
            "last_timestamp": trades[-1].timestamp,
            "first_aggregate_trade_id": trades[0].aggregate_trade_id,
            "last_aggregate_trade_id": trades[-1].aggregate_trade_id,
            "total_underlying_trade_count": sum(
                item.underlying_trade_count for item in trades
            ),
            "taker_buy_aggtrade_count": sum(
                item.aggressor_side is AggressorSide.TAKER_BUY for item in trades
            ),
            "taker_sell_aggtrade_count": sum(
                item.aggressor_side is AggressorSide.TAKER_SELL for item in trades
            ),
        },
        "reconciliation": {
            key: value
            for key, value in asdict(result).items()
            if key != "bucket_rows"
        },
        "data_policy": {
            "status": "CONSUMED_RESEARCH_DATA",
            "downloaded_utc_days": 1,
            "blind_holdout": "LOCKED_NOT_LOADED_NOT_REVEALED_NOT_CONSUMED_NOT_EVALUATED",
            "strategy_or_backtest_executed": False,
        },
    }
    summary_path, csv_path = _write_reports(
        summary, result.bucket_rows, _deterministic_id(day)
    )
    return summary, summary_path, csv_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate exactly one deterministic safe day of V7 order flow."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Acquire and validate the single deterministic consumed-data day.",
    )
    args = parser.parse_args(argv)
    if not args.execute:
        print("NO VALIDATION EXECUTED — pass --execute for the one-day integrity sample")
        return 0
    try:
        summary, summary_path, csv_path = execute_validation()
    except Exception as exc:
        print(f"VALIDATION_FAILED — {exc}")
        return 1
    print(f"aggTrade rows: {summary['raw_aggtrades']['rows']}")
    print(f"15m buckets: {summary['reconciliation']['bucket_count']}")
    print(f"Report: {summary_path}")
    print(f"Bucket comparison: {csv_path}")
    print(summary["classification"])
    return 0 if summary["classification"] == "VALIDATED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
