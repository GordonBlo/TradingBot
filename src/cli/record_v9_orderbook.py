"""Record time-bounded public BTCUSDC Spot L2 depth without authentication."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from src.orderbook.recorder import DepthSynchronizer, V9DepthRecorder, session_paths
from src.orderbook.storage import RawDepthEventStore


UTC = timezone.utc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    duration = parser.add_mutually_exclusive_group(required=True)
    duration.add_argument(
        "--validate-seconds", type=int,
        help="Safe validation mode; must be between 5 and 300 seconds.",
    )
    duration.add_argument(
        "--duration-seconds", type=int,
        help="Explicit bounded recording duration; maximum 86400 seconds.",
    )
    parser.add_argument("--output-root", default="data/orderbook/v9/raw")
    parser.add_argument("--snapshot-limit", type=int, choices=(100, 500, 1000, 5000), default=1000)
    parser.add_argument("--max-levels", type=int, default=5000)
    return parser


async def _run(args: argparse.Namespace) -> tuple[dict, Path, Path]:
    duration = args.validate_seconds if args.validate_seconds is not None else args.duration_seconds
    if args.validate_seconds is not None and not 5 <= duration <= 300:
        raise ValueError("Validation duration must be between 5 and 300 seconds.")
    if not 0 < duration <= 86_400:
        raise ValueError("Recording duration must be between 1 and 86400 seconds.")
    started = datetime.now(UTC)
    session_id, raw_path, summary_path = session_paths(args.output_root, started_at=started)
    with RawDepthEventStore(raw_path, session_id=session_id) as store:
        synchronizer = DepthSynchronizer(store, max_levels_per_side=args.max_levels)
        recorder = V9DepthRecorder(
            synchronizer=synchronizer, snapshot_limit=args.snapshot_limit
        )
        counters = await recorder.run(duration_seconds=duration)
        validation_passed = bool(
            counters.snapshots >= 1
            and counters.diff_events >= 1
            and counters.reconstructed_updates >= 1
            and synchronizer.book.last_update_id is not None
            and synchronizer.book.is_valid
        )
        payload = {
            "version": "V9_DEPTH_FOUNDATION_1",
            "session_id": session_id,
            "symbol": "BTCUSDC",
            "source": "BINANCE_PUBLIC_SPOT",
            "rest_snapshot": "PUBLIC_API_V3_DEPTH",
            "websocket_stream": "BTCUSDC_DIFF_DEPTH_100MS",
            "uses_authentication": False,
            "orders_enabled": False,
            "duration_seconds": duration,
            "started_at_utc": started.isoformat(),
            "ended_at_utc": datetime.now(UTC).isoformat(),
            "raw_event_log": str(raw_path),
            "integrity": counters.as_dict(),
            "validation_status": (
                "PASSED" if validation_passed else "INSUFFICIENT_VALID_DEPTH_DATA"
            ),
            "collector_left_running": False,
        }
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload, raw_path, summary_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload, raw_path, summary_path = asyncio.run(_run(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"V9 DEPTH RECORDING FAILED: {exc}")
        return 1
    print(json.dumps(payload["integrity"], indent=2, sort_keys=True))
    print(f"RAW: {raw_path}")
    print(f"SUMMARY: {summary_path}")
    print("V9 DEPTH RECORDING COMPLETE - COLLECTOR STOPPED")
    return 0 if payload["validation_status"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
