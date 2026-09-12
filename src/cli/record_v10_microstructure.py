"""Record bounded public BTCUSDC Spot L2 depth and aggTrades together."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from src.microstructure.v10 import collect_live_session


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    duration = parser.add_mutually_exclusive_group(required=True)
    duration.add_argument(
        "--validate-seconds",
        type=int,
        help="Short smoke mode; must be between 5 and 300 seconds.",
    )
    duration.add_argument(
        "--duration-seconds",
        type=int,
        help="Explicit bounded duration; maximum 86400 seconds.",
    )
    parser.add_argument("--output-root", default="data/microstructure/v10")
    parser.add_argument(
        "--snapshot-limit",
        type=int,
        choices=(100, 500, 1000, 5000),
        default=5000,
    )
    parser.add_argument("--max-levels", type=int, default=5000)
    return parser


def _duration(args: argparse.Namespace) -> int:
    value = (
        args.validate_seconds
        if args.validate_seconds is not None
        else args.duration_seconds
    )
    if args.validate_seconds is not None and not 5 <= value <= 300:
        raise ValueError("Validation duration must be between 5 and 300 seconds.")
    if not 0 < value <= 86_400:
        raise ValueError("Recording duration must be between 1 and 86400 seconds.")
    if args.max_levels < 1:
        raise ValueError("Maximum levels must be positive.")
    return value


async def _run(args: argparse.Namespace) -> tuple[dict[str, object], Path]:
    summary, paths = await collect_live_session(
        duration_seconds=_duration(args),
        output_root=args.output_root,
        snapshot_limit=args.snapshot_limit,
        max_levels_per_side=args.max_levels,
    )
    return summary, paths.directory


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary, session_directory = asyncio.run(_run(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"V10 PUBLIC MICROSTRUCTURE RECORDING FAILED: {exc}")
        return 1
    print(json.dumps(summary["live_integrity"], indent=2, sort_keys=True))
    print(json.dumps(summary["integrity_accounting"], indent=2, sort_keys=True))
    print(f"SESSION: {summary['session_id']}")
    print(f"ARTIFACTS: {session_directory}")
    print(f"CLASSIFICATION: {summary['collection_classification']}")
    print("PUBLIC DATA ONLY - NO AUTHENTICATION - NO ORDERS - NO PREDICTIVE EVALUATION")
    return 0 if summary["collection_classification"] == "CLOSED_INTEGRITY_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
