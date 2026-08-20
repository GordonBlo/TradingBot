"""Explicitly register, but never execute, a future blind holdout."""

from __future__ import annotations

import argparse

from src.cli.common import parse_utc_datetime
from src.hypotheses.manifest import ResearchManifestStore
from src.utils.logger import configure_logging, get_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register one non-overlapping blind holdout without downloading or running it."
    )
    parser.add_argument("--symbol", default="BTCUSDC")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--manifest", default="research/hypothesis_manifest.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    logger = get_logger(__name__)
    try:
        manifest = ResearchManifestStore(args.manifest).register_holdout(
            symbol=args.symbol,
            interval=args.interval,
            start=parse_utc_datetime(args.start),
            end=parse_utc_datetime(args.end),
        )
        holdout = manifest.blind_holdout
        assert holdout is not None
        logger.info("Blind holdout registered without strategy execution or download.")
        logger.info("Range: [%s, %s) UTC", holdout.start, holdout.end)
        logger.info("Status: %s", holdout.status.value)
        return 0
    except (OSError, ValueError) as exc:
        logger.error("Holdout registration refused: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
