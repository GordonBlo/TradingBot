"""Download or incrementally update validated public Binance candle history."""

from __future__ import annotations

import argparse

from src.cli.common import parse_utc_datetime
from src.exchange.public_market_client import PublicMarketDataClient
from src.historical.downloader import DownloadProgress, HistoricalRangeDownloader
from src.historical.manager import HistoricalDatasetManager
from src.historical.storage import HistoricalDatasetStore
from src.historical.validator import DatasetValidator
from src.utils.logger import configure_logging, get_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download REAL Binance PUBLIC Spot candles into a validated CSV cache."
    )
    parser.add_argument("--symbol", default="BTCUSDC")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--start", required=True, help="Inclusive UTC ISO date/time")
    parser.add_argument("--end", required=True, help="Exclusive UTC ISO date/time")
    parser.add_argument("--data-root", default="data/historical")
    parser.add_argument(
        "--allow-source-gaps",
        action="store_true",
        help=(
            "Record real exchange-source gaps instead of synthesizing candles. "
            "Research must split replay regions at every recorded gap."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    logger = get_logger(__name__)
    try:
        start = parse_utc_datetime(args.start)
        end = parse_utc_datetime(args.end)
        validator = DatasetValidator()
        client = PublicMarketDataClient(args.symbol)
        downloader = HistoricalRangeDownloader(client)
        store = HistoricalDatasetStore(
            args.data_root,
            validator=validator,
            allow_source_gaps=args.allow_source_gaps,
        )
        manager = HistoricalDatasetManager(
            downloader,
            store,
            validator=validator,
            allow_source_gaps=args.allow_source_gaps,
        )

        logger.info(
            "Downloading/updating %s %s from %s to %s (end exclusive)",
            args.symbol.upper(),
            args.interval,
            start.isoformat(),
            end.isoformat(),
        )

        def progress(update: DownloadProgress) -> None:
            logger.info(
                "Downloaded %d candles through %s in %d request(s)",
                update.candle_count,
                update.last_timestamp.isoformat()
                if update.last_timestamp is not None
                else "N/A",
                update.request_count,
            )

        result = manager.update(
            args.symbol,
            args.interval,
            start,
            end,
            progress=progress,
        )
        validation = validator.validate(
            result.requested.candles,
            symbol=result.requested.symbol,
            interval=result.requested.interval,
            allow_gaps=args.allow_source_gaps,
        )
        logger.info("Validation: PASSED (%d candles)", validation.candle_count)
        if validation.gaps:
            logger.warning(
                "Recorded %d real source gap(s); no missing candle was synthesized.",
                len(validation.gaps),
            )
            for gap_start, gap_end in validation.gaps:
                logger.warning(
                    "Source gap: [%s, %s)",
                    gap_start.isoformat(),
                    gap_end.isoformat(),
                )
        logger.info(
            "Actual requested range: %s to %s",
            validation.first_timestamp.isoformat(),
            validation.last_timestamp.isoformat(),
        )
        logger.info(
            "Downloaded now: %d candles in %d request(s); cache total: %d",
            result.downloaded_candle_count,
            result.request_count,
            len(result.dataset.candles),
        )
        logger.info(
            "Saved candles: %s",
            store.candles_path(args.symbol, args.interval).resolve(),
        )
        logger.info(
            "Saved metadata: %s",
            store.metadata_path(args.symbol, args.interval).resolve(),
        )
        return 0
    except (RuntimeError, ValueError) as exc:
        logger.error("Historical download failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
