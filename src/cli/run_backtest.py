"""Run the clearly labeled V2 buy-and-hold accounting benchmark offline."""

from __future__ import annotations

import argparse

from src.backtest.benchmark import BuyAndHoldBenchmark
from src.backtest.engine import BacktestEngine
from src.backtest.models import AmbiguousBarPolicy, BacktestConfig
from src.backtest.report import BacktestReportWriter
from src.cli.common import parse_utc_datetime
from src.config.settings import load_settings
from src.historical.storage import HistoricalDatasetStore
from src.historical.validator import DatasetValidator
from src.utils.logger import configure_logging, get_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an OFFLINE BUY_AND_HOLD_BENCHMARK for V2 accounting validation. "
            "This is not a trading strategy."
        )
    )
    parser.add_argument("--symbol", default="BTCUSDC")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--start", help="Optional inclusive UTC ISO date/time")
    parser.add_argument("--end", help="Optional exclusive UTC ISO date/time")
    parser.add_argument("--data-root", default="data/historical")
    parser.add_argument("--reports-root", default="reports/backtests")
    parser.add_argument(
        "--mode",
        choices=("buy-and-hold-benchmark",),
        default="buy-and-hold-benchmark",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    logger = get_logger(__name__)
    try:
        if bool(args.start) != bool(args.end):
            raise ValueError("--start and --end must be supplied together.")
        settings = load_settings()
        validator = DatasetValidator()
        store = HistoricalDatasetStore(args.data_root, validator=validator)
        dataset = store.load(args.symbol, args.interval)
        if dataset is None:
            raise ValueError("No cached dataset exists; run download_history first.")
        if args.start and args.end:
            dataset = dataset.slice(
                parse_utc_datetime(args.start), parse_utc_datetime(args.end)
            )
        validator.validate(
            dataset.candles, symbol=dataset.symbol, interval=dataset.interval
        )

        config = BacktestConfig(
            initial_capital_usdc=settings.backtest_initial_capital_usdc,
            fee_bps=settings.backtest_fee_bps,
            slippage_bps=settings.backtest_slippage_bps,
            ambiguous_bar_policy=AmbiguousBarPolicy(
                settings.backtest_ambiguous_bar_policy
            ),
        )
        logger.info("V2 OFFLINE benchmark starting; no Binance client is initialized")
        logger.info(
            "Mode: BUY_AND_HOLD_BENCHMARK (accounting baseline, not a strategy)"
        )
        result = BacktestEngine(config, validator=validator).run(
            dataset,
            BuyAndHoldBenchmark(config.initial_capital_usdc, config.fee_bps),
        )
        paths = BacktestReportWriter(args.reports_root).write(result)
        metrics = result.metrics
        logger.info(
            "Completed %d candles and %d simulated trade(s)",
            len(dataset.candles),
            metrics.total_trades,
        )
        logger.info(
            "Final equity: %s | Net PnL: %s | Return: %s%% | Max drawdown: %s%%",
            metrics.final_equity,
            metrics.net_profit,
            metrics.total_return_percent,
            metrics.maximum_drawdown_percent,
        )
        logger.info("Total simulated fees: %s", metrics.total_fees_paid)
        logger.info("Report directory: %s", paths.directory.resolve())
        return 0
    except (RuntimeError, ValueError) as exc:
        logger.error("Backtest failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
