"""Run the V3 baseline research workflow entirely from local historical data."""

from __future__ import annotations

import argparse

from src.backtest.models import AmbiguousBarPolicy, BacktestConfig
from src.cli.common import parse_utc_datetime
from src.config.settings import load_settings
from src.historical.storage import HistoricalDatasetStore
from src.historical.validator import DatasetValidator
from src.research.dataset_split import ResearchPeriod, ResearchPeriodRange
from src.research.report import ResearchReportWriter
from src.research.runner import ResearchConfig, StrategyResearchRunner
from src.utils.logger import configure_logging, get_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed V3 TrendMomentumBaselineStrategy on independent "
            "development, validation, and out-of-sample periods. Offline only."
        )
    )
    parser.add_argument("--symbol", default="BTCUSDC")
    parser.add_argument("--interval", default="15m")
    parser.add_argument("--development-start", required=True)
    parser.add_argument("--development-end", required=True)
    parser.add_argument("--validation-start", required=True)
    parser.add_argument("--validation-end", required=True)
    parser.add_argument("--oos-start", required=True)
    parser.add_argument("--oos-end", required=True)
    parser.add_argument("--data-root", default="data/historical")
    parser.add_argument("--reports-root", default="reports/research")
    parser.add_argument(
        "--cost-stress",
        action="store_true",
        help="Rerun the same frozen strategy with 2x fee and slippage.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    logger = get_logger(__name__)
    try:
        settings = load_settings()
        ranges = (
            ResearchPeriodRange(
                ResearchPeriod.DEVELOPMENT,
                parse_utc_datetime(args.development_start),
                parse_utc_datetime(args.development_end),
            ),
            ResearchPeriodRange(
                ResearchPeriod.VALIDATION,
                parse_utc_datetime(args.validation_start),
                parse_utc_datetime(args.validation_end),
            ),
            ResearchPeriodRange(
                ResearchPeriod.OUT_OF_SAMPLE,
                parse_utc_datetime(args.oos_start),
                parse_utc_datetime(args.oos_end),
            ),
        )
        validator = DatasetValidator()
        dataset = HistoricalDatasetStore(
            args.data_root, validator=validator
        ).load(args.symbol, args.interval)
        if dataset is None:
            raise ValueError(
                "No cached dataset exists. Download it first with: "
                f".\\.venv\\Scripts\\python.exe -m src.cli.download_history "
                f"--symbol {args.symbol} --interval {args.interval} "
                f"--start {args.development_start} --end {args.oos_end}"
            )
        backtest = BacktestConfig(
            initial_capital_usdc=settings.backtest_initial_capital_usdc,
            fee_bps=settings.backtest_fee_bps,
            slippage_bps=settings.backtest_slippage_bps,
            ambiguous_bar_policy=AmbiguousBarPolicy(
                settings.backtest_ambiguous_bar_policy
            ),
        )
        config = ResearchConfig(
            development=ranges[0],
            validation=ranges[1],
            out_of_sample=ranges[2],
            backtest=backtest,
            strategy=settings.trend_momentum_config(),
            minimum_trades_warning=settings.research_min_trades_warning,
            cost_stress=args.cost_stress,
        )

        logger.info("V3 Strategy Research starting (offline; no Binance client)")
        logger.info("Strategy: TrendMomentumBaselineStrategy 3.0")
        logger.info("Symbol: %s | Interval: %s", args.symbol, args.interval)
        logger.info(
            "Development: [%s, %s)", ranges[0].start, ranges[0].end
        )
        logger.info("Validation: [%s, %s)", ranges[1].start, ranges[1].end)
        logger.info("Out-of-sample: [%s, %s)", ranges[2].start, ranges[2].end)
        logger.info(
            "Initial capital: %s | Fee: %s bps | Slippage: %s bps",
            backtest.initial_capital_usdc,
            backtest.fee_bps,
            backtest.slippage_bps,
        )

        result = StrategyResearchRunner(validator=validator).run(dataset, config)
        paths = ResearchReportWriter(args.reports_root).write(result)
        logger.info("Research complete")
        logger.info("PERIOD          TRADES   RETURN%%      WIN RATE%%    PF        MAX DD%%")
        for row in result.comparison:
            warning = " LOW SAMPLE SIZE" if row.low_sample_size else ""
            logger.info(
                "%-15s %-8d %-12s %-12s %-9s %s%s",
                row.period.value,
                row.trades,
                row.return_percent,
                row.win_rate_percent,
                row.profit_factor if row.profit_factor is not None else "N/A",
                row.maximum_drawdown_percent,
                warning,
            )
            logger.info(
                "  benchmark return=%s%% benchmark max DD=%s%%",
                row.benchmark_return_percent,
                row.benchmark_maximum_drawdown_percent,
            )
        logger.info("Report directory: %s", paths.directory.resolve())
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("Research failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

