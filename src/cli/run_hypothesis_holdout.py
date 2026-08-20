"""Explicit, irreversible V3.2 blind-holdout reveal workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from src.backtest.models import AmbiguousBarPolicy, BacktestConfig
from src.backtest.report import BacktestReportWriter
from src.config.settings import load_settings
from src.historical.storage import HistoricalDatasetStore
from src.historical.validator import DatasetValidator
from src.hypotheses.candidates import build_candidate
from src.hypotheses.manifest import HoldoutStatus, ResearchManifestStore
from src.hypotheses.models import HypothesisId
from src.research.evaluation import evaluate_strategy_period
from src.market.intervals import next_open_time
from src.utils.logger import configure_logging, get_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Irreversibly reveal and consume a registered V3.2 blind holdout."
    )
    parser.add_argument("--manifest", default="research/hypothesis_manifest.json")
    parser.add_argument("--data-root", default="data/historical")
    parser.add_argument("--reports-root", default="reports/hypotheses")
    parser.add_argument(
        "--confirm-reveal",
        action="store_true",
        help="Required acknowledgement that viewing results destroys blind status.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    logger = get_logger(__name__)
    try:
        if not args.confirm_reveal:
            raise ValueError(
                "Explicit --confirm-reveal is required; no holdout was revealed."
            )
        settings = load_settings()
        store = ResearchManifestStore(args.manifest)
        manifest = store.load()
        holdout = manifest.blind_holdout
        if holdout is None:
            raise ValueError("No blind holdout is registered.")
        if holdout.status is HoldoutStatus.CONSUMED:
            raise ValueError("This holdout is already CONSUMED and cannot be blind again.")
        if holdout.status not in (
            HoldoutStatus.LOCKED_BLIND_HOLDOUT,
            HoldoutStatus.REVEALED,
        ):
            raise ValueError(f"Holdout status cannot be evaluated: {holdout.status.value}")

        dataset = HistoricalDatasetStore(args.data_root).load(
            holdout.symbol, holdout.interval
        )
        download_command = (
            ".\\.venv\\Scripts\\python.exe -m src.cli.download_history "
            f"--symbol {holdout.symbol} --interval {holdout.interval} "
            f"--start {holdout.start.isoformat()} --end {holdout.end.isoformat()}"
        )
        if dataset is None:
            raise ValueError(f"Holdout candles are missing. Download explicitly with: {download_command}")
        selected = dataset.slice(holdout.start, holdout.end)
        if not selected.candles:
            raise ValueError(f"Holdout candles are missing. Download explicitly with: {download_command}")
        actual_end = next_open_time(selected.candles[-1].timestamp, selected.interval)
        if selected.candles[0].timestamp != holdout.start or actual_end != holdout.end:
            raise ValueError(f"Holdout cache coverage is incomplete. Download explicitly with: {download_command}")
        DatasetValidator().validate(
            selected.candles, symbol=selected.symbol, interval=selected.interval
        )

        if holdout.status is HoldoutStatus.LOCKED_BLIND_HOLDOUT:
            manifest = store.reveal()
        backtest_config = BacktestConfig(
            initial_capital_usdc=settings.backtest_initial_capital_usdc,
            fee_bps=settings.backtest_fee_bps,
            slippage_bps=settings.backtest_slippage_bps,
            ambiguous_bar_policy=AmbiguousBarPolicy(
                settings.backtest_ambiguous_bar_policy
            ),
        )
        suite = settings.hypothesis_suite_config()
        fingerprint = hashlib.sha256(
            f"{holdout.symbol}|{holdout.interval}|{holdout.start}|{holdout.end}".encode()
        ).hexdigest()[:16]
        directory = Path(args.reports_root) / f"holdout_{fingerprint}"
        rows = {}
        for hypothesis_id in HypothesisId:
            evaluation = evaluate_strategy_period(
                selected,
                strategy=build_candidate(hypothesis_id, settings.trend_momentum_config(), suite),
                strategy_config=settings.trend_momentum_config(),
                backtest_config=backtest_config,
            )
            BacktestReportWriter(directory / hypothesis_id.value).write(
                evaluation.backtest, run_id="revealed"
            )
            metrics = evaluation.backtest.metrics
            rows[hypothesis_id.value] = {
                "trades": metrics.total_trades,
                "net_pnl": str(metrics.net_profit),
                "return_percent": str(metrics.total_return_percent),
                "win_rate_percent": str(metrics.win_rate_percent),
                "profit_factor": (
                    str(metrics.profit_factor) if metrics.profit_factor is not None else None
                ),
                "expectancy": str(metrics.expectancy_per_trade),
            }
            logger.info(
                "%s holdout: trades=%d net=%s expectancy=%s",
                hypothesis_id.value,
                metrics.total_trades,
                metrics.net_profit,
                metrics.expectancy_per_trade,
            )
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "holdout_summary.json").write_text(
            json.dumps(
                {
                    "data_status": "REVEALED_THEN_CONSUMED",
                    "range": asdict(holdout),
                    "results": rows,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = store.consume()
        logger.warning("Blind holdout results were viewed; status is permanently CONSUMED.")
        logger.info("Report directory: %s", directory.resolve())
        logger.info("Manifest status: %s", manifest.holdout_status.value)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("Holdout reveal refused/failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
