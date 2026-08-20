"""Run V3.1 diagnostics from an existing frozen V3 research report."""

from __future__ import annotations

import argparse

from src.diagnostics.analyzer import StrategyDiagnosticsAnalyzer
from src.diagnostics.loader import ResearchRunLoader
from src.diagnostics.report import DiagnosticsReportWriter
from src.utils.logger import configure_logging, get_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze an existing V3 research run offline. No strategy replay, "
            "parameter search, network request, or Binance client is used."
        )
    )
    parser.add_argument("--research-run", required=True, help="Research run ID or path")
    parser.add_argument("--research-root", default="reports/research")
    parser.add_argument("--data-root", default="data/historical")
    parser.add_argument("--reports-root", default="reports/diagnostics")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    logger = get_logger(__name__)
    try:
        run = ResearchRunLoader(
            research_root=args.research_root, data_root=args.data_root
        ).load(args.research_run)
        report = StrategyDiagnosticsAnalyzer().analyze(run)
        paths = DiagnosticsReportWriter(args.reports_root).write(report)
        logger.info("V3.1 STRATEGY DIAGNOSTICS (offline post-trade analysis)")
        logger.info("Source research run: %s", report.research_run_id)
        for period in report.periods:
            logger.info("PERIOD: %s | Trades: %d", period.period.value, period.total_trades)
            logger.info(
                "Frictionless: %s | Gross after slippage: %s | Slippage: %s | Fees: -%s | Net: %s",
                period.costs.frictionless_pnl,
                period.costs.slippage_adjusted_gross_pnl,
                period.costs.slippage_drag,
                period.costs.fee_drag,
                period.costs.net_pnl,
            )
            logger.info(
                "Win rate: %s%% | PF: %s | Expectancy: %s | Avg win/loss: %s / %s",
                period.win_rate_percent,
                period.profit_factor if period.profit_factor is not None else "N/A",
                period.expectancy,
                period.outcomes.average_win,
                period.outcomes.average_loss,
            )
            logger.info(
                "Median MFE/MAE: %sR / %sR | Average hold: %s bars | Evidence: %s",
                period.excursions.median_mfe_r,
                period.excursions.median_mae_r,
                period.holding.average_bars,
                period.evidence_strength.value,
            )
            exit_text = " | ".join(
                f"{row.exit_reason.value}: {row.percentage_of_trades}%"
                for row in period.exit_reasons
            )
            logger.info("Exits: %s", exit_text)
        logger.info("DIAGNOSTIC QUESTIONS")
        for period in report.periods:
            cost_answer = (
                "positive pre-cost movement was reversed by costs"
                if period.costs.frictionless_pnl > 0 and period.costs.net_pnl < 0
                else "losses existed before costs"
                if period.costs.frictionless_pnl < 0
                else "costs did not reverse the PnL sign"
            )
            reached = {row.group: row for row in period.excursions.mfe_thresholds}
            logger.info(
                "%s: %s; reached 1R=%s%%, 2R=%s%%; early loser MAE>=0.5R=%s%%",
                period.period.value,
                cost_answer,
                reached[">=1R"].percentage_of_trades,
                reached[">=2R"].percentage_of_trades,
                period.excursions.losing_mae_half_r_first_four_percent,
            )
        logger.info("Diagnostic report: %s", paths.directory.resolve())
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("Diagnostics failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

