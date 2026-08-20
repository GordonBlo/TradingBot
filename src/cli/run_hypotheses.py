"""Run exactly H0-H4 offline on an existing, consumed V3 research dataset."""

from __future__ import annotations

import argparse

from src.config.settings import load_settings
from src.diagnostics.loader import ResearchRunLoader
from src.hypotheses.manifest import ResearchManifestStore
from src.hypotheses.report import HypothesisReportWriter
from src.hypotheses.runner import HypothesisResearchRunner
from src.utils.logger import configure_logging, get_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pre-registered V3.2 H0-H4 suite on CONSUMED_RESEARCH_DATA. "
            "Offline only; this is not fresh out-of-sample evidence."
        )
    )
    parser.add_argument("--research-run", required=True)
    parser.add_argument("--research-root", default="reports/research")
    parser.add_argument("--data-root", default="data/historical")
    parser.add_argument("--reports-root", default="reports/hypotheses")
    parser.add_argument("--manifest", default="research/hypothesis_manifest.json")
    parser.add_argument(
        "--cost-stress",
        action="store_true",
        help="Replay the same H0-H4 mechanisms with exactly 2x fees and slippage.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    logger = get_logger(__name__)
    try:
        settings = load_settings()
        source = ResearchRunLoader(
            research_root=args.research_root,
            data_root=args.data_root,
        ).load(args.research_run)
        suite = settings.hypothesis_suite_config()
        manifest_store = ResearchManifestStore(args.manifest)
        manifest = manifest_store.ensure_for_research_run(source, suite)

        logger.info("V3.2 HYPOTHESIS RESEARCH")
        logger.info("Dataset status: CONSUMED_RESEARCH_DATA")
        logger.warning(
            "These results are exploratory. The former OOS period was inspected "
            "and is NOT fresh out-of-sample evidence."
        )
        result = HypothesisResearchRunner(suite).run(
            source, cost_stress=args.cost_stress
        )
        manifest = manifest_store.mark_hypotheses_tested_on_consumed_data()
        paths = HypothesisReportWriter(args.reports_root).write(result, manifest)

        logger.info("ID  TRADES  FRICTIONLESS  NET  PF  EXPECTANCY  MAX DD%%")
        for item in result.experiments:
            metric = item.combined_metrics
            logger.info(
                "%s  %d  %s  %s  %s  %s  %s",
                item.hypothesis_id.value,
                metric.trades,
                metric.frictionless_pnl,
                metric.net_pnl,
                metric.profit_factor if metric.profit_factor is not None else "N/A",
                metric.expectancy,
                metric.maximum_drawdown_percent,
            )
            logger.info(
                "  %s | %s | shared/baseline-only/candidate-only=%d/%d/%d",
                item.classification.value,
                item.verification_status,
                item.alignment.shared_signals,
                item.alignment.baseline_only_signals,
                item.alignment.candidate_only_signals,
            )
            if item.delta_to_baseline is not None:
                delta = item.delta_to_baseline
                logger.info(
                    "  vs H0: delta net=%s frictionless=%s expectancy=%s PF=%s trades=%d",
                    delta.net_pnl,
                    delta.frictionless_pnl,
                    delta.expectancy,
                    delta.profit_factor if delta.profit_factor is not None else "N/A",
                    delta.trade_count,
                )
            for warning in item.warnings:
                logger.warning("  %s", warning)
            if item.stress_combined_metrics is not None:
                logger.info(
                    "  cost stress: base expectancy=%s | stress expectancy=%s | stress net=%s",
                    metric.expectancy,
                    item.stress_combined_metrics.expectancy,
                    item.stress_combined_metrics.net_pnl,
                )
        logger.info("Frozen H0 reproduction: VERIFIED")
        logger.info("Blind holdout registered: %s", "YES" if manifest.blind_holdout else "NO")
        logger.info("Blind holdout status: %s", manifest.holdout_status.value)
        logger.info("Report directory: %s", paths.directory.resolve())
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error("V3.2 hypothesis research failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
