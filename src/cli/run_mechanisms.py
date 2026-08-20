"""Run preregistered V3.2.2 H0/H1/H5/H6/H7 on consumed data only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path

from src.cli.run_multiregime import (
    CONSUMED_END,
    CONSUMED_START,
    EXPANSION_END,
    INTERVAL,
    SYMBOL,
    _assert_holdout_absent,
    _load_exact_range,
    _load_expansion,
    _partition_metadata,
    _require_locked_holdout,
    _verify_previous_h0,
)
from src.config.settings import load_settings
from src.diagnostics.loader import ResearchRunLoader
from src.hypotheses.manifest import ResearchManifestStore
from src.hypotheses.mechanism_manifest import MechanismManifestStore
from src.hypotheses.mechanism_report import MechanismReportWriter
from src.hypotheses.mechanisms import MechanismHypothesisId
from src.research.multiregime.manifest import plain
from src.research.multiregime.mechanism_runner import MechanismResearchRunner
from src.research.multiregime.models import PartitionKind, ResearchRegion
from src.research.multiregime.report import MultiRegimeReportWriter
from src.utils.logger import configure_logging, get_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen V3.2.2 mechanisms on existing CONSUMED_RESEARCH_DATA. "
            "No downloader or exchange client is available in this command."
        )
    )
    parser.add_argument("--manifest", default="research/hypothesis_manifest.json")
    parser.add_argument(
        "--source-multiregime-run", default="3db30d3c8eacef4e"
    )
    parser.add_argument("--source-reports-root", default="reports/multiregime")
    parser.add_argument("--source-manifests-root", default="research/multiregime")
    parser.add_argument(
        "--expansion-data-root", default="data/historical/multiregime_expansion"
    )
    parser.add_argument("--consumed-data-root", default="data/historical")
    parser.add_argument("--research-root", default="reports/research")
    parser.add_argument("--research-run", default="adeef00722e9704d")
    parser.add_argument("--reports-root", default="reports/mechanisms")
    parser.add_argument("--preregistration-root", default="research/mechanisms")
    parser.add_argument(
        "--cost-stress",
        action="store_true",
        help="Also replay identical signals with exactly 2x execution costs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    logger = get_logger(__name__)
    try:
        if not args.cost_stress:
            raise ValueError(
                "V3.2.2 requires --cost-stress so both frozen base costs and the "
                "preregistered doubled-cost replay are evaluated."
            )
        settings = load_settings()
        root_store = ResearchManifestStore(args.manifest)
        root_manifest = root_store.load()
        _require_locked_holdout(root_manifest)
        source_record = next(
            (
                item
                for item in root_manifest.multiregime_runs
                if item["run_id"] == args.source_multiregime_run
            ),
            None,
        )
        if source_record is None:
            raise ValueError("The frozen V3.2.1 source run is not registered.")

        source_manifest_path = (
            Path(args.source_manifests_root)
            / args.source_multiregime_run
            / "manifest.json"
        )
        source_manifest_bytes = source_manifest_path.read_bytes()
        source_manifest = json.loads(source_manifest_bytes)
        if (
            source_manifest.get("configuration_sha256")
            != source_record["configuration_sha256"]
        ):
            raise ValueError("The registered V3.2.1 source manifest changed.")

        expansion_segments = _load_expansion(args.expansion_data_root)
        consumed = _load_exact_range(
            args.consumed_data_root, CONSUMED_START, CONSUMED_END
        )
        _assert_holdout_absent(*expansion_segments, consumed)
        partitions = _partition_metadata(
            expansion_segments=expansion_segments,
            consumed=consumed,
            expansion_data_root=args.expansion_data_root,
            consumed_data_root=args.consumed_data_root,
            holdout_registered_at=root_manifest.blind_holdout.registered_at,
        )
        if plain([asdict(item) for item in partitions]) != source_manifest["partitions"]:
            raise ValueError("V3.2.2 partitions differ from the frozen V3.2.1 run.")

        previous_loader = ResearchRunLoader(
            research_root=args.research_root,
            data_root=args.consumed_data_root,
        )
        previous = previous_loader.load(args.research_run)
        _verify_previous_h0(previous, previous_loader.resolve(args.research_run))

        source_report_directory = (
            Path(args.source_reports_root) / args.source_multiregime_run
        )
        source_stability = _rows_by_hypothesis(
            source_report_directory / "candidate_stability.csv"
        )
        source_windows = _control_window_rows(
            source_report_directory / "window_comparison.csv"
        )
        source_control_metrics = {
            hypothesis_id: {
                key: row[key]
                for key in (
                    "combined_trades",
                    "combined_frictionless_pnl",
                    "combined_net_pnl",
                    "combined_frictionless_expectancy",
                    "combined_net_expectancy",
                    "combined_profit_factor",
                    "stress_expectancy",
                )
            }
            for hypothesis_id, row in source_stability.items()
            if hypothesis_id in {"H0", "H1"}
        }

        mechanism_config = settings.mechanism_suite_config()
        v32_config = settings.hypothesis_suite_config()
        multiregime_config = settings.multiregime_config()
        prepared = MechanismManifestStore(args.preregistration_root).prepare(
            symbol=SYMBOL,
            interval=INTERVAL,
            source_multiregime_run_id=args.source_multiregime_run,
            source_multiregime_manifest_sha256=hashlib.sha256(
                source_manifest_bytes
            ).hexdigest(),
            source_control_metrics=source_control_metrics,
            partitions=partitions,
            multiregime_config=multiregime_config,
            v32_config=v32_config,
            mechanism_config=mechanism_config,
            strategy_config=previous.strategy_config,
            backtest_config=previous.backtest_config,
            cost_stress_enabled=args.cost_stress,
        )
        root_store.record_mechanism_run(
            run_id=prepared.run_id,
            manifest_path=str(prepared.path),
            configuration_sha256=prepared.configuration_sha256,
            source_multiregime_run_id=args.source_multiregime_run,
        )

        expansion_partitions = tuple(
            item for item in partitions if item.kind is PartitionKind.RESEARCH_EXPANSION
        )
        consumed_partition = next(
            item for item in partitions if item.kind is PartitionKind.CONSUMED_RESEARCH
        )
        regions = (
            *(
                ResearchRegion(partition, segment)
                for partition, segment in zip(
                    expansion_partitions, expansion_segments, strict=True
                )
            ),
            ResearchRegion(consumed_partition, consumed),
        )
        logger.info("V3.2.2 NEW MECHANISM HYPOTHESES (OFFLINE)")
        logger.info("Candidates: H0, H1, H5, H6, H7")
        logger.info("Data: CONSUMED_RESEARCH_DATA | source windows: %s", args.source_multiregime_run)
        logger.warning("Blind holdout: LOCKED | NOT LOADED | NOT EVALUATED")
        result = MechanismResearchRunner(
            multiregime_config=multiregime_config,
            v32_config=v32_config,
            mechanism_config=mechanism_config,
            strategy_config=previous.strategy_config,
            backtest_config=previous.backtest_config,
            minimum_trades_warning=previous.minimum_trades_warning,
        ).run(
            run_id=prepared.run_id,
            regions=regions,
            partitions=partitions,
            cost_stress=args.cost_stress,
            previous_h0_reproduction_verified=True,
            previous_h1_reproduction_verified=True,
        )
        _verify_frozen_control_windows(result, source_windows)
        _verify_frozen_control_stability(result, source_stability)
        _require_locked_holdout(root_store.load())
        paths = MechanismReportWriter(args.reports_root).write(result, prepared)

        logger.info("Windows: %d | eligible: %d", len(result.windows), sum(item.eligible for item in result.windows))
        logger.info("ID  TRADES  FRICTIONLESS EXP  NET EXP  PF  CLASSIFICATION")
        for item in result.stability:
            metric = item.combined_metrics
            logger.info(
                "%s  %d  %s  %s  %s  %s",
                item.hypothesis_id.value,
                metric.trades,
                metric.average_frictionless_pnl_per_trade,
                metric.expectancy,
                metric.profit_factor if metric.profit_factor is not None else "N/A",
                item.classification.value,
            )
            if item.hypothesis_id in {
                MechanismHypothesisId.H5,
                MechanismHypothesisId.H6,
                MechanismHypothesisId.H7,
            }:
                logger.info(
                    "  consistency vs H0: frictionless=%d/%d | net=%d/%d | "
                    "PF=%d/%d | trade ratio=%s",
                    item.frictionless_better_windows,
                    item.eligible_windows,
                    item.net_better_windows,
                    item.eligible_windows,
                    item.profit_factor_better_windows,
                    item.eligible_windows,
                    item.trade_count_ratio,
                )
        logger.info("Frozen controls: H0 VERIFIED | H1 VERIFIED")
        logger.info("Blind holdout: LOCKED | NOT REVEALED | NOT CONSUMED")
        logger.info("Report directory: %s", paths.base.directory.resolve())
        return 0
    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        logger.error("V3.2.2 mechanism research failed: %s", exc)
        return 1


def _rows_by_hypothesis(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {row["hypothesis"]: row for row in csv.DictReader(stream)}


def _control_window_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {
            (row["window_id"], row["hypothesis"]): row
            for row in csv.DictReader(stream)
            if row["hypothesis"] in {"H0", "H1"}
        }


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _verify_frozen_control_windows(result: object, expected: dict[tuple[str, str], dict[str, str]]) -> None:
    for window in result.windows:
        for candidate in window.candidates:
            if candidate.hypothesis_id not in {
                MechanismHypothesisId.H0,
                MechanismHypothesisId.H1,
            }:
                continue
            key = (window.window.window_id, candidate.hypothesis_id.value)
            row = MultiRegimeReportWriter._window_row(window, candidate)
            recorded = expected.get(key)
            mismatches = (
                {"row": ("missing", "missing")}
                if recorded is None
                else {
                    name: (_cell(value), recorded[name])
                    for name, value in row.items()
                    if _cell(value) != recorded[name]
                }
            )
            if mismatches:
                name, values = next(iter(mismatches.items()))
                raise ValueError(
                    f"Frozen {candidate.hypothesis_id.value} window reproduction failed "
                    f"at {key[0]} for {name}: actual={values[0]!r}, "
                    f"expected={values[1]!r}."
                )


def _verify_frozen_control_stability(result: object, expected: dict[str, dict[str, str]]) -> None:
    metric_fields = (
        "combined_trades",
        "combined_frictionless_pnl",
        "combined_slippage_drag",
        "combined_fees",
        "combined_net_pnl",
        "combined_frictionless_expectancy",
        "combined_net_expectancy",
        "combined_profit_factor",
        "combined_return_percent",
        "combined_maximum_drawdown_percent",
        "combined_win_rate_percent",
        "combined_payoff_ratio",
        "stress_expectancy",
        "frictionless_better_windows",
        "net_better_windows",
        "profit_factor_better_windows",
        "lower_drawdown_windows",
        "eligible_windows",
        "trade_count_ratio",
    )
    for item in result.stability:
        if item.hypothesis_id not in {
            MechanismHypothesisId.H0,
            MechanismHypothesisId.H1,
        }:
            continue
        row = MultiRegimeReportWriter._stability_row(item)
        recorded = expected[item.hypothesis_id.value]
        if any(_cell(row[name]) != recorded[name] for name in metric_fields):
            raise ValueError(
                f"Frozen {item.hypothesis_id.value} combined reproduction failed."
            )


if __name__ == "__main__":
    raise SystemExit(main())
