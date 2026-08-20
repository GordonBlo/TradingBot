"""Audit H5 capital sizing and actual stop risk on frozen consumed research data."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

from src.cli.run_multiregime import (
    CONSUMED_END,
    CONSUMED_START,
    _assert_holdout_absent,
    _load_exact_range,
    _load_expansion,
    _partition_metadata,
    _require_locked_holdout,
)
from src.config.settings import load_settings
from src.diagnostics.loader import ResearchRunLoader
from src.diagnostics.risk_capital_audit import (
    build_risk_capital_audit,
    summarize_risk_capital_audit,
)
from src.diagnostics.r_normalized import (
    build_r_normalized_trades,
    summarize_r_normalized_trades,
)
from src.hypotheses.manifest import ResearchManifestStore
from src.hypotheses.mechanisms import (
    MechanismHypothesisId,
    build_mechanism_candidate,
)
from src.research.evaluation import evaluate_strategy_period
from src.research.multiregime.models import PartitionKind, ResearchRegion
from src.research.multiregime.windows import construct_windows
from src.utils.logger import configure_logging, get_logger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline H5 risk/capital audit on consumed research data."
    )
    parser.add_argument(
        "--manifest",
        default="research/hypothesis_manifest.json",
    )
    parser.add_argument(
        "--mechanism-run",
        default="bc2496aed05555b5",
    )
    parser.add_argument(
        "--mechanism-reports-root",
        default="reports/mechanisms",
    )
    parser.add_argument(
        "--expansion-data-root",
        default="data/historical/multiregime_expansion",
    )
    parser.add_argument(
        "--consumed-data-root",
        default="data/historical",
    )
    parser.add_argument(
        "--research-root",
        default="reports/research",
    )
    parser.add_argument(
        "--research-run",
        default="adeef00722e9704d",
    )
    parser.add_argument(
        "--output-root",
        default="reports/audits/h5_risk_capital_v322",
    )
    return parser


def _read_expected_windows(
    report_directory: Path,
) -> dict[str, int]:
    path = report_directory / "window_comparison.csv"

    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = tuple(csv.DictReader(stream))

    eligible_windows = {
        row["window_id"]
        for row in rows
        if row["hypothesis"] == "H0"
        and row["eligible"].strip().lower() == "true"
    }

    expected_h5 = {
        row["window_id"]: int(row["trades"])
        for row in rows
        if row["hypothesis"] == "H5"
        and row["window_id"] in eligible_windows
    }

    if not expected_h5:
        raise ValueError("No eligible H5 windows found in frozen V3.2.2 report.")

    return expected_h5


def _json_safe(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    logger = get_logger(__name__)

    try:
        settings = load_settings()

        root_store = ResearchManifestStore(args.manifest)
        root_manifest = root_store.load()
        _require_locked_holdout(root_manifest)

        expansion_segments = _load_expansion(args.expansion_data_root)
        consumed = _load_exact_range(
            args.consumed_data_root,
            CONSUMED_START,
            CONSUMED_END,
        )
        _assert_holdout_absent(*expansion_segments, consumed)

        partitions = _partition_metadata(
            expansion_segments=expansion_segments,
            consumed=consumed,
            expansion_data_root=args.expansion_data_root,
            consumed_data_root=args.consumed_data_root,
            holdout_registered_at=root_manifest.blind_holdout.registered_at,
        )

        expansion_partitions = tuple(
            item
            for item in partitions
            if item.kind is PartitionKind.RESEARCH_EXPANSION
        )
        consumed_partition = next(
            item
            for item in partitions
            if item.kind is PartitionKind.CONSUMED_RESEARCH
        )

        regions = (
            *(
                ResearchRegion(partition, segment)
                for partition, segment in zip(
                    expansion_partitions,
                    expansion_segments,
                    strict=True,
                )
            ),
            ResearchRegion(consumed_partition, consumed),
        )

        previous = ResearchRunLoader(
            research_root=args.research_root,
            data_root=args.consumed_data_root,
        ).load(args.research_run)

        strategy_config = previous.strategy_config
        backtest_config = previous.backtest_config

        mechanism_config = settings.mechanism_suite_config()
        v32_config = settings.hypothesis_suite_config()
        multiregime_config = settings.multiregime_config()

        windows = construct_windows(regions, multiregime_config)

        source_report = (
            Path(args.mechanism_reports_root)
            / args.mechanism_run
        )
        expected_windows = _read_expected_windows(source_report)

        all_records = []
        all_r_records = []

        csv_rows: list[dict[str, object]] = []
        r_csv_rows: list[dict[str, object]] = []

        logger.info("H5 RISK/CAPITAL AUDIT (OFFLINE)")
        logger.info("Eligible frozen windows: %d", len(expected_windows))
        logger.warning(
            "Blind holdout: LOCKED | NOT LOADED | NOT EVALUATED"
        )

        for window in windows:
            if window.window_id not in expected_windows:
                continue

            strategy = build_mechanism_candidate(
                MechanismHypothesisId.H5,
                strategy_config,
                mechanism_config,
                v32_config,
                backtest_config,
            )

            evaluation = evaluate_strategy_period(
                window.replay_dataset,
                strategy=strategy,
                strategy_config=strategy_config,
                backtest_config=backtest_config,
                evaluation_start_index=window.evaluation_start_index,
            )

            expected_trades = expected_windows[window.window_id]
            actual_trades = len(evaluation.backtest.trades)

            if actual_trades != expected_trades:
                raise ValueError(
                    f"H5 reproduction failed in {window.window_id}: "
                    f"expected {expected_trades}, got {actual_trades}."
                )

            records = build_risk_capital_audit(
                trades=evaluation.backtest.trades,
                signals=evaluation.signals,
                equity_curve=evaluation.backtest.equity_curve,
                backtest_config=backtest_config,
                strategy_config=strategy_config,
            )

            if len(records) != actual_trades:
                raise ValueError(
                    f"Audit record count mismatch in {window.window_id}."
                )

            risky_by_trade_id = {
                record.trade_id: record.actual_stop_risk
                for record in records
            }

            r_records = build_r_normalized_trades(
                trades=evaluation.backtest.trades,
                actual_stop_risk_by_trade_id=risky_by_trade_id,
                slippage_bps=backtest_config.slippage_bps,
            )

            if len(r_records) != actual_trades:
                raise ValueError(
                    f"R-normalized record count mismatch in {window.window_id}."
                )

            all_r_records.extend(r_records)

            for record in r_records:
                r_csv_rows.append(
                    {
                        "window_id": window.window_id,
                        **asdict(record),
                    }
                )

            all_records.extend(records)

            for record in records:
                row = {
                    "window_id": window.window_id,
                    **asdict(record),
                }
                csv_rows.append(row)

            logger.info(
                "%s | H5 trades: %d | audit records: %d",
                window.window_id,
                actual_trades,
                len(records),
            )

        records_tuple = tuple(all_records)
        expected_total = sum(expected_windows.values())

        if len(records_tuple) != expected_total:
            raise ValueError(
                f"Combined H5 trade reproduction failed: "
                f"expected {expected_total}, got {len(records_tuple)}."
            )

        summary = summarize_risk_capital_audit(records_tuple)

        r_records_tuple = tuple(all_r_records)

        if len(r_records_tuple) != expected_total:
            raise ValueError(
                f"Combined R-normalized trade audit failed: "
                f"expected {expected_total}, got {len(r_records_tuple)}."
            )

        r_summary = summarize_r_normalized_trades(
            r_records_tuple
        )

        # Both diagnostic systems calculate friction / actual initial risk.
        # They must agree exactly apart from negligible Decimal arithmetic.

        expecred_average_friciton_r = (
            summary.average_friction_to_actual_risk_percent
            / Decimal("100")
        )

        if (
            abs(
                r_summary.average_total_friction_r
                - expecred_average_friciton_r
            )
            > Decimal("1e-20")
        ):
            raise ValueError(
                "Risk/capital audit and R-normalized friction disagree: "
            )

        output = Path(args.output_root)
        output.mkdir(parents=True, exist_ok=True)

        csv_path = output / "trade_audit.csv"
        r_csv_path = output / "r_normalized_trade_audit.csv"
        summary_path = output / "summary.json"

        if r_csv_rows:
            with r_csv_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=tuple(r_csv_rows[0].keys()),
                )
                writer.writeheader()
                writer.writerows(r_csv_rows)

        if csv_rows:
            with csv_path.open(
                "w",
                encoding="utf-8",
                newline="",
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=tuple(csv_rows[0].keys()),
                )
                writer.writeheader()
                writer.writerows(csv_rows)

        summary_payload = {
            "hypothesis": "H5",
            "dataset_status": "CONSUMED_RESEARCH_DATA",
            "mechanism_run": args.mechanism_run,
            "eligible_windows": len(expected_windows),
            "configured_initial_capital_usdc": (
                backtest_config.initial_capital_usdc
            ),
            "configured_max_position_notional_usdc": (
                strategy_config.maximum_position_notional_usdc
            ),
            "configured_risk_per_trade_percent": (
                strategy_config.risk_per_trade_percent
            ),
            "audit": asdict(summary),
            "r_normalized_audit": asdict(r_summary),
            "holdout": {
                "status": "LOCKED_BLIND_HOLDOUT",
                "revealed": False,
                "consumed": False,
                "evaluated": False,
            },
        }

        summary_path.write_text(
            json.dumps(
                _json_safe(summary_payload),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        logger.info("")
        logger.info("H5 RISK/CAPITAL SUMMARY")
        logger.info("Trades: %d", summary.trade_count)

        logger.info(
            "Risk-budget limited: %d",
            summary.risk_budget_limited_count,
        )
        logger.info(
            "Position-cap limited: %d (%s%%)",
            summary.position_cap_limited_count,
            summary.position_cap_limited_percent,
        )
        logger.info(
            "Available-cash limited: %d",
            summary.available_cash_limited_count,
        )

        logger.info(
            "Configured risk/trade: %s%%",
            summary.average_requested_risk_percent,
        )
        logger.info(
            "Average actual stop-risk: %s%% of equity",
            summary.average_actual_risk_percent,
        )
        logger.info(
            "Maximum actual stop-risk: %s%% of equity",
            summary.maximum_actual_risk_percent,
        )
        logger.info(
            "Average risk-budget utilization: %s%%",
            summary.average_risk_budget_utilization_percent,
        )
        logger.info(
            "Average capital utilization: %s%%",
            summary.average_capital_utilization_percent,
        )
        logger.info(
            "Average friction / actual stop-risk: %s%%",
            summary.average_friction_to_actual_risk_percent,
        )

        logger.info("Audit CSV: %s", csv_path.resolve())
        logger.info("Summary: %s", summary_path.resolve())
        logger.info(
            "Blind holdout: LOCKED | NOT REVEALED | NOT CONSUMED"
        )

        logger.info("")
        logger.info("H5 R-NORMALIZED PERFORMANCE")
        logger.info(
            "Trades: %d | Winners: %d | Losers: %d | Breakeven: %d",
            r_summary.trade_count,
            r_summary.winning_trades,
            r_summary.losing_trades,
            r_summary.breakeven_trades,
        )

        logger.info(
            "Win rate: %s%%",
            r_summary.win_rate_percent,
        )

        logger.info(
            "Frictionless expectancy: %s R/trade",
            r_summary.frictionless_expectancy_r,
        )

        logger.info(
            "Gross after slippage expectancy: %s R/trade",
            r_summary.gross_after_slippage_expectancy_r,
        )

        logger.info(
            "Fee cost: %s R/trade",
            r_summary.average_fee_r,
        )

        logger.info(
            "Slippage cost: %s R/trade",
            r_summary.average_slippage_cost_r,
        )

        logger.info(
            "Total friction: %s R/trade",
            r_summary.average_total_friction_r,
        )

        logger.info(
            "NET expectancy: %s R/trade",
            r_summary.net_expectancy_r,
        )

        logger.info(
            "Average winner: %s R | Average loser: %s R",
            r_summary.average_winner_r,
            r_summary.average_loser_r,
        )

        logger.info(
            "Payoff ratio: %s | Profit factor: %s",
            r_summary.payoff_ratio_r,
            r_summary.profit_factor_r,
        )

        logger.info(
            "Actual win rate: %s%% | Break-even win rate: %s%%",
            r_summary.win_rate_percent,
            r_summary.break_even_win_rate_percent,
        )

        logger.info(
            "R-normalized CSV: %s",
            r_csv_path.resolve(),
        )
        return 0

    except (OSError, RuntimeError, ValueError, KeyError) as exc:
        logger.error("Risk/capital audit failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())