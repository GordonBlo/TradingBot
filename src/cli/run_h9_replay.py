"""Execute the single preregistered H9 Early Failure counterfactual replay."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from pathlib import Path

from src.cli.run_h9 import (
    EXPECTED_H9_RUN_ID,
    EXPECTED_REFERENCE_TRADES,
    validate_early_failure_report,
    validate_h9_manifest,
)
from src.cli.run_multiregime import _require_locked_holdout
from src.cli.run_v33_h5 import (
    _build_regions,
    _max_drawdown_percent,
    _read_reference_windows,
)
from src.config.settings import load_settings
from src.diagnostics.loader import ResearchRunLoader
from src.diagnostics.r_normalized import (
    build_r_normalized_trades,
    summarize_r_normalized_trades,
)
from src.diagnostics.risk_capital_audit import (
    build_risk_capital_audit,
)
from src.hypotheses.h5_parameter_research import (
    H5ParameterCandidateId,
    build_h5_parameter_candidate,
    validate_h5_parameter_registry,
)
from src.hypotheses.h9_early_failure import H9EarlyFailureExit
from src.hypotheses.manifest import (
    HoldoutStatus,
    ResearchManifestStore,
)
from src.research.evaluation import evaluate_strategy_period
from src.research.h9_stability import (
    H9Metrics,
    H9WindowComparison,
    evaluate_h9_gate,
    h9_v34_eligible,
)
from src.research.multiregime.windows import construct_windows
from src.utils.logger import configure_logging, get_logger


EXPECTED_V33_RUN_ID = "b104ea78b4c88bcf"
EXPECTED_V331_RUN_ID = "ef593be9a0bb2697"

TOLERANCE = Decimal("1e-18")


@dataclass(frozen=True, slots=True)
class WindowResult:
    window_id: str
    variant: str

    trades: int

    frictionless_expectancy_r: Decimal
    net_expectancy_r: Decimal
    profit_factor_r: Decimal | None

    maximum_drawdown_percent: Decimal

    h9_triggers: int
    h9_exits: int


@dataclass(slots=True)
class CounterfactualTotals:
    triggered_trades: int = 0
    h9_exit_trades: int = 0

    matched_baseline_entries: int = 0
    unmatched_baseline_entries: int = 0
    new_h9_entries: int = 0

    baseline_stop_losses_improved: int = 0
    baseline_trend_exits_improved: int = 0
    baseline_take_profits_cut_early: int = 0

    gross_r_saved_or_lost: Decimal = Decimal("0")
    net_r_saved_or_lost: Decimal = Decimal("0")

    h9_exit_net_r_total: Decimal = Decimal("0")


def _json_safe(value):
    if isinstance(value, Decimal):
        return str(value)

    if hasattr(value, "value"):
        return value.value

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item)
            for item in value
        ]

    return value


def _load_json(path: str | Path) -> dict:
    path = Path(path)

    if not path.is_file():
        raise ValueError(
            f"Required research artifact missing: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _evaluate_window(
    *,
    window,
    strategy_config,
    backtest_config,
    use_h9: bool,
):
    strategy = build_h5_parameter_candidate(
        H5ParameterCandidateId.H5_Q25,
        strategy_config,
    )

    h9 = (
        H9EarlyFailureExit()
        if use_h9
        else None
    )

    evaluation = evaluate_strategy_period(
        window.replay_dataset,
        strategy=strategy,
        strategy_config=strategy_config,
        backtest_config=backtest_config,
        evaluation_start_index=(
            window.evaluation_start_index
        ),
        position_exits=h9,
    )

    risk_records = build_risk_capital_audit(
        trades=evaluation.backtest.trades,
        signals=evaluation.signals,
        equity_curve=evaluation.backtest.equity_curve,
        backtest_config=backtest_config,
        strategy_config=strategy_config,
    )

    actual_risk_by_trade_id = {
        record.trade_id: record.actual_stop_risk
        for record in risk_records
    }

    r_records = build_r_normalized_trades(
        trades=evaluation.backtest.trades,
        actual_stop_risk_by_trade_id=(
            actual_risk_by_trade_id
        ),
        slippage_bps=backtest_config.slippage_bps,
    )

    if len(r_records) != len(
        evaluation.backtest.trades
    ):
        raise ValueError(
            f"R-record mismatch in {window.window_id}."
        )

    summary = summarize_r_normalized_trades(
        r_records
    )

    return (
        evaluation,
        r_records,
        summary,
        h9,
    )


def _window_result(
    *,
    window_id: str,
    variant: str,
    evaluation,
    summary,
    h9: H9EarlyFailureExit | None,
) -> WindowResult:
    h9_exits = sum(
        trade.exit_reason.value == "EARLY_FAILURE_EXIT"
        for trade in evaluation.backtest.trades
    )

    return WindowResult(
        window_id=window_id,
        variant=variant,
        trades=len(evaluation.backtest.trades),
        frictionless_expectancy_r=(
            summary.frictionless_expectancy_r
        ),
        net_expectancy_r=summary.net_expectancy_r,
        profit_factor_r=summary.profit_factor_r,
        maximum_drawdown_percent=(
            _max_drawdown_percent(
                evaluation.backtest.equity_curve
            )
        ),
        h9_triggers=(
            h9.trigger_count
            if h9 is not None
            else 0
        ),
        h9_exits=h9_exits,
    )


def _update_counterfactual(
    totals: CounterfactualTotals,
    *,
    reference_evaluation,
    reference_records,
    h9_evaluation,
    h9_records,
    h9: H9EarlyFailureExit,
) -> None:
    totals.triggered_trades += h9.trigger_count

    reference_r = {
        record.trade_id: record
        for record in reference_records
    }

    h9_r = {
        record.trade_id: record
        for record in h9_records
    }

    reference_by_signal = {
        trade.entry_signal_time: trade
        for trade in reference_evaluation.backtest.trades
    }

    h9_by_signal = {
        trade.entry_signal_time: trade
        for trade in h9_evaluation.backtest.trades
    }

    reference_signals = set(reference_by_signal)
    h9_signals = set(h9_by_signal)

    matched = (
        reference_signals
        & h9_signals
    )

    totals.matched_baseline_entries += len(
        matched
    )

    totals.unmatched_baseline_entries += len(
        reference_signals - h9_signals
    )

    totals.new_h9_entries += len(
        h9_signals - reference_signals
    )

    for trade in h9_evaluation.backtest.trades:
        if (
            trade.exit_reason.value
            != "EARLY_FAILURE_EXIT"
        ):
            continue

        totals.h9_exit_trades += 1

        record = h9_r[trade.trade_id]

        totals.h9_exit_net_r_total += (
            record.net_r
        )

    for signal_time in matched:
        reference_trade = reference_by_signal[
            signal_time
        ]

        h9_trade = h9_by_signal[
            signal_time
        ]

        reference_record = reference_r[
            reference_trade.trade_id
        ]

        h9_record = h9_r[
            h9_trade.trade_id
        ]

        gross_delta = (
            h9_record.frictionless_r
            - reference_record.frictionless_r
        )

        net_delta = (
            h9_record.net_r
            - reference_record.net_r
        )

        totals.gross_r_saved_or_lost += (
            gross_delta
        )

        totals.net_r_saved_or_lost += (
            net_delta
        )

        if (
            reference_trade.exit_reason.value
            == "STOP_LOSS"
            and h9_record.net_r
            > reference_record.net_r
        ):
            totals.baseline_stop_losses_improved += 1

        if (
            reference_trade.exit_reason.value
            == "TREND_EXIT"
            and h9_record.net_r
            > reference_record.net_r
        ):
            totals.baseline_trend_exits_improved += 1

        if (
            reference_trade.exit_reason.value
            == "TAKE_PROFIT"
            and h9_trade.exit_reason.value
            == "EARLY_FAILURE_EXIT"
        ):
            totals.baseline_take_profits_cut_early += 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Execute preregistered H9 "
            "Early Failure counterfactual research."
        )
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Explicitly authorize the single "
            "H9 consumed-data replay."
        ),
    )

    parser.add_argument(
        "--h9-manifest",
        default=(
            "research/h9/"
            "eb7a43ccaad63dfc/"
            "manifest.json"
        ),
    )

    parser.add_argument(
        "--early-failure-report",
        default=(
            "reports/diagnostics/v331_h5/"
            "ef593be9a0bb2697/"
            "early_failure_summary.json"
        ),
    )

    parser.add_argument(
        "--v33-summary",
        default=(
            "reports/v3_3_h5/"
            "b104ea78b4c88bcf/"
            "summary.json"
        ),
    )

    parser.add_argument(
        "--v331-summary",
        default=(
            "reports/diagnostics/v331_h5/"
            "ef593be9a0bb2697/"
            "summary.json"
        ),
    )

    parser.add_argument(
        "--root-manifest",
        default=(
            "research/hypothesis_manifest.json"
        ),
    )

    parser.add_argument(
        "--research-run",
        default="adeef00722e9704d",
    )

    parser.add_argument(
        "--research-root",
        default="reports/research",
    )

    parser.add_argument(
        "--mechanism-report",
        default=(
            "reports/mechanisms/"
            "bc2496aed05555b5"
        ),
    )

    parser.add_argument(
        "--expansion-data-root",
        default=(
            "data/historical/"
            "multiregime_expansion"
        ),
    )

    parser.add_argument(
        "--consumed-data-root",
        default="data/historical",
    )

    parser.add_argument(
        "--output-root",
        default="reports/h9",
    )

    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = build_parser().parse_args(argv)

    configure_logging()
    logger = get_logger(__name__)

    if not args.execute:
        logger.error(
            "H9 replay not executed. "
            "Explicit --execute is required."
        )
        return 1

    try:
        validate_h5_parameter_registry()

        h9_manifest = _load_json(
            args.h9_manifest
        )

        validate_h9_manifest(
            h9_manifest
        )

        early_failure_path = Path(
            args.early_failure_report
        )

        early_failure_raw = (
            early_failure_path.read_bytes()
        )

        early_failure = json.loads(
            early_failure_raw.decode("utf-8")
        )

        validate_early_failure_report(
            early_failure,
            raw=early_failure_raw,
            h9_manifest=h9_manifest,
        )

        if (
            h9_manifest["run_id"]
            != EXPECTED_H9_RUN_ID
        ):
            raise ValueError(
                "Unexpected H9 run ID."
            )

        if (
            H9EarlyFailureExit.ADVERSE_TRIGGER_R
            != Decimal("0.5")
        ):
            raise ValueError(
                "H9 implementation trigger changed."
            )

        if (
            H9EarlyFailureExit.OBSERVATION_BARS
            != 4
        ):
            raise ValueError(
                "H9 observation window changed."
            )

        v33 = _load_json(
            args.v33_summary
        )

        v331 = _load_json(
            args.v331_summary
        )

        if (
            v33.get("run_id")
            != EXPECTED_V33_RUN_ID
        ):
            raise ValueError(
                "Unexpected V3.3 source."
            )

        if (
            v331.get("run_id")
            != EXPECTED_V331_RUN_ID
        ):
            raise ValueError(
                "Unexpected V3.3.1 source."
            )

        root_manifest = ResearchManifestStore(
            args.root_manifest
        ).load()

        _require_locked_holdout(
            root_manifest
        )

        if (
            root_manifest.holdout_status
            is not HoldoutStatus.LOCKED_BLIND_HOLDOUT
        ):
            raise ValueError(
                "Blind holdout is not locked."
            )

        holdout = (
            root_manifest.blind_holdout
        )

        if (
            holdout.reveal_timestamp
            is not None
            or holdout.consumed_timestamp
            is not None
        ):
            raise ValueError(
                "Blind holdout contamination detected."
            )

        reference_windows = (
            _read_reference_windows(
                Path(args.mechanism_report)
            )
        )

        previous = ResearchRunLoader(
            research_root=args.research_root,
            data_root=args.consumed_data_root,
        ).load(
            args.research_run
        )

        strategy_config = (
            previous.strategy_config
        )

        base_config = (
            previous.backtest_config
        )

        if (
            base_config.ambiguous_bar_policy.value
            != "STOP_FIRST"
        ):
            raise ValueError(
                "STOP_FIRST semantics changed."
            )

        stress_config = replace(
            base_config,
            fee_bps=(
                base_config.fee_bps
                * Decimal("2")
            ),
            slippage_bps=(
                base_config.slippage_bps
                * Decimal("2")
            ),
        )

        regions = _build_regions(
            root_manifest=root_manifest,
            expansion_data_root=(
                args.expansion_data_root
            ),
            consumed_data_root=(
                args.consumed_data_root
            ),
        )

        windows = construct_windows(
            regions,
            load_settings().multiregime_config(),
        )

        eligible_windows = tuple(
            window
            for window in windows
            if (
                window.window_id
                in reference_windows
            )
        )

        if len(eligible_windows) != 11:
            raise ValueError(
                "H9 requires exactly "
                f"11 windows; got "
                f"{len(eligible_windows)}."
            )

        output = (
            Path(args.output_root)
            / EXPECTED_H9_RUN_ID
        )

        summary_path = (
            output
            / "summary.json"
        )

        if summary_path.exists():
            raise ValueError(
                "H9 result already exists. "
                "Do not rerun the preregistered experiment."
            )

        logger.info(
            "H9 PREREGISTERED COUNTERFACTUAL REPLAY"
        )

        logger.info(
            "Reference: H5_Q25"
        )

        logger.info(
            "Candidate: H5_Q25 + H9"
        )

        logger.info(
            "Trigger: -0.5R in first 4 held bars"
        )

        logger.info(
            "Execution: NEXT BAR OPEN"
        )

        logger.info(
            "Windows: 11 | CONSUMED_RESEARCH_DATA"
        )

        logger.warning(
            "Blind holdout: LOCKED | NOT LOADED | "
            "NOT REVEALED | NOT CONSUMED | NOT EVALUATED"
        )

        reference_records = []
        h9_records = []

        reference_rows: list[
            WindowResult
        ] = []

        h9_rows: list[
            WindowResult
        ] = []

        reference_max_dd = Decimal("0")
        h9_max_dd = Decimal("0")

        counterfactual = (
            CounterfactualTotals()
        )

        # ==================================================
        # BASE COST REPLAY
        # ==================================================

        for window in eligible_windows:
            (
                reference_evaluation,
                reference_r,
                reference_summary,
                _,
            ) = _evaluate_window(
                window=window,
                strategy_config=strategy_config,
                backtest_config=base_config,
                use_h9=False,
            )

            expected = reference_windows[
                window.window_id
            ]

            if (
                len(
                    reference_evaluation
                    .backtest
                    .trades
                )
                != expected
            ):
                raise ValueError(
                    f"Frozen Q25 mismatch in "
                    f"{window.window_id}."
                )

            (
                h9_evaluation,
                h9_r,
                h9_summary,
                h9,
            ) = _evaluate_window(
                window=window,
                strategy_config=strategy_config,
                backtest_config=base_config,
                use_h9=True,
            )

            assert h9 is not None

            reference_row = _window_result(
                window_id=window.window_id,
                variant="H5_Q25_REFERENCE",
                evaluation=reference_evaluation,
                summary=reference_summary,
                h9=None,
            )

            h9_row = _window_result(
                window_id=window.window_id,
                variant="H9",
                evaluation=h9_evaluation,
                summary=h9_summary,
                h9=h9,
            )

            reference_rows.append(
                reference_row
            )

            h9_rows.append(
                h9_row
            )

            reference_records.extend(
                reference_r
            )

            h9_records.extend(
                h9_r
            )

            reference_max_dd = max(
                reference_max_dd,
                reference_row.maximum_drawdown_percent,
            )

            h9_max_dd = max(
                h9_max_dd,
                h9_row.maximum_drawdown_percent,
            )

            _update_counterfactual(
                counterfactual,
                reference_evaluation=(
                    reference_evaluation
                ),
                reference_records=reference_r,
                h9_evaluation=h9_evaluation,
                h9_records=h9_r,
                h9=h9,
            )

            logger.info(
                "%s | REF trades=%d net=%sR | "
                "H9 trades=%d net=%sR | "
                "triggers=%d | exits=%d",
                window.window_id,
                reference_row.trades,
                reference_row.net_expectancy_r,
                h9_row.trades,
                h9_row.net_expectancy_r,
                h9_row.h9_triggers,
                h9_row.h9_exits,
            )

        if (
            len(reference_records)
            != EXPECTED_REFERENCE_TRADES
        ):
            raise ValueError(
                "Frozen H5_Q25 did not "
                "reproduce exactly 197 trades."
            )

        reference_summary = (
            summarize_r_normalized_trades(
                tuple(reference_records)
            )
        )

        h9_summary = (
            summarize_r_normalized_trades(
                tuple(h9_records)
            )
        )

        stored_reference = (
            v331["r_accounting"]["summary"]
        )

        for field in (
            "frictionless_expectancy_r",
            "net_expectancy_r",
            "profit_factor_r",
        ):
            expected = Decimal(
                str(
                    stored_reference[field]
                )
            )

            actual = getattr(
                reference_summary,
                field,
            )

            if (
                abs(actual - expected)
                > TOLERANCE
            ):
                raise ValueError(
                    "Frozen reference mismatch: "
                    f"{field}."
                )

        logger.info(
            "Frozen H5_Q25 "
            "197-trade R-reference VERIFIED."
        )

        # ==================================================
        # DOUBLED-COST STRESS
        # ==================================================

        reference_stress_records = []
        h9_stress_records = []

        logger.info(
            "Running doubled-cost stress replay."
        )

        for window in eligible_windows:
            (
                _,
                reference_r,
                _,
                _,
            ) = _evaluate_window(
                window=window,
                strategy_config=strategy_config,
                backtest_config=stress_config,
                use_h9=False,
            )

            (
                _,
                h9_r,
                _,
                _,
            ) = _evaluate_window(
                window=window,
                strategy_config=strategy_config,
                backtest_config=stress_config,
                use_h9=True,
            )

            reference_stress_records.extend(
                reference_r
            )

            h9_stress_records.extend(
                h9_r
            )

        reference_stress = (
            summarize_r_normalized_trades(
                tuple(
                    reference_stress_records
                )
            )
        )

        h9_stress = (
            summarize_r_normalized_trades(
                tuple(
                    h9_stress_records
                )
            )
        )

        stored_stress = Decimal(
            str(
                v33[
                    "candidates"
                ][
                    "H5_Q25"
                ][
                    "stress"
                ][
                    "net_expectancy_r"
                ]
            )
        )

        if (
            abs(
                reference_stress.net_expectancy_r
                - stored_stress
            )
            > TOLERANCE
        ):
            raise ValueError(
                "Frozen doubled-cost "
                "reference mismatch."
            )

        # ==================================================
        # GATES
        # ==================================================

        reference_metrics = H9Metrics(
            trades=len(reference_records),
            frictionless_expectancy_r=(
                reference_summary
                .frictionless_expectancy_r
            ),
            net_expectancy_r=(
                reference_summary
                .net_expectancy_r
            ),
            profit_factor_r=(
                reference_summary
                .profit_factor_r
            ),
            maximum_drawdown_percent=(
                reference_max_dd
            ),
            stress_net_expectancy_r=(
                reference_stress
                .net_expectancy_r
            ),
        )

        h9_metrics = H9Metrics(
            trades=len(h9_records),
            frictionless_expectancy_r=(
                h9_summary
                .frictionless_expectancy_r
            ),
            net_expectancy_r=(
                h9_summary
                .net_expectancy_r
            ),
            profit_factor_r=(
                h9_summary
                .profit_factor_r
            ),
            maximum_drawdown_percent=(
                h9_max_dd
            ),
            stress_net_expectancy_r=(
                h9_stress
                .net_expectancy_r
            ),
        )

        reference_by_window = {
            row.window_id: row
            for row in reference_rows
        }

        comparisons = tuple(
            H9WindowComparison(
                h9_frictionless_r=(
                    row.frictionless_expectancy_r
                ),
                reference_frictionless_r=(
                    reference_by_window[
                        row.window_id
                    ].frictionless_expectancy_r
                ),
                h9_net_r=(
                    row.net_expectancy_r
                ),
                reference_net_r=(
                    reference_by_window[
                        row.window_id
                    ].net_expectancy_r
                ),
                h9_net_positive=(
                    row.net_expectancy_r > 0
                ),
            )
            for row in h9_rows
        )

        gate = evaluate_h9_gate(
            h9=h9_metrics,
            reference=reference_metrics,
            windows=comparisons,
            matched_reference_entries=(
                counterfactual
                .matched_baseline_entries
            ),
            reference_entry_count=(
                EXPECTED_REFERENCE_TRADES
            ),
        )

        v34_eligible = (
            h9_v34_eligible(
                h9=h9_metrics,
                gate=gate,
            )
        )

        average_h9_exit_net_r = (
            counterfactual.h9_exit_net_r_total
            / Decimal(
                counterfactual.h9_exit_trades
            )
            if (
                counterfactual.h9_exit_trades
            )
            else Decimal("0")
        )

        counterfactual_payload = {
            "h9_triggered_trades": (
                counterfactual.triggered_trades
            ),
            "h9_exit_trades": (
                counterfactual.h9_exit_trades
            ),
            "matched_baseline_entries": (
                counterfactual.matched_baseline_entries
            ),
            "unmatched_baseline_entries": (
                counterfactual.unmatched_baseline_entries
            ),
            "new_h9_entries": (
                counterfactual.new_h9_entries
            ),
            "baseline_stop_losses_improved": (
                counterfactual
                .baseline_stop_losses_improved
            ),
            "baseline_trend_exits_improved": (
                counterfactual
                .baseline_trend_exits_improved
            ),
            "baseline_take_profits_cut_early": (
                counterfactual
                .baseline_take_profits_cut_early
            ),
            "gross_R_saved_or_lost_on_matched_entries": (
                counterfactual
                .gross_r_saved_or_lost
            ),
            "net_R_saved_or_lost_on_matched_entries": (
                counterfactual
                .net_r_saved_or_lost
            ),
            "average_net_R_of_h9_exits": (
                average_h9_exit_net_r
            ),
        }

        # ==================================================
        # REPORT
        # ==================================================

        output.mkdir(
            parents=True,
            exist_ok=True,
        )

        csv_path = (
            output
            / "window_comparison.csv"
        )

        temporary_csv = (
            output
            / "window_comparison.csv.tmp"
        )

        rows = (
            *reference_rows,
            *h9_rows,
        )

        with temporary_csv.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=tuple(
                    asdict(rows[0]).keys()
                ),
            )

            writer.writeheader()

            for row in rows:
                writer.writerow(
                    _json_safe(
                        asdict(row)
                    )
                )

        temporary_csv.replace(
            csv_path
        )

        payload = {
            "version": "3.3.3",
            "run_id": EXPECTED_H9_RUN_ID,

            "source_v33_run_id": (
                EXPECTED_V33_RUN_ID
            ),

            "source_v331_run_id": (
                EXPECTED_V331_RUN_ID
            ),

            "dataset_status": (
                "CONSUMED_RESEARCH_DATA"
            ),

            "eligible_windows": 11,

            "blind_holdout": {
                "status": (
                    "LOCKED_BLIND_HOLDOUT"
                ),
                "loaded": False,
                "revealed": False,
                "consumed": False,
                "evaluated": False,
            },

            "reference": {
                "candidate": "H5_Q25",
                "trades": len(
                    reference_records
                ),
                "base": asdict(
                    reference_summary
                ),
                "maximum_drawdown_percent": (
                    reference_max_dd
                ),
                "stress": asdict(
                    reference_stress
                ),
            },

            "h9": {
                "definition": (
                    h9_manifest["definition"]
                ),
                "trades": len(
                    h9_records
                ),
                "base": asdict(
                    h9_summary
                ),
                "maximum_drawdown_percent": (
                    h9_max_dd
                ),
                "stress": asdict(
                    h9_stress
                ),
            },

            "gate": asdict(
                gate
            ),

            "all_support_gates_met": (
                gate.all_support_gates_met
            ),

            "v3_4_eligible": (
                v34_eligible
            ),

            "counterfactual": (
                counterfactual_payload
            ),

            "windows": [
                asdict(row)
                for row in rows
            ],
        }

        temporary_summary = (
            output
            / "summary.json.tmp"
        )

        temporary_summary.write_text(
            json.dumps(
                _json_safe(payload),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_summary.replace(
            summary_path
        )

        # ==================================================
        # FINAL CONSOLE SUMMARY
        # ==================================================

        logger.info("")
        logger.info(
            "H9 RESULTS"
        )

        logger.info(
            "REFERENCE H5_Q25 | "
            "trades=%d | gross=%sR | "
            "net=%sR | PF=%s | stress=%sR",
            len(reference_records),
            reference_summary.frictionless_expectancy_r,
            reference_summary.net_expectancy_r,
            reference_summary.profit_factor_r,
            reference_stress.net_expectancy_r,
        )

        logger.info(
            "H9 | trades=%d | gross=%sR | "
            "net=%sR | PF=%s | stress=%sR",
            len(h9_records),
            h9_summary.frictionless_expectancy_r,
            h9_summary.net_expectancy_r,
            h9_summary.profit_factor_r,
            h9_stress.net_expectancy_r,
        )

        logger.info(
            "Consistency: gross=%d/11 | "
            "net=%d/11 | positive-net=%d/11",
            gate.frictionless_better_windows,
            gate.net_better_windows,
            gate.positive_net_windows,
        )

        logger.info(
            "Trade ratio=%s | "
            "entry-match ratio=%s",
            gate.trade_count_ratio,
            gate.baseline_entry_match_ratio,
        )

        logger.info(
            "H9 triggers=%d | exits=%d | "
            "SL improved=%d | TREND improved=%d | "
            "TP cut early=%d",
            counterfactual.triggered_trades,
            counterfactual.h9_exit_trades,
            counterfactual.baseline_stop_losses_improved,
            counterfactual.baseline_trend_exits_improved,
            counterfactual.baseline_take_profits_cut_early,
        )

        logger.info(
            "Matched entries=%d | "
            "unmatched baseline=%d | "
            "new H9 entries=%d",
            counterfactual.matched_baseline_entries,
            counterfactual.unmatched_baseline_entries,
            counterfactual.new_h9_entries,
        )

        logger.info(
            "Matched-entry delta: "
            "gross=%sR | net=%sR",
            counterfactual.gross_r_saved_or_lost,
            counterfactual.net_r_saved_or_lost,
        )

        logger.info(
            "Average H9 exit: %sR net",
            average_h9_exit_net_r,
        )

        logger.info(
            "Support gates met: %s",
            gate.all_support_gates_met,
        )

        logger.info(
            "V3.4 eligible: %s",
            v34_eligible,
        )

        logger.info(
            "Report: %s",
            summary_path.resolve(),
        )

        logger.warning(
            "Blind holdout: LOCKED | NOT LOADED | "
            "NOT REVEALED | NOT CONSUMED | NOT EVALUATED"
        )

        return 0

    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        logger.error(
            "H9 replay failed: %s",
            exc,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())