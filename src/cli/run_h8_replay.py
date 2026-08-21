"""Execute the single preregistered H8 counterfactual replay."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from pathlib import Path

from src.cli.run_h8 import (
    EXPECTED_H8_RUN_ID,
    EXPECTED_V33_RUN_ID,
    EXPECTED_V331_RUN_ID,
    validate_h8_manifest,
    validate_v33_summary,
    validate_v331_summary,
)
from src.cli.run_multiregime import _require_locked_holdout
from src.cli.run_v33_h5 import (
    _build_regions,
    _max_drawdown_percent,
    _read_reference_windows,
)
from src.diagnostics.loader import ResearchRunLoader
from src.diagnostics.r_normalized import (
    build_r_normalized_trades,
    summarize_r_normalized_trades,
)
from src.diagnostics.risk_capital_audit import build_risk_capital_audit
from src.hypotheses.h5_parameter_research import (
    H5ParameterCandidateId,
    build_h5_parameter_candidate,
    validate_h5_parameter_registry,
)
from src.hypotheses.h8_protection import H8BreakEvenProtection
from src.hypotheses.manifest import (
    HoldoutStatus,
    ResearchManifestStore,
)
from src.research.evaluation import evaluate_strategy_period
from src.research.h8_stability import (
    H8Metrics,
    H8WindowComparison,
    evaluate_h8_gate,
    h8_v34_eligible,
)
from src.research.multiregime.windows import construct_windows
from src.utils.logger import configure_logging, get_logger


REFERENCE_TRADES = 197
TOLERANCE = Decimal("1e-18")


@dataclass(frozen=True, slots=True)
class WindowMetrics:
    window_id: str
    variant: str
    trades: int
    frictionless_expectancy_r: Decimal
    net_expectancy_r: Decimal
    profit_factor_r: Decimal | None
    maximum_drawdown_percent: Decimal
    h8_activations: int
    break_even_exits: int


@dataclass(slots=True)
class CounterfactualTotals:
    activated_trades: int = 0
    protective_stop_exits: int = 0
    matched_trades: int = 0
    unmatched_reference_trades: int = 0
    unmatched_h8_trades: int = 0

    baseline_stop_losses_improved: int = 0
    baseline_take_profits_cut_early: int = 0

    gross_r_saved_by_protection: Decimal = Decimal("0")
    net_r_saved_by_protection: Decimal = Decimal("0")

    protective_exit_net_r_total: Decimal = Decimal("0")


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
        return [_json_safe(item) for item in value]

    return value


def _load_json(path: str | Path) -> dict:
    path = Path(path)

    if not path.is_file():
        raise ValueError(f"Required artifact missing: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def _evaluate_window(
    *,
    window,
    strategy_config,
    backtest_config,
    use_h8: bool,
):
    strategy = build_h5_parameter_candidate(
        H5ParameterCandidateId.H5_Q25,
        strategy_config,
    )

    protection = (
        H8BreakEvenProtection()
        if use_h8
        else None
    )

    evaluation = evaluate_strategy_period(
        window.replay_dataset,
        strategy=strategy,
        strategy_config=strategy_config,
        backtest_config=backtest_config,
        evaluation_start_index=window.evaluation_start_index,
        stop_updates=protection,
    )

    risk_records = build_risk_capital_audit(
        trades=evaluation.backtest.trades,
        signals=evaluation.signals,
        equity_curve=evaluation.backtest.equity_curve,
        backtest_config=backtest_config,
        strategy_config=strategy_config,
    )

    risk_by_trade_id = {
        record.trade_id: record.actual_stop_risk
        for record in risk_records
    }

    r_records = build_r_normalized_trades(
        trades=evaluation.backtest.trades,
        actual_stop_risk_by_trade_id=risk_by_trade_id,
        slippage_bps=backtest_config.slippage_bps,
    )

    if len(r_records) != len(evaluation.backtest.trades):
        raise ValueError(
            f"R-record mismatch in {window.window_id}."
        )

    return (
        evaluation,
        r_records,
        summarize_r_normalized_trades(r_records),
        protection,
    )


def _window_metrics(
    *,
    window_id: str,
    variant: str,
    evaluation,
    summary,
    protection,
) -> WindowMetrics:
    break_even_exits = sum(
        trade.exit_reason.value == "BREAK_EVEN_STOP"
        for trade in evaluation.backtest.trades
    )

    return WindowMetrics(
        window_id=window_id,
        variant=variant,
        trades=len(evaluation.backtest.trades),
        frictionless_expectancy_r=(
            summary.frictionless_expectancy_r
        ),
        net_expectancy_r=summary.net_expectancy_r,
        profit_factor_r=summary.profit_factor_r,
        maximum_drawdown_percent=_max_drawdown_percent(
            evaluation.backtest.equity_curve
        ),
        h8_activations=(
            protection.activation_count
            if protection is not None
            else 0
        ),
        break_even_exits=break_even_exits,
    )


def _update_counterfactual(
    totals: CounterfactualTotals,
    *,
    reference_evaluation,
    reference_records,
    h8_evaluation,
    h8_records,
    protection: H8BreakEvenProtection,
) -> None:
    totals.activated_trades += protection.activation_count

    reference_r = {
        record.trade_id: record
        for record in reference_records
    }

    h8_r = {
        record.trade_id: record
        for record in h8_records
    }

    reference_by_signal = {
        trade.entry_signal_time: trade
        for trade in reference_evaluation.backtest.trades
    }

    h8_by_signal = {
        trade.entry_signal_time: trade
        for trade in h8_evaluation.backtest.trades
    }

    reference_signals = set(reference_by_signal)
    h8_signals = set(h8_by_signal)

    matched = reference_signals & h8_signals

    totals.matched_trades += len(matched)
    totals.unmatched_reference_trades += len(
        reference_signals - h8_signals
    )
    totals.unmatched_h8_trades += len(
        h8_signals - reference_signals
    )

    for trade in h8_evaluation.backtest.trades:
        if trade.exit_reason.value != "BREAK_EVEN_STOP":
            continue

        totals.protective_stop_exits += 1

        record = h8_r[trade.trade_id]
        totals.protective_exit_net_r_total += record.net_r

    for signal_time in matched:
        reference_trade = reference_by_signal[signal_time]
        h8_trade = h8_by_signal[signal_time]

        reference_record = reference_r[
            reference_trade.trade_id
        ]
        h8_record = h8_r[h8_trade.trade_id]

        if (
            reference_trade.exit_reason.value == "STOP_LOSS"
            and h8_record.net_r > reference_record.net_r
        ):
            totals.baseline_stop_losses_improved += 1

        if (
            reference_trade.exit_reason.value == "TAKE_PROFIT"
            and h8_trade.exit_reason.value == "BREAK_EVEN_STOP"
        ):
            totals.baseline_take_profits_cut_early += 1

        if h8_trade.exit_reason.value == "BREAK_EVEN_STOP":
            totals.gross_r_saved_by_protection += (
                h8_record.frictionless_r
                - reference_record.frictionless_r
            )

            totals.net_r_saved_by_protection += (
                h8_record.net_r
                - reference_record.net_r
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute preregistered H8 research replay."
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly authorize the single H8 consumed-data replay.",
    )

    parser.add_argument(
        "--h8-manifest",
        default=(
            "research/h8/"
            "369586c25375c0ef/manifest.json"
        ),
    )

    parser.add_argument(
        "--v33-summary",
        default=(
            "reports/v3_3_h5/"
            "b104ea78b4c88bcf/summary.json"
        ),
    )

    parser.add_argument(
        "--v331-summary",
        default=(
            "reports/diagnostics/v331_h5/"
            "ef593be9a0bb2697/summary.json"
        ),
    )

    parser.add_argument(
        "--root-manifest",
        default="research/hypothesis_manifest.json",
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
        default="reports/mechanisms/bc2496aed05555b5",
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
        "--output-root",
        default="reports/h8",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    configure_logging()
    logger = get_logger(__name__)

    if not args.execute:
        logger.error(
            "H8 replay not executed. Explicit --execute is required."
        )
        return 1

    try:
        validate_h5_parameter_registry()

        h8_manifest = _load_json(args.h8_manifest)
        v33 = _load_json(args.v33_summary)
        v331 = _load_json(args.v331_summary)

        validate_h8_manifest(h8_manifest)
        validate_v33_summary(v33)
        validate_v331_summary(v331)

        if h8_manifest["run_id"] != EXPECTED_H8_RUN_ID:
            raise ValueError("Unexpected H8 run ID.")

        if H8BreakEvenProtection.TRIGGER_R != Decimal("1"):
            raise ValueError("H8 implementation trigger changed.")

        root_manifest = ResearchManifestStore(
            args.root_manifest
        ).load()

        _require_locked_holdout(root_manifest)

        if (
            root_manifest.holdout_status
            is not HoldoutStatus.LOCKED_BLIND_HOLDOUT
        ):
            raise ValueError("Blind holdout is not locked.")

        if (
            root_manifest.blind_holdout.reveal_timestamp is not None
            or root_manifest.blind_holdout.consumed_timestamp is not None
        ):
            raise ValueError("Blind holdout contamination detected.")

        reference_windows = _read_reference_windows(
            Path(args.mechanism_report)
        )

        previous = ResearchRunLoader(
            research_root=args.research_root,
            data_root=args.consumed_data_root,
        ).load(args.research_run)

        strategy_config = previous.strategy_config
        base_config = previous.backtest_config

        if base_config.ambiguous_bar_policy.value != "STOP_FIRST":
            raise ValueError("STOP_FIRST semantics changed.")

        stress_config = replace(
            base_config,
            fee_bps=base_config.fee_bps * Decimal("2"),
            slippage_bps=(
                base_config.slippage_bps * Decimal("2")
            ),
        )

        from src.config.settings import load_settings

        regions = _build_regions(
            root_manifest=root_manifest,
            expansion_data_root=args.expansion_data_root,
            consumed_data_root=args.consumed_data_root,
        )

        windows = construct_windows(
            regions,
            load_settings().multiregime_config(),
        )

        eligible_windows = tuple(
            window
            for window in windows
            if window.window_id in reference_windows
        )

        if len(eligible_windows) != 11:
            raise ValueError(
                f"Expected 11 windows, got {len(eligible_windows)}."
            )

        output = (
            Path(args.output_root)
            / EXPECTED_H8_RUN_ID
        )

        summary_path = output / "summary.json"

        if summary_path.exists():
            raise ValueError(
                "H8 result already exists. "
                "Do not rerun the preregistered experiment."
            )

        logger.info("H8 PREREGISTERED COUNTERFACTUAL REPLAY")
        logger.info("Reference: H5_Q25")
        logger.info("Candidate: H5_Q25 + H8")
        logger.info("Windows: 11 | CONSUMED_RESEARCH_DATA")
        logger.warning(
            "Blind holdout: LOCKED | NOT LOADED | "
            "NOT REVEALED | NOT CONSUMED | NOT EVALUATED"
        )

        reference_base_records = []
        h8_base_records = []

        reference_rows: list[WindowMetrics] = []
        h8_rows: list[WindowMetrics] = []

        reference_max_dd = Decimal("0")
        h8_max_dd = Decimal("0")

        counterfactual = CounterfactualTotals()

        # --------------------------------------------------
        # BASE COST
        # --------------------------------------------------

        for window in eligible_windows:
            (
                reference_eval,
                reference_r,
                reference_summary,
                _,
            ) = _evaluate_window(
                window=window,
                strategy_config=strategy_config,
                backtest_config=base_config,
                use_h8=False,
            )

            expected = reference_windows[window.window_id]

            if len(reference_eval.backtest.trades) != expected:
                raise ValueError(
                    f"Frozen Q25 mismatch in {window.window_id}: "
                    f"expected {expected}, got "
                    f"{len(reference_eval.backtest.trades)}."
                )

            (
                h8_eval,
                h8_r,
                h8_summary,
                protection,
            ) = _evaluate_window(
                window=window,
                strategy_config=strategy_config,
                backtest_config=base_config,
                use_h8=True,
            )

            assert protection is not None

            reference_row = _window_metrics(
                window_id=window.window_id,
                variant="H5_Q25_REFERENCE",
                evaluation=reference_eval,
                summary=reference_summary,
                protection=None,
            )

            h8_row = _window_metrics(
                window_id=window.window_id,
                variant="H8",
                evaluation=h8_eval,
                summary=h8_summary,
                protection=protection,
            )

            reference_rows.append(reference_row)
            h8_rows.append(h8_row)

            reference_base_records.extend(reference_r)
            h8_base_records.extend(h8_r)

            reference_max_dd = max(
                reference_max_dd,
                reference_row.maximum_drawdown_percent,
            )

            h8_max_dd = max(
                h8_max_dd,
                h8_row.maximum_drawdown_percent,
            )

            _update_counterfactual(
                counterfactual,
                reference_evaluation=reference_eval,
                reference_records=reference_r,
                h8_evaluation=h8_eval,
                h8_records=h8_r,
                protection=protection,
            )

            logger.info(
                "%s | REF trades=%d net=%sR | "
                "H8 trades=%d net=%sR | activations=%d | BE exits=%d",
                window.window_id,
                reference_row.trades,
                reference_row.net_expectancy_r,
                h8_row.trades,
                h8_row.net_expectancy_r,
                h8_row.h8_activations,
                h8_row.break_even_exits,
            )

        if len(reference_base_records) != REFERENCE_TRADES:
            raise ValueError(
                "Frozen reference did not reproduce 197 trades."
            )

        reference_summary = summarize_r_normalized_trades(
            tuple(reference_base_records)
        )

        h8_summary = summarize_r_normalized_trades(
            tuple(h8_base_records)
        )

        stored_reference = v331["r_accounting"]["summary"]

        for field in (
            "frictionless_expectancy_r",
            "net_expectancy_r",
        ):
            expected = Decimal(str(stored_reference[field]))
            actual = getattr(reference_summary, field)

            if abs(actual - expected) > TOLERANCE:
                raise ValueError(
                    f"Frozen reference mismatch: {field}."
                )

        logger.info(
            "Frozen H5_Q25 197-trade R-reference VERIFIED."
        )

        # --------------------------------------------------
        # DOUBLED COST STRESS
        # --------------------------------------------------

        reference_stress_records = []
        h8_stress_records = []

        logger.info("Running doubled-cost stress replay.")

        for window in eligible_windows:
            _, reference_r, _, _ = _evaluate_window(
                window=window,
                strategy_config=strategy_config,
                backtest_config=stress_config,
                use_h8=False,
            )

            _, h8_r, _, _ = _evaluate_window(
                window=window,
                strategy_config=strategy_config,
                backtest_config=stress_config,
                use_h8=True,
            )

            reference_stress_records.extend(reference_r)
            h8_stress_records.extend(h8_r)

        reference_stress = summarize_r_normalized_trades(
            tuple(reference_stress_records)
        )

        h8_stress = summarize_r_normalized_trades(
            tuple(h8_stress_records)
        )

        stored_stress = Decimal(
            str(
                v33["candidates"]["H5_Q25"]["stress"][
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
                "Frozen doubled-cost reference mismatch."
            )

        # --------------------------------------------------
        # PREREGISTERED GATES
        # --------------------------------------------------

        reference_metrics = H8Metrics(
            trades=len(reference_base_records),
            frictionless_expectancy_r=(
                reference_summary.frictionless_expectancy_r
            ),
            net_expectancy_r=reference_summary.net_expectancy_r,
            profit_factor_r=reference_summary.profit_factor_r,
            maximum_drawdown_percent=reference_max_dd,
            stress_net_expectancy_r=(
                reference_stress.net_expectancy_r
            ),
        )

        h8_metrics = H8Metrics(
            trades=len(h8_base_records),
            frictionless_expectancy_r=(
                h8_summary.frictionless_expectancy_r
            ),
            net_expectancy_r=h8_summary.net_expectancy_r,
            profit_factor_r=h8_summary.profit_factor_r,
            maximum_drawdown_percent=h8_max_dd,
            stress_net_expectancy_r=h8_stress.net_expectancy_r,
        )

        reference_by_window = {
            row.window_id: row
            for row in reference_rows
        }

        comparisons = tuple(
            H8WindowComparison(
                h8_frictionless_r=(
                    row.frictionless_expectancy_r
                ),
                reference_frictionless_r=(
                    reference_by_window[
                        row.window_id
                    ].frictionless_expectancy_r
                ),
                h8_net_r=row.net_expectancy_r,
                reference_net_r=(
                    reference_by_window[
                        row.window_id
                    ].net_expectancy_r
                ),
                h8_net_positive=(
                    row.net_expectancy_r > 0
                ),
            )
            for row in h8_rows
        )

        gate = evaluate_h8_gate(
            h8=h8_metrics,
            reference=reference_metrics,
            windows=comparisons,
        )

        v34_eligible = h8_v34_eligible(
            h8=h8_metrics,
            gate=gate,
        )

        average_protective_exit_r = (
            counterfactual.protective_exit_net_r_total
            / Decimal(counterfactual.protective_stop_exits)
            if counterfactual.protective_stop_exits
            else Decimal("0")
        )

        counterfactual_payload = {
            "protective_stop_activated_trades": (
                counterfactual.activated_trades
            ),
            "protective_stop_exit_trades": (
                counterfactual.protective_stop_exits
            ),
            "matched_reference_h8_trades": (
                counterfactual.matched_trades
            ),
            "unmatched_reference_trades": (
                counterfactual.unmatched_reference_trades
            ),
            "unmatched_h8_trades": (
                counterfactual.unmatched_h8_trades
            ),
            "baseline_stop_losses_improved": (
                counterfactual.baseline_stop_losses_improved
            ),
            "baseline_take_profits_cut_early": (
                counterfactual.baseline_take_profits_cut_early
            ),
            "gross_R_saved_by_protection_on_matched_BE_exits": (
                counterfactual.gross_r_saved_by_protection
            ),
            "net_R_saved_by_protection_on_matched_BE_exits": (
                counterfactual.net_r_saved_by_protection
            ),
            "average_net_R_of_protective_exits": (
                average_protective_exit_r
            ),
        }

        # --------------------------------------------------
        # REPORTS
        # --------------------------------------------------

        output.mkdir(parents=True, exist_ok=True)

        csv_path = output / "window_comparison.csv"
        temp_csv = output / "window_comparison.csv.tmp"

        with temp_csv.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            fieldnames = tuple(
                asdict(reference_rows[0]).keys()
            )

            writer = csv.DictWriter(
                stream,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for row in (*reference_rows, *h8_rows):
                writer.writerow(
                    _json_safe(asdict(row))
                )

        temp_csv.replace(csv_path)

        summary_payload = {
            "version": "3.3.2",
            "run_id": EXPECTED_H8_RUN_ID,
            "source_v33_run_id": EXPECTED_V33_RUN_ID,
            "source_v331_run_id": EXPECTED_V331_RUN_ID,
            "dataset_status": "CONSUMED_RESEARCH_DATA",
            "eligible_windows": 11,
            "blind_holdout": {
                "status": "LOCKED_BLIND_HOLDOUT",
                "loaded": False,
                "revealed": False,
                "consumed": False,
                "evaluated": False,
            },
            "reference": {
                "candidate": "H5_Q25",
                "trades": len(reference_base_records),
                "base": asdict(reference_summary),
                "maximum_drawdown_percent": reference_max_dd,
                "stress": asdict(reference_stress),
            },
            "h8": {
                "definition": h8_manifest["definition"],
                "trades": len(h8_base_records),
                "base": asdict(h8_summary),
                "maximum_drawdown_percent": h8_max_dd,
                "stress": asdict(h8_stress),
            },
            "gate": asdict(gate),
            "all_support_gates_met": (
                gate.all_support_gates_met
            ),
            "v3_4_eligible": v34_eligible,
            "counterfactual": counterfactual_payload,
        }

        temporary_summary = output / "summary.json.tmp"

        temporary_summary.write_text(
            json.dumps(
                _json_safe(summary_payload),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_summary.replace(summary_path)

        # --------------------------------------------------
        # FINAL RESULT
        # --------------------------------------------------

        logger.info("")
        logger.info("H8 RESULTS")

        logger.info(
            "REFERENCE H5_Q25 | trades=%d | gross=%sR | "
            "net=%sR | PF=%s | stress=%sR",
            len(reference_base_records),
            reference_summary.frictionless_expectancy_r,
            reference_summary.net_expectancy_r,
            reference_summary.profit_factor_r,
            reference_stress.net_expectancy_r,
        )

        logger.info(
            "H8 | trades=%d | gross=%sR | net=%sR | "
            "PF=%s | stress=%sR",
            len(h8_base_records),
            h8_summary.frictionless_expectancy_r,
            h8_summary.net_expectancy_r,
            h8_summary.profit_factor_r,
            h8_stress.net_expectancy_r,
        )

        logger.info(
            "Consistency: gross=%d/11 | net=%d/11 | "
            "positive-net=%d/11 | trade-ratio=%s",
            gate.frictionless_better_windows,
            gate.net_better_windows,
            gate.positive_net_windows,
            gate.trade_count_ratio,
        )

        logger.info(
            "H8 activations=%d | BE exits=%d | "
            "baseline SL improved=%d | baseline TP cut early=%d",
            counterfactual.activated_trades,
            counterfactual.protective_stop_exits,
            counterfactual.baseline_stop_losses_improved,
            counterfactual.baseline_take_profits_cut_early,
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
        logger.error("H8 replay failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())