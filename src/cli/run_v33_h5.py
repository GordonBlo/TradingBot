"""Run preregistered V3.3 H5 parameter research on consumed data only."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
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
from src.diagnostics.r_normalized import (
    RNormalizedSummary,
    build_r_normalized_trades,
    summarize_r_normalized_trades,
)
from src.diagnostics.risk_capital_audit import (
    build_risk_capital_audit,
)
from src.diagnostics.loader import ResearchRunLoader
from src.hypotheses.h5_parameter_manifest import (
    H5ParameterSupportCriteria,
)
from src.hypotheses.h5_parameter_research import (
    H5ParameterCandidateId,
    build_h5_parameter_candidate,
    h5_parameter_registry,
    validate_h5_parameter_registry,
)
from src.hypotheses.manifest import (
    HoldoutStatus,
    ResearchManifestStore,
)
from src.research.evaluation import evaluate_strategy_period
from src.research.h5_parameter_stability import (
    H5ParameterClassification,
    H5ParameterMetrics,
    H5WindowComparison,
    classify_h5_parameter_candidate,
    evaluate_h5_parameter_gate,
)
from src.research.multiregime.models import (
    PartitionKind,
    ResearchRegion,
)
from src.research.multiregime.windows import construct_windows
from src.utils.logger import configure_logging, get_logger


EXPECTED_V33_RUN_ID = "b104ea78b4c88bcf"
EXPECTED_REFERENCE_TRADES = 197


@dataclass(frozen=True, slots=True)
class WindowResult:
    window_id: str
    candidate_id: H5ParameterCandidateId
    trades: int
    frictionless_expectancy_r: Decimal
    net_expectancy_r: Decimal
    profit_factor_r: Decimal | None
    maximum_drawdown_percent: Decimal


@dataclass(frozen=True, slots=True)
class CandidateResult:
    candidate_id: H5ParameterCandidateId
    trades: int
    summary: RNormalizedSummary
    maximum_drawdown_percent: Decimal
    stress_summary: RNormalizedSummary


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


def _max_drawdown_percent(equity_curve) -> Decimal:
    if not equity_curve:
        return Decimal("0")

    peak = equity_curve[0].total_equity
    maximum = Decimal("0")

    for point in equity_curve:
        equity = point.total_equity

        if equity > peak:
            peak = equity

        if peak <= 0:
            continue

        drawdown = (
            (peak - equity)
            / peak
            * Decimal("100")
        )

        if drawdown > maximum:
            maximum = drawdown

    return maximum


def _read_reference_windows(
    report_directory: Path,
) -> dict[str, int]:
    path = report_directory / "window_comparison.csv"

    if not path.is_file():
        raise ValueError(
            f"Frozen V3.2.2 window report missing: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        rows = tuple(csv.DictReader(stream))

    eligible = {
        row["window_id"]
        for row in rows
        if row["hypothesis"] == "H0"
        and row["eligible"].strip().lower() == "true"
    }

    reference = {
        row["window_id"]: int(row["trades"])
        for row in rows
        if row["hypothesis"] == "H5"
        and row["window_id"] in eligible
    }

    if len(reference) != 11:
        raise ValueError(
            f"Expected 11 frozen eligible windows, got {len(reference)}."
        )

    if sum(reference.values()) != EXPECTED_REFERENCE_TRADES:
        raise ValueError(
            "Frozen H5 reference no longer reproduces "
            f"{EXPECTED_REFERENCE_TRADES} trades."
        )

    return reference


def _load_v33_manifest(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(
            f"V3.3 preregistration manifest missing: {path}"
        )

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if payload.get("run_id") != EXPECTED_V33_RUN_ID:
        raise ValueError(
            "Unexpected V3.3 preregistration run ID."
        )

    policy = payload.get("holdout_policy", {})

    if policy.get("required_status") != "LOCKED_BLIND_HOLDOUT":
        raise ValueError("V3.3 holdout policy is invalid.")

    if any(
        bool(policy.get(name))
        for name in ("revealed", "consumed", "evaluated")
    ):
        raise ValueError(
            "V3.3 preregistration indicates holdout contamination."
        )

    candidate_ids = [
        item["candidate_id"]
        for item in payload.get("candidates", [])
    ]

    if candidate_ids != [
        "H5_Q25",
        "H5_Q35",
        "H5_Q45",
        "H5_Q50",
    ]:
        raise ValueError(
            "V3.3 candidate set differs from preregistration."
        )

    return payload


def _evaluate_window(
    *,
    window,
    candidate_id: H5ParameterCandidateId,
    strategy_config,
    backtest_config,
):
    strategy = build_h5_parameter_candidate(
        candidate_id,
        strategy_config,
    )

    evaluation = evaluate_strategy_period(
        window.replay_dataset,
        strategy=strategy,
        strategy_config=strategy_config,
        backtest_config=backtest_config,
        evaluation_start_index=window.evaluation_start_index,
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
            f"R record mismatch in {window.window_id} / "
            f"{candidate_id.value}."
        )

    return (
        evaluation,
        r_records,
        summarize_r_normalized_trades(r_records),
    )


def _build_regions(
    *,
    root_manifest,
    expansion_data_root: str,
    consumed_data_root: str,
):
    expansion_segments = _load_expansion(
        expansion_data_root
    )

    consumed = _load_exact_range(
        consumed_data_root,
        CONSUMED_START,
        CONSUMED_END,
    )

    _assert_holdout_absent(
        *expansion_segments,
        consumed,
    )

    partitions = _partition_metadata(
        expansion_segments=expansion_segments,
        consumed=consumed,
        expansion_data_root=expansion_data_root,
        consumed_data_root=consumed_data_root,
        holdout_registered_at=(
            root_manifest.blind_holdout.registered_at
        ),
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

    return (
        *(
            ResearchRegion(partition, segment)
            for partition, segment in zip(
                expansion_partitions,
                expansion_segments,
                strict=True,
            )
        ),
        ResearchRegion(
            consumed_partition,
            consumed,
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run preregistered V3.3 controlled H5 research."
        )
    )

    parser.add_argument(
        "--root-manifest",
        default="research/hypothesis_manifest.json",
    )

    parser.add_argument(
        "--v33-manifest",
        default=(
            "research/v3_3_h5/"
            "b104ea78b4c88bcf/"
            "manifest.json"
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
        default="reports/mechanisms/bc2496aed05555b5",
    )

    parser.add_argument(
        "--reference-audit",
        default=(
            "reports/audits/"
            "h5_risk_capital_v322/"
            "summary.json"
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
        default="reports/v3_3_h5",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate V3.3 preregistration, data boundaries, "
            "reference windows and holdout integrity without "
            "running candidate backtests."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    configure_logging()
    logger = get_logger(__name__)

    try:
        validate_h5_parameter_registry()

        preregistration = _load_v33_manifest(
            Path(args.v33_manifest)
        )

        root_store = ResearchManifestStore(
            args.root_manifest
        )

        root_manifest = root_store.load()

        _require_locked_holdout(root_manifest)

        if (
            root_manifest.holdout_status
            is not HoldoutStatus.LOCKED_BLIND_HOLDOUT
        ):
            raise ValueError(
                "Blind holdout is not locked."
            )

        holdout = root_manifest.blind_holdout

        if (
            holdout.reveal_timestamp is not None
            or holdout.consumed_timestamp is not None
        ):
            raise ValueError(
                "Blind holdout has been contaminated."
            )

        reference_windows = _read_reference_windows(
            Path(args.mechanism_report)
        )

        previous = ResearchRunLoader(
            research_root=args.research_root,
            data_root=args.consumed_data_root,
        ).load(args.research_run)

        strategy_config = previous.strategy_config
        base_config = previous.backtest_config

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

        from src.config.settings import load_settings

        settings = load_settings()

        regions = _build_regions(
            root_manifest=root_manifest,
            expansion_data_root=args.expansion_data_root,
            consumed_data_root=args.consumed_data_root,
        )

        windows = construct_windows(
            regions,
            settings.multiregime_config(),
        )

        eligible_windows = tuple(
            window
            for window in windows
            if window.window_id in reference_windows
        )

        if len(eligible_windows) != 11:
            raise ValueError(
                f"Expected 11 replay windows, got "
                f"{len(eligible_windows)}."
            )

        if args.dry_run:
            logger.info("V3.3 DRY-RUN INTEGRITY CHECK")
            logger.info(
                "Preregistration run: %s",
                preregistration["run_id"],
            )
            logger.info(
                "Candidates: H5_Q25, H5_Q35, H5_Q45, H5_Q50"
            )
            logger.info(
                "Eligible consumed windows: %d",
                len(eligible_windows),
            )
            logger.info(
                "Frozen H5_Q25 reference trades: %d",
                sum(reference_windows.values()),
            )
            logger.info(
                "Data status: CONSUMED_RESEARCH_DATA"
            )
            logger.warning(
                "Blind holdout: LOCKED | NOT LOADED | "
                "NOT REVEALED | NOT CONSUMED | NOT EVALUATED"
            )
            logger.info(
                "DRY RUN PASSED — NO PARAMETER REPLAY EXECUTED"
            )
            return 0

        logger.info("V3.3 CONTROLLED H5 PARAMETER RESEARCH")
        logger.info(
            "Candidates: H5_Q25, H5_Q35, H5_Q45, H5_Q50"
        )
        logger.info(
            "Data: CONSUMED_RESEARCH_DATA | windows: 11"
        )
        logger.warning(
            "Blind holdout: LOCKED | NOT LOADED | NOT EVALUATED"
        )

        candidate_base_records = {}
        candidate_stress_records = {}

        base_windows: dict[
            H5ParameterCandidateId,
            list[WindowResult],
        ] = {}

        max_dd: dict[
            H5ParameterCandidateId,
            Decimal,
        ] = {}

        # -------------------------------------------------
        # BASE COST REPLAY
        # -------------------------------------------------

        for definition in h5_parameter_registry():
            candidate_id = definition.candidate_id

            logger.info(
                "BASE replay: %s",
                candidate_id.value,
            )

            combined_records = []
            results: list[WindowResult] = []
            candidate_max_dd = Decimal("0")

            for window in eligible_windows:
                evaluation, r_records, summary = _evaluate_window(
                    window=window,
                    candidate_id=candidate_id,
                    strategy_config=strategy_config,
                    backtest_config=base_config,
                )

                trades = len(evaluation.backtest.trades)

                if candidate_id is H5ParameterCandidateId.H5_Q25:
                    expected = reference_windows[
                        window.window_id
                    ]

                    if trades != expected:
                        raise ValueError(
                            f"Frozen Q25 mismatch in "
                            f"{window.window_id}: "
                            f"expected {expected}, got {trades}."
                        )

                dd = _max_drawdown_percent(
                    evaluation.backtest.equity_curve
                )

                candidate_max_dd = max(
                    candidate_max_dd,
                    dd,
                )

                combined_records.extend(r_records)

                results.append(
                    WindowResult(
                        window_id=window.window_id,
                        candidate_id=candidate_id,
                        trades=trades,
                        frictionless_expectancy_r=(
                            summary.frictionless_expectancy_r
                        ),
                        net_expectancy_r=(
                            summary.net_expectancy_r
                        ),
                        profit_factor_r=(
                            summary.profit_factor_r
                        ),
                        maximum_drawdown_percent=dd,
                    )
                )

                logger.info(
                    "%s | %s | trades=%d | "
                    "gross=%sR | net=%sR",
                    window.window_id,
                    candidate_id.value,
                    trades,
                    summary.frictionless_expectancy_r,
                    summary.net_expectancy_r,
                )

            candidate_base_records[candidate_id] = tuple(
                combined_records
            )

            base_windows[candidate_id] = results
            max_dd[candidate_id] = candidate_max_dd

        q25_records = candidate_base_records[
            H5ParameterCandidateId.H5_Q25
        ]

        if len(q25_records) != EXPECTED_REFERENCE_TRADES:
            raise ValueError(
                "Frozen H5_Q25 did not reproduce 197 trades."
            )

        # -------------------------------------------------
        # REFERENCE R-AUDIT REPRODUCTION
        # -------------------------------------------------

        reference_payload = json.loads(
            Path(args.reference_audit).read_text(
                encoding="utf-8"
            )
        )

        stored_r = reference_payload.get(
            "r_normalized_audit",
            {},
        )

        q25_summary = summarize_r_normalized_trades(
            q25_records
        )

        if int(stored_r.get("trade_count", -1)) != 197:
            raise ValueError(
                "Stored V3.2.2 R-audit reference is invalid."
            )

        for field in (
            "frictionless_expectancy_r",
            "net_expectancy_r",
        ):
            expected = Decimal(str(stored_r[field]))
            actual = getattr(q25_summary, field)

            if abs(actual - expected) > Decimal("1e-18"):
                raise ValueError(
                    f"Frozen Q25 R metric mismatch: {field}."
                )

        logger.info(
            "Frozen H5_Q25 R-reference VERIFIED."
        )

        # -------------------------------------------------
        # DOUBLED COST REPLAY
        # -------------------------------------------------

        for definition in h5_parameter_registry():
            candidate_id = definition.candidate_id

            logger.info(
                "STRESS replay: %s",
                candidate_id.value,
            )

            combined_records = []

            for window in eligible_windows:
                _, r_records, _ = _evaluate_window(
                    window=window,
                    candidate_id=candidate_id,
                    strategy_config=strategy_config,
                    backtest_config=stress_config,
                )

                combined_records.extend(r_records)

            candidate_stress_records[candidate_id] = tuple(
                combined_records
            )

        # -------------------------------------------------
        # COMBINED METRICS
        # -------------------------------------------------

        combined: dict[
            H5ParameterCandidateId,
            CandidateResult,
        ] = {}

        for definition in h5_parameter_registry():
            candidate_id = definition.candidate_id

            base_records = candidate_base_records[
                candidate_id
            ]

            stress_records = candidate_stress_records[
                candidate_id
            ]

            combined[candidate_id] = CandidateResult(
                candidate_id=candidate_id,
                trades=len(base_records),
                summary=summarize_r_normalized_trades(
                    base_records
                ),
                maximum_drawdown_percent=max_dd[
                    candidate_id
                ],
                stress_summary=(
                    summarize_r_normalized_trades(
                        stress_records
                    )
                ),
            )

        reference = combined[
            H5ParameterCandidateId.H5_Q25
        ]

        reference_metrics = H5ParameterMetrics(
            trades=reference.trades,
            frictionless_expectancy_r=(
                reference.summary.frictionless_expectancy_r
            ),
            net_expectancy_r=(
                reference.summary.net_expectancy_r
            ),
            profit_factor_r=(
                reference.summary.profit_factor_r
            ),
            maximum_drawdown_percent=(
                reference.maximum_drawdown_percent
            ),
            stress_net_expectancy_r=(
                reference.stress_summary.net_expectancy_r
            ),
        )

        criteria = H5ParameterSupportCriteria()

        classifications = {}
        gate_results = {}

        reference_window_map = {
            row.window_id: row
            for row in base_windows[
                H5ParameterCandidateId.H5_Q25
            ]
        }

        for definition in h5_parameter_registry():
            candidate_id = definition.candidate_id
            result = combined[candidate_id]

            metrics = H5ParameterMetrics(
                trades=result.trades,
                frictionless_expectancy_r=(
                    result.summary.frictionless_expectancy_r
                ),
                net_expectancy_r=(
                    result.summary.net_expectancy_r
                ),
                profit_factor_r=(
                    result.summary.profit_factor_r
                ),
                maximum_drawdown_percent=(
                    result.maximum_drawdown_percent
                ),
                stress_net_expectancy_r=(
                    result.stress_summary.net_expectancy_r
                ),
            )

            comparisons = tuple(
                H5WindowComparison(
                    candidate_frictionless_r=(
                        row.frictionless_expectancy_r
                    ),
                    reference_frictionless_r=(
                        reference_window_map[
                            row.window_id
                        ].frictionless_expectancy_r
                    ),
                    candidate_net_r=(
                        row.net_expectancy_r
                    ),
                    reference_net_r=(
                        reference_window_map[
                            row.window_id
                        ].net_expectancy_r
                    ),
                    candidate_net_positive=(
                        row.net_expectancy_r > 0
                    ),
                )
                for row in base_windows[candidate_id]
            )

            gate = evaluate_h5_parameter_gate(
                candidate=metrics,
                reference=reference_metrics,
                windows=comparisons,
                criteria=criteria,
            )

            classification = (
                classify_h5_parameter_candidate(
                    candidate_id=candidate_id,
                    candidate=metrics,
                    reference=reference_metrics,
                    gate=gate,
                )
            )

            gate_results[candidate_id] = gate
            classifications[candidate_id] = classification

        # -------------------------------------------------
        # REPORTS
        # -------------------------------------------------

        output = (
            Path(args.output_root)
            / preregistration["run_id"]
        )

        output.mkdir(
            parents=True,
            exist_ok=True,
        )

        window_csv = output / "window_comparison.csv"

        with window_csv.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as stream:
            fieldnames = (
                "window_id",
                "candidate_id",
                "trades",
                "frictionless_expectancy_r",
                "net_expectancy_r",
                "profit_factor_r",
                "maximum_drawdown_percent",
            )

            writer = csv.DictWriter(
                stream,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            for candidate_id, rows in base_windows.items():
                for row in rows:
                    writer.writerow(
                        _json_safe(asdict(row))
                    )

        summary_payload = {
            "version": "3.3",
            "run_id": preregistration["run_id"],
            "dataset_status": "CONSUMED_RESEARCH_DATA",
            "blind_holdout": {
                "status": "LOCKED_BLIND_HOLDOUT",
                "revealed": False,
                "consumed": False,
                "evaluated": False,
            },
            "reference": "H5_Q25",
            "candidates": {},
        }

        logger.info("")
        logger.info("V3.3 RESULTS")

        for definition in h5_parameter_registry():
            candidate_id = definition.candidate_id
            result = combined[candidate_id]
            gate = gate_results[candidate_id]
            classification = classifications[candidate_id]

            candidate_payload = {
                "definition": asdict(definition),
                "trades": result.trades,
                "base": asdict(result.summary),
                "maximum_drawdown_percent": (
                    result.maximum_drawdown_percent
                ),
                "stress": asdict(
                    result.stress_summary
                ),
                "gate": asdict(gate),
                "classification": (
                    classification.value
                ),
            }

            summary_payload["candidates"][
                candidate_id.value
            ] = candidate_payload

            logger.info(
                "%s | trades=%d | gross=%sR | "
                "net=%sR | PF=%s | stress=%sR | %s",
                candidate_id.value,
                result.trades,
                result.summary.frictionless_expectancy_r,
                result.summary.net_expectancy_r,
                result.summary.profit_factor_r,
                result.stress_summary.net_expectancy_r,
                classification.value,
            )

            logger.info(
                "  consistency gross=%d/%d | "
                "net=%d/%d | positive-net=%d/%d | "
                "trade-ratio=%s",
                gate.frictionless_better_windows,
                gate.eligible_windows,
                gate.net_better_windows,
                gate.eligible_windows,
                gate.positive_net_windows,
                gate.eligible_windows,
                gate.trade_count_ratio,
            )

        summary_path = output / "summary.json"

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
        logger.info(
            "Report: %s",
            summary_path.resolve(),
        )

        logger.warning(
            "Blind holdout: LOCKED | NOT REVEALED | "
            "NOT CONSUMED | NOT EVALUATED"
        )

        eligible = [
            candidate_id.value
            for candidate_id, classification
            in classifications.items()
            if classification
            is H5ParameterClassification.V3_4_ELIGIBLE
        ]

        logger.info(
            "V3.4 eligible candidates: %s",
            ", ".join(eligible)
            if eligible
            else "NONE",
        )

        return 0

    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        logger.error("V3.3 replay failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())