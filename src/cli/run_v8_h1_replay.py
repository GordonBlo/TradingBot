"""Execute the single preregistered V8-H1 open-interest replay."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from src.backtest.models import BacktestConfig
from src.cli.run_multiregime import _require_locked_holdout
from src.cli.run_v33_h5 import _build_regions, _read_reference_windows
from src.cli.run_v6_h0_replay import (
    evaluation_days_total,
    expand_complete_region_history,
    validate_eligible_windows as validate_v6_windows,
    validate_replay_configs as validate_v6_replay_configs,
)
from src.cli.run_v7_h0_replay import summarize_combined_records
from src.cli.run_v8_h0_replay import (
    EXPECTED_WINDOW_IDS,
    HOLDOUT_END,
    HOLDOUT_START,
    FrozenV6Candidate,
    V8WindowResult,
    _evaluate_window,
    _gate_payload,
    _load_json,
    build_window_result,
    doubled_cost_stress_records,
    load_frozen_v6_reference,
    validate_derivatives_manifest,
    write_reports,
)
from src.diagnostics.r_normalized import RNormalizedSummary, RNormalizedTrade
from src.hypotheses.manifest import HoldoutStatus, ResearchManifestStore
from src.research.multiregime.windows import construct_windows
from src.research.v6_mtf_continuation_preregistration import (
    build_manifest as build_v6_manifest,
)
from src.research.v8_funding_context_stability import (
    V8H0Metrics,
    V8H0WindowMetrics,
    evaluate_v8_h0_progression_gate,
    evaluate_v8_h0_support_gate,
)
from src.research.v8_open_interest_preregistration import (
    OPEN_INTEREST_LOOKBACK_MINUTES,
    build_manifest,
    open_interest_expansion_accepts,
    require_v8_h1_subset_of_frozen_v6,
)
from src.strategy.models import TrendMomentumConfig
from src.strategy.v8_funding_schedule import V8FundingScheduleStrategy
from src.utils.logger import configure_logging, get_logger


EXPECTED_RUN_ID = "6e7c9147b8c3187e"


@dataclass(frozen=True, slots=True)
class OpenInterestObservation:
    bucket_close_time: datetime
    sum_open_interest: Decimal | None
    source_timestamp: datetime | None


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    window_id: str
    signal_close_utc: str
    status: str
    oi_now: Decimal | None
    oi_now_timestamp_utc: str | None
    oi_60m: Decimal | None
    oi_60m_timestamp_utc: str | None


def _utc(value: str | datetime) -> datetime:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("V8-H1 timestamps must be timezone-aware.")
    return parsed.astimezone(timezone.utc)


def validate_v8_h1_manifest(payload: dict) -> None:
    if payload.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("Unexpected V8-H1 preregistration run ID.")
    if payload != build_manifest():
        raise ValueError("V8-H1 manifest differs from the frozen definition.")


def load_open_interest_context(
    dataset_manifest: dict,
) -> dict[datetime, OpenInterestObservation]:
    observations: dict[datetime, OpenInterestObservation] = {}
    for path_text in dataset_manifest.get("partition_paths", ()):
        path = Path(path_text)
        if not path.is_file():
            raise FileNotFoundError(f"V8 context partition is missing: {path}")
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                bucket_close = _utc(row["bucket_close_time"])
                if HOLDOUT_START <= bucket_close < HOLDOUT_END:
                    raise ValueError("V8 context partition intersects the blind holdout.")
                if bucket_close in observations:
                    raise ValueError("V8 context bucket close is duplicated.")
                quantity_text = row["open_interest"].strip()
                timestamp_text = row["open_interest_timestamp"].strip()
                observations[bucket_close] = OpenInterestObservation(
                    bucket_close_time=bucket_close,
                    sum_open_interest=(
                        Decimal(quantity_text) if quantity_text else None
                    ),
                    source_timestamp=(
                        _utc(timestamp_text) if timestamp_text else None
                    ),
                )
    if len(observations) != dataset_manifest["context_buckets"]:
        raise ValueError("V8 loaded context bucket count is not exact.")
    return observations


def filter_frozen_candidates(
    *,
    candidates: tuple[FrozenV6Candidate, ...],
    open_interest_by_bucket_close: dict[datetime, OpenInterestObservation],
) -> tuple[tuple[FrozenV6Candidate, ...], tuple[CandidateDecision, ...]]:
    retained = []
    decisions = []
    lookback = timedelta(minutes=OPEN_INTEREST_LOOKBACK_MINUTES)
    for candidate in candidates:
        current = open_interest_by_bucket_close.get(candidate.signal_close)
        historical = open_interest_by_bucket_close.get(
            candidate.signal_close - lookback
        )
        oi_now = current.sum_open_interest if current else None
        now_timestamp = current.source_timestamp if current else None
        oi_60m = historical.sum_open_interest if historical else None
        historical_timestamp = historical.source_timestamp if historical else None

        if oi_now is None or now_timestamp is None:
            status = "FILTERED_MISSING_CURRENT_OI"
        elif now_timestamp > candidate.signal_close:
            status = "FILTERED_FUTURE_CURRENT_OI"
        elif oi_60m is None or historical_timestamp is None:
            status = "FILTERED_MISSING_60M_OI"
        elif historical_timestamp > candidate.signal_close - lookback:
            status = "FILTERED_FUTURE_60M_OI"
        elif open_interest_expansion_accepts(
            oi_now=oi_now,
            oi_now_timestamp=now_timestamp,
            oi_60m=oi_60m,
            oi_60m_timestamp=historical_timestamp,
            signal_close=candidate.signal_close,
        ):
            status = "RETAINED_OI_EXPANDING_60M"
            retained.append(candidate)
        else:
            status = "FILTERED_OI_NOT_EXPANDING_60M"

        decisions.append(
            CandidateDecision(
                window_id=candidate.window_id,
                signal_close_utc=candidate.signal_close.isoformat(),
                status=status,
                oi_now=oi_now,
                oi_now_timestamp_utc=(
                    now_timestamp.isoformat() if now_timestamp else None
                ),
                oi_60m=oi_60m,
                oi_60m_timestamp_utc=(
                    historical_timestamp.isoformat()
                    if historical_timestamp
                    else None
                ),
            )
        )

    require_v8_h1_subset_of_frozen_v6(
        (row.signal_close for row in candidates),
        (row.signal_close for row in retained),
    )
    return tuple(retained), tuple(decisions)


def validate_replay_configs(
    *,
    manifest: dict,
    strategy_config: TrendMomentumConfig,
    base_config: BacktestConfig,
    stress_config: BacktestConfig,
) -> None:
    validate_v6_replay_configs(
        manifest=build_v6_manifest(),
        strategy_config=strategy_config,
        base_config=base_config,
        stress_config=stress_config,
    )
    for config, name in ((base_config, "base"), (stress_config, "stress")):
        expected = manifest["definition"]["costs"][name]
        if (
            config.fee_bps != Decimal(expected["fee_bps_per_side"])
            or config.slippage_bps != Decimal(expected["slippage_bps_per_side"])
        ):
            raise ValueError(f"Frozen V8-H1 {name} costs changed.")
    frozen = manifest["definition"]["execution_and_risk"]
    if (
        V8FundingScheduleStrategy.REWARD_RISK_RATIO != Decimal("2")
        or V8FundingScheduleStrategy.MAXIMUM_HOLD_BARS
        != frozen["maximum_hold_bars"]
        or V8FundingScheduleStrategy.COOLDOWN_BARS != frozen["cooldown_bars"]
        or frozen["entry_execution"] != "NEXT_BAR_OPEN"
        or frozen["ambiguous_bar_policy"] != "STOP_FIRST"
    ):
        raise ValueError("Frozen V8-H1 V6 execution/risk constants changed.")


def build_summary_payload(
    *,
    manifest: dict,
    dataset_manifest: dict,
    v6_summary: RNormalizedSummary,
    v8_summary: RNormalizedSummary,
    stress_summary: RNormalizedSummary,
    window_results: tuple[V8WindowResult, ...],
    candidate_decisions: tuple[CandidateDecision, ...],
    maximum_drawdown_percent: Decimal,
    evaluation_days: Decimal,
) -> dict:
    if (
        tuple(row.window_id for row in window_results) != EXPECTED_WINDOW_IDS
        or sum(row.v6_candidates for row in window_results)
        != v6_summary.trade_count
        or sum(row.v8_retained for row in window_results)
        != v8_summary.trade_count
    ):
        raise ValueError("V8-H1 combined records and windows are inconsistent.")
    non_v6_entries = sum(row.non_v6_entries for row in window_results)
    gate_windows = tuple(
        V8H0WindowMetrics(
            trades=row.v8_retained,
            net_result_r=(row.v8_net_result_r if row.v8_retained else None),
            v6_net_result_r=row.v6_net_result_r,
        )
        for row in window_results
    )
    metrics = V8H0Metrics(
        gross_expectancy_r=v8_summary.frictionless_expectancy_r,
        net_expectancy_r=v8_summary.net_expectancy_r,
        profit_factor_r=v8_summary.profit_factor_r,
        non_v6_entry_count=non_v6_entries,
    )
    support = evaluate_v8_h0_support_gate(metrics=metrics, windows=gate_windows)
    progression = evaluate_v8_h0_progression_gate(
        metrics=metrics, windows=gate_windows
    )
    classification = (
        "NOT_SUPPORTED"
        if not support.all_conditions_met
        else (
            "PROGRESSION_ELIGIBLE"
            if progression.all_conditions_met
            else "SUPPORTED_NOT_PROFITABLE"
        )
    )
    retained = v8_summary.trade_count
    v6_count = v6_summary.trade_count
    status_counts: dict[str, int] = {}
    for row in candidate_decisions:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1
    definition = manifest["definition"]
    return {
        "version": "8.1",
        "baseline_id": "V8_H1",
        "strategy_id": "V8_OPEN_INTEREST_FILTERED_FROZEN_V6_SCHEDULE",
        "run_id": EXPECTED_RUN_ID,
        "derivatives_dataset": {
            "dataset_id": dataset_manifest["dataset_id"],
            "definition_sha256": dataset_manifest["dataset_definition_sha256"],
            "classification": dataset_manifest["classification"],
            "context_buckets": dataset_manifest["context_buckets"],
            "future_data_violations": dataset_manifest["future_data_violations"],
        },
        "market": definition["market"],
        "costs": definition["costs"],
        "eligible_windows": list(EXPECTED_WINDOW_IDS),
        "window_results_csv": "window_results.csv",
        "candidate_decisions_csv": "candidate_decisions.csv",
        "v6_frozen_reference": {
            **definition["frozen_v6_reference"],
            "candidate_count": v6_count,
            "metrics": asdict(v6_summary),
        },
        "v8_base": {
            "v6_candidate_count": v6_count,
            "retained_candidate_count": retained,
            "filtered_candidate_count": v6_count - retained,
            "retention_ratio": Decimal(retained) / Decimal(v6_count),
            "non_v6_entry_count": non_v6_entries,
            "gross_expectancy_r_per_trade": v8_summary.frictionless_expectancy_r,
            "net_expectancy_r_per_trade": v8_summary.net_expectancy_r,
            "profit_factor_r": v8_summary.profit_factor_r,
            "win_rate_percent": v8_summary.win_rate_percent,
            "maximum_drawdown_percent": maximum_drawdown_percent,
            "positive_net_windows": sum(row.positive_net for row in window_results),
            "net_better_windows": sum(
                row.net_result_better_than_v6 for row in window_results
            ),
            "trades_per_day": Decimal(retained) / evaluation_days,
            "net_r_per_day": (
                v8_summary.net_expectancy_r * Decimal(retained) / evaluation_days
            ),
            "metrics": asdict(v8_summary),
        },
        "comparison_vs_v6": {
            "gross_expectancy_v6": v6_summary.frictionless_expectancy_r,
            "gross_expectancy_v8": v8_summary.frictionless_expectancy_r,
            "net_expectancy_v6": v6_summary.net_expectancy_r,
            "net_expectancy_v8": v8_summary.net_expectancy_r,
            "profit_factor_v6": v6_summary.profit_factor_r,
            "profit_factor_v8": v8_summary.profit_factor_r,
            "net_better_windows": sum(
                row.net_result_better_than_v6 for row in window_results
            ),
        },
        "candidate_filtering": {
            "status_counts": status_counts,
            "rule": "SUM_OPEN_INTEREST_T_STRICTLY_GREATER_THAN_T_MINUS_60M",
            "field": "SUM_OPEN_INTEREST",
            "open_interest_value_used": False,
            "missing_or_future_rejected": True,
            "alternative_horizons_thresholds_or_variants_tested": False,
        },
        "subset_diagnostics": {
            "v6_reference_entries": v6_count,
            "v8_retained_entries": retained,
            "filtered_v6_entries": v6_count - retained,
            "non_v6_entries": non_v6_entries,
            "required_non_v6_entries": 0,
        },
        "stress_2x": {
            "fee_bps_per_side": Decimal("20"),
            "slippage_bps_per_side": Decimal("4"),
            "frozen_stop_floor_bps": Decimal("96"),
            "trade_count": stress_summary.trade_count,
            "net_expectancy_r_per_trade": stress_summary.net_expectancy_r,
            "profit_factor_r": stress_summary.profit_factor_r,
            "method": "FIXED_BASE_TRADE_PATH_DOUBLE_RECORDED_FRICTION",
            "candidate_schedule_and_exit_path_frozen": True,
            "reporting_only_not_gate": True,
        },
        "support_gate": _gate_payload(support),
        "progression_gate": _gate_payload(progression),
        "blind_holdout_integrity": definition["data_policy"]["blind_holdout"],
        "final_classification": classification,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the single preregistered V8-H1 replay."
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--manifest",
        default="research/v8_open_interest/6e7c9147b8c3187e/manifest.json",
    )
    parser.add_argument(
        "--dataset-manifest",
        default=(
            "data/derivatives/processed/15m/BTCUSDT/eec764735d270f9d/manifest.json"
        ),
    )
    parser.add_argument(
        "--v6-summary",
        default="reports/v6_mtf_continuation/e1eef7bdd0c37ad4/summary.json",
    )
    parser.add_argument(
        "--v6-windows",
        default="reports/v6_mtf_continuation/e1eef7bdd0c37ad4/window_results.csv",
    )
    parser.add_argument(
        "--v6-schedule",
        default=(
            "reports/diagnostics/v6_signal_quality/e1eef7bdd0c37ad4/trade_ledger.csv"
        ),
    )
    parser.add_argument("--root-manifest", default="research/hypothesis_manifest.json")
    parser.add_argument(
        "--mechanism-report", default="reports/mechanisms/bc2496aed05555b5"
    )
    parser.add_argument(
        "--expansion-data-root", default="data/historical/multiregime_expansion"
    )
    parser.add_argument("--consumed-data-root", default="data/historical")
    parser.add_argument("--output-root", default="reports/v8_open_interest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    logger = get_logger(__name__)
    if not args.execute:
        logger.error("V8-H1 replay not executed. Explicit --execute is required.")
        return 1

    try:
        manifest = _load_json(args.manifest)
        dataset_manifest = _load_json(args.dataset_manifest)
        validate_v8_h1_manifest(manifest)
        validate_derivatives_manifest(dataset_manifest)
        output = Path(args.output_root) / EXPECTED_RUN_ID
        if (output / "summary.json").exists():
            raise ValueError("V8-H1 result already exists; refusing overwrite or rerun.")

        root_manifest = ResearchManifestStore(args.root_manifest).load()
        _require_locked_holdout(root_manifest)
        if (
            root_manifest.holdout_status is not HoldoutStatus.LOCKED_BLIND_HOLDOUT
            or root_manifest.blind_holdout.reveal_timestamp is not None
            or root_manifest.blind_holdout.consumed_timestamp is not None
        ):
            raise ValueError("Blind holdout contamination detected.")

        strategy_config = TrendMomentumConfig()
        base_config = BacktestConfig()
        stress_config = replace(
            base_config, fee_bps=Decimal("20"), slippage_bps=Decimal("4")
        )
        validate_replay_configs(
            manifest=manifest,
            strategy_config=strategy_config,
            base_config=base_config,
            stress_config=stress_config,
        )
        v6_summary, v6_windows, candidates = load_frozen_v6_reference(
            summary_path=args.v6_summary,
            windows_path=args.v6_windows,
            schedule_path=args.v6_schedule,
            manifest=manifest["definition"],
        )
        open_interest = load_open_interest_context(dataset_manifest)
        retained, candidate_decisions = filter_frozen_candidates(
            candidates=candidates,
            open_interest_by_bucket_close=open_interest,
        )

        from src.config.settings import load_settings

        regions = _build_regions(
            root_manifest=root_manifest,
            expansion_data_root=args.expansion_data_root,
            consumed_data_root=args.consumed_data_root,
        )
        reference_windows = _read_reference_windows(Path(args.mechanism_report))
        if tuple(sorted(reference_windows)) != EXPECTED_WINDOW_IDS:
            raise ValueError("V8-H1 requires exactly the frozen 11 consumed windows.")
        windows = construct_windows(regions, load_settings().multiregime_config())
        eligible = tuple(
            window for window in windows if window.window_id in reference_windows
        )
        eligible_windows = expand_complete_region_history(eligible, regions)
        validate_v6_windows(eligible_windows)
        if tuple(window.window_id for window in eligible_windows) != EXPECTED_WINDOW_IDS:
            raise ValueError("V8-H1 eligible window order or identity changed.")

        base_groups: list[tuple[RNormalizedTrade, ...]] = []
        stress_groups: list[tuple[RNormalizedTrade, ...]] = []
        window_results = []
        maximum_drawdown = Decimal("0")
        for window in eligible_windows:
            window_candidates = tuple(
                row for row in retained if row.window_id == window.window_id
            )
            evaluation, records, summary = _evaluate_window(
                window=window,
                candidates=window_candidates,
                strategy_config=strategy_config,
                backtest_config=base_config,
            )
            row = build_window_result(
                window=window,
                v6=v6_windows[window.window_id],
                retained_count=len(window_candidates),
                evaluation=evaluation,
                summary=summary,
            )
            window_results.append(row)
            base_groups.append(records)
            maximum_drawdown = max(
                maximum_drawdown, row.v8_maximum_drawdown_percent
            )
            stress_groups.append(doubled_cost_stress_records(records))

        v8_summary = summarize_combined_records(tuple(base_groups))
        stress_summary = summarize_combined_records(tuple(stress_groups))
        summary_payload = build_summary_payload(
            manifest=manifest,
            dataset_manifest=dataset_manifest,
            v6_summary=v6_summary,
            v8_summary=v8_summary,
            stress_summary=stress_summary,
            window_results=tuple(window_results),
            candidate_decisions=candidate_decisions,
            maximum_drawdown_percent=maximum_drawdown,
            evaluation_days=evaluation_days_total(eligible_windows),
        )
        paths = write_reports(
            output=output,
            summary_payload=summary_payload,
            window_results=tuple(window_results),
            candidate_decisions=candidate_decisions,
        )
        logger.info(
            "V8-H1 classification: %s", summary_payload["final_classification"]
        )
        logger.info("Reports: %s", ", ".join(str(path.resolve()) for path in paths))
        logger.warning(
            "Blind holdout: LOCKED | NOT DOWNLOADED | NOT LOADED | "
            "NOT REVEALED | NOT CONSUMED | NOT EVALUATED"
        )
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.error("V8-H1 replay failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
