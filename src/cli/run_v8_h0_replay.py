"""Execute the single preregistered V8-H0 funding-context replay."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from src.backtest.models import BacktestConfig
from src.cli.run_multiregime import _require_locked_holdout
from src.cli.run_v33_h5 import (
    _build_regions,
    _max_drawdown_percent,
    _read_reference_windows,
)
from src.cli.run_v6_h0_replay import (
    StopObservation,
    _risk_audit_signals,
    classify_stop_source,
    evaluation_days_total,
    expand_complete_region_history,
    validate_eligible_windows as validate_v6_windows,
    validate_replay_configs as validate_v6_replay_configs,
)
from src.cli.run_v7_h0_replay import summarize_combined_records
from src.diagnostics.r_normalized import (
    RNormalizedSummary,
    RNormalizedTrade,
    build_r_normalized_trades,
    summarize_r_normalized_trades,
)
from src.diagnostics.risk_capital_audit import build_risk_capital_audit
from src.hypotheses.manifest import HoldoutStatus, ResearchManifestStore
from src.research.evaluation import evaluate_strategy_period
from src.research.multiregime.windows import construct_windows
from src.research.v6_mtf_continuation_preregistration import (
    build_manifest as build_v6_manifest,
)
from src.research.v8_funding_context_preregistration import (
    V8_DERIVATIVES_DATASET_DEFINITION_SHA256,
    V8_DERIVATIVES_DATASET_ID,
    build_manifest,
    funding_filter_accepts,
    require_v8_subset_of_frozen_v6,
)
from src.research.v8_funding_context_stability import (
    V8H0Metrics,
    V8H0WindowMetrics,
    evaluate_v8_h0_progression_gate,
    evaluate_v8_h0_support_gate,
)
from src.strategy.models import StrategyAction, TrendMomentumConfig
from src.strategy.v8_funding_schedule import V8FundingScheduleStrategy
from src.utils.logger import configure_logging, get_logger


EXPECTED_RUN_ID = "089b62c96a7de979"
V6_RUN_ID = "e1eef7bdd0c37ad4"
EXPECTED_WINDOW_IDS = (
    "W002",
    "W003",
    "W004",
    "W005",
    "W006",
    "W007",
    "W008",
    "W009",
    "W010",
    "W012",
    "W013",
)
HOLDOUT_START = datetime(2025, 8, 1, tzinfo=timezone.utc)
HOLDOUT_END = datetime(2026, 2, 1, tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class FrozenV6Candidate:
    window_id: str
    signal_close: datetime
    atr14_4h: Decimal


@dataclass(frozen=True, slots=True)
class FundingObservation:
    bucket_close_time: datetime
    funding_rate: Decimal | None
    funding_timestamp: datetime | None


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    window_id: str
    signal_close_utc: str
    status: str
    funding_rate: Decimal | None
    funding_timestamp_utc: str | None


@dataclass(frozen=True, slots=True)
class FrozenV6Window:
    window_id: str
    trades: int
    gross_expectancy_r: Decimal
    net_expectancy_r: Decimal
    profit_factor_r: Decimal | None

    @property
    def net_result_r(self) -> Decimal:
        return self.net_expectancy_r * Decimal(self.trades)


@dataclass(frozen=True, slots=True)
class V8WindowResult:
    window_id: str
    status: str
    evaluation_days: Decimal
    v6_candidates: int
    v8_retained: int
    filtered_candidates: int
    non_v6_entries: int
    v6_gross_expectancy_r: Decimal
    v6_net_expectancy_r: Decimal
    v6_profit_factor_r: Decimal | None
    v6_net_result_r: Decimal
    v8_gross_expectancy_r: Decimal
    v8_net_expectancy_r: Decimal
    v8_profit_factor_r: Decimal | None
    v8_net_result_r: Decimal
    v8_win_rate_percent: Decimal
    v8_maximum_drawdown_percent: Decimal
    positive_net: bool
    net_result_better_than_v6: bool


def _utc(value: str | datetime) -> datetime:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("V8-H0 timestamps must be timezone-aware.")
    return parsed.astimezone(timezone.utc)


def _load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required V8-H0 file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_v8_manifest(payload: dict) -> None:
    if payload.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("Unexpected V8-H0 preregistration run ID.")
    if payload != build_manifest():
        raise ValueError("V8-H0 manifest differs from the frozen definition.")


def validate_derivatives_manifest(payload: dict) -> None:
    if (
        payload.get("dataset_id") != V8_DERIVATIVES_DATASET_ID
        or payload.get("dataset_definition_sha256")
        != V8_DERIVATIVES_DATASET_DEFINITION_SHA256
        or payload.get("classification") == "INCOMPLETE"
        or payload.get("required_context_buckets") != 95040
        or payload.get("context_buckets") != 95040
        or payload.get("future_data_violations") != 0
        or tuple(payload.get("window_ids", ())) != EXPECTED_WINDOW_IDS
    ):
        raise ValueError("V8 derivatives dataset identity or integrity changed.")
    holdout = payload.get("blind_holdout", {})
    if (
        holdout.get("status") != "LOCKED_BLIND_HOLDOUT"
        or holdout.get("intersection") != "NONE"
        or any(
            holdout.get(name) is not False
            for name in ("downloaded", "loaded", "revealed", "consumed", "evaluated")
        )
    ):
        raise ValueError("V8 derivatives dataset holdout contamination detected.")
    integrity = payload.get("integrity", {})
    if integrity.get("future_fill") is not False or integrity.get("interpolation") is not False:
        raise ValueError("V8 derivatives dataset causal policy changed.")


def _summary_from_payload(payload: dict) -> RNormalizedSummary:
    fields = payload["aggregate"]["base_cost_metrics"]
    decimal_fields = {
        "win_rate_percent",
        "frictionless_expectancy_r",
        "gross_after_slippage_expectancy_r",
        "net_expectancy_r",
        "average_fee_r",
        "average_slippage_cost_r",
        "average_total_friction_r",
        "average_winner_r",
        "average_loser_r",
        "payoff_ratio_r",
        "profit_factor_r",
        "break_even_win_rate_percent",
    }
    values = {}
    for name in RNormalizedSummary.__dataclass_fields__:
        value = fields[name]
        values[name] = (
            Decimal(str(value))
            if name in decimal_fields and value is not None
            else value
        )
    return RNormalizedSummary(**values)


def load_frozen_v6_reference(
    *,
    summary_path: str | Path,
    windows_path: str | Path,
    schedule_path: str | Path,
    manifest: dict,
) -> tuple[
    RNormalizedSummary,
    dict[str, FrozenV6Window],
    tuple[FrozenV6Candidate, ...],
]:
    summary_payload = _load_json(summary_path)
    if summary_payload.get("run_id") != V6_RUN_ID:
        raise ValueError("Frozen V6 result run ID changed.")
    summary = _summary_from_payload(summary_payload)
    reference = manifest["frozen_v6_reference"]
    if (
        summary.frictionless_expectancy_r
        != Decimal(reference["gross_expectancy_r_per_trade"])
        or summary.net_expectancy_r
        != Decimal(reference["net_expectancy_r_per_trade"])
        or summary.profit_factor_r != Decimal(reference["profit_factor_r"])
    ):
        raise ValueError("Frozen V6 aggregate reference metrics changed.")

    with Path(windows_path).open("r", encoding="utf-8", newline="") as stream:
        window_rows = tuple(csv.DictReader(stream))
    windows: dict[str, FrozenV6Window] = {}
    for row in window_rows:
        window_id = row["window_id"]
        if window_id in windows:
            raise ValueError("Frozen V6 window result is duplicated.")
        windows[window_id] = FrozenV6Window(
            window_id=window_id,
            trades=int(row["trades"]),
            gross_expectancy_r=Decimal(row["frictionless_expectancy_r"]),
            net_expectancy_r=Decimal(row["net_expectancy_r"]),
            profit_factor_r=(
                Decimal(row["profit_factor_r"])
                if row["profit_factor_r"]
                else None
            ),
        )
    if tuple(windows) != EXPECTED_WINDOW_IDS:
        raise ValueError("Frozen V6 window identity or order changed.")

    with Path(schedule_path).open("r", encoding="utf-8", newline="") as stream:
        schedule_rows = tuple(csv.DictReader(stream))
    candidates = tuple(
        FrozenV6Candidate(
            window_id=row["window_id"],
            signal_close=_utc(row["entry_signal_time"]),
            atr14_4h=Decimal(row["atr14_4h"]),
        )
        for row in schedule_rows
    )
    keys = tuple((row.window_id, row.signal_close) for row in candidates)
    if len(set(keys)) != len(keys):
        raise ValueError("Frozen V6 candidate schedule contains duplicates.")
    if any(row.atr14_4h <= 0 or row.window_id not in windows for row in candidates):
        raise ValueError("Frozen V6 candidate schedule is invalid.")
    if len(candidates) != summary.trade_count or sum(
        window.trades for window in windows.values()
    ) != len(candidates):
        raise ValueError("Frozen V6 schedule count does not match its result.")
    for window_id, window in windows.items():
        if sum(row.window_id == window_id for row in candidates) != window.trades:
            raise ValueError("Frozen V6 schedule/window counts are inconsistent.")
    return summary, windows, candidates


def load_funding_context(dataset_manifest: dict) -> dict[datetime, FundingObservation]:
    observations: dict[datetime, FundingObservation] = {}
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
                rate_text = row["latest_known_funding_rate"].strip()
                timestamp_text = row["funding_timestamp"].strip()
                observations[bucket_close] = FundingObservation(
                    bucket_close_time=bucket_close,
                    funding_rate=Decimal(rate_text) if rate_text else None,
                    funding_timestamp=_utc(timestamp_text) if timestamp_text else None,
                )
    if len(observations) != dataset_manifest["context_buckets"]:
        raise ValueError("V8 loaded context bucket count is not exact.")
    return observations


def filter_frozen_candidates(
    *,
    candidates: tuple[FrozenV6Candidate, ...],
    funding_by_signal_close: dict[datetime, FundingObservation],
) -> tuple[tuple[FrozenV6Candidate, ...], tuple[CandidateDecision, ...]]:
    retained = []
    decisions = []
    for candidate in candidates:
        observation = funding_by_signal_close.get(candidate.signal_close)
        funding_rate = observation.funding_rate if observation else None
        funding_timestamp = observation.funding_timestamp if observation else None
        if funding_rate is None or funding_timestamp is None:
            status = "FILTERED_MISSING_FUNDING"
        elif funding_timestamp > candidate.signal_close:
            status = "FILTERED_FUTURE_FUNDING"
        elif funding_filter_accepts(
            funding_rate=funding_rate,
            funding_timestamp=funding_timestamp,
            signal_close=candidate.signal_close,
        ):
            status = "RETAINED_NON_POSITIVE_FUNDING"
            retained.append(candidate)
        else:
            status = "FILTERED_POSITIVE_FUNDING"
        decisions.append(
            CandidateDecision(
                window_id=candidate.window_id,
                signal_close_utc=candidate.signal_close.isoformat(),
                status=status,
                funding_rate=funding_rate,
                funding_timestamp_utc=(
                    funding_timestamp.isoformat() if funding_timestamp else None
                ),
            )
        )
    require_v8_subset_of_frozen_v6(
        frozen_v6_candidate_signal_closes=(row.signal_close for row in candidates),
        v8_candidate_signal_closes=(row.signal_close for row in retained),
    )
    return tuple(retained), tuple(decisions)


def doubled_cost_stress_records(
    records: tuple[RNormalizedTrade, ...],
) -> tuple[RNormalizedTrade, ...]:
    """Double recorded base friction while freezing the base trade path.

    V8's immutable candidate schedule must not change under reporting-only
    cost stress.  Fee and slippage drag are linear in their bps inputs, so the
    frozen-path 2x result is the frictionless R less twice each base cost.
    """

    output = []
    for record in records:
        fee = record.fee_cost_r * Decimal("2")
        slippage = record.slippage_cost_r * Decimal("2")
        total = fee + slippage
        output.append(
            replace(
                record,
                gross_after_slippage_r=record.frictionless_r - slippage,
                fee_cost_r=fee,
                slippage_cost_r=slippage,
                total_friction_r=total,
                net_r=record.frictionless_r - total,
            )
        )
    return tuple(output)


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
    for config, name in ((base_config, "base"), (stress_config, "stress_2x")):
        expected = manifest["costs"][name]
        if (
            config.fee_bps != Decimal(expected["fee_bps_per_side"])
            or config.slippage_bps != Decimal(expected["slippage_bps_per_side"])
        ):
            raise ValueError(f"Frozen V8-H0 {name} costs changed.")
    frozen = manifest["execution_risk_and_exit"]
    if (
        V8FundingScheduleStrategy.REWARD_RISK_RATIO
        != Decimal(frozen["reward_risk_ratio"])
        or V8FundingScheduleStrategy.MIN_STOP_DISTANCE_FRACTION
        != Decimal(frozen["minimum_stop_distance_bps"]) / Decimal("10000")
        or V8FundingScheduleStrategy.MAXIMUM_HOLD_BARS
        != frozen["maximum_hold_15m_bars"]
        or V8FundingScheduleStrategy.COOLDOWN_BARS != frozen["cooldown_bars"]
    ):
        raise ValueError("Frozen V8-H0 V6 execution/risk constants changed.")


def _evaluate_window(
    *,
    window,
    candidates: tuple[FrozenV6Candidate, ...],
    strategy_config: TrendMomentumConfig,
    backtest_config: BacktestConfig,
):
    schedule = {row.signal_close: row.atr14_4h for row in candidates}
    strategy = V8FundingScheduleStrategy(
        strategy_config,
        retained_atr_by_signal_close=schedule,
    )
    evaluation = evaluate_strategy_period(
        window.replay_dataset,
        strategy=strategy,
        strategy_config=strategy_config,
        backtest_config=backtest_config,
        evaluation_start_index=window.evaluation_start_index,
    )
    if strategy.scheduled_candidate_conflict_timestamps:
        conflicts = ", ".join(
            timestamp.isoformat()
            for timestamp in sorted(
                strategy.scheduled_candidate_conflict_timestamps
            )
        )
        recent_trades = "; ".join(
            f"signal={trade.entry_signal_time.isoformat()},"
            f"fill={trade.entry_time.isoformat()}@{trade.entry_price},"
            f"exit={trade.exit_time.isoformat()}({trade.exit_reason.value})"
            for trade in evaluation.backtest.trades[-3:]
        )
        raise ValueError(
            f"V8-H0 account-state conflict in frozen schedule for "
            f"{window.window_id}: {conflicts}; recent trades: {recent_trades}."
        )
    emitted = strategy.emitted_candidate_timestamps
    if emitted != frozenset(schedule):
        raise ValueError(f"V8-H0 failed to emit its exact schedule in {window.window_id}.")
    entry_signals = {
        signal.timestamp.astimezone(timezone.utc)
        for signal in evaluation.signals
        if signal.action is StrategyAction.ENTER_LONG
    }
    if entry_signals != emitted:
        raise ValueError(f"V8-H0 signal journal differs from schedule in {window.window_id}.")

    by_signal = {row.signal_close: row for row in candidates}
    observations: list[StopObservation] = []
    for trade in evaluation.backtest.trades:
        candidate = by_signal.get(trade.entry_signal_time.astimezone(timezone.utc))
        if candidate is None:
            raise ValueError(f"V8-H0 created a non-V6 trade in {window.window_id}.")
        observations.append(
            classify_stop_source(
                trade_id=trade.trade_id,
                entry_signal_time=trade.entry_signal_time,
                entry_price=trade.entry_price,
                atr_distance=candidate.atr14_4h,
                minimum_stop_distance_fraction=(
                    V8FundingScheduleStrategy.MIN_STOP_DISTANCE_FRACTION
                ),
            )
        )
    risk_records = build_risk_capital_audit(
        trades=evaluation.backtest.trades,
        signals=_risk_audit_signals(
            signals=evaluation.signals,
            observations=tuple(observations),
            strategy_config=strategy_config,
        ),
        equity_curve=evaluation.backtest.equity_curve,
        backtest_config=backtest_config,
        strategy_config=strategy_config,
    )
    risk_by_trade = {
        record.trade_id: record.actual_stop_risk for record in risk_records
    }
    r_records = build_r_normalized_trades(
        trades=evaluation.backtest.trades,
        actual_stop_risk_by_trade_id=risk_by_trade,
        slippage_bps=backtest_config.slippage_bps,
    )
    if len(r_records) != len(candidates):
        raise ValueError(f"V8-H0 trade count differs from retained schedule in {window.window_id}.")
    return evaluation, r_records, summarize_r_normalized_trades(r_records)


def build_window_result(
    *,
    window,
    v6: FrozenV6Window,
    retained_count: int,
    evaluation,
    summary: RNormalizedSummary,
) -> V8WindowResult:
    if summary.trade_count != retained_count:
        raise ValueError("V8-H0 retained candidate/trade count mismatch.")
    net_result = summary.net_expectancy_r * Decimal(summary.trade_count)
    return V8WindowResult(
        window_id=window.window_id,
        status="ZERO_TRADES" if not summary.trade_count else "TRADES",
        evaluation_days=window.duration_days,
        v6_candidates=v6.trades,
        v8_retained=summary.trade_count,
        filtered_candidates=v6.trades - summary.trade_count,
        non_v6_entries=0,
        v6_gross_expectancy_r=v6.gross_expectancy_r,
        v6_net_expectancy_r=v6.net_expectancy_r,
        v6_profit_factor_r=v6.profit_factor_r,
        v6_net_result_r=v6.net_result_r,
        v8_gross_expectancy_r=summary.frictionless_expectancy_r,
        v8_net_expectancy_r=summary.net_expectancy_r,
        v8_profit_factor_r=summary.profit_factor_r,
        v8_net_result_r=net_result,
        v8_win_rate_percent=summary.win_rate_percent,
        v8_maximum_drawdown_percent=_max_drawdown_percent(
            evaluation.backtest.equity_curve
        ),
        positive_net=summary.trade_count > 0 and net_result > 0,
        net_result_better_than_v6=(
            summary.trade_count > 0 and net_result > v6.net_result_r
        ),
    )


def _gate_payload(gate) -> dict:
    return {**asdict(gate), "all_conditions_met": gate.all_conditions_met}


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
        or sum(row.v6_candidates for row in window_results) != v6_summary.trade_count
        or sum(row.v8_retained for row in window_results) != v8_summary.trade_count
    ):
        raise ValueError("V8-H0 combined records and windows are inconsistent.")
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
    return {
        "version": "8.0",
        "baseline_id": "V8_H0",
        "strategy_id": V8FundingScheduleStrategy.NAME,
        "run_id": EXPECTED_RUN_ID,
        "derivatives_dataset": {
            "dataset_id": dataset_manifest["dataset_id"],
            "definition_sha256": dataset_manifest["dataset_definition_sha256"],
            "classification": dataset_manifest["classification"],
            "context_buckets": dataset_manifest["context_buckets"],
            "future_data_violations": dataset_manifest["future_data_violations"],
        },
        "market": manifest["market"],
        "costs": manifest["costs"],
        "eligible_windows": list(EXPECTED_WINDOW_IDS),
        "window_results_csv": "window_results.csv",
        "candidate_decisions_csv": "candidate_decisions.csv",
        "v6_frozen_reference": {
            **manifest["frozen_v6_reference"],
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
            "missing_or_future_rejected": True,
            "threshold": "FUNDING_RATE_LESS_THAN_OR_EQUAL_TO_ZERO",
            "alternative_thresholds_tested": False,
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
        "blind_holdout_integrity": manifest["dataset_policy"]["blind_holdout"],
        "final_classification": classification,
    }


def _json_safe(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_csv(path: Path, rows: tuple[object, ...]) -> None:
    if not rows:
        raise ValueError(f"V8-H0 report rows are empty: {path.name}.")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(asdict(rows[0])))
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(asdict(row)))
    temporary.replace(path)


def write_reports(
    *,
    output: Path,
    summary_payload: dict,
    window_results: tuple[V8WindowResult, ...],
    candidate_decisions: tuple[CandidateDecision, ...],
) -> tuple[Path, Path, Path]:
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise ValueError("V8-H0 result already exists; refusing overwrite or rerun.")
    output.mkdir(parents=True, exist_ok=True)
    window_path = output / "window_results.csv"
    candidate_path = output / "candidate_decisions.csv"
    _write_csv(window_path, window_results)
    _write_csv(candidate_path, candidate_decisions)
    temporary = output / "summary.json.tmp"
    temporary.write_text(
        json.dumps(_json_safe(summary_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    return summary_path, window_path, candidate_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the single preregistered V8-H0 replay."
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--manifest",
        default="research/v8_funding_context/089b62c96a7de979/manifest.json",
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
    parser.add_argument("--output-root", default="reports/v8_funding_context")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    logger = get_logger(__name__)
    if not args.execute:
        logger.error("V8-H0 replay not executed. Explicit --execute is required.")
        return 1

    try:
        manifest = _load_json(args.manifest)
        dataset_manifest = _load_json(args.dataset_manifest)
        validate_v8_manifest(manifest)
        validate_derivatives_manifest(dataset_manifest)
        output = Path(args.output_root) / EXPECTED_RUN_ID
        if (output / "summary.json").exists():
            raise ValueError("V8-H0 result already exists; refusing overwrite or rerun.")

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
            manifest=manifest,
        )
        funding = load_funding_context(dataset_manifest)
        retained, candidate_decisions = filter_frozen_candidates(
            candidates=candidates,
            funding_by_signal_close=funding,
        )

        from src.config.settings import load_settings

        regions = _build_regions(
            root_manifest=root_manifest,
            expansion_data_root=args.expansion_data_root,
            consumed_data_root=args.consumed_data_root,
        )
        reference_windows = _read_reference_windows(Path(args.mechanism_report))
        if tuple(sorted(reference_windows)) != EXPECTED_WINDOW_IDS:
            raise ValueError("V8-H0 requires exactly the frozen 11 consumed windows.")
        windows = construct_windows(regions, load_settings().multiregime_config())
        eligible = tuple(
            window for window in windows if window.window_id in reference_windows
        )
        eligible_windows = expand_complete_region_history(eligible, regions)
        validate_v6_windows(eligible_windows)
        if tuple(window.window_id for window in eligible_windows) != EXPECTED_WINDOW_IDS:
            raise ValueError("V8-H0 eligible window order or identity changed.")

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
        logger.info("V8-H0 classification: %s", summary_payload["final_classification"])
        logger.info("Reports: %s", ", ".join(str(path.resolve()) for path in paths))
        logger.warning(
            "Blind holdout: LOCKED | NOT DOWNLOADED | NOT LOADED | "
            "NOT REVEALED | NOT CONSUMED | NOT EVALUATED"
        )
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.error("V8-H0 replay failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
