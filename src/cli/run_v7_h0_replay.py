"""Execute the single preregistered V7-H0 consumed-data replay."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from statistics import median

from src.backtest.models import BacktestConfig
from src.cli.run_multiregime import _require_locked_holdout
from src.cli.run_v33_h5 import (
    _build_regions,
    _max_drawdown_percent,
    _read_reference_windows,
)
from src.cli.run_v6_h0_replay import (
    _evaluate_window as evaluate_v6_window,
    _risk_audit_signals,
    _stop_observations,
    evaluation_days_total,
    expand_complete_region_history,
    prepare_v6_region_contexts,
    prepared_context_for_window,
    validate_eligible_windows as validate_v6_windows,
    validate_replay_configs as validate_v6_replay_configs,
)
from src.cli.run_v6_signal_quality import verify_frozen_reproduction
from src.cli.run_v7_h0 import (
    EXPECTED_BUCKETS,
    EXPECTED_DATASET_ID,
    EXPECTED_DATASET_SHA256,
    EXPECTED_RUN_ID,
    EXPECTED_WINDOW_IDS,
    HOLDOUT_END,
    HOLDOUT_START,
    _load_json,
    validate_eligible_windows,
    validate_v7_dataset_manifest,
    validate_v7_h0_implementation,
    validate_v7_h0_manifest,
)
from src.diagnostics.r_normalized import (
    RNormalizedSummary,
    RNormalizedTrade,
    build_r_normalized_trades,
    summarize_r_normalized_trades,
)
from src.diagnostics.risk_capital_audit import build_risk_capital_audit
from src.hypotheses.manifest import HoldoutStatus, ResearchManifestStore
from src.orderflow.aggregation import OrderFlowBucket
from src.orderflow.storage import OrderFlowBucketStore
from src.research.evaluation import SignalRecord, evaluate_strategy_period
from src.research.multiregime.windows import construct_windows
from src.research.v6_mtf_continuation_preregistration import (
    build_manifest as build_v6_manifest,
)
from src.research.v7_orderflow_stability import (
    V7H0Metrics,
    V7H0WindowMetrics,
    evaluate_v7_h0_progression_gate,
    evaluate_v7_h0_support_gate,
)
from src.strategy.models import StrategyAction, TrendMomentumConfig
from src.strategy.v6_mtf_continuation import (
    SOURCE_BAR_DURATION,
    V6MTFContinuationStrategy,
)
from src.strategy.v7_orderflow_confirmation import (
    V7OrderFlowConfirmationStrategy,
    assert_v7_entries_are_v6_subset,
)
from src.utils.logger import configure_logging, get_logger


@dataclass(frozen=True, slots=True)
class SubsetDiagnostics:
    v6_reference_entries: int
    v7_retained_entries: int
    filtered_v6_entries: int
    retention_ratio: Decimal
    non_v6_entries: int


@dataclass(frozen=True, slots=True)
class V7WindowResult:
    window_id: str
    status: str
    evaluation_days: Decimal
    v6_trades: int
    v7_trades: int
    retention_ratio: Decimal
    v6_frictionless_expectancy_r: Decimal
    v6_net_expectancy_r: Decimal
    v6_profit_factor_r: Decimal | None
    v7_frictionless_expectancy_r: Decimal
    v7_net_expectancy_r: Decimal
    v7_profit_factor_r: Decimal | None
    v7_win_rate_percent: Decimal
    v7_average_winner_r: Decimal
    v7_average_loser_r: Decimal
    v7_payoff_ratio: Decimal | None
    v7_average_friction_r: Decimal
    v7_maximum_drawdown_percent: Decimal
    v7_trades_per_day: Decimal
    v7_net_r_per_day: Decimal
    positive_net: bool
    gross_delta_vs_v6: Decimal | None
    net_delta_vs_v6: Decimal | None
    profit_factor_delta_vs_v6: Decimal | None
    gross_better_than_v6: bool
    net_better_than_v6: bool
    v6_reference_entries: int
    v7_retained_entries: int
    filtered_v6_entries: int
    non_v6_entries: int


@dataclass(frozen=True, slots=True)
class EntryComparison:
    window_id: str
    signal_timestamp_utc: str
    status: str
    quote_volume_imbalance: Decimal
    taker_buy_quote_ratio: Decimal


def summarize_combined_records(
    groups: tuple[tuple[RNormalizedTrade, ...], ...],
) -> RNormalizedSummary:
    """Aggregate real trade records before applying shared R accounting."""

    return summarize_r_normalized_trades(
        tuple(record for group in groups for record in group)
    )


def entry_signal_timestamps(signals: tuple[SignalRecord, ...]) -> tuple[datetime, ...]:
    return tuple(
        signal.timestamp
        for signal in signals
        if signal.action is StrategyAction.ENTER_LONG
    )


def validate_subset_matching(
    *,
    v6_entry_timestamps: tuple[datetime, ...],
    v7_entry_timestamps: tuple[datetime, ...],
) -> SubsetDiagnostics:
    """Enforce and summarize the frozen V7 subset relationship."""

    assert_v7_entries_are_v6_subset(v7_entry_timestamps, v6_entry_timestamps)
    v6 = {timestamp.astimezone(timezone.utc) for timestamp in v6_entry_timestamps}
    v7 = {timestamp.astimezone(timezone.utc) for timestamp in v7_entry_timestamps}
    non_v6 = v7.difference(v6)
    retained = len(v7)
    reference = len(v6)
    return SubsetDiagnostics(
        v6_reference_entries=reference,
        v7_retained_entries=retained,
        filtered_v6_entries=len(v6.difference(v7)),
        retention_ratio=(
            Decimal(retained) / Decimal(reference)
            if reference
            else Decimal("0")
        ),
        non_v6_entries=len(non_v6),
    )


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
    costs = manifest["costs"]
    for config, name in ((base_config, "base"), (stress_config, "stress_2x")):
        if (
            config.fee_bps != Decimal(costs[name]["fee_bps_per_side"])
            or config.slippage_bps != Decimal(costs[name]["slippage_bps_per_side"])
        ):
            raise ValueError(f"Frozen V7-H0 {name} costs changed.")
    if (
        V6MTFContinuationStrategy.MIN_STOP_DISTANCE_BPS != Decimal("96")
        or V6MTFContinuationStrategy.MIN_STOP_DISTANCE_FRACTION
        != Decimal("0.0096")
    ):
        raise ValueError("V7-H0 stress changed the frozen 96-bps stop floor.")


def _partition_month(path_text: str) -> tuple[int, int]:
    path = Path(path_text)
    try:
        return int(path.parent.parent.name), int(path.parent.name)
    except (TypeError, ValueError) as exc:
        raise ValueError("V7 order-flow partition path is invalid.") from exc


def validate_partition_paths(dataset_manifest: dict) -> tuple[tuple[int, int], ...]:
    months = tuple(
        _partition_month(path) for path in dataset_manifest.get("partition_paths", ())
    )
    if not months or len(months) != len(set(months)):
        raise ValueError("V7 order-flow partition index is empty or duplicated.")
    for year, month in months:
        if not 1 <= month <= 12:
            raise ValueError("V7 order-flow partition month is invalid.")
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        end = (
            datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            if month == 12
            else datetime(year, month + 1, 1, tzinfo=timezone.utc)
        )
        if start < HOLDOUT_END and HOLDOUT_START < end:
            raise ValueError("V7 order-flow partition index overlaps blind holdout.")
    return months


def load_orderflow_buckets(
    *, dataset_manifest: dict, root: str | Path
) -> tuple[OrderFlowBucket, ...]:
    """Load each frozen safe partition once through the shared storage parser."""

    store = OrderFlowBucketStore(root)
    buckets = tuple(
        bucket
        for year, month in validate_partition_paths(dataset_manifest)
        for bucket in store.load_partition(year=year, month=month)
    )
    timestamps = tuple(bucket.bucket_open_time for bucket in buckets)
    if len(buckets) != EXPECTED_BUCKETS or len(set(timestamps)) != EXPECTED_BUCKETS:
        raise ValueError("V7 order-flow loaded bucket coverage is not exact.")
    if any(
        later <= earlier
        for earlier, later in zip(timestamps, timestamps[1:], strict=False)
    ):
        raise ValueError("V7 order-flow loaded partitions are not chronological.")
    if any(HOLDOUT_START <= timestamp < HOLDOUT_END for timestamp in timestamps):
        raise ValueError("V7 order-flow loaded data intersects blind holdout.")
    return buckets


def _evaluate_v7_window(
    *,
    window,
    strategy_config: TrendMomentumConfig,
    backtest_config: BacktestConfig,
    prepared_context,
    buckets: tuple[OrderFlowBucket, ...],
    frozen_v6_entry_signal_times: tuple[datetime, ...],
):
    strategy = V7OrderFlowConfirmationStrategy(
        strategy_config,
        buckets=buckets,
        dataset_id=EXPECTED_DATASET_ID,
        dataset_definition_sha256=EXPECTED_DATASET_SHA256,
        frozen_v6_entry_signal_times=frozen_v6_entry_signal_times,
        v6_strategy=V6MTFContinuationStrategy(
            strategy_config,
            prepared_context=prepared_context,
        ),
    )
    evaluation = evaluate_strategy_period(
        window.replay_dataset,
        strategy=strategy,
        strategy_config=strategy_config,
        backtest_config=backtest_config,
        evaluation_start_index=window.evaluation_start_index,
        prepared_indicators=prepared_context.indicators_15m_by_timestamp,
    )
    if strategy.missing_bucket_timestamps:
        raise ValueError(
            f"V7-H0 exact order-flow buckets missing in {window.window_id}."
        )
    if strategy.scheduled_candidate_conflict_timestamps:
        raise ValueError(
            f"V7-H0 frozen candidate conflicts with V7 state in {window.window_id}."
        )
    observations = _stop_observations(
        trades=evaluation.backtest.trades,
        prepared_context=prepared_context,
    )
    risk_records = build_risk_capital_audit(
        trades=evaluation.backtest.trades,
        signals=_risk_audit_signals(
            signals=evaluation.signals,
            observations=observations,
            strategy_config=strategy_config,
        ),
        equity_curve=evaluation.backtest.equity_curve,
        backtest_config=backtest_config,
        strategy_config=strategy_config,
    )
    r_records = build_r_normalized_trades(
        trades=evaluation.backtest.trades,
        actual_stop_risk_by_trade_id={
            record.trade_id: record.actual_stop_risk for record in risk_records
        },
        slippage_bps=backtest_config.slippage_bps,
    )
    if len(r_records) != len(evaluation.backtest.trades):
        raise ValueError(f"V7-H0 R-record mismatch in {window.window_id}.")
    return evaluation, r_records, summarize_r_normalized_trades(r_records)


def build_window_result(
    *, window, v6_evaluation, v6_summary, v7_evaluation, v7_summary
) -> V7WindowResult:
    subset = validate_subset_matching(
        v6_entry_timestamps=entry_signal_timestamps(v6_evaluation.signals),
        v7_entry_timestamps=entry_signal_timestamps(v7_evaluation.signals),
    )
    days = window.duration_days
    zero_trades = v7_summary.trade_count == 0
    pf_delta = (
        v7_summary.profit_factor_r - v6_summary.profit_factor_r
        if not zero_trades
        and v7_summary.profit_factor_r is not None
        and v6_summary.profit_factor_r is not None
        else None
    )
    return V7WindowResult(
        window_id=window.window_id,
        status="ZERO_TRADES" if zero_trades else "TRADES",
        evaluation_days=days,
        v6_trades=v6_summary.trade_count,
        v7_trades=v7_summary.trade_count,
        retention_ratio=(
            Decimal(v7_summary.trade_count) / Decimal(v6_summary.trade_count)
            if v6_summary.trade_count
            else Decimal("0")
        ),
        v6_frictionless_expectancy_r=v6_summary.frictionless_expectancy_r,
        v6_net_expectancy_r=v6_summary.net_expectancy_r,
        v6_profit_factor_r=v6_summary.profit_factor_r,
        v7_frictionless_expectancy_r=v7_summary.frictionless_expectancy_r,
        v7_net_expectancy_r=v7_summary.net_expectancy_r,
        v7_profit_factor_r=v7_summary.profit_factor_r,
        v7_win_rate_percent=v7_summary.win_rate_percent,
        v7_average_winner_r=v7_summary.average_winner_r,
        v7_average_loser_r=v7_summary.average_loser_r,
        v7_payoff_ratio=v7_summary.payoff_ratio_r,
        v7_average_friction_r=v7_summary.average_total_friction_r,
        v7_maximum_drawdown_percent=_max_drawdown_percent(
            v7_evaluation.backtest.equity_curve
        ),
        v7_trades_per_day=(
            Decimal(v7_summary.trade_count) / days if days > 0 else Decimal("0")
        ),
        v7_net_r_per_day=(
            v7_summary.net_expectancy_r * Decimal(v7_summary.trade_count) / days
            if days > 0
            else Decimal("0")
        ),
        positive_net=(not zero_trades and v7_summary.net_expectancy_r > 0),
        gross_delta_vs_v6=(
            None
            if zero_trades
            else v7_summary.frictionless_expectancy_r
            - v6_summary.frictionless_expectancy_r
        ),
        net_delta_vs_v6=(
            None
            if zero_trades
            else v7_summary.net_expectancy_r - v6_summary.net_expectancy_r
        ),
        profit_factor_delta_vs_v6=pf_delta,
        gross_better_than_v6=(
            not zero_trades
            and v7_summary.frictionless_expectancy_r
            > v6_summary.frictionless_expectancy_r
        ),
        net_better_than_v6=(
            not zero_trades
            and v7_summary.net_expectancy_r > v6_summary.net_expectancy_r
        ),
        v6_reference_entries=subset.v6_reference_entries,
        v7_retained_entries=subset.v7_retained_entries,
        filtered_v6_entries=subset.filtered_v6_entries,
        non_v6_entries=subset.non_v6_entries,
    )


def build_entry_comparison(
    *,
    window_id: str,
    v6_entry_timestamps: tuple[datetime, ...],
    v7_entry_timestamps: tuple[datetime, ...],
    bucket_index: dict[datetime, OrderFlowBucket],
) -> tuple[EntryComparison, ...]:
    validate_subset_matching(
        v6_entry_timestamps=v6_entry_timestamps,
        v7_entry_timestamps=v7_entry_timestamps,
    )
    retained = {timestamp.astimezone(timezone.utc) for timestamp in v7_entry_timestamps}
    rows = []
    for signal_time in sorted(
        timestamp.astimezone(timezone.utc) for timestamp in v6_entry_timestamps
    ):
        bucket = bucket_index.get(signal_time - SOURCE_BAR_DURATION)
        if bucket is None:
            raise ValueError("V7 entry diagnostics lack an exact signal bucket.")
        rows.append(
            EntryComparison(
                window_id=window_id,
                signal_timestamp_utc=signal_time.isoformat(),
                status="RETAINED" if signal_time in retained else "FILTERED_OUT",
                quote_volume_imbalance=bucket.quote_volume_imbalance,
                taker_buy_quote_ratio=bucket.taker_buy_quote_ratio,
            )
        )
    return tuple(rows)


def summarize_orderflow_diagnostics(rows: tuple[EntryComparison, ...]) -> dict:
    retained = tuple(row for row in rows if row.status == "RETAINED")
    filtered = tuple(row for row in rows if row.status == "FILTERED_OUT")

    def mean(values: tuple[Decimal, ...]) -> Decimal | None:
        return (
            sum(values, start=Decimal("0")) / Decimal(len(values))
            if values
            else None
        )

    retained_imbalance = tuple(row.quote_volume_imbalance for row in retained)
    retained_ratio = tuple(row.taker_buy_quote_ratio for row in retained)
    filtered_imbalance = tuple(row.quote_volume_imbalance for row in filtered)
    return {
        "retained_entries": {
            "count": len(retained),
            "mean_quote_volume_imbalance": mean(retained_imbalance),
            "median_quote_volume_imbalance": (
                median(retained_imbalance) if retained_imbalance else None
            ),
            "mean_taker_buy_quote_ratio": mean(retained_ratio),
            "median_taker_buy_quote_ratio": (
                median(retained_ratio) if retained_ratio else None
            ),
        },
        "filtered_v6_entries": {
            "count": len(filtered),
            "mean_quote_volume_imbalance": mean(filtered_imbalance),
            "median_quote_volume_imbalance": (
                median(filtered_imbalance) if filtered_imbalance else None
            ),
        },
        "diagnostic_only_no_threshold_derivation": True,
    }


def _gate_payload(gate) -> dict:
    return {**asdict(gate), "all_conditions_met": gate.all_conditions_met}


def build_summary_payload(
    *,
    manifest: dict,
    dataset_manifest: dict,
    v6_summary: RNormalizedSummary,
    v7_summary: RNormalizedSummary,
    stress_summary: RNormalizedSummary,
    window_results: tuple[V7WindowResult, ...],
    entry_comparison: tuple[EntryComparison, ...],
    maximum_drawdown_percent: Decimal,
    evaluation_days: Decimal,
) -> dict:
    if (
        tuple(row.window_id for row in window_results) != EXPECTED_WINDOW_IDS
        or sum(row.v6_trades for row in window_results) != v6_summary.trade_count
        or sum(row.v7_trades for row in window_results) != v7_summary.trade_count
    ):
        raise ValueError("V7-H0 combined records and window results are inconsistent.")
    non_v6_entries = sum(row.non_v6_entries for row in window_results)
    gate_windows = tuple(
        V7H0WindowMetrics(
            trades=row.v7_trades,
            net_expectancy_r=(
                row.v7_net_expectancy_r if row.v7_trades else None
            ),
            v6_net_expectancy_r=row.v6_net_expectancy_r,
        )
        for row in window_results
    )
    metrics = V7H0Metrics(
        frictionless_expectancy_r=v7_summary.frictionless_expectancy_r,
        net_expectancy_r=v7_summary.net_expectancy_r,
        profit_factor_r=v7_summary.profit_factor_r,
        non_v6_entry_count=non_v6_entries,
    )
    support = evaluate_v7_h0_support_gate(metrics=metrics, windows=gate_windows)
    progression = evaluate_v7_h0_progression_gate(
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
    v6_trades = v6_summary.trade_count
    v7_trades = v7_summary.trade_count
    trades_per_day = (
        Decimal(v7_trades) / evaluation_days if evaluation_days > 0 else Decimal("0")
    )
    net_r_per_day = (
        v7_summary.net_expectancy_r * Decimal(v7_trades) / evaluation_days
        if evaluation_days > 0
        else Decimal("0")
    )
    stress_net_r_per_day = (
        stress_summary.net_expectancy_r
        * Decimal(stress_summary.trade_count)
        / evaluation_days
        if evaluation_days > 0
        else Decimal("0")
    )
    positive_windows = sum(row.positive_net for row in window_results)
    return {
        "version": "7.0",
        "baseline_id": "V7_H0",
        "strategy_id": "V7OrderFlowConfirmationStrategy",
        "run_id": EXPECTED_RUN_ID,
        "orderflow_dataset": {
            "dataset_id": EXPECTED_DATASET_ID,
            "definition_sha256": EXPECTED_DATASET_SHA256,
            "aggregated_buckets": dataset_manifest["aggregated_buckets"],
            "coverage": dataset_manifest["coverage"],
            "reconciliation": {
                "bucket_count": dataset_manifest["reconciliation"]["bucket_count"],
                "kline_count": dataset_manifest["reconciliation"]["kline_count"],
            },
        },
        "market": manifest["market"],
        "costs": manifest["costs"],
        "eligible_windows": list(EXPECTED_WINDOW_IDS),
        "window_results_csv": "window_results.csv",
        "entry_comparison_csv": "entry_comparison.csv",
        "v6_frozen_reference": {
            **manifest["frozen_v6_reference"],
            "reproduced_metrics": asdict(v6_summary),
        },
        "v7_base": {
            "v6_reference_trade_count": v6_trades,
            "v7_retained_trade_count": v7_trades,
            "retention_ratio": (
                Decimal(v7_trades) / Decimal(v6_trades)
                if v6_trades
                else Decimal("0")
            ),
            "filtered_trade_count": v6_trades - v7_trades,
            "non_v6_entry_count": non_v6_entries,
            "frictionless_expectancy_r_per_trade": (
                v7_summary.frictionless_expectancy_r
            ),
            "net_expectancy_r_per_trade": v7_summary.net_expectancy_r,
            "profit_factor_r": v7_summary.profit_factor_r,
            "win_rate_percent": v7_summary.win_rate_percent,
            "average_winner_r": v7_summary.average_winner_r,
            "average_loser_r": v7_summary.average_loser_r,
            "payoff_ratio": v7_summary.payoff_ratio_r,
            "average_friction_r_per_trade": v7_summary.average_total_friction_r,
            "maximum_drawdown_percent": maximum_drawdown_percent,
            "trades_per_day": trades_per_day,
            "net_r_per_day": net_r_per_day,
            "positive_net_windows": positive_windows,
            "positive_net_windows_total": len(window_results),
            "positive_net_window_ratio": (
                Decimal(positive_windows) / Decimal(len(window_results))
            ),
            "metrics": asdict(v7_summary),
        },
        "comparison_vs_v6": {
            "gross_delta_r": (
                v7_summary.frictionless_expectancy_r
                - v6_summary.frictionless_expectancy_r
            ),
            "net_delta_r": v7_summary.net_expectancy_r - v6_summary.net_expectancy_r,
            "profit_factor_delta_r": (
                v7_summary.profit_factor_r - v6_summary.profit_factor_r
                if v7_summary.profit_factor_r is not None
                and v6_summary.profit_factor_r is not None
                else None
            ),
            "gross_better_windows": sum(
                row.gross_better_than_v6 for row in window_results
            ),
            "net_better_windows": sum(
                row.net_better_than_v6 for row in window_results
            ),
            "eligible_windows": len(window_results),
        },
        "subset_diagnostics": {
            "v6_reference_entries": sum(
                row.v6_reference_entries for row in window_results
            ),
            "v7_retained_entries": sum(
                row.v7_retained_entries for row in window_results
            ),
            "filtered_v6_entries": sum(
                row.filtered_v6_entries for row in window_results
            ),
            "non_v6_entries": non_v6_entries,
            "required_non_v6_entries": 0,
        },
        "orderflow_diagnostics": summarize_orderflow_diagnostics(entry_comparison),
        "stress_2x": {
            "fee_bps_per_side": Decimal("20"),
            "slippage_bps_per_side": Decimal("4"),
            "frozen_stop_floor_bps": Decimal("96"),
            "trade_count": stress_summary.trade_count,
            "net_expectancy_r_per_trade": stress_summary.net_expectancy_r,
            "profit_factor_r": stress_summary.profit_factor_r,
            "net_r_per_day": stress_net_r_per_day,
            "reporting_only_not_gate": True,
        },
        "window_consistency": {
            "expected": list(EXPECTED_WINDOW_IDS),
            "actual": [row.window_id for row in window_results],
            "all_11_exact": tuple(row.window_id for row in window_results)
            == EXPECTED_WINDOW_IDS,
        },
        "support_gate": _gate_payload(support),
        "progression_gate": _gate_payload(progression),
        "blind_holdout_integrity": manifest["dataset_policy"]["blind_holdout"],
        "final_classification": classification,
    }


def _write_csv(path: Path, rows: tuple[object, ...]) -> None:
    if not rows:
        raise ValueError(f"V7-H0 report rows are empty: {path.name}.")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        fields = tuple(asdict(rows[0]))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(asdict(row)))
    temporary.replace(path)


def _json_safe(value):
    if isinstance(value, Decimal):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def write_reports(
    *,
    output: Path,
    summary_payload: dict,
    window_results: tuple[V7WindowResult, ...],
    entry_comparison: tuple[EntryComparison, ...],
) -> tuple[Path, Path, Path]:
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise ValueError("V7-H0 result already exists; refusing overwrite or rerun.")
    output.mkdir(parents=True, exist_ok=True)
    window_path = output / "window_results.csv"
    entry_path = output / "entry_comparison.csv"
    _write_csv(window_path, window_results)
    _write_csv(entry_path, entry_comparison)
    temporary = output / "summary.json.tmp"
    temporary.write_text(
        json.dumps(_json_safe(summary_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    return summary_path, window_path, entry_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the single preregistered V7-H0 replay."
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--manifest",
        default="research/v7_orderflow/74b2458cb2812de2/manifest.json",
    )
    parser.add_argument(
        "--dataset-manifest",
        default="data/orderflow/aggregated/15m/BTCUSDC/dataset_manifest.json",
    )
    parser.add_argument(
        "--orderflow-root", default="data/orderflow/aggregated/15m"
    )
    parser.add_argument("--root-manifest", default="research/hypothesis_manifest.json")
    parser.add_argument(
        "--mechanism-report", default="reports/mechanisms/bc2496aed05555b5"
    )
    parser.add_argument(
        "--expansion-data-root",
        default="data/historical/multiregime_expansion",
    )
    parser.add_argument("--consumed-data-root", default="data/historical")
    parser.add_argument("--output-root", default="reports/v7_orderflow")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    logger = get_logger(__name__)
    if not args.execute:
        logger.error("V7-H0 replay not executed. Explicit --execute is required.")
        return 1

    try:
        manifest = _load_json(args.manifest)
        dataset_manifest = _load_json(args.dataset_manifest)
        validate_v7_h0_manifest(manifest)
        validate_v7_dataset_manifest(dataset_manifest)
        validate_v7_h0_implementation(manifest)
        validate_partition_paths(dataset_manifest)

        output = Path(args.output_root) / EXPECTED_RUN_ID
        if (output / "summary.json").exists():
            raise ValueError("V7-H0 result already exists; refusing overwrite or rerun.")

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

        from src.config.settings import load_settings

        regions = _build_regions(
            root_manifest=root_manifest,
            expansion_data_root=args.expansion_data_root,
            consumed_data_root=args.consumed_data_root,
        )
        reference_windows = _read_reference_windows(Path(args.mechanism_report))
        validate_eligible_windows(reference_windows)
        windows = construct_windows(regions, load_settings().multiregime_config())
        eligible = tuple(
            window for window in windows if window.window_id in reference_windows
        )
        eligible_windows = expand_complete_region_history(eligible, regions)
        validate_v6_windows(eligible_windows)
        if tuple(window.window_id for window in eligible_windows) != EXPECTED_WINDOW_IDS:
            raise ValueError("V7-H0 eligible window order or identity changed.")
        region_contexts = prepare_v6_region_contexts(
            windows=eligible_windows, regions=regions
        )

        v6_evaluations = []
        v6_summaries = []
        v6_record_groups = []
        for window in eligible_windows:
            prepared = prepared_context_for_window(
                window=window, regions=regions, region_contexts=region_contexts
            )
            evaluation, records, summary, _ = evaluate_v6_window(
                window=window,
                strategy_config=strategy_config,
                backtest_config=base_config,
                prepared_context=prepared,
            )
            v6_evaluations.append(evaluation)
            v6_summaries.append(summary)
            v6_record_groups.append(records)

        v6_summary = summarize_combined_records(tuple(v6_record_groups))
        verify_frozen_reproduction(
            summary=v6_summary,
            positive_windows=sum(
                summary.net_expectancy_r > 0 for summary in v6_summaries
            ),
        )
        logger.info("Frozen V6-H0 reference reproduced before V7 evaluation.")

        buckets = load_orderflow_buckets(
            dataset_manifest=dataset_manifest, root=args.orderflow_root
        )
        bucket_index = {bucket.bucket_open_time: bucket for bucket in buckets}
        v7_record_groups = []
        stress_record_groups = []
        window_results = []
        entry_rows = []
        maximum_drawdown = Decimal("0")

        for window, v6_evaluation, v6_window_summary in zip(
            eligible_windows, v6_evaluations, v6_summaries, strict=True
        ):
            prepared = prepared_context_for_window(
                window=window, regions=regions, region_contexts=region_contexts
            )
            v7_evaluation, records, v7_window_summary = _evaluate_v7_window(
                window=window,
                strategy_config=strategy_config,
                backtest_config=base_config,
                prepared_context=prepared,
                buckets=buckets,
                frozen_v6_entry_signal_times=entry_signal_timestamps(
                    v6_evaluation.signals
                ),
            )
            row = build_window_result(
                window=window,
                v6_evaluation=v6_evaluation,
                v6_summary=v6_window_summary,
                v7_evaluation=v7_evaluation,
                v7_summary=v7_window_summary,
            )
            window_results.append(row)
            v7_record_groups.append(records)
            maximum_drawdown = max(
                maximum_drawdown, row.v7_maximum_drawdown_percent
            )
            entry_rows.extend(
                build_entry_comparison(
                    window_id=window.window_id,
                    v6_entry_timestamps=entry_signal_timestamps(
                        v6_evaluation.signals
                    ),
                    v7_entry_timestamps=entry_signal_timestamps(
                        v7_evaluation.signals
                    ),
                    bucket_index=bucket_index,
                )
            )
            _, stress_records, _ = _evaluate_v7_window(
                window=window,
                strategy_config=strategy_config,
                backtest_config=stress_config,
                prepared_context=prepared,
                buckets=buckets,
                frozen_v6_entry_signal_times=entry_signal_timestamps(
                    v6_evaluation.signals
                ),
            )
            stress_record_groups.append(stress_records)

        v7_summary = summarize_combined_records(tuple(v7_record_groups))
        stress_summary = summarize_combined_records(tuple(stress_record_groups))
        summary_payload = build_summary_payload(
            manifest=manifest,
            dataset_manifest=dataset_manifest,
            v6_summary=v6_summary,
            v7_summary=v7_summary,
            stress_summary=stress_summary,
            window_results=tuple(window_results),
            entry_comparison=tuple(entry_rows),
            maximum_drawdown_percent=maximum_drawdown,
            evaluation_days=evaluation_days_total(eligible_windows),
        )
        paths = write_reports(
            output=output,
            summary_payload=summary_payload,
            window_results=tuple(window_results),
            entry_comparison=tuple(entry_rows),
        )
        logger.info("V7-H0 classification: %s", summary_payload["final_classification"])
        logger.info("Reports: %s", ", ".join(str(path.resolve()) for path in paths))
        logger.warning(
            "Blind holdout: LOCKED | NOT DOWNLOADED | NOT LOADED | "
            "NOT REVEALED | NOT CONSUMED | NOT EVALUATED"
        )
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.error("V7-H0 replay failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
