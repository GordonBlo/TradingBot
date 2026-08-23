"""Execute the single preregistered V6-H0 consumed-data replay."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from statistics import median

from src.backtest.models import BacktestConfig, Trade
from src.cli.run_multiregime import _require_locked_holdout
from src.cli.run_v33_h5 import (
    _build_regions,
    _max_drawdown_percent,
    _read_reference_windows,
)
from src.cli.run_v5_h0_replay import _json_safe, summarize_combined_records
from src.cli.run_v6_h0 import (
    EXPECTED_RUN_ID,
    _load_json,
    validate_v6_h0_implementation,
    validate_v6_h0_manifest,
)
from src.diagnostics.r_normalized import (
    RNormalizedSummary,
    RNormalizedTrade,
    build_r_normalized_trades,
    summarize_r_normalized_trades,
)
from src.diagnostics.risk_capital_audit import build_risk_capital_audit
from src.historical.dataset import HistoricalDataset
from src.hypotheses.manifest import HoldoutStatus, ResearchManifestStore
from src.market.intervals import next_open_time
from src.research.evaluation import SignalRecord, evaluate_strategy_period
from src.research.multiregime.models import PartitionStatus, ResearchWindow
from src.research.multiregime.windows import construct_windows
from src.research.v6_mtf_continuation_stability import (
    REQUIRED_ELIGIBLE_WINDOWS,
    V6H0Metrics,
    evaluate_v6_h0_gate,
)
from src.strategy.models import StrategyAction, TrendMomentumConfig
from src.strategy.v6_mtf_continuation import (
    V6PreparedContext,
    V6MTFContinuationStrategy,
    prepare_v6_context,
)
from src.utils.logger import configure_logging, get_logger


ATR_STOP_SOURCE = "ATR14_4H"
COST_FLOOR_STOP_SOURCE = "96_BPS_COST_FLOOR"


@dataclass(frozen=True, slots=True)
class WindowResult:
    window_id: str
    trades: int
    evaluation_days: Decimal
    trades_per_day: Decimal
    frictionless_expectancy_r: Decimal
    net_expectancy_r: Decimal
    profit_factor_r: Decimal | None
    win_rate_percent: Decimal
    average_winner_r: Decimal
    average_loser_r: Decimal
    payoff_ratio: Decimal | None
    maximum_drawdown_percent: Decimal
    net_r_per_day: Decimal
    positive_net: bool


@dataclass(frozen=True, slots=True)
class StopObservation:
    trade_id: str
    entry_signal_time: datetime
    entry_price: Decimal
    atr_distance: Decimal
    cost_floor_distance: Decimal
    initial_stop_distance: Decimal
    initial_stop_distance_bps: Decimal
    stop_source: str


def _mean(values: tuple[Decimal, ...]) -> Decimal:
    if not values:
        return Decimal("0")
    return sum(values, start=Decimal("0")) / Decimal(len(values))


def classify_stop_source(
    *,
    trade_id: str,
    entry_signal_time: datetime,
    entry_price: Decimal,
    atr_distance: Decimal,
    minimum_stop_distance_fraction: Decimal,
) -> StopObservation:
    """Classify the frozen max(ATR4h, actual-fill floor) stop."""

    if entry_price <= 0 or atr_distance <= 0:
        raise ValueError("V6-H0 stop diagnostics require positive distances.")
    cost_floor = entry_price * minimum_stop_distance_fraction
    initial_distance = max(atr_distance, cost_floor)
    return StopObservation(
        trade_id=trade_id,
        entry_signal_time=entry_signal_time,
        entry_price=entry_price,
        atr_distance=atr_distance,
        cost_floor_distance=cost_floor,
        initial_stop_distance=initial_distance,
        initial_stop_distance_bps=(
            initial_distance / entry_price * Decimal("10000")
        ),
        stop_source=(
            ATR_STOP_SOURCE
            if atr_distance >= cost_floor
            else COST_FLOOR_STOP_SOURCE
        ),
    )


def build_cost_diagnostics(
    *,
    observations: tuple[StopObservation, ...],
    records: tuple[RNormalizedTrade, ...],
    final_buy_signals: int,
) -> dict:
    """Summarize safely observable V6 cost/stop diagnostics."""

    if len(observations) != len(records) or any(
        observation.trade_id != record.trade_id
        for observation, record in zip(observations, records, strict=True)
    ):
        raise ValueError("V6-H0 stop/friction diagnostic records differ.")

    stop_bps = tuple(item.initial_stop_distance_bps for item in observations)
    friction_r = tuple(record.total_friction_r for record in records)
    atr_count = sum(item.stop_source == ATR_STOP_SOURCE for item in observations)
    floor_count = sum(
        item.stop_source == COST_FLOOR_STOP_SOURCE for item in observations
    )
    trade_count = len(observations)
    return {
        "average_initial_stop_distance_bps": _mean(stop_bps),
        "median_initial_stop_distance_bps": (
            median(stop_bps) if stop_bps else Decimal("0")
        ),
        "average_modeled_base_friction_r_per_trade": _mean(friction_r),
        "median_modeled_base_friction_r_per_trade": (
            median(friction_r) if friction_r else Decimal("0")
        ),
        "average_friction_initial_risk_ratio": _mean(friction_r),
        "atr_determined_trades": atr_count,
        "cost_floor_determined_trades": floor_count,
        "atr_determined_percent": (
            Decimal(atr_count) / Decimal(trade_count) * Decimal("100")
            if trade_count
            else Decimal("0")
        ),
        "cost_floor_determined_percent": (
            Decimal(floor_count) / Decimal(trade_count) * Decimal("100")
            if trade_count
            else Decimal("0")
        ),
        "signal_funnel": {
            "final_buy_signals": final_buy_signals,
            "higher_timeframe_regime_active": None,
            "pullback_reclaim_candidates": None,
            "previous_high_confirmations": None,
            "unavailable_reason": (
                "Intermediate HOLD-state funnel events are not exposed by the "
                "existing adapter; omitted to avoid changing strategy behavior."
            ),
        },
    }


def validate_replay_configs(
    *,
    manifest: dict,
    strategy_config: TrendMomentumConfig,
    base_config: BacktestConfig,
    stress_config: BacktestConfig,
) -> None:
    risk = manifest["risk_and_exit"]
    initial_risk = manifest["volatility_and_initial_risk"]
    costs = manifest["costs"]
    expected_strategy = {
        "fast_ema_period": manifest["entry"]["ema_period_15m"],
        "atr_period": initial_risk["atr_period"],
        "reward_risk_ratio": Decimal(risk["reward_risk_ratio"]),
        "maximum_bars_in_position": risk["maximum_hold_bars"],
        "cooldown_bars": risk["cooldown_bars"],
    }
    for name, expected in expected_strategy.items():
        if getattr(strategy_config, name) != expected:
            raise ValueError(f"Frozen V6-H0 strategy config changed: {name}.")

    for config, cost_name in (
        (base_config, "base"),
        (stress_config, "stress_2x"),
    ):
        expected = costs[cost_name]
        if config.fee_bps != Decimal(expected["fee_bps_per_side"]):
            raise ValueError(f"Frozen V6-H0 {cost_name} fee changed.")
        if config.slippage_bps != Decimal(expected["slippage_bps_per_side"]):
            raise ValueError(f"Frozen V6-H0 {cost_name} slippage changed.")
        if config.execution_timing is not base_config.execution_timing:
            raise ValueError("V6-H0 stress execution timing changed.")
        if config.ambiguous_bar_policy is not base_config.ambiguous_bar_policy:
            raise ValueError("V6-H0 stress STOP_FIRST policy changed.")

    cls = V6MTFContinuationStrategy
    if (
        cls.MIN_STOP_DISTANCE_BPS != Decimal("96")
        or cls.MIN_STOP_DISTANCE_FRACTION != Decimal("0.0096")
        or cls.MIN_STOP_DISTANCE_FRACTION
        != cls.MIN_STOP_DISTANCE_BPS / Decimal("10000")
        or Decimal(initial_risk["minimum_stop_distance_bps"])
        != cls.MIN_STOP_DISTANCE_BPS
        or Decimal(initial_risk["minimum_stop_distance_entry_fraction"])
        != cls.MIN_STOP_DISTANCE_FRACTION
    ):
        raise ValueError("Frozen V6-H0 96-bps stop floor changed.")


def validate_eligible_windows(windows: tuple[ResearchWindow, ...]) -> None:
    if len(windows) != REQUIRED_ELIGIBLE_WINDOWS:
        raise ValueError(
            f"V6-H0 requires exactly 11 eligible windows, got {len(windows)}."
        )
    for window in windows:
        if window.data_status is not PartitionStatus.CONSUMED_RESEARCH_DATA:
            raise ValueError("V6-H0 window is not consumed research data.")
        if (
            window.replay_dataset.symbol != "BTCUSDC"
            or window.replay_dataset.interval != "15m"
        ):
            raise ValueError("V6-H0 window market changed.")


def expand_complete_region_history(
    windows: tuple[ResearchWindow, ...],
    regions: tuple,
) -> tuple[ResearchWindow, ...]:
    """Retain each window boundary but supply all safe regional history."""

    expanded: list[ResearchWindow] = []
    for window in windows:
        owner = next(
            (
                region
                for region in regions
                if region.metadata.kind is window.partition_kind
                and region.metadata.start <= window.start
                and window.end <= region.metadata.end
            ),
            None,
        )
        if owner is None:
            raise ValueError("V6-H0 window has no consumed-data owner region.")
        prior = tuple(
            candle
            for candle in owner.dataset.candles
            if candle.timestamp < window.start
        )
        evaluation = tuple(
            candle
            for candle in owner.dataset.candles
            if window.start <= candle.timestamp < window.end
        )
        if not evaluation:
            raise ValueError("V6-H0 window evaluation candles are empty.")
        replay = HistoricalDataset(
            symbol=owner.dataset.symbol,
            interval=owner.dataset.interval,
            source=owner.dataset.source,
            candles=prior + evaluation,
        )
        expanded.append(
            replace(
                window,
                replay_dataset=replay,
                evaluation_start_index=len(prior),
            )
        )
    return tuple(expanded)


def evaluation_days_total(windows: tuple[ResearchWindow, ...]) -> Decimal:
    """Sum declared evaluation durations; warmup candles are excluded."""

    return sum(
        (window.duration_days for window in windows),
        start=Decimal("0"),
    )


def _stop_observations(
    *,
    trades: tuple[Trade, ...],
    dataset: HistoricalDataset,
    prepared_context: V6PreparedContext,
) -> tuple[StopObservation, ...]:
    signal_candles = {
        next_open_time(candle.timestamp, dataset.interval): candle
        for candle in dataset.candles
    }
    output: list[StopObservation] = []
    for trade in trades:
        try:
            signal_candle = signal_candles[trade.entry_signal_time]
        except KeyError as exc:
            raise ValueError(
                f"V6-H0 signal candle missing for {trade.trade_id}."
            ) from exc
        state = prepared_context.state_at(signal_candle.timestamp)
        if state is None:
            raise ValueError(f"V6-H0 4h ATR missing for {trade.trade_id}.")
        output.append(
            classify_stop_source(
                trade_id=trade.trade_id,
                entry_signal_time=trade.entry_signal_time,
                entry_price=trade.entry_price,
                atr_distance=state.atr_14,
                minimum_stop_distance_fraction=(
                    V6MTFContinuationStrategy.MIN_STOP_DISTANCE_FRACTION
                ),
            )
        )
    return tuple(output)


def _risk_audit_signals(
    *,
    signals: tuple[SignalRecord, ...],
    observations: tuple[StopObservation, ...],
    strategy_config: TrendMomentumConfig,
) -> tuple[SignalRecord, ...]:
    """Feed actual V6 initial distance through the shared risk audit."""

    by_signal_time = {item.entry_signal_time: item for item in observations}
    adjusted: list[SignalRecord] = []
    for signal in signals:
        observation = by_signal_time.get(signal.timestamp)
        if signal.action is StrategyAction.ENTER_LONG and observation is not None:
            signal = replace(
                signal,
                atr=(
                    observation.initial_stop_distance
                    / strategy_config.atr_stop_multiplier
                ),
            )
        adjusted.append(signal)
    return tuple(adjusted)


def _evaluate_window(
    *,
    window: ResearchWindow,
    strategy_config: TrendMomentumConfig,
    backtest_config: BacktestConfig,
    prepared_context: V6PreparedContext,
):
    evaluation = evaluate_strategy_period(
        window.replay_dataset,
        strategy=V6MTFContinuationStrategy(
            strategy_config,
            prepared_context=prepared_context,
        ),
        strategy_config=strategy_config,
        backtest_config=backtest_config,
        evaluation_start_index=window.evaluation_start_index,
    )
    observations = _stop_observations(
        trades=evaluation.backtest.trades,
        dataset=window.replay_dataset,
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
    risk_by_trade_id = {
        record.trade_id: record.actual_stop_risk for record in risk_records
    }
    r_records = build_r_normalized_trades(
        trades=evaluation.backtest.trades,
        actual_stop_risk_by_trade_id=risk_by_trade_id,
        slippage_bps=backtest_config.slippage_bps,
    )
    if len(r_records) != len(evaluation.backtest.trades):
        raise ValueError(f"R-record mismatch in {window.window_id}.")
    return (
        evaluation,
        r_records,
        summarize_r_normalized_trades(r_records),
        observations,
    )


def _window_result(
    *, window: ResearchWindow, evaluation, summary: RNormalizedSummary
) -> WindowResult:
    days = window.duration_days
    trades = summary.trade_count
    return WindowResult(
        window_id=window.window_id,
        trades=trades,
        evaluation_days=days,
        trades_per_day=(Decimal(trades) / days if days > 0 else Decimal("0")),
        frictionless_expectancy_r=summary.frictionless_expectancy_r,
        net_expectancy_r=summary.net_expectancy_r,
        profit_factor_r=summary.profit_factor_r,
        win_rate_percent=summary.win_rate_percent,
        average_winner_r=summary.average_winner_r,
        average_loser_r=summary.average_loser_r,
        payoff_ratio=summary.payoff_ratio_r,
        maximum_drawdown_percent=_max_drawdown_percent(
            evaluation.backtest.equity_curve
        ),
        net_r_per_day=(
            summary.net_expectancy_r * Decimal(trades) / days
            if days > 0
            else Decimal("0")
        ),
        positive_net=summary.net_expectancy_r > 0,
    )


def build_summary_payload(
    *,
    manifest: dict,
    base_summary: RNormalizedSummary,
    stress_summary: RNormalizedSummary,
    window_results: tuple[WindowResult, ...],
    maximum_drawdown_percent: Decimal,
    evaluation_days: Decimal,
    cost_diagnostics: dict,
) -> dict:
    gate = evaluate_v6_h0_gate(
        metrics=V6H0Metrics(
            net_expectancy_r=base_summary.net_expectancy_r,
            profit_factor_r=base_summary.profit_factor_r,
        ),
        window_net_expectancies_r=tuple(
            row.net_expectancy_r for row in window_results
        ),
    )
    trades = Decimal(base_summary.trade_count)
    trades_per_day = trades / evaluation_days if evaluation_days > 0 else Decimal("0")
    net_r_per_day = (
        base_summary.net_expectancy_r * trades / evaluation_days
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
    return {
        "version": "6.0",
        "strategy_id": "V6MTFContinuationStrategy",
        "baseline_id": "V6_H0",
        "run_id": EXPECTED_RUN_ID,
        "market": manifest["market"],
        "dataset_status": "CONSUMED_RESEARCH_DATA",
        "eligible_windows": REQUIRED_ELIGIBLE_WINDOWS,
        "blind_holdout_integrity": manifest["dataset_policy"]["blind_holdout"],
        "costs": manifest["costs"],
        "window_results_csv": "window_results.csv",
        "aggregate": {
            "total_trades": base_summary.trade_count,
            "total_evaluation_days": evaluation_days,
            "trades_per_day": trades_per_day,
            "frictionless_expectancy_r_per_trade": (
                base_summary.frictionless_expectancy_r
            ),
            "net_expectancy_r_per_trade": base_summary.net_expectancy_r,
            "average_friction_r_per_trade": (
                base_summary.average_total_friction_r
            ),
            "profit_factor_r": base_summary.profit_factor_r,
            "win_rate_percent": base_summary.win_rate_percent,
            "average_winner_r": base_summary.average_winner_r,
            "average_loser_r": base_summary.average_loser_r,
            "payoff_ratio": base_summary.payoff_ratio_r,
            "maximum_drawdown_percent": maximum_drawdown_percent,
            "positive_net_windows": gate.positive_net_windows,
            "positive_net_windows_total": gate.eligible_windows,
            "positive_net_window_ratio": gate.positive_net_window_ratio,
            "net_r_per_day": net_r_per_day,
            "base_cost_metrics": asdict(base_summary),
        },
        "stress_2x": {
            "fee_bps_per_side": Decimal("20"),
            "slippage_bps_per_side": Decimal("4"),
            "frozen_stop_floor_bps": Decimal("96"),
            "frozen_stop_floor_fraction": Decimal("0.0096"),
            "net_expectancy_r_per_trade": stress_summary.net_expectancy_r,
            "profit_factor_r": stress_summary.profit_factor_r,
            "net_r_per_day": stress_net_r_per_day,
            "metrics": asdict(stress_summary),
        },
        "cost_aware_diagnostics": cost_diagnostics,
        "gate": {
            **asdict(gate),
            "positive_net_window_ratio": gate.positive_net_window_ratio,
            "all_conditions_met": gate.all_conditions_met,
        },
        "progression_eligible": gate.all_conditions_met,
    }


def write_reports(
    *,
    output: Path,
    summary_payload: dict,
    window_results: tuple[WindowResult, ...],
) -> tuple[Path, Path]:
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise ValueError(
            "V6-H0 result already exists; refusing to rerun or overwrite."
        )
    if not window_results:
        raise ValueError("V6-H0 window results are empty.")

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "window_results.csv"
    temporary_csv = output / "window_results.csv.tmp"
    with temporary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=tuple(asdict(window_results[0]).keys()),
        )
        writer.writeheader()
        for row in window_results:
            writer.writerow(_json_safe(asdict(row)))
    temporary_csv.replace(csv_path)

    temporary_summary = output / "summary.json.tmp"
    temporary_summary.write_text(
        json.dumps(_json_safe(summary_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)
    return summary_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the single preregistered V6-H0 replay."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly authorize the single V6-H0 consumed-data replay.",
    )
    parser.add_argument(
        "--manifest",
        default=(
            "research/v6_mtf_continuation/e1eef7bdd0c37ad4/manifest.json"
        ),
    )
    parser.add_argument(
        "--root-manifest", default="research/hypothesis_manifest.json"
    )
    parser.add_argument(
        "--mechanism-report", default="reports/mechanisms/bc2496aed05555b5"
    )
    parser.add_argument(
        "--expansion-data-root",
        default="data/historical/multiregime_expansion",
    )
    parser.add_argument("--consumed-data-root", default="data/historical")
    parser.add_argument("--output-root", default="reports/v6_mtf_continuation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    logger = get_logger(__name__)
    if not args.execute:
        logger.error("V6-H0 replay not executed. Explicit --execute is required.")
        return 1

    try:
        manifest = _load_json(args.manifest)
        validate_v6_h0_manifest(manifest)
        validate_v6_h0_implementation(manifest)
        output = Path(args.output_root) / EXPECTED_RUN_ID
        if (output / "summary.json").exists():
            raise ValueError(
                "V6-H0 result already exists; refusing to rerun or overwrite."
            )

        root_manifest = ResearchManifestStore(args.root_manifest).load()
        _require_locked_holdout(root_manifest)
        if (
            root_manifest.holdout_status
            is not HoldoutStatus.LOCKED_BLIND_HOLDOUT
            or root_manifest.blind_holdout.reveal_timestamp is not None
            or root_manifest.blind_holdout.consumed_timestamp is not None
        ):
            raise ValueError("Blind holdout contamination detected.")

        strategy_config = TrendMomentumConfig()
        base_config = BacktestConfig()
        stress_config = replace(
            base_config,
            fee_bps=Decimal("20"),
            slippage_bps=Decimal("4"),
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
        windows = construct_windows(regions, load_settings().multiregime_config())
        reference_windows = _read_reference_windows(Path(args.mechanism_report))
        eligible = tuple(
            window for window in windows if window.window_id in reference_windows
        )
        eligible_windows = expand_complete_region_history(eligible, regions)
        validate_eligible_windows(eligible_windows)

        logger.info("V6-H0 PREREGISTERED REPLAY")
        logger.info("Windows: 11 | CONSUMED_RESEARCH_DATA")
        logger.warning(
            "Blind holdout: LOCKED | NOT LOADED | NOT REVEALED | "
            "NOT CONSUMED | NOT EVALUATED"
        )

        base_window_records: list[tuple[RNormalizedTrade, ...]] = []
        stress_window_records: list[tuple[RNormalizedTrade, ...]] = []
        base_observations: list[StopObservation] = []
        window_results: list[WindowResult] = []
        maximum_drawdown = Decimal("0")
        final_buy_signals = 0

        for window in eligible_windows:
            prepared_context = prepare_v6_context(window.replay_dataset.candles)
            evaluation, records, summary, observations = _evaluate_window(
                window=window,
                strategy_config=strategy_config,
                backtest_config=base_config,
                prepared_context=prepared_context,
            )
            row = _window_result(
                window=window, evaluation=evaluation, summary=summary
            )
            window_results.append(row)
            base_window_records.append(records)
            base_observations.extend(observations)
            final_buy_signals += sum(
                signal.action is StrategyAction.ENTER_LONG
                for signal in evaluation.signals
            )
            maximum_drawdown = max(
                maximum_drawdown, row.maximum_drawdown_percent
            )
            logger.info(
                "%s | trades=%d | gross=%sR | net=%sR | PF=%s | positive=%s",
                row.window_id,
                row.trades,
                row.frictionless_expectancy_r,
                row.net_expectancy_r,
                row.profit_factor_r,
                row.positive_net,
            )
            _, records, _, _ = _evaluate_window(
                window=window,
                strategy_config=strategy_config,
                backtest_config=stress_config,
                prepared_context=prepared_context,
            )
            stress_window_records.append(records)

        base_records = tuple(
            record for group in base_window_records for record in group
        )
        base_summary = summarize_combined_records(tuple(base_window_records))
        stress_summary = summarize_combined_records(tuple(stress_window_records))
        summary_payload = build_summary_payload(
            manifest=manifest,
            base_summary=base_summary,
            stress_summary=stress_summary,
            window_results=tuple(window_results),
            maximum_drawdown_percent=maximum_drawdown,
            evaluation_days=evaluation_days_total(eligible_windows),
            cost_diagnostics=build_cost_diagnostics(
                observations=tuple(base_observations),
                records=base_records,
                final_buy_signals=final_buy_signals,
            ),
        )
        summary_path, csv_path = write_reports(
            output=output,
            summary_payload=summary_payload,
            window_results=tuple(window_results),
        )
        logger.info("Progression eligible: %s", summary_payload["progression_eligible"])
        logger.info("Summary: %s", summary_path.resolve())
        logger.info("Windows: %s", csv_path.resolve())
        logger.warning(
            "Blind holdout: LOCKED | NOT LOADED | NOT REVEALED | "
            "NOT CONSUMED | NOT EVALUATED"
        )
        return 0
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        logger.error("V6-H0 replay failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
