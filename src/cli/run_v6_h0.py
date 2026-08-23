"""Dry-run-only integrity guard for the frozen V6-H0 replay."""

from __future__ import annotations

import argparse
import inspect
import json
from decimal import Decimal
from pathlib import Path

from src.backtest.engine import BacktestEngine
from src.backtest.models import (
    AmbiguousBarPolicy,
    BacktestConfig,
    ExecutionTiming,
)
from src.cli.run_v33_h5 import _read_reference_windows
from src.research.evaluation import StrategyDecisionAdapter
from src.research.v6_mtf_continuation_preregistration import build_manifest
from src.research.v6_mtf_continuation_stability import REQUIRED_ELIGIBLE_WINDOWS
from src.strategy.models import TrendMomentumConfig
from src.strategy.v6_mtf_continuation import (
    V6MTFContinuationStrategy,
    aggregate_completed_4h_candles,
    higher_timeframe_state,
)


EXPECTED_RUN_ID = "e1eef7bdd0c37ad4"


def _load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required V6-H0 manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_v6_h0_manifest(payload: dict) -> None:
    if payload.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("Unexpected V6-H0 preregistration run ID.")
    if payload != build_manifest():
        raise ValueError("V6-H0 manifest differs from the frozen definition.")


def _normalized_source(value: object) -> str:
    return " ".join(inspect.getsource(value).split())


def validate_actual_fill_execution() -> None:
    engine_source = _normalized_source(BacktestEngine._execute_pending)
    steps = (
        "fill_price = self._execution.buy_fill_price(candle.open)",
        "stop_distance = intent.stop_distance",
        "fill_price * intent.minimum_stop_distance_fraction",
        "stop_loss = fill_price - stop_distance",
        "take_profit = fill_price + ( stop_distance * intent.reward_risk_ratio )",
    )
    positions = tuple(engine_source.find(step) for step in steps)
    if any(position < 0 for position in positions) or positions != tuple(
        sorted(positions)
    ):
        raise ValueError("V6-H0 actual-fill stop/target execution changed.")
    if "candle.high" in engine_source or "candle.low" in engine_source:
        raise ValueError("V6-H0 bracket construction uses next-bar range data.")

    adapter_source = _normalized_source(StrategyDecisionAdapter.__call__)
    if (
        "stop_distance=decision.stop_distance" not in adapter_source
        or "decision.minimum_stop_distance_fraction" not in adapter_source
    ):
        raise ValueError("V6-H0 signal-time ATR/fill-floor intent changed.")

    run_source = _normalized_source(BacktestEngine.run)
    pending_position = run_source.find("if pending is not None:")
    protective_position = run_source.find("protective = self._execution.protective_exit")
    if (
        pending_position < 0
        or protective_position < 0
        or pending_position >= protective_position
    ):
        raise ValueError("V6-H0 pending-fill/protective-check timeline changed.")


def validate_v6_h0_implementation(manifest: dict) -> None:
    market = manifest["market"]
    construction = manifest["higher_timeframe_construction"]
    regime = manifest["higher_timeframe_regime"]
    entry = manifest["entry"]
    initial_risk = manifest["volatility_and_initial_risk"]
    position = manifest["risk_and_exit"]
    costs = manifest["costs"]
    filters = manifest["filters"]
    policy = manifest["dataset_policy"]

    if market != {
        "symbol": "BTCUSDC",
        "market_type": "BINANCE_PUBLIC_SPOT",
        "direction": "LONG_ONLY",
        "execution_interval": "15m",
        "higher_timeframe_regime_interval": "4h",
        "signal_candles": "FULLY_CLOSED_15M_CANDLES",
        "higher_timeframe_role": "CONTEXT_ONLY",
    }:
        raise ValueError("V6-H0 15m/4h market configuration changed.")

    required_construction = {
        "source_interval": "15m",
        "target_interval": "4h",
        "consecutive_source_candles_per_target_candle": 16,
        "alignment_timezone": "UTC",
        "boundary_hours_utc": [0, 4, 8, 12, 16, 20],
        "only_fully_completed_4h_candles_allowed": True,
        "forming_4h_candle_allowed_for_ohlc": False,
        "forming_4h_candle_allowed_for_ema": False,
        "forming_4h_candle_allowed_for_atr": False,
        "forming_4h_candle_allowed_for_regime": False,
        "future_15m_candle_may_influence_current_regime": False,
    }
    if any(
        construction.get(name) != expected
        for name, expected in required_construction.items()
    ):
        raise ValueError("V6-H0 completed-4h causality definition changed.")

    if regime != {
        "price_field": "CLOSE",
        "ema_fast_period": 20,
        "ema_slow_period": 50,
        "indicator_candles": "FULLY_COMPLETED_4H_CANDLES_ONLY",
        "state_candle": "LATEST_FULLY_COMPLETED_4H_CANDLE",
        "all_conditions_required": True,
        "conditions": [
            "EMA20_4H_STRICTLY_GREATER_THAN_EMA50_4H",
            "LATEST_COMPLETED_4H_CLOSE_STRICTLY_GREATER_THAN_EMA20_4H",
        ],
        "equality_qualifies": False,
        "crossover_event_required": False,
    }:
        raise ValueError("V6-H0 completed-4h regime changed.")

    if entry["conditions"] != [
        "PREVIOUS_CLOSE_LESS_THAN_OR_EQUAL_TO_PREVIOUS_EMA20_15M",
        "CURRENT_CLOSE_STRICTLY_GREATER_THAN_CURRENT_EMA20_15M",
        "CURRENT_CLOSE_STRICTLY_GREATER_THAN_PREVIOUS_HIGH",
    ] or any(
        (
            entry["ema_period_15m"] != 20,
            entry["ema_input"] != "CLOSED_15M_CANDLES_ONLY",
            entry["previous_close_equal_to_previous_ema_qualifies"] is not True,
            entry["current_close_equal_to_current_ema_qualifies"] is not False,
            entry["current_close_equal_to_previous_high_qualifies"] is not False,
            entry["intrabar_entry"] is not False,
            entry["execution"] != ExecutionTiming.NEXT_BAR_OPEN.value,
        )
    ):
        raise ValueError("V6-H0 causal 15m entry changed.")

    cls = V6MTFContinuationStrategy
    constants = {
        "EXECUTION_INTERVAL": market["execution_interval"],
        "HIGHER_TIMEFRAME_CONTEXT_INTERVAL": market[
            "higher_timeframe_regime_interval"
        ],
        "SOURCE_BARS_PER_4H_CANDLE": construction[
            "consecutive_source_candles_per_target_candle"
        ],
        "HIGHER_TIMEFRAME_FAST_EMA_PERIOD": regime["ema_fast_period"],
        "HIGHER_TIMEFRAME_SLOW_EMA_PERIOD": regime["ema_slow_period"],
        "PULLBACK_EMA_PERIOD": entry["ema_period_15m"],
        "ATR_PERIOD_4H": initial_risk["atr_period"],
        "BASE_FEE_BPS_PER_SIDE": Decimal(costs["base"]["fee_bps_per_side"]),
        "BASE_SLIPPAGE_BPS_PER_SIDE": Decimal(
            costs["base"]["slippage_bps_per_side"]
        ),
        "ROUND_TRIP_FRICTION_BPS": Decimal(
            initial_risk["base_round_trip_friction_bps"]
        ),
        "COST_DISTANCE_MULTIPLE": initial_risk["cost_distance_multiple"],
        "MIN_STOP_DISTANCE_BPS": Decimal(
            initial_risk["minimum_stop_distance_bps"]
        ),
        "MIN_STOP_DISTANCE_FRACTION": Decimal(
            initial_risk["minimum_stop_distance_entry_fraction"]
        ),
        "REWARD_RISK_RATIO": Decimal(position["reward_risk_ratio"]),
        "MAXIMUM_HOLD_BARS": position["maximum_hold_bars"],
        "COOLDOWN_BARS": position["cooldown_bars"],
    }
    if any(
        getattr(cls, name) != expected for name, expected in constants.items()
    ):
        raise ValueError("V6-H0 strategy constants changed.")

    if (
        cls.ROUND_TRIP_FRICTION_BPS
        != Decimal("2")
        * (cls.BASE_FEE_BPS_PER_SIDE + cls.BASE_SLIPPAGE_BPS_PER_SIDE)
        or cls.ROUND_TRIP_FRICTION_BPS != Decimal("24")
        or cls.MIN_STOP_DISTANCE_BPS
        != cls.ROUND_TRIP_FRICTION_BPS * cls.COST_DISTANCE_MULTIPLE
        or cls.MIN_STOP_DISTANCE_BPS != Decimal("96")
        or cls.MIN_STOP_DISTANCE_FRACTION
        != cls.MIN_STOP_DISTANCE_BPS / Decimal("10000")
        or cls.MIN_STOP_DISTANCE_FRACTION != Decimal("0.0096")
    ):
        raise ValueError("V6-H0 96-bps/0.0096 conversion changed.")

    strategy = cls(TrendMomentumConfig())
    if strategy.requires_full_history is not True:
        raise ValueError("V6-H0 complete signal-time history invariant changed.")
    adapter_init_source = _normalized_source(StrategyDecisionAdapter.__init__)
    adapter_call_source = _normalized_source(StrategyDecisionAdapter.__call__)
    if (
        "self._requires_full_history = strategy.requires_full_history"
        not in adapter_init_source
        or "0 if self._requires_full_history else max(" not in adapter_call_source
    ):
        raise ValueError("V6-H0 full-history adapter path changed.")

    aggregate_source = _normalized_source(aggregate_completed_4h_candles)
    state_source = _normalized_source(higher_timeframe_state)
    required_aggregate_source = (
        "if _utc_timestamp(candle.timestamp) <= as_of_utc",
        "source[index : index + SOURCE_BARS_PER_4H_CANDLE]",
        "candle.timestamp != expected_timestamp",
        "or not candle.is_closed",
        "close=group[-1].close",
    )
    if any(
        fragment not in aggregate_source for fragment in required_aggregate_source
    ) or any(
        fragment not in state_source
        for fragment in (
            "completed = aggregate_completed_4h_candles",
            "tuple(candle.close for candle in completed)",
            "atr(completed, V6MTFContinuationStrategy.ATR_PERIOD_4H)",
        )
    ):
        raise ValueError("V6-H0 completed-4h causal implementation changed.")

    strategy_source = _normalized_source(cls)
    required_strategy_source = (
        "previous_indicators = context.previous_indicators",
        "current_ema = context.indicators.ema_fast",
        "previous.close <= previous_ema",
        "current.close > current_ema",
        "current.close > previous.high",
        "stop_distance=state.atr_14",
        "minimum_stop_distance_fraction=self.MIN_STOP_DISTANCE_FRACTION",
    )
    if any(fragment not in strategy_source for fragment in required_strategy_source):
        raise ValueError("V6-H0 causal signal/risk implementation changed.")

    config = TrendMomentumConfig()
    if (
        config.fast_ema_period != cls.PULLBACK_EMA_PERIOD
        or config.atr_period != cls.ATR_PERIOD_4H
        or config.reward_risk_ratio != cls.REWARD_RISK_RATIO
        or config.maximum_bars_in_position != cls.MAXIMUM_HOLD_BARS
        or config.cooldown_bars != cls.COOLDOWN_BARS
    ):
        raise ValueError("V6-H0 frozen strategy configuration changed.")

    execution = BacktestConfig()
    if execution.execution_timing is not ExecutionTiming.NEXT_BAR_OPEN:
        raise ValueError("Backtest next-bar-open execution changed.")
    if execution.ambiguous_bar_policy is not AmbiguousBarPolicy.STOP_FIRST:
        raise ValueError("Backtest STOP_FIRST behavior changed.")
    if execution.fee_bps != Decimal("10") or execution.slippage_bps != Decimal("2"):
        raise ValueError("V6-H0 base execution costs changed.")
    if costs != {
        "base": {"fee_bps_per_side": "10", "slippage_bps_per_side": "2"},
        "stress_2x": {"fee_bps_per_side": "20", "slippage_bps_per_side": "4"},
    }:
        raise ValueError("V6-H0 cost model changed.")

    if any(filters.get(name) is not False for name in filters):
        raise ValueError("V6-H0 forbidden entry logic is enabled.")
    if any(
        position[name] is not False
        for name in (
            "trailing_stop",
            "break_even_stop",
            "early_failure_exit",
            "partial_exits",
            "pyramiding",
            "averaging_down",
        )
    ):
        raise ValueError("V6-H0 forbidden position logic is enabled.")
    if position["take_profit"] != "ACTUAL_ENTRY_FILL_PLUS_2R":
        raise ValueError("V6-H0 +2R target changed.")
    if position["ambiguous_bar_policy"] != AmbiguousBarPolicy.STOP_FIRST.value:
        raise ValueError("V6-H0 STOP_FIRST definition changed.")

    if policy["research_data_status"] != "CONSUMED_RESEARCH_DATA":
        raise ValueError("V6-H0 dataset policy changed.")
    if policy["eligible_windows"] != REQUIRED_ELIGIBLE_WINDOWS:
        raise ValueError("V6-H0 eligible-window count changed.")
    holdout = policy["blind_holdout"]
    if holdout["status"] != "LOCKED_BLIND_HOLDOUT" or any(
        holdout[name] is not False
        for name in ("loaded", "revealed", "consumed", "evaluated")
    ):
        raise ValueError("V6-H0 blind holdout integrity failed.")

    expected_gate = {
        "required_eligible_windows": 11,
        "combined_net_expectancy_r_gt": "0",
        "profit_factor_r_gt": "1",
        "positive_net_window_ratio_gte": "0.60",
        "minimum_positive_net_windows": 7,
        "all_conditions_required": True,
    }
    if manifest["progression_gate"] != expected_gate:
        raise ValueError("V6-H0 progression gate changed.")
    validate_actual_fill_execution()


def validate_eligible_windows(reference_windows: dict) -> None:
    if len(reference_windows) != REQUIRED_ELIGIBLE_WINDOWS:
        raise ValueError("V6-H0 requires exactly 11 eligible research windows.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run-only V6-H0 integrity guard."
    )
    parser.add_argument(
        "--manifest",
        default="research/v6_mtf_continuation/e1eef7bdd0c37ad4/manifest.json",
    )
    parser.add_argument(
        "--mechanism-report",
        default="reports/mechanisms/bc2496aed05555b5",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate V6-H0 only; replay execution remains disabled.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dry_run:
        raise SystemExit("V6-H0 REPLAY DISABLED. Run with --dry-run only.")

    manifest = _load_json(args.manifest)
    validate_v6_h0_manifest(manifest)
    validate_v6_h0_implementation(manifest)
    validate_eligible_windows(
        _read_reference_windows(Path(args.mechanism_report))
    )

    print()
    print("V6-H0 DRY-RUN INTEGRITY CHECK")
    print(f"Preregistration: {EXPECTED_RUN_ID} VERIFIED")
    print("Market: BTCUSDC Binance Public Spot | 15m execution | 4h context")
    print("Causality: UTC 16x15m completed 4h | forming/future excluded")
    print("Indicators: complete-history EMA20/EMA50/ATR14 | causal 15m EMA20 snapshots")
    print("Entry: P.close <= P.EMA20 | C.close > C.EMA20 and P.high")
    print("Execution: NEXT_BAR_OPEN fill | max(ATR4h, fill*0.0096) | +2R | STOP_FIRST")
    print("Costs: base 10/2 bps | stress 20/4 bps per side")
    print("Dataset: CONSUMED_RESEARCH_DATA | eligible windows: 11")
    print("Blind holdout: LOCKED | NOT LOADED | NOT REVEALED | NOT CONSUMED | NOT EVALUATED")
    print()
    print("DRY RUN PASSED — NO V6-H0 REPLAY EXECUTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
