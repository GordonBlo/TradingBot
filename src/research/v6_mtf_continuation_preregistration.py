"""Preregister V6-H0 cost-aware multi-timeframe continuation.

This module defines research intent only. It does not load market data or run
a backtest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


VERSION = "6.0"
BASELINE_ID = "V6_H0"
STRATEGY_FAMILY = "COST_AWARE_MTF_PULLBACK_CONTINUATION"

SYMBOL = "BTCUSDC"
EXECUTION_INTERVAL = "15m"
REGIME_INTERVAL = "4h"
SOURCE_BARS_PER_REGIME_CANDLE = 16

REGIME_FAST_EMA_PERIOD = 20
REGIME_SLOW_EMA_PERIOD = 50
PULLBACK_EMA_PERIOD = 20
ATR_PERIOD_4H = 14

FEE_BPS_PER_SIDE = "10"
SLIPPAGE_BPS_PER_SIDE = "2"
ROUND_TRIP_FRICTION_BPS = "24"
COST_DISTANCE_MULTIPLE = 4
MIN_STOP_DISTANCE_BPS = "96"

REWARD_RISK_RATIO = "2"
MAX_HOLD_15M_BARS = 96
COOLDOWN_BARS = 4
ELIGIBLE_WINDOWS = 11

HOLDOUT_START = "2025-08-01T00:00:00+00:00"
HOLDOUT_END = "2026-02-01T00:00:00+00:00"


def _definition() -> dict:
    return {
        "version": VERSION,
        "baseline_id": BASELINE_ID,
        "strategy_family": STRATEGY_FAMILY,
        "market": {
            "symbol": SYMBOL,
            "market_type": "BINANCE_PUBLIC_SPOT",
            "direction": "LONG_ONLY",
            "execution_interval": EXECUTION_INTERVAL,
            "higher_timeframe_regime_interval": REGIME_INTERVAL,
            "signal_candles": "FULLY_CLOSED_15M_CANDLES",
            "higher_timeframe_role": "CONTEXT_ONLY",
        },
        "higher_timeframe_construction": {
            "source": "EXISTING_15M_DATA",
            "source_interval": EXECUTION_INTERVAL,
            "target_interval": REGIME_INTERVAL,
            "consecutive_source_candles_per_target_candle": (
                SOURCE_BARS_PER_REGIME_CANDLE
            ),
            "aggregation": "DETERMINISTIC_15M_TO_4H_OHLCV",
            "alignment_timezone": "UTC",
            "boundary_hours_utc": [0, 4, 8, 12, 16, 20],
            "only_fully_completed_4h_candles_allowed": True,
            "forming_4h_candle_allowed_for_ohlc": False,
            "forming_4h_candle_allowed_for_ema": False,
            "forming_4h_candle_allowed_for_atr": False,
            "forming_4h_candle_allowed_for_regime": False,
            "future_15m_candle_may_influence_current_regime": False,
        },
        "higher_timeframe_regime": {
            "price_field": "CLOSE",
            "ema_fast_period": REGIME_FAST_EMA_PERIOD,
            "ema_slow_period": REGIME_SLOW_EMA_PERIOD,
            "indicator_candles": "FULLY_COMPLETED_4H_CANDLES_ONLY",
            "state_candle": "LATEST_FULLY_COMPLETED_4H_CANDLE",
            "all_conditions_required": True,
            "conditions": [
                "EMA20_4H_STRICTLY_GREATER_THAN_EMA50_4H",
                "LATEST_COMPLETED_4H_CLOSE_STRICTLY_GREATER_THAN_EMA20_4H",
            ],
            "equality_qualifies": False,
            "crossover_event_required": False,
        },
        "entry": {
            "mechanism": "MTF_PULLBACK_RECLAIM_CONTINUATION",
            "signal_candle": "CURRENT_FULLY_CLOSED_15M_CANDLE",
            "previous_candle": "PREVIOUS_FULLY_CLOSED_15M_CANDLE",
            "ema_period_15m": PULLBACK_EMA_PERIOD,
            "ema_input": "CLOSED_15M_CANDLES_ONLY",
            "higher_timeframe_long_regime_required": True,
            "all_conditions_required": True,
            "conditions": [
                "PREVIOUS_CLOSE_LESS_THAN_OR_EQUAL_TO_PREVIOUS_EMA20_15M",
                "CURRENT_CLOSE_STRICTLY_GREATER_THAN_CURRENT_EMA20_15M",
                "CURRENT_CLOSE_STRICTLY_GREATER_THAN_PREVIOUS_HIGH",
            ],
            "previous_close_equal_to_previous_ema_qualifies": True,
            "current_close_equal_to_current_ema_qualifies": False,
            "current_close_equal_to_previous_high_qualifies": False,
            "intrabar_entry": False,
            "execution": "NEXT_BAR_OPEN",
        },
        "volatility_and_initial_risk": {
            "atr_period": ATR_PERIOD_4H,
            "atr_interval": REGIME_INTERVAL,
            "atr_source": "LATEST_FULLY_COMPLETED_4H_CANDLE_AT_SIGNAL_TIME",
            "forming_4h_candle_excluded": True,
            "base_round_trip_cost_formula": "2_TIMES_(10_BPS_FEE_PLUS_2_BPS_SLIPPAGE)",
            "base_round_trip_friction_bps": ROUND_TRIP_FRICTION_BPS,
            "cost_distance_multiple": COST_DISTANCE_MULTIPLE,
            "minimum_stop_distance_derivation": "4_TIMES_24_BPS_EQUALS_96_BPS",
            "minimum_stop_distance_bps": MIN_STOP_DISTANCE_BPS,
            "minimum_stop_distance_entry_fraction": "0.0096",
            "initial_stop_distance": "MAX_OF_ATR14_4H_AND_ENTRY_FILL_PRICE_TIMES_0.0096",
            "initial_stop": "ACTUAL_ENTRY_FILL_MINUS_INITIAL_STOP_DISTANCE",
            "derivation_is_mechanical_not_optimized": True,
            "geometry_intent": (
                "CAP_BASE_FRICTION_RELATIVE_TO_INITIAL_RISK_AT_APPROXIMATELY_"
                "0.25R_OR_LESS_BEFORE_GAP_AND_FILL_EFFECTS"
            ),
        },
        "risk_and_exit": {
            "reward_risk_ratio": REWARD_RISK_RATIO,
            "take_profit": "ACTUAL_ENTRY_FILL_PLUS_2R",
            "r_definition": "FROZEN_INITIAL_STOP_DISTANCE_FROM_ACTUAL_ENTRY_FILL",
            "maximum_hold_bars": MAX_HOLD_15M_BARS,
            "maximum_hold_interval": EXECUTION_INTERVAL,
            "cooldown_bars": COOLDOWN_BARS,
            "ambiguous_bar_policy": "STOP_FIRST",
            "strategy_exit_execution": "NEXT_BAR_OPEN_WHERE_EXISTING_ARCHITECTURE_REQUIRES",
            "trailing_stop": False,
            "break_even_stop": False,
            "early_failure_exit": False,
            "partial_exits": False,
            "pyramiding": False,
            "averaging_down": False,
        },
        "filters": {
            "rsi_filter": False,
            "volume_filter": False,
            "adx_filter": False,
            "atr_percentile_filter": False,
            "bollinger_band_filter": False,
            "breakout_lookback_filter": False,
            "h5_volatility_band_filter": False,
            "v5_statistical_exhaustion_filter": False,
            "time_of_day_filter": False,
            "day_of_week_filter": False,
        },
        "costs": {
            "base": {
                "fee_bps_per_side": FEE_BPS_PER_SIDE,
                "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
            },
            "stress_2x": {
                "fee_bps_per_side": "20",
                "slippage_bps_per_side": "4",
            },
        },
        "dataset_policy": {
            "research_data_status": "CONSUMED_RESEARCH_DATA",
            "eligible_windows": ELIGIBLE_WINDOWS,
            "blind_holdout": {
                "status": "LOCKED_BLIND_HOLDOUT",
                "start": HOLDOUT_START,
                "end": HOLDOUT_END,
                "loaded": False,
                "revealed": False,
                "consumed": False,
                "evaluated": False,
            },
        },
        "evaluation": {
            "primary_metric": "NET_EXPECTANCY_R",
            "report": [
                "TOTAL_TRADES",
                "TRADES_PER_DAY",
                "FRICTIONLESS_EXPECTANCY_R_PER_TRADE",
                "NET_EXPECTANCY_R_PER_TRADE",
                "AVERAGE_FRICTION_R_PER_TRADE",
                "FRICTION_TO_INITIAL_RISK_RATIO",
                "PROFIT_FACTOR_R",
                "WIN_RATE",
                "AVERAGE_WINNER_R",
                "AVERAGE_LOSER_R",
                "PAYOFF_RATIO",
                "MAXIMUM_DRAWDOWN_PERCENT",
                "POSITIVE_NET_WINDOWS_OUT_OF_11",
                "MEDIAN_TRADES_PER_WINDOW",
                "STRESS_2X_NET_EXPECTANCY_R_PER_TRADE",
                "NET_R_PER_DAY",
            ],
            "diagnostics": [
                "PERCENT_TIME_4H_REGIME_ACTIVE",
                "COUNT_15M_PULLBACK_RECLAIMS",
                "COUNT_SURVIVING_PREVIOUS_HIGH_CONFIRMATION",
                "AVERAGE_INITIAL_STOP_DISTANCE_BPS",
                "MEDIAN_INITIAL_STOP_DISTANCE_BPS",
                "PERCENT_STOPS_DETERMINED_BY_ATR14_4H",
                "PERCENT_STOPS_DETERMINED_BY_96_BPS_COST_FLOOR",
            ],
            "diagnostics_are_optimization_gates": False,
            "stress_2x_is_progression_gate": False,
        },
        "progression_gate": {
            "required_eligible_windows": 11,
            "combined_net_expectancy_r_gt": "0",
            "profit_factor_r_gt": "1",
            "positive_net_window_ratio_gte": "0.60",
            "minimum_positive_net_windows": 7,
            "all_conditions_required": True,
        },
        "anti_overfitting": {
            "parameter_search": False,
            "alternatives_forbidden_if_v6_h0_fails": [
                "1H_INSTEAD_OF_4H",
                "EMA10_30",
                "EMA50_200",
                "COST_DISTANCE_MULTIPLE_3X_OR_5X",
                "ALTERNATE_ATR_PERIODS",
                "ALTERNATE_STOP_FLOORS",
                "ALTERNATE_PULLBACK_EMA",
                "ALTERNATE_PREVIOUS_HIGH_CONFIRMATION",
                "ALTERNATE_MAXIMUM_HOLD",
                "ALTERNATE_COOLDOWN",
            ],
            "future_mechanism_change_rule": (
                "SEPARATE_PREREGISTRATION_AND_NEW_HYPOTHESIS_JUSTIFICATION_REQUIRED"
            ),
        },
        "research_rule": (
            "V6_H0_IS_A_NEW_COST_AWARE_MTF_CONTINUATION_FAMILY_"
            "AND_NOT_POST_HOC_TUNING_OF_V3_V4_OR_V5"
        ),
    }


def _canonical_json(payload: dict) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def build_manifest() -> dict:
    definition = _definition()
    digest = hashlib.sha256(
        _canonical_json(definition).encode("utf-8")
    ).hexdigest()
    return {
        "run_id": digest[:16],
        "definition_sha256": digest,
        **definition,
    }


def write_manifest(
    root: str | Path = "research/v6_mtf_continuation",
) -> Path:
    manifest = build_manifest()
    output_dir = Path(root) / manifest["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"

    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(
                "Existing V6-H0 manifest differs from frozen definition."
            )
        return path

    path.write_text(serialized, encoding="utf-8")
    return path


def main() -> int:
    manifest = build_manifest()
    path = write_manifest()

    print()
    print("V6-H0 COST-AWARE MTF CONTINUATION PREREGISTRATION")
    print(f"Run ID: {manifest['run_id']}")
    print("Market: BTCUSDC Binance Public Spot | 15m | LONG ONLY")
    print("Regime: completed UTC-aligned 4h candles only | EMA20 > EMA50")
    print("Entry: closed 15m EMA20 reclaim + previous-high confirmation")
    print("Risk: max(ATR14_4H, 96 bps cost floor) | +2R")
    print("Dataset: CONSUMED_RESEARCH_DATA | eligible windows: 11")
    print(
        "Blind holdout: LOCKED | NOT LOADED | "
        "NOT REVEALED | NOT CONSUMED | NOT EVALUATED"
    )
    print(f"Manifest: {path}")
    print()
    print("PREREGISTRATION COMPLETE — NO BACKTEST EXECUTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
