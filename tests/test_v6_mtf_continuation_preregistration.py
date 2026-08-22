import json

import pytest

from src.research.v6_mtf_continuation_preregistration import (
    build_manifest,
    write_manifest,
)


EXPECTED_RUN_ID = "e1eef7bdd0c37ad4"


def test_v6_h0_identity_and_market_are_exact() -> None:
    manifest = build_manifest()

    assert manifest["version"] == "6.0"
    assert manifest["baseline_id"] == "V6_H0"
    assert manifest["strategy_family"] == "COST_AWARE_MTF_PULLBACK_CONTINUATION"
    assert manifest["market"] == {
        "symbol": "BTCUSDC",
        "market_type": "BINANCE_PUBLIC_SPOT",
        "direction": "LONG_ONLY",
        "execution_interval": "15m",
        "higher_timeframe_regime_interval": "4h",
        "signal_candles": "FULLY_CLOSED_15M_CANDLES",
        "higher_timeframe_role": "CONTEXT_ONLY",
    }


def test_v6_h0_4h_construction_and_utc_alignment_are_frozen() -> None:
    construction = build_manifest()["higher_timeframe_construction"]

    assert construction["source"] == "EXISTING_15M_DATA"
    assert construction["consecutive_source_candles_per_target_candle"] == 16
    assert construction["alignment_timezone"] == "UTC"
    assert construction["boundary_hours_utc"] == [0, 4, 8, 12, 16, 20]


def test_v6_h0_4h_causality_excludes_forming_and_future_data() -> None:
    construction = build_manifest()["higher_timeframe_construction"]

    assert construction["only_fully_completed_4h_candles_allowed"] is True
    assert construction["forming_4h_candle_allowed_for_ohlc"] is False
    assert construction["forming_4h_candle_allowed_for_ema"] is False
    assert construction["forming_4h_candle_allowed_for_atr"] is False
    assert construction["forming_4h_candle_allowed_for_regime"] is False
    assert construction["future_15m_candle_may_influence_current_regime"] is False


def test_v6_h0_completed_4h_long_regime_is_exact() -> None:
    regime = build_manifest()["higher_timeframe_regime"]

    assert regime["price_field"] == "CLOSE"
    assert regime["ema_fast_period"] == 20
    assert regime["ema_slow_period"] == 50
    assert regime["indicator_candles"] == "FULLY_COMPLETED_4H_CANDLES_ONLY"
    assert regime["conditions"] == [
        "EMA20_4H_STRICTLY_GREATER_THAN_EMA50_4H",
        "LATEST_COMPLETED_4H_CLOSE_STRICTLY_GREATER_THAN_EMA20_4H",
    ]
    assert regime["all_conditions_required"] is True
    assert regime["equality_qualifies"] is False
    assert regime["crossover_event_required"] is False


def test_v6_h0_three_15m_entry_conditions_and_equalities_are_frozen() -> None:
    entry = build_manifest()["entry"]

    assert entry["ema_period_15m"] == 20
    assert entry["conditions"] == [
        "PREVIOUS_CLOSE_LESS_THAN_OR_EQUAL_TO_PREVIOUS_EMA20_15M",
        "CURRENT_CLOSE_STRICTLY_GREATER_THAN_CURRENT_EMA20_15M",
        "CURRENT_CLOSE_STRICTLY_GREATER_THAN_PREVIOUS_HIGH",
    ]
    assert entry["all_conditions_required"] is True
    assert entry["previous_close_equal_to_previous_ema_qualifies"] is True
    assert entry["current_close_equal_to_current_ema_qualifies"] is False
    assert entry["current_close_equal_to_previous_high_qualifies"] is False
    assert entry["intrabar_entry"] is False
    assert entry["execution"] == "NEXT_BAR_OPEN"


def test_v6_h0_atr4h_and_cost_floor_derivation_are_exact() -> None:
    risk = build_manifest()["volatility_and_initial_risk"]

    assert risk["atr_period"] == 14
    assert risk["atr_interval"] == "4h"
    assert risk["atr_source"] == "LATEST_FULLY_COMPLETED_4H_CANDLE_AT_SIGNAL_TIME"
    assert risk["base_round_trip_friction_bps"] == "24"
    assert risk["cost_distance_multiple"] == 4
    assert risk["minimum_stop_distance_bps"] == "96"
    assert risk["minimum_stop_distance_entry_fraction"] == "0.0096"
    assert risk["minimum_stop_distance_derivation"] == "4_TIMES_24_BPS_EQUALS_96_BPS"
    assert risk["initial_stop_distance"] == (
        "MAX_OF_ATR14_4H_AND_ENTRY_FILL_PRICE_TIMES_0.0096"
    )
    assert risk["initial_stop"] == "ACTUAL_ENTRY_FILL_MINUS_INITIAL_STOP_DISTANCE"
    assert risk["derivation_is_mechanical_not_optimized"] is True


def test_v6_h0_exit_and_forbidden_position_logic_are_frozen() -> None:
    risk = build_manifest()["risk_and_exit"]

    assert risk["reward_risk_ratio"] == "2"
    assert risk["take_profit"] == "ACTUAL_ENTRY_FILL_PLUS_2R"
    assert risk["maximum_hold_bars"] == 96
    assert risk["maximum_hold_interval"] == "15m"
    assert risk["cooldown_bars"] == 4
    assert risk["ambiguous_bar_policy"] == "STOP_FIRST"
    assert all(
        risk[name] is False
        for name in (
            "trailing_stop",
            "break_even_stop",
            "early_failure_exit",
            "partial_exits",
            "pyramiding",
            "averaging_down",
        )
    )


def test_v6_h0_filters_and_costs_are_exact() -> None:
    manifest = build_manifest()

    assert all(value is False for value in manifest["filters"].values())
    assert manifest["costs"] == {
        "base": {"fee_bps_per_side": "10", "slippage_bps_per_side": "2"},
        "stress_2x": {"fee_bps_per_side": "20", "slippage_bps_per_side": "4"},
    }


def test_v6_h0_metrics_and_diagnostics_are_reporting_only() -> None:
    evaluation = build_manifest()["evaluation"]

    assert evaluation["report"] == [
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
    ]
    assert len(evaluation["diagnostics"]) == 7
    assert evaluation["diagnostics_are_optimization_gates"] is False
    assert evaluation["stress_2x_is_progression_gate"] is False


def test_v6_h0_progression_gate_is_exact() -> None:
    assert build_manifest()["progression_gate"] == {
        "required_eligible_windows": 11,
        "combined_net_expectancy_r_gt": "0",
        "profit_factor_r_gt": "1",
        "positive_net_window_ratio_gte": "0.60",
        "minimum_positive_net_windows": 7,
        "all_conditions_required": True,
    }


def test_v6_h0_anti_tuning_restrictions_are_frozen() -> None:
    anti = build_manifest()["anti_overfitting"]

    assert anti["parameter_search"] is False
    assert anti["alternatives_forbidden_if_v6_h0_fails"] == [
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
    ]
    assert anti["future_mechanism_change_rule"] == (
        "SEPARATE_PREREGISTRATION_AND_NEW_HYPOTHESIS_JUSTIFICATION_REQUIRED"
    )


def test_v6_h0_keeps_blind_holdout_locked() -> None:
    policy = build_manifest()["dataset_policy"]
    holdout = policy["blind_holdout"]

    assert policy["research_data_status"] == "CONSUMED_RESEARCH_DATA"
    assert policy["eligible_windows"] == 11
    assert holdout["status"] == "LOCKED_BLIND_HOLDOUT"
    assert all(
        holdout[name] is False
        for name in ("loaded", "revealed", "consumed", "evaluated")
    )


def test_v6_h0_manifest_and_run_id_are_deterministic() -> None:
    first = build_manifest()
    second = build_manifest()

    assert first == second
    assert first["run_id"] == EXPECTED_RUN_ID
    assert len(first["definition_sha256"]) == 64


def test_v6_h0_manifest_writing_is_deterministic_and_immutable(tmp_path) -> None:
    first_path = write_manifest(tmp_path)
    second_path = write_manifest(tmp_path)

    assert first_path == second_path
    assert json.loads(first_path.read_text(encoding="utf-8")) == build_manifest()

    first_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from frozen definition"):
        write_manifest(tmp_path)
