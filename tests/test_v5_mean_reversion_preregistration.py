import json

from src.research.v5_mean_reversion_preregistration import (
    build_manifest,
    write_manifest,
)


def test_v5_h0_exact_market_and_reference_window() -> None:
    manifest = build_manifest()
    market = manifest["market"]
    reference = manifest["reference_distribution"]

    assert manifest["baseline_id"] == "V5_H0"
    assert market == {
        "symbol": "BTCUSDC",
        "interval": "15m",
        "market_type": "BINANCE_PUBLIC_SPOT",
        "direction": "LONG_ONLY",
    }
    assert reference["lookback_bars"] == 20
    assert reference["candles"] == "PREVIOUS_EXACTLY_20_FULLY_CLOSED_CANDLES"
    assert reference["current_candle_excluded"] is True
    assert reference["price_field"] == "CLOSE"


def test_v5_h0_population_distribution_is_exactly_frozen() -> None:
    reference = build_manifest()["reference_distribution"]

    assert reference["mean"] == "ARITHMETIC_MEAN_OF_PREVIOUS_20_CLOSES"
    assert reference["standard_deviation"] == "POPULATION_STANDARD_DEVIATION"
    assert reference["standard_deviation_ddof"] == 0
    assert reference["deviation_multiple"] == "2"
    assert reference["lower_band"] == "REFERENCE_MEAN_MINUS_2_TIMES_REFERENCE_STD"
    assert reference["zero_standard_deviation_rule"] == "NO_SIGNAL"


def test_v5_h0_three_strict_entry_conditions_are_frozen() -> None:
    entry = build_manifest()["entry"]

    assert entry["conditions"] == [
        "CURRENT_LOW_STRICTLY_LESS_THAN_LOWER_BAND",
        "CURRENT_CLOSE_STRICTLY_GREATER_THAN_LOWER_BAND",
        "CURRENT_CLOSE_STRICTLY_LESS_THAN_REFERENCE_MEAN",
    ]
    assert entry["all_conditions_required"] is True
    assert entry["strict_inequalities"] is True
    assert entry["equality_qualifies"] is False
    assert entry["execution"] == "NEXT_BAR_OPEN"


def test_v5_h0_risk_exit_and_costs_are_frozen() -> None:
    manifest = build_manifest()
    risk = manifest["risk_and_exit"]

    assert risk["atr_period"] == 14
    assert risk["stop_atr_multiple"] == "2"
    assert risk["initial_stop"] == "ENTRY_MINUS_2_TIMES_SIGNAL_CANDLE_ATR14"
    assert risk["take_profit"] == "PLUS_2R"
    assert risk["reward_risk_ratio"] == "2"
    assert risk["maximum_hold_bars"] == 96
    assert risk["cooldown_bars"] == 4
    assert risk["ambiguous_bar_policy"] == "STOP_FIRST"
    assert manifest["costs"] == {
        "base": {"fee_bps_per_side": "10", "slippage_bps_per_side": "2"},
        "stress_2x": {"fee_bps_per_side": "20", "slippage_bps_per_side": "4"},
    }


def test_v5_h0_has_no_old_filters_or_adaptive_exits() -> None:
    manifest = build_manifest()

    assert manifest["filters"] == {
        "ema_filter": False,
        "rsi_filter": False,
        "volume_filter": False,
        "trend_regime_filter": False,
        "atr_percentile_filter": False,
        "breakout_filter": False,
    }
    assert manifest["risk_and_exit"]["trailing_stop"] is False
    assert manifest["risk_and_exit"]["break_even_stop"] is False
    assert manifest["risk_and_exit"]["early_failure_exit"] is False


def test_v5_h0_progression_gate_is_exact() -> None:
    gate = build_manifest()["progression_gate"]

    assert gate == {
        "required_eligible_windows": 11,
        "combined_net_expectancy_r_gt": "0",
        "profit_factor_r_gt": "1",
        "positive_net_window_ratio_gte": "0.60",
        "minimum_positive_net_windows": 7,
        "all_conditions_required": True,
    }


def test_v5_h0_evaluation_metrics_are_preregistered() -> None:
    assert build_manifest()["evaluation"]["report"] == [
        "TOTAL_TRADES",
        "FRICTIONLESS_EXPECTANCY_R_PER_TRADE",
        "NET_EXPECTANCY_R_PER_TRADE",
        "PROFIT_FACTOR_R",
        "WIN_RATE",
        "MAXIMUM_DRAWDOWN_PERCENT",
        "POSITIVE_NET_WINDOWS_OUT_OF_11",
        "STRESS_2X_NET_EXPECTANCY_R",
        "TRADES_PER_DAY",
        "NET_R_PER_DAY",
    ]


def test_v5_h0_keeps_blind_holdout_locked() -> None:
    policy = build_manifest()["dataset_policy"]
    holdout = policy["blind_holdout"]

    assert policy["research_data_status"] == "CONSUMED_RESEARCH_DATA"
    assert policy["eligible_windows"] == 11
    assert holdout["status"] == "LOCKED_BLIND_HOLDOUT"
    assert holdout["loaded"] is False
    assert holdout["revealed"] is False
    assert holdout["consumed"] is False
    assert holdout["evaluated"] is False


def test_v5_h0_manifest_and_run_id_are_deterministic() -> None:
    first = build_manifest()
    second = build_manifest()

    assert first == second
    assert len(first["run_id"]) == 16
    assert len(first["definition_sha256"]) == 64


def test_v5_h0_manifest_writes_deterministically(tmp_path) -> None:
    first_path = write_manifest(tmp_path)
    second_path = write_manifest(tmp_path)

    assert first_path == second_path
    assert json.loads(first_path.read_text(encoding="utf-8")) == build_manifest()
