from __future__ import annotations

import json

import pytest

from src.research.v7_orderflow_preregistration import (
    DATASET_DEFINITION_SHA256,
    DATASET_ID,
    build_manifest,
    write_manifest,
)


EXPECTED_RUN_ID = "74b2458cb2812de2"


def test_v7_h0_dataset_and_market_are_frozen() -> None:
    manifest = build_manifest()

    assert manifest["market"] == {
        "symbol": "BTCUSDC",
        "market_type": "BINANCE_PUBLIC_SPOT",
        "direction": "LONG_ONLY",
        "execution_interval": "15m",
        "regime_interval": "4h",
    }
    assert manifest["orderflow_dataset"] == {
        "dataset_id": DATASET_ID,
        "definition_sha256": DATASET_DEFINITION_SHA256,
        "source": "BINANCE_PUBLIC_SPOT_AGGTRADES",
        "interval": "15m",
        "required_research_buckets": 95040,
        "reconciled_buckets": 95040,
        "missing_buckets": 0,
        "extra_buckets": 0,
        "duplicate_buckets": 0,
        "buyer_is_maker_false": "TAKER_BUY",
        "buyer_is_maker_true": "TAKER_SELL",
    }


def test_v7_h0_frozen_v6_reference_is_exact() -> None:
    reference = build_manifest()["frozen_v6_reference"]

    assert reference["run_id"] == "e1eef7bdd0c37ad4"
    assert reference["trades"] == 443
    assert reference["frictionless_expectancy_r_per_trade"] == (
        "+0.04722009245553381286395292467"
    )
    assert reference["net_expectancy_r_per_trade"] == (
        "-0.1366917232474779417819851766"
    )
    assert reference["profit_factor_r"] == "0.7695960517596068371318908235"
    assert reference["positive_net_windows"] == 1
    assert reference["eligible_windows"] == 11
    assert reference["average_friction_r_per_trade"] == (
        "0.1839118157030117546459381019"
    )
    assert reference["progression"] is False
    assert reference["entry_regime_risk_execution_and_exits_must_remain_unchanged"]


def test_v7_h0_uses_only_strict_current_quote_flow_direction() -> None:
    confirmation = build_manifest()["orderflow_confirmation"]

    assert confirmation["new_feature_count"] == 1
    assert confirmation["feature"] == (
        "CURRENT_SIGNAL_CANDLE_QUOTE_VOLUME_AGGRESSOR_DIRECTION"
    )
    assert confirmation["conditions"] == [
        "FROZEN_V6_H0_EMITS_BUY",
        "TOTAL_QUOTE_VOLUME_STRICTLY_GREATER_THAN_ZERO",
        "TAKER_BUY_QUOTE_VOLUME_STRICTLY_GREATER_THAN_TAKER_SELL_QUOTE_VOLUME",
        "QUOTE_VOLUME_IMBALANCE_STRICTLY_GREATER_THAN_ZERO",
    ]
    assert confirmation["threshold"] == "0"
    assert confirmation["comparison_is_strict"] is True
    assert confirmation["equality_qualifies"] is False
    assert confirmation["zero_total_quote_volume_rule"] == "NO_ENTRY"
    assert build_manifest()["feature_scope"]["allowed"] == [
        "CURRENT_SIGNAL_CANDLE_QUOTE_VOLUME_AGGRESSOR_DIRECTION"
    ]


def test_v7_h0_causality_requires_exact_completed_signal_bucket() -> None:
    causality = build_manifest()["causality"]

    assert causality["join_rule"] == (
        "BUCKET_OPEN_TIME_EXACTLY_EQUALS_V6_SIGNAL_CANDLE_OPEN_TIME"
    )
    assert causality["bucket_must_be_fully_completed_at_decision_time"] is True
    assert causality["signal_candle_bucket_only"] is True
    assert causality["next_bucket_allowed"] is False
    assert causality["partial_future_bucket_allowed"] is False
    assert causality["later_aggregate_trade_allowed"] is False
    assert causality["future_orderflow_forward_fill_allowed"] is False
    assert causality["nearest_neighbor_lookup_allowed"] is False
    assert causality["timestamp_approximation_allowed"] is False
    assert causality["missing_bucket_entry_rule"] == "NO_ENTRY"
    assert causality["missing_bucket_integrity_rule"] == "REPORT_INTEGRITY_FAILURE"


def test_v7_h0_is_strict_subset_of_v6_entries() -> None:
    relationship = build_manifest()["reference_relationship"]

    assert relationship["relationship"] == "STRICT_FILTER_OF_V6_H0_ENTRIES"
    assert relationship["may_remove_v6_entries"] is True
    assert relationship["may_create_non_v6_entry"] is False
    assert relationship["required_non_v6_entry_count"] == 0
    assert "NON_V6_ENTRY_COUNT" in relationship["diagnostics"]


def test_v7_h0_preserves_v6_execution_risk_exit_and_costs() -> None:
    manifest = build_manifest()
    frozen = manifest["execution_risk_and_exit"]

    assert frozen["execution"] == "NEXT_BAR_OPEN"
    assert frozen["initial_stop_distance"] == (
        "MAX_OF_FROZEN_ATR14_4H_AND_ACTUAL_ENTRY_FILL_TIMES_0.0096"
    )
    assert frozen["minimum_stop_distance_bps"] == "96"
    assert frozen["stress_does_not_change_stop_floor"] is True
    assert frozen["reward_risk_ratio"] == "2"
    assert frozen["maximum_hold_15m_bars"] == 96
    assert frozen["cooldown_bars"] == 4
    assert frozen["ambiguous_bar_policy"] == "STOP_FIRST"
    assert manifest["costs"] == {
        "base": {"fee_bps_per_side": "10", "slippage_bps_per_side": "2"},
        "stress_2x": {"fee_bps_per_side": "20", "slippage_bps_per_side": "4"},
    }


def test_v7_h0_required_metrics_and_stress_are_reporting_only() -> None:
    evaluation = build_manifest()["evaluation"]

    assert len(evaluation["base_metrics"]) == 14
    assert evaluation["orderflow_metrics"] == [
        "MEAN_RETAINED_SIGNAL_QUOTE_VOLUME_IMBALANCE",
        "MEDIAN_RETAINED_SIGNAL_QUOTE_VOLUME_IMBALANCE",
        "MEAN_RETAINED_SIGNAL_TAKER_BUY_QUOTE_RATIO",
        "MEDIAN_RETAINED_SIGNAL_TAKER_BUY_QUOTE_RATIO",
    ]
    assert len(evaluation["reference_comparison_metrics"]) == 5
    assert len(evaluation["stress_metrics"]) == 3
    assert evaluation["stress_is_mandatory_reporting"] is True
    assert evaluation["stress_is_support_gate"] is False
    assert evaluation["stress_is_progression_gate"] is False


def test_v7_h0_support_gate_is_exact_and_requires_seven_of_eleven() -> None:
    assert build_manifest()["support_gate"] == {
        "meaning": "ORDERFLOW_INFORMATION_HYPOTHESIS_SUPPORTED",
        "required_eligible_consumed_windows": 11,
        "combined_frictionless_expectancy_r_gt_frozen_v6": (
            "+0.04722009245553381286395292467"
        ),
        "combined_net_expectancy_r_gt_frozen_v6": (
            "-0.1366917232474779417819851766"
        ),
        "profit_factor_r_gt_frozen_v6": "0.7695960517596068371318908235",
        "minimum_windows_net_expectancy_better_than_v6": 7,
        "required_non_v6_entry_count": 0,
        "all_conditions_required": True,
    }


def test_v7_h0_progression_gate_is_exact() -> None:
    assert build_manifest()["progression_gate"] == {
        "support_gate_must_pass": True,
        "combined_net_expectancy_r_gt": "0",
        "profit_factor_r_gt": "1",
        "minimum_positive_net_windows": 7,
        "required_eligible_windows": 11,
        "stress_is_gate": False,
        "all_conditions_required": True,
    }


def test_v7_h0_zero_trade_window_semantics_are_frozen() -> None:
    assert build_manifest()["zero_trade_window_semantics"] == {
        "status": "ZERO_TRADES",
        "counts_as_positive_net": False,
        "counts_as_net_better_than_v6": False,
        "fabricate_zero_expectancy_as_improvement": False,
        "must_report_explicitly": True,
    }


def test_v7_h0_anti_threshold_mining_restrictions_are_frozen() -> None:
    anti = build_manifest()["anti_overfitting"]

    assert anti["parameter_search"] is False
    assert anti["alternate_threshold_testing"] is False
    assert anti["forbidden_automatic_followups_if_v7_h0_fails"] == [
        "QUOTE_IMBALANCE_GT_0.05",
        "QUOTE_IMBALANCE_GT_0.10",
        "QUOTE_IMBALANCE_GT_0.20",
        "BUY_RATIO_GT_55_PERCENT",
        "BUY_RATIO_GT_60_PERCENT",
        "BASE_IMBALANCE_INSTEAD_OF_QUOTE_IMBALANCE",
        "PREVIOUS_CANDLE_IMBALANCE",
        "TWO_CANDLE_CUMULATIVE_DELTA",
        "FOUR_CANDLE_CUMULATIVE_DELTA",
    ]
    assert anti["separate_justification_and_preregistration_required"] is True


def test_v7_h0_data_policy_keeps_holdout_locked_and_untouched() -> None:
    policy = build_manifest()["dataset_policy"]
    holdout = policy["blind_holdout"]

    assert policy["research_data_status"] == "CONSUMED_RESEARCH_DATA"
    assert policy["eligible_windows"] == 11
    assert policy["only_dataset_id"] == DATASET_ID
    assert policy["only_dataset_definition_sha256"] == DATASET_DEFINITION_SHA256
    assert holdout["status"] == "LOCKED_BLIND_HOLDOUT"
    assert all(
        holdout[name] is False
        for name in ("downloaded", "loaded", "revealed", "consumed", "evaluated")
    )


def test_v7_h0_manifest_and_run_id_are_deterministic() -> None:
    first = build_manifest()
    second = build_manifest()

    assert first == second
    assert first["run_id"] == EXPECTED_RUN_ID
    assert first["definition_sha256"].startswith(EXPECTED_RUN_ID)
    assert len(first["definition_sha256"]) == 64


def test_v7_h0_manifest_writing_is_deterministic_and_immutable(tmp_path) -> None:
    first = write_manifest(tmp_path)
    second = write_manifest(tmp_path)

    assert first == second
    assert json.loads(first.read_text(encoding="utf-8")) == build_manifest()

    first.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from frozen definition"):
        write_manifest(tmp_path)
