"""Preregister V7-H0: strict quote-flow confirmation of frozen V6-H0 entries.

This module defines research intent only. It does not load data, implement a
strategy, or run a replay.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


VERSION = "7.0"
BASELINE_ID = "V7_H0"
STRATEGY_FAMILY = "ORDER_FLOW_CONFIRMED_V6_CONTINUATION"

DATASET_ID = "8fdfcee8d68b6f48"
DATASET_DEFINITION_SHA256 = (
    "dbca5d298f870068cfdc99ad0d550eda3d0af52b452c484c2e6d1e80e3c79567"
)
V6_RUN_ID = "e1eef7bdd0c37ad4"
ELIGIBLE_WINDOWS = 11

V6_TRADES = 443
V6_FRICTIONLESS_EXPECTANCY_R = "+0.04722009245553381286395292467"
V6_NET_EXPECTANCY_R = "-0.1366917232474779417819851766"
V6_PROFIT_FACTOR_R = "0.7695960517596068371318908235"
V6_POSITIVE_NET_WINDOWS = 1
V6_AVERAGE_FRICTION_R = "0.1839118157030117546459381019"

HOLDOUT_START = "2025-08-01T00:00:00+00:00"
HOLDOUT_END = "2026-02-01T00:00:00+00:00"


def _definition() -> dict:
    return {
        "version": VERSION,
        "baseline_id": BASELINE_ID,
        "strategy_family": STRATEGY_FAMILY,
        "research_question": (
            "DOES_CONTEMPORANEOUS_AGGRESSIVE_BUYER_DOMINANCE_ON_THE_CLOSED_"
            "15M_V6_SIGNAL_CANDLE_IMPROVE_FROZEN_V6_CONTINUATION_ENTRIES"
        ),
        "market": {
            "symbol": "BTCUSDC",
            "market_type": "BINANCE_PUBLIC_SPOT",
            "direction": "LONG_ONLY",
            "execution_interval": "15m",
            "regime_interval": "4h",
        },
        "orderflow_dataset": {
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
        },
        "frozen_v6_reference": {
            "run_id": V6_RUN_ID,
            "strategy": "V6_H0_COST_AWARE_MTF_PULLBACK_CONTINUATION",
            "trades": V6_TRADES,
            "frictionless_expectancy_r_per_trade": V6_FRICTIONLESS_EXPECTANCY_R,
            "net_expectancy_r_per_trade": V6_NET_EXPECTANCY_R,
            "profit_factor_r": V6_PROFIT_FACTOR_R,
            "positive_net_windows": V6_POSITIVE_NET_WINDOWS,
            "eligible_windows": ELIGIBLE_WINDOWS,
            "average_friction_r_per_trade": V6_AVERAGE_FRICTION_R,
            "progression": False,
            "entry_regime_risk_execution_and_exits_must_remain_unchanged": True,
        },
        "frozen_v6_signal": {
            "completed_4h_regime_all_required": [
                "EMA20_4H_STRICTLY_GREATER_THAN_EMA50_4H",
                "LATEST_COMPLETED_4H_CLOSE_STRICTLY_GREATER_THAN_EMA20_4H",
            ],
            "setup_15m_all_required": [
                "PREVIOUS_CLOSE_LESS_THAN_OR_EQUAL_TO_PREVIOUS_EMA20_15M",
                "CURRENT_CLOSE_STRICTLY_GREATER_THAN_CURRENT_EMA20_15M",
                "CURRENT_CLOSE_STRICTLY_GREATER_THAN_PREVIOUS_HIGH",
            ],
            "v7_requires_v6_buy": True,
        },
        "orderflow_confirmation": {
            "new_feature_count": 1,
            "feature": "CURRENT_SIGNAL_CANDLE_QUOTE_VOLUME_AGGRESSOR_DIRECTION",
            "source_bucket": "EXACT_CORRESPONDING_FULLY_COMPLETED_15M_BUCKET",
            "formula": (
                "(TAKER_BUY_QUOTE_VOLUME_MINUS_TAKER_SELL_QUOTE_VOLUME)"
                "/TOTAL_QUOTE_VOLUME"
            ),
            "all_conditions_required": True,
            "conditions": [
                "FROZEN_V6_H0_EMITS_BUY",
                "TOTAL_QUOTE_VOLUME_STRICTLY_GREATER_THAN_ZERO",
                "TAKER_BUY_QUOTE_VOLUME_STRICTLY_GREATER_THAN_TAKER_SELL_QUOTE_VOLUME",
                "QUOTE_VOLUME_IMBALANCE_STRICTLY_GREATER_THAN_ZERO",
            ],
            "threshold": "0",
            "threshold_basis": "NATURAL_ECONOMIC_BALANCE_POINT_NOT_OPTIMIZED",
            "comparison_is_strict": True,
            "equality_qualifies": False,
            "zero_total_quote_volume_rule": "NO_ENTRY",
        },
        "causality": {
            "signal_candle_timestamp_semantics": "15M_BUCKET_OPEN_TIME_UTC",
            "join_rule": "BUCKET_OPEN_TIME_EXACTLY_EQUALS_V6_SIGNAL_CANDLE_OPEN_TIME",
            "bucket_must_be_fully_completed_at_decision_time": True,
            "signal_candle_bucket_only": True,
            "next_bucket_allowed": False,
            "partial_future_bucket_allowed": False,
            "later_aggregate_trade_allowed": False,
            "rolling_future_bucket_feature_allowed": False,
            "future_orderflow_forward_fill_allowed": False,
            "nearest_neighbor_lookup_allowed": False,
            "timestamp_approximation_allowed": False,
            "missing_bucket_entry_rule": "NO_ENTRY",
            "missing_bucket_integrity_rule": "REPORT_INTEGRITY_FAILURE",
        },
        "reference_relationship": {
            "relationship": "STRICT_FILTER_OF_V6_H0_ENTRIES",
            "may_remove_v6_entries": True,
            "may_create_non_v6_entry": False,
            "required_non_v6_entry_count": 0,
            "diagnostics": [
                "V6_REFERENCE_ENTRIES",
                "V7_RETAINED_ENTRIES",
                "FILTERED_OUT_V6_ENTRIES",
                "TRADE_RETENTION_RATIO_VS_V6",
                "NON_V6_ENTRY_COUNT",
            ],
        },
        "execution_risk_and_exit": {
            "execution": "NEXT_BAR_OPEN",
            "initial_stop_distance": "MAX_OF_FROZEN_ATR14_4H_AND_ACTUAL_ENTRY_FILL_TIMES_0.0096",
            "minimum_stop_distance_bps": "96",
            "stress_does_not_change_stop_floor": True,
            "reward_risk_ratio": "2",
            "take_profit": "ACTUAL_ENTRY_FILL_PLUS_2R",
            "maximum_hold_15m_bars": 96,
            "cooldown_bars": 4,
            "ambiguous_bar_policy": "STOP_FIRST",
            "trailing_stop": False,
            "break_even_stop": False,
            "early_failure_exit": False,
        },
        "costs": {
            "base": {"fee_bps_per_side": "10", "slippage_bps_per_side": "2"},
            "stress_2x": {
                "fee_bps_per_side": "20",
                "slippage_bps_per_side": "4",
            },
        },
        "feature_scope": {
            "allowed": ["CURRENT_SIGNAL_CANDLE_QUOTE_VOLUME_AGGRESSOR_DIRECTION"],
            "forbidden": [
                "BASE_VOLUME_IMBALANCE_FILTER",
                "PRIOR_CANDLE_IMBALANCE",
                "MULTI_CANDLE_CUMULATIVE_DELTA",
                "ROLLING_IMBALANCE",
                "TRADE_SIZE",
                "LARGE_TRADE_CONCENTRATION",
                "TRADE_COUNT",
                "BUY_SELL_AGGTRADE_COUNT",
                "VOLUME_PERCENTILE",
                "DIVERGENCE",
                "ABSORPTION",
                "ORDER_BOOK_DATA",
            ],
        },
        "evaluation": {
            "base_metrics": [
                "TOTAL_TRADES",
                "TRADE_RETENTION_RATIO_VS_V6",
                "FRICTIONLESS_EXPECTANCY_R_PER_TRADE",
                "NET_EXPECTANCY_R_PER_TRADE",
                "PROFIT_FACTOR_R",
                "WIN_RATE",
                "AVERAGE_WINNER_R",
                "AVERAGE_LOSER_R",
                "PAYOFF_RATIO",
                "AVERAGE_FRICTION_R_PER_TRADE",
                "MAXIMUM_DRAWDOWN_PERCENT",
                "TRADES_PER_DAY",
                "NET_R_PER_DAY",
                "POSITIVE_NET_WINDOWS_OUT_OF_11",
            ],
            "orderflow_metrics": [
                "MEAN_RETAINED_SIGNAL_QUOTE_VOLUME_IMBALANCE",
                "MEDIAN_RETAINED_SIGNAL_QUOTE_VOLUME_IMBALANCE",
                "MEAN_RETAINED_SIGNAL_TAKER_BUY_QUOTE_RATIO",
                "MEDIAN_RETAINED_SIGNAL_TAKER_BUY_QUOTE_RATIO",
            ],
            "reference_comparison_metrics": [
                "V7_MINUS_V6_FRICTIONLESS_EXPECTANCY_R",
                "V7_MINUS_V6_NET_EXPECTANCY_R",
                "V7_MINUS_V6_PROFIT_FACTOR_R",
                "WINDOWS_V7_FRICTIONLESS_EXPECTANCY_GT_V6",
                "WINDOWS_V7_NET_EXPECTANCY_GT_V6",
            ],
            "stress_metrics": [
                "STRESS_2X_NET_EXPECTANCY_R_PER_TRADE",
                "STRESS_2X_PROFIT_FACTOR_R",
                "STRESS_2X_NET_R_PER_DAY",
            ],
            "stress_is_mandatory_reporting": True,
            "stress_is_support_gate": False,
            "stress_is_progression_gate": False,
        },
        "support_gate": {
            "meaning": "ORDERFLOW_INFORMATION_HYPOTHESIS_SUPPORTED",
            "required_eligible_consumed_windows": 11,
            "combined_frictionless_expectancy_r_gt_frozen_v6": V6_FRICTIONLESS_EXPECTANCY_R,
            "combined_net_expectancy_r_gt_frozen_v6": V6_NET_EXPECTANCY_R,
            "profit_factor_r_gt_frozen_v6": V6_PROFIT_FACTOR_R,
            "minimum_windows_net_expectancy_better_than_v6": 7,
            "required_non_v6_entry_count": 0,
            "all_conditions_required": True,
        },
        "progression_gate": {
            "support_gate_must_pass": True,
            "combined_net_expectancy_r_gt": "0",
            "profit_factor_r_gt": "1",
            "minimum_positive_net_windows": 7,
            "required_eligible_windows": 11,
            "stress_is_gate": False,
            "all_conditions_required": True,
        },
        "zero_trade_window_semantics": {
            "status": "ZERO_TRADES",
            "counts_as_positive_net": False,
            "counts_as_net_better_than_v6": False,
            "fabricate_zero_expectancy_as_improvement": False,
            "must_report_explicitly": True,
        },
        "anti_overfitting": {
            "parameter_search": False,
            "alternate_threshold_testing": False,
            "forbidden_automatic_followups_if_v7_h0_fails": [
                "QUOTE_IMBALANCE_GT_0.05",
                "QUOTE_IMBALANCE_GT_0.10",
                "QUOTE_IMBALANCE_GT_0.20",
                "BUY_RATIO_GT_55_PERCENT",
                "BUY_RATIO_GT_60_PERCENT",
                "BASE_IMBALANCE_INSTEAD_OF_QUOTE_IMBALANCE",
                "PREVIOUS_CANDLE_IMBALANCE",
                "TWO_CANDLE_CUMULATIVE_DELTA",
                "FOUR_CANDLE_CUMULATIVE_DELTA",
            ],
            "separate_justification_and_preregistration_required": True,
        },
        "dataset_policy": {
            "research_data_status": "CONSUMED_RESEARCH_DATA",
            "eligible_windows": ELIGIBLE_WINDOWS,
            "only_dataset_id": DATASET_ID,
            "only_dataset_definition_sha256": DATASET_DEFINITION_SHA256,
            "blind_holdout": {
                "status": "LOCKED_BLIND_HOLDOUT",
                "start": HOLDOUT_START,
                "end": HOLDOUT_END,
                "downloaded": False,
                "loaded": False,
                "revealed": False,
                "consumed": False,
                "evaluated": False,
            },
        },
        "research_rule": (
            "V7_H0_TESTS_EXACTLY_ONE_UNOPTIMIZED_ORDERFLOW_FEATURE_AS_A_STRICT_"
            "FILTER_OF_FROZEN_V6_H0"
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
    digest = hashlib.sha256(_canonical_json(definition).encode("utf-8")).hexdigest()
    return {"run_id": digest[:16], "definition_sha256": digest, **definition}


def write_manifest(root: str | Path = "research/v7_orderflow") -> Path:
    manifest = build_manifest()
    output = Path(root) / manifest["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError("Existing V7-H0 manifest differs from frozen definition.")
        return path
    path.write_text(serialized, encoding="utf-8")
    return path


def main() -> int:
    manifest = build_manifest()
    path = write_manifest()
    print()
    print("V7-H0 ORDER-FLOW CONFIRMATION PREREGISTRATION")
    print(f"Run ID: {manifest['run_id']}")
    print("Market: BTCUSDC Binance Public Spot | 15m | LONG ONLY")
    print("Reference: strict filter of frozen V6-H0 BUY entries")
    print("Order flow: signal-candle quote-volume imbalance > 0")
    print(f"Dataset: {DATASET_ID} | eligible windows: 11")
    print(
        "Blind holdout: LOCKED | NOT DOWNLOADED | NOT LOADED | "
        "NOT REVEALED | NOT CONSUMED | NOT EVALUATED"
    )
    print(f"Manifest: {path}")
    print()
    print("PREREGISTRATION COMPLETE — NO BACKTEST EXECUTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
