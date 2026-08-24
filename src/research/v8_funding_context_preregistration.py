"""Preregister V8-H0: non-positive funding filter of frozen V6-H0 entries.

This module records research intent only.  It neither loads the derivatives
dataset nor runs a V6 or V8 replay.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterable


VERSION = "8.0"
BASELINE_ID = "V8_H0"
STRATEGY_FAMILY = "FUNDING_FILTERED_V6_CONTINUATION"

V6_RUN_ID = "e1eef7bdd0c37ad4"
V8_DERIVATIVES_DATASET_ID = "eec764735d270f9d"
V8_DERIVATIVES_DATASET_DEFINITION_SHA256 = (
    "ea656661ad25924077f9c8a7f7dca709b8762046e60b6c30c2ca8cefdaa2bea2"
)
ELIGIBLE_WINDOWS = 11

V6_GROSS_EXPECTANCY_R = "+0.04722009245553381286395292467"
V6_NET_EXPECTANCY_R = "-0.1366917232474779417819851766"
V6_PROFIT_FACTOR_R = "0.7695960517596068371318908235"

HOLDOUT_START = "2025-08-01T00:00:00+00:00"
HOLDOUT_END = "2026-02-01T00:00:00+00:00"


def funding_filter_accepts(
    *,
    funding_rate: Decimal | None,
    funding_timestamp: datetime | None,
    signal_close: datetime,
) -> bool:
    """Apply the single V8-H0 condition without substituting missing data.

    The caller supplies the latest causally known funding observation from the
    immutable V8 context.  An absent or future observation is never eligible.
    """

    if funding_rate is None or funding_timestamp is None:
        return False
    if signal_close.tzinfo is None or signal_close.utcoffset() is None:
        raise ValueError("V8-H0 signal close must be timezone-aware.")
    if funding_timestamp.tzinfo is None or funding_timestamp.utcoffset() is None:
        raise ValueError("V8-H0 funding timestamp must be timezone-aware.")
    return (
        funding_timestamp.astimezone(timezone.utc)
        <= signal_close.astimezone(timezone.utc)
        and funding_rate <= Decimal("0")
    )


def require_v8_subset_of_frozen_v6(
    *,
    frozen_v6_candidate_signal_closes: Iterable[datetime],
    v8_candidate_signal_closes: Iterable[datetime],
) -> None:
    """Reject a candidate that is not present in the frozen V6 schedule."""

    frozen = {
        timestamp.astimezone(timezone.utc)
        for timestamp in frozen_v6_candidate_signal_closes
    }
    non_v6 = {
        timestamp.astimezone(timezone.utc)
        for timestamp in v8_candidate_signal_closes
    } - frozen
    if non_v6:
        raise ValueError("V8-H0 may not create non-V6 entries.")


def _definition() -> dict:
    return {
        "version": VERSION,
        "baseline_id": BASELINE_ID,
        "strategy_family": STRATEGY_FAMILY,
        "research_question": (
            "DO_FROZEN_V6_H0_LONG_ENTRIES_IMPROVE_WHEN_BTCUSDT_USDM_PERPETUAL_"
            "FUNDING_IS_NON_POSITIVE_AT_SPOT_SIGNAL_CLOSE"
        ),
        "market": {
            "symbol": "BTCUSDC",
            "market_type": "BINANCE_PUBLIC_SPOT",
            "direction": "LONG_ONLY",
            "execution_interval": "15m",
            "regime_interval": "4h",
        },
        "derivatives_context_dataset": {
            "dataset_id": V8_DERIVATIVES_DATASET_ID,
            "definition_sha256": V8_DERIVATIVES_DATASET_DEFINITION_SHA256,
            "context_market": "BTCUSDT_BINANCE_PUBLIC_USDM_PERPETUAL",
            "source": "BINANCE_PUBLIC_USDM_FUNDINGRATE",
            "funding_field": "LATEST_KNOWN_FUNDING_RATE",
            "funding_timestamp_field": "FUNDING_TIMESTAMP",
            "spot_context_interval": "15m",
            "required_research_buckets": 95040,
        },
        "frozen_v6_reference": {
            "run_id": V6_RUN_ID,
            "strategy": "V6_H0_COST_AWARE_MTF_PULLBACK_CONTINUATION",
            "candidate_schedule": "FROZEN_V6_H0_CLOSED_SIGNAL_CANDIDATES",
            "schedule_must_be_used_directly": True,
            "v6_must_not_be_reevaluated_against_v8_account_state": True,
            "gross_expectancy_r_per_trade": V6_GROSS_EXPECTANCY_R,
            "net_expectancy_r_per_trade": V6_NET_EXPECTANCY_R,
            "profit_factor_r": V6_PROFIT_FACTOR_R,
            "eligible_windows": ELIGIBLE_WINDOWS,
        },
        "funding_filter": {
            "new_feature_count": 1,
            "input": "LATEST_CAUSALLY_KNOWN_BTCUSDT_USDM_FUNDING_RATE",
            "decision_time": "BTCUSDC_SPOT_SIGNAL_CANDLE_CLOSE",
            "all_conditions_required": True,
            "conditions": [
                "CANDIDATE_IS_IN_FROZEN_V6_H0_SCHEDULE",
                "FUNDING_RECORD_PRESENT",
                "FUNDING_TIMESTAMP_LESS_THAN_OR_EQUAL_TO_SPOT_SIGNAL_CLOSE",
                "FUNDING_RATE_LESS_THAN_OR_EQUAL_TO_ZERO",
            ],
            "comparison": "FUNDING_RATE_LESS_THAN_OR_EQUAL_TO_ZERO",
            "zero_funding_qualifies": True,
            "positive_funding_qualifies": False,
            "missing_funding_entry_rule": "REJECT_CANDIDATE",
            "future_funding_entry_rule": "REJECT_CANDIDATE",
            "interpolation_allowed": False,
            "future_fill_allowed": False,
        },
        "reference_relationship": {
            "relationship": "STRICT_FILTER_OF_FROZEN_V6_H0_CANDIDATES",
            "may_remove_v6_entries": True,
            "may_create_non_v6_entry": False,
            "required_non_v6_entry_count": 0,
            "diagnostics": [
                "V6_FROZEN_CANDIDATE_COUNT",
                "V8_RETAINED_CANDIDATE_COUNT",
                "FILTERED_OUT_POSITIVE_OR_MISSING_FUNDING_COUNT",
                "NON_V6_ENTRY_COUNT",
            ],
        },
        "execution_risk_and_exit": {
            "execution": "NEXT_BAR_OPEN",
            "initial_stop_distance": "MAX_OF_FROZEN_ATR14_4H_AND_ACTUAL_ENTRY_FILL_TIMES_0.0096",
            "minimum_stop_distance_bps": "96",
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
        "evaluation": {
            "metrics": [
                "TOTAL_TRADES",
                "GROSS_EXPECTANCY_R_PER_TRADE",
                "NET_EXPECTANCY_R_PER_TRADE",
                "PROFIT_FACTOR_R",
                "WIN_RATE",
                "MAXIMUM_DRAWDOWN_PERCENT",
                "POSITIVE_NET_WINDOWS_OUT_OF_11",
                "STRESS_2X_NET_EXPECTANCY_R_PER_TRADE",
                "TRADES_PER_DAY",
                "NET_R_PER_DAY",
            ],
            "reference_comparison_metrics": [
                "V8_MINUS_V6_GROSS_EXPECTANCY_R",
                "V8_MINUS_V6_NET_EXPECTANCY_R",
                "V8_MINUS_V6_PROFIT_FACTOR_R",
                "WINDOWS_V8_NET_RESULT_BETTER_THAN_V6",
            ],
        },
        "support_gate": {
            "required_eligible_consumed_windows": 11,
            "required_non_v6_entry_count": 0,
            "combined_gross_expectancy_r_gt_frozen_v6": V6_GROSS_EXPECTANCY_R,
            "combined_net_expectancy_r_gt_frozen_v6": V6_NET_EXPECTANCY_R,
            "profit_factor_r_gt_frozen_v6": V6_PROFIT_FACTOR_R,
            "minimum_windows_net_result_better_than_v6": 7,
            "all_conditions_required": True,
        },
        "progression_gate": {
            "support_gate_must_pass": True,
            "combined_net_expectancy_r_gt": "0",
            "profit_factor_r_gt": "1",
            "minimum_positive_net_windows": 7,
            "required_eligible_windows": 11,
            "all_conditions_required": True,
        },
        "zero_trade_window_semantics": {
            "status": "ZERO_TRADES",
            "counts_as_positive_net": False,
            "counts_as_net_result_better_than_v6": False,
            "fabricate_zero_expectancy_as_improvement": False,
            "must_report_explicitly": True,
        },
        "anti_overfitting": {
            "parameter_search": False,
            "alternate_funding_threshold_testing": False,
            "forbidden_automatic_followups_if_v8_h0_fails": [
                "FUNDING_RATE_LESS_THAN_ZERO",
                "FUNDING_RATE_LESS_THAN_NEGATIVE_0_0001",
                "FUNDING_PERCENTILE_FILTER",
                "FUNDING_CHANGE_FILTER",
                "FUNDING_AND_OPEN_INTEREST_FILTER",
                "FUNDING_AND_PREMIUM_FILTER",
            ],
            "separate_justification_and_preregistration_required": True,
        },
        "dataset_policy": {
            "research_data_status": "CONSUMED_RESEARCH_DATA",
            "eligible_windows": ELIGIBLE_WINDOWS,
            "only_dataset_id": V8_DERIVATIVES_DATASET_ID,
            "only_dataset_definition_sha256": V8_DERIVATIVES_DATASET_DEFINITION_SHA256,
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
            "V8_H0_TESTS_ONE_FROZEN_NON_POSITIVE_FUNDING_FILTER_AS_A_STRICT_"
            "FILTER_OF_THE_FROZEN_V6_H0_CANDIDATE_SCHEDULE"
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


def write_manifest(
    root: str | Path = "research/v8_funding_context",
) -> Path:
    manifest = build_manifest()
    output = Path(root) / manifest["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError("Existing V8-H0 manifest differs from frozen definition.")
        return path
    path.write_text(serialized, encoding="utf-8")
    return path


def main() -> int:
    manifest = build_manifest()
    path = write_manifest()
    print()
    print("V8-H0 FUNDING-CONTEXT PREREGISTRATION")
    print(f"Run ID: {manifest['run_id']}")
    print("Market: BTCUSDC Binance Public Spot | 15m | LONG ONLY")
    print("Reference: strict filter of frozen V6-H0 candidate schedule")
    print("Rule: latest causal BTCUSDT USD-M funding rate <= 0 at signal close")
    print(f"Dataset: {V8_DERIVATIVES_DATASET_ID} | eligible windows: 11")
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
