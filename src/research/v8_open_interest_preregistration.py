"""Immutable V8-H1 open-interest context preregistration; no replay code."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Iterable

from src.research.v8_funding_context_preregistration import (
    V6_GROSS_EXPECTANCY_R,
    V6_NET_EXPECTANCY_R,
    V6_PROFIT_FACTOR_R,
    V6_RUN_ID,
    V8_DERIVATIVES_DATASET_DEFINITION_SHA256,
    V8_DERIVATIVES_DATASET_ID,
)


VERSION = "8.1"
BASELINE_ID = "V8_H1"
STRATEGY_FAMILY = "OPEN_INTEREST_EXPANSION_FILTERED_V6_CONTINUATION"
ELIGIBLE_WINDOWS = 11
OPEN_INTEREST_LOOKBACK_MINUTES = 60
HOLDOUT_START = "2025-10-01T00:00:00Z"
HOLDOUT_END = "2026-01-01T00:00:00Z"


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def open_interest_expansion_accepts(
    *,
    oi_now: Decimal | None,
    oi_now_timestamp: datetime | None,
    oi_60m: Decimal | None,
    oi_60m_timestamp: datetime | None,
    signal_close: datetime,
) -> bool:
    """Return whether the two required causal quantity observations strictly expand."""
    if None in (oi_now, oi_now_timestamp, oi_60m, oi_60m_timestamp):
        return False
    signal_close_utc = _utc(signal_close)
    return (
        _utc(oi_now_timestamp) <= signal_close_utc
        and _utc(oi_60m_timestamp)
        <= signal_close_utc - timedelta(minutes=OPEN_INTEREST_LOOKBACK_MINUTES)
        and oi_now > oi_60m
    )


def require_v8_h1_subset_of_frozen_v6(
    v6_candidate_ids: Iterable[str], v8_candidate_ids: Iterable[str]
) -> None:
    non_v6_entries = set(v8_candidate_ids) - set(v6_candidate_ids)
    if non_v6_entries:
        raise ValueError("V8-H1 may not create non-V6 entries")


def _definition() -> dict:
    return {
        "version": VERSION,
        "baseline_id": BASELINE_ID,
        "strategy_family": STRATEGY_FAMILY,
        "hypothesis": (
            "Frozen V6-H0 bullish candidates may improve when BTCUSDT perpetual "
            "open interest is expanding over the preceding 60 minutes."
        ),
        "market": {
            "symbol": "BTCUSDC",
            "venue": "BINANCE_PUBLIC_SPOT",
            "interval": "15m",
            "side": "LONG_ONLY",
        },
        "derivatives_context_dataset": {
            "dataset_id": V8_DERIVATIVES_DATASET_ID,
            "definition_sha256": V8_DERIVATIVES_DATASET_DEFINITION_SHA256,
            "context_symbol": "BTCUSDT",
            "context_venue": "BINANCE_PUBLIC_USDM",
            "source": "BINANCE_PUBLIC_USDM_METRICS",
            "source_field": "SUM_OPEN_INTEREST",
            "forbidden_source_field": "SUM_OPEN_INTEREST_VALUE",
            "spot_context_interval": "15m",
        },
        "frozen_v6_reference": {
            "run_id": V6_RUN_ID,
            "candidate_schedule": "FROZEN_V6_H0_CANDIDATE_SCHEDULE_DIRECT",
            "account_state": "FROZEN_V6_H0_ACCOUNT_STATE_DIRECT",
            "gross_expectancy_r_per_trade": V6_GROSS_EXPECTANCY_R,
            "net_expectancy_r_per_trade": V6_NET_EXPECTANCY_R,
            "profit_factor_r": V6_PROFIT_FACTOR_R,
        },
        "open_interest_filter": {
            "new_feature_count": 1,
            "decision_time": "BTCUSDC_SPOT_SIGNAL_CANDLE_CLOSE",
            "oi_now": "CAUSAL_SUM_OPEN_INTEREST_AT_SIGNAL_CLOSE",
            "oi_60m": "CAUSAL_SUM_OPEN_INTEREST_AT_SIGNAL_CLOSE_MINUS_60_MINUTES",
            "comparison_horizon_minutes": OPEN_INTEREST_LOOKBACK_MINUTES,
            "comparison_horizon": "PRECEDING_60_MINUTES_EXACTLY",
            "acceptance_rule": "OI_NOW_STRICTLY_GREATER_THAN_OI_60M",
            "comparison_is_strict": True,
            "equality_qualifies": False,
            "required_conditions": [
                "CANDIDATE_IS_IN_FROZEN_V6_H0_SCHEDULE",
                "OI_NOW_OBSERVATION_PRESENT",
                "OI_60M_OBSERVATION_PRESENT",
                "OI_NOW_TIMESTAMP_LESS_THAN_OR_EQUAL_TO_SPOT_SIGNAL_CLOSE",
                "OI_60M_TIMESTAMP_LESS_THAN_OR_EQUAL_TO_SPOT_SIGNAL_CLOSE_MINUS_60_MINUTES",
                "OI_NOW_STRICTLY_GREATER_THAN_OI_60M",
            ],
            "missing_or_unavailable_comparison": "REJECT_CANDIDATE",
            "future_observation": "REJECT_CANDIDATE",
            "interpolation": False,
            "future_fill": False,
        },
        "subset_relationship": {
            "v8_may_only": "REMOVE_FROZEN_V6_H0_CANDIDATES",
            "non_v6_entry_count_required": 0,
            "diagnostic_counts": [
                "FROZEN_V6_CANDIDATE_COUNT",
                "V8_RETAINED_COUNT",
                "FILTERED_OUT_NONEXPANDING_OR_MISSING_OPEN_INTEREST_COUNT",
                "NON_V6_ENTRY_COUNT",
            ],
        },
        "execution_and_risk": {
            "entry_execution": "NEXT_BAR_OPEN",
            "atr": "V6_H0_CAUSAL_SIGNAL_CANDLE_ATR14",
            "initial_stop": "V6_H0_ACTUAL_ENTRY_FILL_MINUS_2X_SIGNAL_ATR14",
            "take_profit": "V6_H0_PLUS_2R",
            "maximum_hold_bars": 96,
            "cooldown_bars": 4,
            "ambiguous_bar_policy": "STOP_FIRST",
        },
        "costs": {
            "base": {"fee_bps_per_side": 10, "slippage_bps_per_side": 2},
            "stress": {"fee_bps_per_side": 20, "slippage_bps_per_side": 4},
        },
        "evaluation": {
            "eligible_research_windows": ELIGIBLE_WINDOWS,
            "metrics": [
                "TOTAL_TRADES",
                "V6_AND_V8_GROSS_EXPECTANCY_R_PER_TRADE",
                "V6_AND_V8_NET_EXPECTANCY_R_PER_TRADE",
                "V6_AND_V8_PROFIT_FACTOR_R",
                "POSITIVE_NET_WINDOWS",
                "NET_BETTER_THAN_V6_WINDOWS",
                "DOUBLED_COST_NET_EXPECTANCY_R_PER_TRADE",
            ],
        },
        "support_gate": {
            "all_required": True,
            "conditions": [
                "ELIGIBLE_RESEARCH_WINDOWS_EQUALS_11",
                "NON_V6_ENTRY_COUNT_EQUALS_0",
                "V8_GROSS_EXPECTANCY_R_STRICTLY_GREATER_THAN_FROZEN_V6_GROSS_EXPECTANCY_R",
                "V8_NET_EXPECTANCY_R_STRICTLY_GREATER_THAN_FROZEN_V6_NET_EXPECTANCY_R",
                "V8_PROFIT_FACTOR_R_STRICTLY_GREATER_THAN_FROZEN_V6_PROFIT_FACTOR_R",
                "V8_NET_RESULT_BETTER_THAN_V6_IN_AT_LEAST_7_OF_11_WINDOWS",
            ],
        },
        "progression_gate": {
            "all_required": True,
            "requires_support_gate_pass": True,
            "conditions": [
                "V8_NET_EXPECTANCY_R_STRICTLY_GREATER_THAN_0",
                "V8_PROFIT_FACTOR_R_STRICTLY_GREATER_THAN_1",
                "V8_POSITIVE_NET_WINDOWS_AT_LEAST_7_OF_11",
            ],
        },
        "data_policy": {
            "research_data": "CONSUMED_RESEARCH_DATA_ONLY",
            "eligible_research_windows": ELIGIBLE_WINDOWS,
            "blind_holdout": {
                "status": "LOCKED",
                "start": HOLDOUT_START,
                "end": HOLDOUT_END,
                "loaded": False,
                "revealed": False,
                "consumed": False,
                "evaluated": False,
            },
        },
        "anti_overfit": {
            "parameter_search": False,
            "alternate_oi_horizon_testing": False,
            "alternate_oi_threshold_testing": False,
            "alternate_oi_ratio_testing": False,
            "open_interest_value_variant_testing": False,
            "forbidden": [
                "OI_LOOKBACK_15_MINUTES",
                "OI_LOOKBACK_30_MINUTES",
                "OI_LOOKBACK_120_MINUTES",
                "OI_NOW_GREATER_THAN_OR_EQUAL_TO_OI_60M",
                "OI_GROWTH_RATIO_THRESHOLD",
                "SUM_OPEN_INTEREST_VALUE",
                "OPEN_INTEREST_AND_FUNDING_FILTER",
                "OPEN_INTEREST_AND_PREMIUM_FILTER",
            ],
        },
        "research_rule": "PREREGISTRATION_ONLY_NO_REPLAY_OR_OUTCOME_INSPECTION",
    }


def _canonical_json(definition: dict) -> str:
    return json.dumps(definition, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def build_manifest() -> dict:
    definition = _definition()
    definition_sha256 = hashlib.sha256(_canonical_json(definition).encode("ascii")).hexdigest()
    return {
        "run_id": definition_sha256[:16],
        "definition_sha256": definition_sha256,
        "definition": definition,
    }


def write_manifest(root: Path = Path("research/v8_open_interest")) -> Path:
    manifest = build_manifest()
    path = root / manifest["run_id"] / "manifest.json"
    serialized = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(f"immutable manifest mismatch: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")
    return path


def main() -> None:
    manifest = build_manifest()
    path = write_manifest()
    print("V8-H1 OPEN-INTEREST-CONTEXT PREREGISTRATION")
    print(f"Run ID: {manifest['run_id']}")
    print("Market: BTCUSDC Binance PUBLIC Spot 15m LONG ONLY")
    print("Rule: causal sum_open_interest at signal close > causal sum_open_interest 60m earlier")
    print(f"Manifest: {path}")
    print("PREREGISTRATION COMPLETE — NO BACKTEST EXECUTED")


if __name__ == "__main__":
    main()
