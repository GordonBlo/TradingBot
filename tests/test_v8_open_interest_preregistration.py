from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.research.v8_open_interest_preregistration import (
    ELIGIBLE_WINDOWS,
    HOLDOUT_END,
    HOLDOUT_START,
    OPEN_INTEREST_LOOKBACK_MINUTES,
    V6_RUN_ID,
    build_manifest,
    open_interest_expansion_accepts,
    require_v8_h1_subset_of_frozen_v6,
    write_manifest,
)


T = datetime(2025, 8, 1, 12, tzinfo=timezone.utc)


def _accepts(now=Decimal("101"), old=Decimal("100"), now_at=T, old_at=None):
    return open_interest_expansion_accepts(
        oi_now=now,
        oi_now_timestamp=now_at,
        oi_60m=old,
        oi_60m_timestamp=old_at or T - timedelta(minutes=60),
        signal_close=T,
    )


def test_oi_increase_is_accepted():
    assert _accepts()


@pytest.mark.parametrize("now,old", [(Decimal("100"), Decimal("100")), (Decimal("99"), Decimal("100"))])
def test_equal_or_decreasing_oi_is_rejected(now, old):
    assert not _accepts(now, old)


@pytest.mark.parametrize(
    "now,old",
    [(None, Decimal("100")), (Decimal("101"), None)],
)
def test_missing_current_or_historical_oi_is_rejected(now, old):
    assert not _accepts(now, old)


def test_future_observations_are_rejected():
    assert not _accepts(now_at=T + timedelta(seconds=1))
    assert not _accepts(old_at=T - timedelta(minutes=59))


def test_comparison_is_exactly_preceding_60_minutes():
    assert OPEN_INTEREST_LOOKBACK_MINUTES == 60
    assert _accepts(old_at=T - timedelta(minutes=60))
    assert not _accepts(old_at=T - timedelta(minutes=59, seconds=59))


def test_v8_h1_candidates_are_a_subset_of_frozen_v6_schedule():
    require_v8_h1_subset_of_frozen_v6(["a", "b"], ["b"])
    with pytest.raises(ValueError, match="non-V6"):
        require_v8_h1_subset_of_frozen_v6(["a"], ["a", "v8-only"])


def test_manifest_freezes_market_dataset_and_quantity_only_rule():
    manifest = build_manifest()
    definition = manifest["definition"]
    assert manifest["run_id"] == "6e7c9147b8c3187e"
    assert definition["market"] == {
        "symbol": "BTCUSDC",
        "venue": "BINANCE_PUBLIC_SPOT",
        "interval": "15m",
        "side": "LONG_ONLY",
    }
    assert definition["derivatives_context_dataset"]["source_field"] == "SUM_OPEN_INTEREST"
    assert definition["derivatives_context_dataset"]["forbidden_source_field"] == "SUM_OPEN_INTEREST_VALUE"
    assert definition["frozen_v6_reference"]["run_id"] == V6_RUN_ID
    assert definition["frozen_v6_reference"]["candidate_schedule"] == "FROZEN_V6_H0_CANDIDATE_SCHEDULE_DIRECT"
    assert definition["open_interest_filter"]["comparison_horizon_minutes"] == 60
    assert definition["open_interest_filter"]["acceptance_rule"] == "OI_NOW_STRICTLY_GREATER_THAN_OI_60M"
    assert definition["open_interest_filter"]["equality_qualifies"] is False


def test_manifest_freezes_v6_semantics_and_all_gate_conditions():
    definition = build_manifest()["definition"]
    assert definition["execution_and_risk"] == {
        "entry_execution": "NEXT_BAR_OPEN",
        "atr": "V6_H0_CAUSAL_SIGNAL_CANDLE_ATR14",
        "initial_stop": "V6_H0_ACTUAL_ENTRY_FILL_MINUS_2X_SIGNAL_ATR14",
        "take_profit": "V6_H0_PLUS_2R",
        "maximum_hold_bars": 96,
        "cooldown_bars": 4,
        "ambiguous_bar_policy": "STOP_FIRST",
    }
    assert definition["costs"] == {
        "base": {"fee_bps_per_side": 10, "slippage_bps_per_side": 2},
        "stress": {"fee_bps_per_side": 20, "slippage_bps_per_side": 4},
    }
    assert definition["support_gate"]["all_required"] is True
    assert definition["support_gate"]["conditions"] == [
        "ELIGIBLE_RESEARCH_WINDOWS_EQUALS_11",
        "NON_V6_ENTRY_COUNT_EQUALS_0",
        "V8_GROSS_EXPECTANCY_R_STRICTLY_GREATER_THAN_FROZEN_V6_GROSS_EXPECTANCY_R",
        "V8_NET_EXPECTANCY_R_STRICTLY_GREATER_THAN_FROZEN_V6_NET_EXPECTANCY_R",
        "V8_PROFIT_FACTOR_R_STRICTLY_GREATER_THAN_FROZEN_V6_PROFIT_FACTOR_R",
        "V8_NET_RESULT_BETTER_THAN_V6_IN_AT_LEAST_7_OF_11_WINDOWS",
    ]
    assert definition["progression_gate"] == {
        "all_required": True,
        "requires_support_gate_pass": True,
        "conditions": [
            "V8_NET_EXPECTANCY_R_STRICTLY_GREATER_THAN_0",
            "V8_PROFIT_FACTOR_R_STRICTLY_GREATER_THAN_1",
            "V8_POSITIVE_NET_WINDOWS_AT_LEAST_7_OF_11",
        ],
    }


def test_holdout_is_locked_and_alternative_rules_are_forbidden():
    definition = build_manifest()["definition"]
    assert definition["data_policy"]["research_data"] == "CONSUMED_RESEARCH_DATA_ONLY"
    assert definition["data_policy"]["eligible_research_windows"] == ELIGIBLE_WINDOWS == 11
    assert definition["data_policy"]["blind_holdout"] == {
        "status": "LOCKED",
        "start": HOLDOUT_START,
        "end": HOLDOUT_END,
        "loaded": False,
        "revealed": False,
        "consumed": False,
        "evaluated": False,
    }
    assert definition["anti_overfit"] == {
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
    }


def test_manifest_and_writing_are_deterministic_and_immutable(tmp_path):
    assert build_manifest() == build_manifest()
    path = write_manifest(tmp_path)
    assert path == tmp_path / "6e7c9147b8c3187e" / "manifest.json"
    original = path.read_text(encoding="utf-8")
    assert write_manifest(tmp_path) == path
    assert path.read_text(encoding="utf-8") == original
    path.write_text("different\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="immutable manifest mismatch"):
        write_manifest(tmp_path)
