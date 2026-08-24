from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.research.v8_funding_context_preregistration import (
    V8_DERIVATIVES_DATASET_DEFINITION_SHA256,
    V8_DERIVATIVES_DATASET_ID,
    build_manifest,
    funding_filter_accepts,
    require_v8_subset_of_frozen_v6,
    write_manifest,
)


EXPECTED_RUN_ID = "089b62c96a7de979"
SIGNAL_CLOSE = datetime(2024, 1, 1, 12, 15, tzinfo=timezone.utc)


def test_v8_h0_non_positive_causal_funding_is_accepted() -> None:
    assert funding_filter_accepts(
        funding_rate=Decimal("-0.0001"),
        funding_timestamp=SIGNAL_CLOSE,
        signal_close=SIGNAL_CLOSE,
    )


def test_v8_h0_positive_funding_is_rejected() -> None:
    assert not funding_filter_accepts(
        funding_rate=Decimal("0.0001"),
        funding_timestamp=SIGNAL_CLOSE,
        signal_close=SIGNAL_CLOSE,
    )


def test_v8_h0_exact_zero_funding_is_accepted() -> None:
    assert funding_filter_accepts(
        funding_rate=Decimal("0"),
        funding_timestamp=SIGNAL_CLOSE,
        signal_close=SIGNAL_CLOSE,
    )


@pytest.mark.parametrize(
    ("funding_rate", "funding_timestamp"),
    ((None, SIGNAL_CLOSE), (Decimal("0"), None)),
)
def test_v8_h0_missing_funding_is_rejected(
    funding_rate: Decimal | None, funding_timestamp: datetime | None
) -> None:
    assert not funding_filter_accepts(
        funding_rate=funding_rate,
        funding_timestamp=funding_timestamp,
        signal_close=SIGNAL_CLOSE,
    )


def test_v8_h0_future_funding_is_rejected() -> None:
    assert not funding_filter_accepts(
        funding_rate=Decimal("-0.0001"),
        funding_timestamp=SIGNAL_CLOSE + timedelta(microseconds=1),
        signal_close=SIGNAL_CLOSE,
    )


def test_v8_h0_candidates_must_be_a_subset_of_the_frozen_v6_schedule() -> None:
    v6_schedule = (SIGNAL_CLOSE, SIGNAL_CLOSE + timedelta(minutes=15))
    require_v8_subset_of_frozen_v6(
        frozen_v6_candidate_signal_closes=v6_schedule,
        v8_candidate_signal_closes=(SIGNAL_CLOSE,),
    )
    with pytest.raises(ValueError, match="non-V6"):
        require_v8_subset_of_frozen_v6(
            frozen_v6_candidate_signal_closes=v6_schedule,
            v8_candidate_signal_closes=(SIGNAL_CLOSE + timedelta(minutes=30),),
        )


def test_v8_h0_dataset_market_and_frozen_v6_schedule_are_exact() -> None:
    manifest = build_manifest()

    assert manifest["market"] == {
        "symbol": "BTCUSDC",
        "market_type": "BINANCE_PUBLIC_SPOT",
        "direction": "LONG_ONLY",
        "execution_interval": "15m",
        "regime_interval": "4h",
    }
    assert manifest["derivatives_context_dataset"] == {
        "dataset_id": V8_DERIVATIVES_DATASET_ID,
        "definition_sha256": V8_DERIVATIVES_DATASET_DEFINITION_SHA256,
        "context_market": "BTCUSDT_BINANCE_PUBLIC_USDM_PERPETUAL",
        "source": "BINANCE_PUBLIC_USDM_FUNDINGRATE",
        "funding_field": "LATEST_KNOWN_FUNDING_RATE",
        "funding_timestamp_field": "FUNDING_TIMESTAMP",
        "spot_context_interval": "15m",
        "required_research_buckets": 95040,
    }
    reference = manifest["frozen_v6_reference"]
    assert reference["run_id"] == "e1eef7bdd0c37ad4"
    assert reference["candidate_schedule"] == "FROZEN_V6_H0_CLOSED_SIGNAL_CANDIDATES"
    assert reference["schedule_must_be_used_directly"] is True
    assert reference["v6_must_not_be_reevaluated_against_v8_account_state"] is True


def test_v8_h0_funding_rule_and_v6_execution_are_frozen() -> None:
    manifest = build_manifest()

    assert manifest["funding_filter"] == {
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
    }
    assert manifest["execution_risk_and_exit"] == {
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
    }


def test_v8_h0_support_and_progression_gates_are_exact() -> None:
    manifest = build_manifest()

    assert manifest["support_gate"] == {
        "required_eligible_consumed_windows": 11,
        "required_non_v6_entry_count": 0,
        "combined_gross_expectancy_r_gt_frozen_v6": "+0.04722009245553381286395292467",
        "combined_net_expectancy_r_gt_frozen_v6": "-0.1366917232474779417819851766",
        "profit_factor_r_gt_frozen_v6": "0.7695960517596068371318908235",
        "minimum_windows_net_result_better_than_v6": 7,
        "all_conditions_required": True,
    }
    assert manifest["progression_gate"] == {
        "support_gate_must_pass": True,
        "combined_net_expectancy_r_gt": "0",
        "profit_factor_r_gt": "1",
        "minimum_positive_net_windows": 7,
        "required_eligible_windows": 11,
        "all_conditions_required": True,
    }


def test_v8_h0_holdout_remains_inaccessible_and_no_alternatives_are_frozen() -> None:
    manifest = build_manifest()

    holdout = manifest["dataset_policy"]["blind_holdout"]
    assert manifest["dataset_policy"]["research_data_status"] == "CONSUMED_RESEARCH_DATA"
    assert manifest["dataset_policy"]["eligible_windows"] == 11
    assert holdout["status"] == "LOCKED_BLIND_HOLDOUT"
    assert all(
        holdout[name] is False
        for name in ("downloaded", "loaded", "revealed", "consumed", "evaluated")
    )
    assert manifest["anti_overfitting"]["parameter_search"] is False
    assert manifest["anti_overfitting"]["alternate_funding_threshold_testing"] is False


def test_v8_h0_manifest_and_run_id_are_deterministic() -> None:
    first = build_manifest()
    second = build_manifest()

    assert first == second
    assert first["run_id"] == EXPECTED_RUN_ID
    assert first["definition_sha256"].startswith(EXPECTED_RUN_ID)
    assert len(first["definition_sha256"]) == 64


def test_v8_h0_manifest_writing_is_deterministic_and_immutable(tmp_path) -> None:
    first = write_manifest(tmp_path)
    second = write_manifest(tmp_path)

    assert first == second
    assert json.loads(first.read_text(encoding="utf-8")) == build_manifest()

    first.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from frozen definition"):
        write_manifest(tmp_path)
