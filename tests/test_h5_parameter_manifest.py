from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.backtest.models import BacktestConfig
from src.hypotheses.h5_parameter_manifest import (
    H5ParameterManifestStore,
    H5ParameterSupportCriteria,
)
from src.strategy.models import TrendMomentumConfig


UTC = timezone.utc


def fixed_clock() -> datetime:
    return datetime(2026, 8, 20, 18, 0, tzinfo=UTC)


def prepare(store: H5ParameterManifestStore):
    return store.prepare(
        symbol="BTCUSDC",
        interval="15m",
        source_mechanism_run_id="mechanism123",
        source_mechanism_manifest_sha256="a" * 64,
        source_multiregime_run_id="multiregime123",
        source_r_audit_sha256="b" * 64,
        strategy_config=TrendMomentumConfig(),
        backtest_config=BacktestConfig(),
    )


def test_support_criteria_are_frozen() -> None:
    criteria = H5ParameterSupportCriteria()

    assert criteria.minimum_eligible_windows == 6
    assert criteria.consistency_percent_required == Decimal("60")
    assert criteria.trade_count_ratio_required == Decimal("0.50")
    assert criteria.maximum_drawdown_worse_ratio == Decimal("1.25")


def test_support_criteria_cannot_be_changed() -> None:
    with pytest.raises(ValueError, match="frozen"):
        H5ParameterSupportCriteria(
            consistency_percent_required=Decimal("55")
        )


def test_manifest_contains_exact_four_candidates(tmp_path) -> None:
    result = prepare(
        H5ParameterManifestStore(
            tmp_path,
            clock=fixed_clock,
        )
    )

    candidates = result.payload["candidates"]

    assert [item["candidate_id"] for item in candidates] == [
        "H5_Q25",
        "H5_Q35",
        "H5_Q45",
        "H5_Q50",
    ]


def test_manifest_locks_blind_holdout(tmp_path) -> None:
    result = prepare(
        H5ParameterManifestStore(
            tmp_path,
            clock=fixed_clock,
        )
    )

    policy = result.payload["holdout_policy"]

    assert policy["required_status"] == "LOCKED_BLIND_HOLDOUT"
    assert policy["revealed"] is False
    assert policy["consumed"] is False
    assert policy["evaluated"] is False


def test_manifest_primary_metric_is_net_r_expectancy(tmp_path) -> None:
    result = prepare(
        H5ParameterManifestStore(
            tmp_path,
            clock=fixed_clock,
        )
    )

    assert (
        result.payload["primary_metric"]
        == "combined_net_expectancy_R_per_trade"
    )


def test_manifest_is_deterministic(tmp_path) -> None:
    store = H5ParameterManifestStore(
        tmp_path,
        clock=fixed_clock,
    )

    first = prepare(store)
    second = prepare(store)

    assert first.run_id == second.run_id
    assert (
        first.configuration_sha256
        == second.configuration_sha256
    )


def test_v34_requires_positive_net_expectancy(tmp_path) -> None:
    result = prepare(
        H5ParameterManifestStore(
            tmp_path,
            clock=fixed_clock,
        )
    )

    gate = result.payload["v3_4_eligibility"]

    assert gate["combined_net_expectancy_R"] == "> 0"
    assert gate["combined_profit_factor"] == "> 1"
    assert gate["positive_net_windows_percent"] == ">= 60"