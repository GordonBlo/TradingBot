from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.hypotheses.h8_preregistration import (
    H8Definition,
    H8PreregistrationStore,
    H8SupportCriteria,
)


UTC = timezone.utc


def fixed_clock() -> datetime:
    return datetime(
        2026,
        8,
        21,
        12,
        0,
        tzinfo=UTC,
    )


def test_h8_definition_is_frozen() -> None:
    definition = H8Definition()

    assert definition.trigger_r == Decimal("1.0")
    assert definition.take_profit_r == Decimal("2.0")

    assert (
        definition.activation_timing
        == "NEXT_BAR_AFTER_CLOSED_TRIGGER"
    )

    assert definition.protective_stop == "ENTRY_PRICE"

    assert definition.preserve_h5_q25_entries is True
    assert definition.partial_exit is False
    assert definition.ambiguous_bar_policy == "STOP_FIRST"


def test_h8_trigger_cannot_be_tuned() -> None:
    with pytest.raises(ValueError, match="frozen"):
        H8Definition(
            trigger_r=Decimal("0.5")
        )


def test_h8_support_gate_is_frozen() -> None:
    criteria = H8SupportCriteria()

    assert criteria.minimum_eligible_windows == 11

    assert (
        criteria.consistency_percent_required
        == Decimal("60")
    )

    assert (
        criteria.minimum_trade_count_ratio
        == Decimal("0.80")
    )


def test_h8_support_gate_cannot_be_relaxed() -> None:
    with pytest.raises(ValueError, match="frozen"):
        H8SupportCriteria(
            consistency_percent_required=Decimal("50")
        )


def test_manifest_is_deterministic(tmp_path) -> None:
    store = H8PreregistrationStore(
        tmp_path,
        clock=fixed_clock,
    )

    first = store.prepare(
        source_v33_run_id="b104ea78b4c88bcf",
        source_v331_diagnostic_run_id="ef593be9a0bb2697",
    )

    second = store.prepare(
        source_v33_run_id="b104ea78b4c88bcf",
        source_v331_diagnostic_run_id="ef593be9a0bb2697",
    )

    assert first[0] == second[0]

    assert (
        first[2]["configuration_sha256"]
        == second[2]["configuration_sha256"]
    )


def test_manifest_keeps_holdout_locked(tmp_path) -> None:
    store = H8PreregistrationStore(
        tmp_path,
        clock=fixed_clock,
    )

    _, _, payload = store.prepare(
        source_v33_run_id="b104ea78b4c88bcf",
        source_v331_diagnostic_run_id="ef593be9a0bb2697",
    )

    assert (
        payload["blind_holdout_status"]
        == "LOCKED_BLIND_HOLDOUT"
    )

    assert payload["blind_holdout_revealed"] is False
    assert payload["blind_holdout_consumed"] is False
    assert payload["blind_holdout_evaluated"] is False


def test_next_stage_requires_real_positive_net_edge(tmp_path) -> None:
    store = H8PreregistrationStore(
        tmp_path,
        clock=fixed_clock,
    )

    _, _, payload = store.prepare(
        source_v33_run_id="b104ea78b4c88bcf",
        source_v331_diagnostic_run_id="ef593be9a0bb2697",
    )

    gate = payload["next_stage_eligibility"]

    assert gate["combined_net_expectancy_r"] == "> 0"
    assert gate["profit_factor"] == "> 1"
    assert gate["positive_net_windows_percent"] == ">= 60"