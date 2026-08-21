from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.hypotheses.h9_preregistration import (
    H9Definition,
    H9PreregistrationStore,
    H9SupportCriteria,
)


UTC = timezone.utc


def clock() -> datetime:
    return datetime(
        2026,
        8,
        21,
        12,
        45,
        tzinfo=UTC,
    )


def test_h9_definition_is_frozen() -> None:
    definition = H9Definition()

    assert definition.adverse_trigger_r == Decimal("0.5")
    assert definition.observation_bars == 4
    assert definition.execution_timing == "NEXT_BAR_OPEN"


def test_h9_trigger_cannot_be_tuned() -> None:
    with pytest.raises(ValueError, match="frozen"):
        H9Definition(
            adverse_trigger_r=Decimal("0.4")
        )


def test_h9_window_cannot_be_tuned() -> None:
    with pytest.raises(ValueError, match="frozen"):
        H9Definition(
            observation_bars=3
        )


def test_h9_gate_is_frozen() -> None:
    criteria = H9SupportCriteria()

    assert (
        criteria.consistency_percent_required
        == Decimal("60")
    )

    assert (
        criteria.minimum_trade_count_ratio
        == Decimal("0.80")
    )

    assert (
        criteria.minimum_baseline_entry_match_ratio
        == Decimal("0.80")
    )


def test_h9_gate_cannot_be_relaxed() -> None:
    with pytest.raises(ValueError, match="frozen"):
        H9SupportCriteria(
            consistency_percent_required=Decimal("50")
        )


def test_manifest_is_deterministic(tmp_path) -> None:
    store = H9PreregistrationStore(
        tmp_path,
        clock=clock,
    )

    kwargs = {
        "source_v33_run_id": "b104ea78b4c88bcf",
        "source_v331_run_id": "ef593be9a0bb2697",
        "source_early_failure_sha256": "abc123",
    }

    first = store.prepare(**kwargs)
    second = store.prepare(**kwargs)

    assert first[0] == second[0]

    assert (
        first[2]["configuration_sha256"]
        == second[2]["configuration_sha256"]
    )


def test_manifest_keeps_holdout_locked(tmp_path) -> None:
    store = H9PreregistrationStore(
        tmp_path,
        clock=clock,
    )

    _, _, payload = store.prepare(
        source_v33_run_id="b104ea78b4c88bcf",
        source_v331_run_id="ef593be9a0bb2697",
        source_early_failure_sha256="abc123",
    )

    assert (
        payload["blind_holdout_status"]
        == "LOCKED_BLIND_HOLDOUT"
    )

    assert payload["blind_holdout_revealed"] is False
    assert payload["blind_holdout_consumed"] is False
    assert payload["blind_holdout_evaluated"] is False