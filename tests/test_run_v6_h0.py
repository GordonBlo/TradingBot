from copy import deepcopy

import pytest

import src.cli.run_v6_h0 as guard
from src.backtest.engine import BacktestEngine
from src.cli.run_v6_h0 import (
    main,
    validate_actual_fill_execution,
    validate_eligible_windows,
    validate_v6_h0_implementation,
    validate_v6_h0_manifest,
)
from src.research.v6_mtf_continuation_preregistration import build_manifest
from src.strategy.v6_mtf_continuation import V6MTFContinuationStrategy


def test_v6_h0_guard_accepts_frozen_configuration() -> None:
    manifest = build_manifest()

    validate_v6_h0_manifest(manifest)
    validate_v6_h0_implementation(manifest)


@pytest.mark.parametrize("change_definition", [False, True])
def test_v6_h0_guard_rejects_run_id_or_manifest_mismatch(
    change_definition: bool,
) -> None:
    manifest = deepcopy(build_manifest())
    if change_definition:
        manifest["entry"]["ema_period_15m"] = 21
        match = "frozen definition"
    else:
        manifest["run_id"] = "wrong"
        match = "run ID"

    with pytest.raises(ValueError, match=match):
        validate_v6_h0_manifest(manifest)


def test_v6_h0_guard_rejects_15m_4h_configuration_mismatch() -> None:
    manifest = deepcopy(build_manifest())
    manifest["market"]["higher_timeframe_regime_interval"] = "1h"

    with pytest.raises(ValueError, match="15m/4h"):
        validate_v6_h0_implementation(manifest)


def test_v6_h0_guard_rejects_strategy_constant_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(V6MTFContinuationStrategy, "ATR_PERIOD_4H", 13)

    with pytest.raises(ValueError, match="strategy constants"):
        validate_v6_h0_implementation(build_manifest())


def test_v6_h0_guard_rejects_cost_floor_conversion_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        V6MTFContinuationStrategy,
        "MIN_STOP_DISTANCE_FRACTION",
        guard.Decimal("0.0095"),
    )

    with pytest.raises(ValueError, match="strategy constants|conversion"):
        validate_v6_h0_implementation(build_manifest())


def test_v6_h0_guard_rejects_wrong_eligible_window_count() -> None:
    with pytest.raises(ValueError, match="exactly 11"):
        validate_eligible_windows({str(index): {} for index in range(10)})


def test_v6_h0_guard_rejects_holdout_contamination() -> None:
    manifest = deepcopy(build_manifest())
    manifest["dataset_policy"]["blind_holdout"]["evaluated"] = True

    with pytest.raises(ValueError, match="holdout integrity"):
        validate_v6_h0_implementation(manifest)


def test_v6_h0_real_execution_remains_disabled() -> None:
    with pytest.raises(SystemExit, match="REPLAY DISABLED"):
        main([])


def test_v6_h0_guard_protects_completed_4h_causality() -> None:
    manifest = deepcopy(build_manifest())
    manifest["higher_timeframe_construction"][
        "forming_4h_candle_allowed_for_atr"
    ] = True

    with pytest.raises(ValueError, match="completed-4h causality"):
        validate_v6_h0_implementation(manifest)


def test_v6_h0_guard_protects_complete_signal_time_history(monkeypatch) -> None:
    monkeypatch.setattr(
        V6MTFContinuationStrategy,
        "requires_full_history",
        property(lambda self: False),
    )

    with pytest.raises(ValueError, match="complete signal-time history"):
        validate_v6_h0_implementation(build_manifest())


def test_v6_h0_guard_protects_actual_fill_stop_floor(monkeypatch) -> None:
    original = guard.inspect.getsource

    def altered_source(value: object) -> str:
        if value is BacktestEngine._execute_pending:
            return "def _execute_pending(): pass"
        return original(value)

    monkeypatch.setattr(guard.inspect, "getsource", altered_source)

    with pytest.raises(ValueError, match="actual-fill"):
        validate_actual_fill_execution()
