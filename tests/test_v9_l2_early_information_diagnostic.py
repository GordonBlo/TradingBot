from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from src.diagnostics.v9_l2_early_information import (
    DiagnosticSample,
    build_chronological_folds,
    classify_l2_information,
    run_diagnostic,
    samples_from_one_second_rows,
)
from src.research.v9_l2_early_information_preregistration import FROZEN_FEATURES
from src.research.v9_l2_readiness import ReadinessReport, SessionEligibility


def _row(timestamp: datetime, mid: float, signal: float = 0.0) -> dict[str, object]:
    row: dict[str, object] = {
        "bucket_open_utc": (timestamp - timedelta(seconds=1)).isoformat(),
        "bucket_close_utc": timestamp.isoformat(),
        "last_feature_available_at_utc": timestamp.isoformat(),
        "mid_price_last": str(mid),
        "spread_bps_last": str(signal),
    }
    return row


def test_samples_use_exact_causal_one_second_inputs_and_exact_forward_target() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [_row(start + timedelta(seconds=index), 100 + index, index) for index in range(61)]
    samples = samples_from_one_second_rows("S1", rows)
    first = samples[0]
    assert first.timestamp == start
    assert len(first.features) == len(FROZEN_FEATURES) == 15
    assert first.target(30) == pytest.approx(math.log(130 / 100))
    assert first.target(60) == pytest.approx(math.log(160 / 100))
    assert len(samples) == 31


def test_samples_reject_future_feature_timestamp_and_do_not_fill_missing_target() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    future_row = _row(start, 100)
    future_row["last_feature_available_at_utc"] = (start + timedelta(microseconds=1)).isoformat()
    with pytest.raises(ValueError, match="future feature"):
        samples_from_one_second_rows("S", [future_row])
    rows = [_row(start, 100), _row(start + timedelta(seconds=31), 101)]
    assert samples_from_one_second_rows("S", rows) == ()


def _sample(session_id: str, timestamp: datetime, signal: float) -> DiagnosticSample:
    target = signal * 0.01
    horizons = (1, 5, 30, 60)
    return DiagnosticSample(
        session_id=session_id,
        timestamp=timestamp,
        features=(signal,) + (0.0,) * 14,
        targets=(target,) * 4,
        target_timestamps=tuple(timestamp + timedelta(seconds=value) for value in horizons),
    )


def _synthetic_inputs() -> tuple[tuple[DiagnosticSample, ...], ReadinessReport]:
    samples = []
    sessions = []
    for session_index in range(8):
        start = datetime(2026, 1, 1 + session_index, 1, tzinfo=UTC)
        session_id = f"S{session_index}"
        sessions.append(
            SessionEligibility(
                session_id, start.isoformat(), (start + timedelta(hours=2.5)).isoformat(),
                2.5, "ELIGIBLE", True, "synthetic integrity passed"
            )
        )
        for row_index in range(12):
            signal = float(row_index - 5) + session_index * 0.01
            samples.append(_sample(session_id, start + timedelta(seconds=row_index), signal))
    readiness = ReadinessReport(
        cutoff_utc="2026-01-01T00:15:00Z",
        discovered_sessions=8,
        engineering_only_sessions=0,
        eligible_closed_sessions=8,
        eligible_hours=20.0,
        eligible_utc_dates=8,
        straddling_sessions=0,
        active_sessions=0,
        integrity_failed_sessions=0,
        ready=True,
        reasons=(),
        sessions=tuple(sessions),
    )
    return tuple(samples), readiness


def test_folds_are_session_isolated_and_training_targets_are_known_before_holdout() -> None:
    samples, _ = _synthetic_inputs()
    folds = build_chronological_folds(samples)
    assert len(folds) == 7
    for fold in folds:
        training_sessions = {samples[index].session_id for index in fold.prepared.training_indices}
        held_sessions = {samples[index].session_id for index in fold.prepared.held_out_indices}
        assert held_sessions == {fold.held_out_session}
        assert fold.held_out_session not in training_sessions
        assert all(
            samples[index].target_timestamp(30) <= fold.held_out_start
            for index in fold.prepared.training_indices
        )


def test_diagnostic_is_deterministic_on_synthetic_data_and_guarded_by_readiness() -> None:
    samples, readiness = _synthetic_inputs()
    first = run_diagnostic(samples, readiness)
    second = run_diagnostic(samples, readiness)
    assert first == second
    assert first.held_out_session_count == 7
    assert first.held_out_spearman is not None and first.held_out_spearman > 0
    assert first.permutation_p_value >= 1 / 1001
    blocked = replace(readiness, ready=False, reasons=("not ready",))
    with pytest.raises(RuntimeError, match="readiness gate"):
        run_diagnostic(samples, blocked)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(held_out_spearman=0.1, permutation_p_value=0.049, model_mse=0.9,
                 baseline_mse=1.0, positive_direction_sessions=7, held_out_session_count=10),
            "STABLE_L2_INFORMATION",
        ),
        (
            dict(held_out_spearman=0.1, permutation_p_value=0.05, model_mse=0.9,
                 baseline_mse=1.0, positive_direction_sessions=7, held_out_session_count=10),
            "WEAK_OR_UNSTABLE_L2_INFORMATION",
        ),
        (
            dict(held_out_spearman=-0.1, permutation_p_value=0.5, model_mse=1.0,
                 baseline_mse=1.0, positive_direction_sessions=0, held_out_session_count=10),
            "NO_STABLE_L2_INFORMATION",
        ),
    ],
)
def test_classification_is_frozen(kwargs: dict[str, object], expected: str) -> None:
    assert classify_l2_information(**kwargs) == expected
