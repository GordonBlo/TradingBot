from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from random import Random

import pytest

import src.diagnostics.v9_l2_early_information as diagnostic
from src.diagnostics.v8_derivatives_discovery import pearson, within_window_permutations
from src.diagnostics.v9_l2_early_information import (
    DiagnosticIntegrityError,
    DiagnosticSample,
    build_chronological_folds,
    classify_l2_information,
    run_diagnostic,
    samples_from_one_second_rows,
    session_circular_shift_permutations,
    valid_circular_offsets,
)
from src.research.v9_l2_early_information_preregistration_v2 import FROZEN_FEATURES
from src.research.v9_l2_readiness import (
    ReadinessReport,
    SessionArtifactBinding,
    SessionEligibility,
)


def _binding(session_id: str) -> SessionArtifactBinding:
    return SessionArtifactBinding(
        session_id=session_id,
        raw_event_log=f"{session_id}.jsonl",
        raw_sha256="1" * 64,
        closure_summary=f"{session_id}.summary.json",
        closure_summary_sha256="2" * 64,
        features_1s=f"{session_id}.features_1s.csv",
        features_1s_sha256="3" * 64,
    )


def _session(
    session_id: str,
    start: datetime,
    *,
    end: datetime | None = None,
) -> SessionEligibility:
    ended = end or start + timedelta(hours=2.5)
    return SessionEligibility(
        session_id=session_id,
        started_at_utc=start.isoformat(),
        ended_at_utc=ended.isoformat(),
        duration_hours=(ended - start).total_seconds() / 3600,
        classification="ELIGIBLE",
        integrity_passed=True,
        reason="synthetic integrity passed",
        artifact_binding=_binding(session_id),
    )


def _row(
    timestamp: datetime,
    mid: float,
    signal: float = 0.0,
    *,
    available_at: datetime | None = None,
) -> dict[str, object]:
    return {
        "bucket_open_utc": (timestamp - timedelta(seconds=1)).isoformat(),
        "bucket_close_utc": timestamp.isoformat(),
        "last_feature_available_at_utc": (
            timestamp if available_at is None else available_at
        ).isoformat(),
        "mid_price_last": str(mid),
        "spread_bps_last": str(signal),
    }


def test_samples_use_exact_causal_one_second_inputs_and_exact_forward_target() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [_row(start + timedelta(seconds=index), 100 + index, index) for index in range(61)]
    samples = samples_from_one_second_rows(_session("S1", start), rows)
    first = samples[0]
    assert first.timestamp == start
    assert len(first.features) == len(FROZEN_FEATURES) == 15
    assert first.target(30) == pytest.approx(math.log(130 / 100))
    assert first.target(60) == pytest.approx(math.log(160 / 100))
    assert len(samples) == 31


def test_missing_or_future_causal_provenance_is_rejected() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    missing = _row(start, 100)
    missing["last_feature_available_at_utc"] = ""
    with pytest.raises(DiagnosticIntegrityError, match="provenance is missing"):
        samples_from_one_second_rows(
            _session("S", start), [missing, _row(start + timedelta(seconds=30), 101)]
        )

    future = _row(start, 100, available_at=start + timedelta(microseconds=1))
    with pytest.raises(DiagnosticIntegrityError, match="future feature/target-start"):
        samples_from_one_second_rows(
            _session("S", start), [future, _row(start + timedelta(seconds=30), 101)]
        )


def test_missing_exact_target_is_not_filled() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [_row(start, 100), _row(start + timedelta(seconds=31), 101)]
    assert samples_from_one_second_rows(_session("S", start), rows) == ()


def test_target_beyond_authoritative_session_end_is_rejected() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    session = _session("S", start, end=start + timedelta(seconds=29, milliseconds=999))
    rows = [_row(start, 100), _row(start + timedelta(seconds=30), 101)]
    assert samples_from_one_second_rows(session, rows) == ()


def test_target_construction_cannot_cross_sessions() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    first = samples_from_one_second_rows(_session("A", start), [_row(start, 100)])
    second = samples_from_one_second_rows(
        _session("B", start + timedelta(seconds=30)),
        [_row(start + timedelta(seconds=30), 101)],
    )
    assert first == second == ()


def _sample(
    session: SessionEligibility,
    timestamp: datetime,
    signal: float,
    *,
    target: float | None = None,
) -> DiagnosticSample:
    value = signal * 0.01 if target is None else target
    horizons = (1, 5, 30, 60)
    target_timestamps = tuple(timestamp + timedelta(seconds=item) for item in horizons)
    return DiagnosticSample(
        session_id=session.session_id,
        timestamp=timestamp,
        feature_source_available_at=timestamp,
        features=(signal,) + (0.0,) * 14,
        targets=(value,) * 4,
        target_timestamps=target_timestamps,
        target_source_available_at=target_timestamps,
        artifact_binding=session.artifact_binding,
    )


def _synthetic_inputs(
    *, sample_count: int = 130,
) -> tuple[tuple[DiagnosticSample, ...], ReadinessReport]:
    samples = []
    sessions = []
    for session_index in range(8):
        start = datetime(2026, 1, 1 + session_index, 1, tzinfo=UTC)
        session = _session(f"S{session_index}", start)
        sessions.append(session)
        for row_index in range(sample_count):
            signal = float(row_index - sample_count // 2) + session_index * 0.01
            samples.append(_sample(session, start + timedelta(seconds=row_index), signal))
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


def test_all_missing_training_feature_fails_explicitly() -> None:
    samples, _ = _synthetic_inputs()
    changed = tuple(
        replace(sample, features=(None,) + sample.features[1:])
        if sample.session_id == "S0"
        else sample
        for sample in samples
    )
    with pytest.raises(DiagnosticIntegrityError, match="entirely missing: spread_bps"):
        build_chronological_folds(changed)


def test_circular_null_preserves_overlap_dependence_better_than_row_shuffle() -> None:
    randomizer = Random(20260909)
    count = 600
    increments = [randomizer.gauss(0.0, 1.0) for _ in range(count + 30)]
    path = [0.0]
    for value in increments:
        path.append(path[-1] + value)
    targets = [path[index + 30] - path[index] for index in range(count)]
    start = datetime(2026, 1, 1, tzinfo=UTC)
    session = _session("S", start)
    samples = tuple(
        _sample(session, start + timedelta(seconds=index), 0.0, target=value)
        for index, value in enumerate(targets)
    )
    circular = next(session_circular_shift_permutations(samples, seed=42, count=1))
    shuffled = next(within_window_permutations(["S"] * count, seed=42, count=1))
    original_lag = pearson(targets[:-1], targets[1:])
    circular_targets = [targets[source] for source in circular]
    shuffled_targets = [targets[source] for source in shuffled]
    circular_lag = pearson(circular_targets[:-1], circular_targets[1:])
    shuffled_lag = pearson(shuffled_targets[:-1], shuffled_targets[1:])
    assert original_lag is not None and circular_lag is not None and shuffled_lag is not None
    assert abs(circular_lag - original_lag) < 0.05
    assert circular_lag > shuffled_lag + 0.75


def test_circular_offsets_exclude_zero_and_near_zero_and_are_deterministic() -> None:
    offsets = valid_circular_offsets(130)
    assert offsets == tuple(range(61, 70))
    samples, _ = _synthetic_inputs()
    first = tuple(session_circular_shift_permutations(samples, seed=20260825, count=20))
    second = tuple(session_circular_shift_permutations(samples, seed=20260825, count=20))
    assert first == second
    for permutation in first:
        for session_index in range(8):
            group_start = session_index * 130
            offset = permutation[group_start] - group_start
            assert min(offset, 130 - offset) > 60


def test_circular_generator_produces_exactly_one_thousand_valid_mappings() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    session = _session("S", start)
    samples = tuple(
        _sample(session, start + timedelta(seconds=index), float(index))
        for index in range(122)
    )
    permutations = tuple(
        session_circular_shift_permutations(samples, seed=20260825, count=1_000)
    )
    assert len(permutations) == 1_000
    assert all(len(mapping) == 122 for mapping in permutations)


def test_diagnostic_requires_exact_null_count(monkeypatch) -> None:
    samples, readiness = _synthetic_inputs()
    monkeypatch.setattr(diagnostic, "PERMUTATION_COUNT", 2)
    identity = tuple(range(len(samples)))
    monkeypatch.setattr(
        diagnostic,
        "session_circular_shift_permutations",
        lambda *args, **kwargs: iter((identity,)),
    )
    with pytest.raises(DiagnosticIntegrityError, match="exactly 2 valid"):
        run_diagnostic(samples, readiness)


def test_undefined_null_statistic_fails_closed(monkeypatch) -> None:
    samples, readiness = _synthetic_inputs()
    monkeypatch.setattr(diagnostic, "PERMUTATION_COUNT", 1)
    identity = tuple(range(len(samples)))
    monkeypatch.setattr(
        diagnostic,
        "session_circular_shift_permutations",
        lambda *args, **kwargs: iter((identity,)),
    )
    evaluations = iter(
        (
            ([0.0, 1.0], [0.0, 1.0], [0.5, 0.5], 1),
            ([0.0, 1.0], [0.5, 0.5], [0.5, 0.5], 0),
        )
    )
    monkeypatch.setattr(diagnostic, "_evaluate_targets", lambda *args: next(evaluations))
    with pytest.raises(DiagnosticIntegrityError, match="permutation statistic is undefined"):
        run_diagnostic(samples, readiness)


def test_diagnostic_is_deterministic_and_guarded_by_readiness(monkeypatch) -> None:
    samples, readiness = _synthetic_inputs()
    monkeypatch.setattr(diagnostic, "PERMUTATION_COUNT", 20)
    first = run_diagnostic(tuple(reversed(samples)), readiness)
    second = run_diagnostic(samples, readiness)
    assert first == second
    assert first.held_out_session_count == 7
    assert first.held_out_spearman > 0
    assert first.permutation_p_value >= 1 / 21
    blocked = replace(readiness, ready=False, reasons=("not ready",))
    with pytest.raises(DiagnosticIntegrityError, match="readiness gate"):
        run_diagnostic(samples, blocked)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(
                held_out_spearman=0.1,
                permutation_p_value=0.049,
                model_mse=0.9,
                baseline_mse=1.0,
                positive_direction_sessions=7,
                held_out_session_count=10,
            ),
            "STABLE_L2_INFORMATION",
        ),
        (
            dict(
                held_out_spearman=0.1,
                permutation_p_value=0.05,
                model_mse=0.9,
                baseline_mse=1.0,
                positive_direction_sessions=7,
                held_out_session_count=10,
            ),
            "WEAK_OR_UNSTABLE_L2_INFORMATION",
        ),
        (
            dict(
                held_out_spearman=-0.1,
                permutation_p_value=0.5,
                model_mse=1.0,
                baseline_mse=1.0,
                positive_direction_sessions=0,
                held_out_session_count=10,
            ),
            "NO_STABLE_L2_INFORMATION",
        ),
    ],
)
def test_classification_is_frozen(kwargs: dict[str, object], expected: str) -> None:
    assert classify_l2_information(**kwargs) == expected
