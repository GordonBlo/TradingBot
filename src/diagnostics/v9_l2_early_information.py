"""Frozen, leakage-safe statistical engine for the prospective V9 L2 diagnostic."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Iterable, Sequence

from src.diagnostics.v8_derivatives_discovery import (
    PreparedFold,
    fit_prepared_fold,
    mean_squared_error,
    prepare_fold,
    spearman,
    within_window_permutations,
)
from src.research.v9_l2_early_information_preregistration import (
    FROZEN_FEATURES,
    PERMUTATION_COUNT,
    PERMUTATION_SEED,
    PRIMARY_HORIZON_SECONDS,
    RIDGE_ALPHA,
    SECONDARY_HORIZONS_SECONDS,
    SIGNIFICANCE_ALPHA,
)
from src.research.v9_l2_readiness import ReadinessReport


TARGET_HORIZONS = tuple(sorted((*SECONDARY_HORIZONS_SECONDS, PRIMARY_HORIZON_SECONDS)))
_INPUT_COLUMNS = (
    "spread_bps_last",
    "microprice_minus_mid_bps_last",
    "depth_imbalance_1_last",
    "depth_imbalance_5_last",
    "depth_imbalance_10_last",
    "depth_imbalance_20_last",
    "bid_depth_top_20_last",
    "ask_depth_top_20_last",
    "bid_depth_concentration_top1_over_top20_last",
    "ask_depth_concentration_top1_over_top20_last",
    "bid_depth_added",
    "bid_depth_removed",
    "ask_depth_added",
    "ask_depth_removed",
    "update_intensity_per_second",
)
if len(_INPUT_COLUMNS) != len(FROZEN_FEATURES):  # pragma: no cover - frozen invariant
    raise RuntimeError("V9 frozen feature mapping is incomplete")


@dataclass(frozen=True, slots=True)
class DiagnosticSample:
    session_id: str
    timestamp: datetime
    features: tuple[float | None, ...]
    targets: tuple[float | None, ...]
    target_timestamps: tuple[datetime | None, ...]

    def target(self, horizon_seconds: int) -> float | None:
        return self.targets[TARGET_HORIZONS.index(horizon_seconds)]

    def target_timestamp(self, horizon_seconds: int) -> datetime | None:
        return self.target_timestamps[TARGET_HORIZONS.index(horizon_seconds)]


@dataclass(frozen=True, slots=True)
class ChronologicalFold:
    held_out_session: str
    held_out_start: datetime
    prepared: PreparedFold


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    sample_count: int
    held_out_sample_count: int
    held_out_session_count: int
    held_out_spearman: float | None
    model_mse: float
    baseline_mse: float
    positive_direction_sessions: int
    permutation_p_value: float
    permutation_null_mean: float | None
    classification: str


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("V9 diagnostic timestamps must be timezone-aware")
    return parsed.astimezone(UTC)


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("V9 diagnostic values must be finite")
    return parsed


def samples_from_one_second_rows(
    session_id: str,
    rows: Iterable[dict[str, Any]],
) -> tuple[DiagnosticSample, ...]:
    """Build exact-time forward targets without fill, interpolation, or future inputs."""

    materialized = list(rows)
    by_timestamp: dict[datetime, dict[str, Any]] = {}
    for row in materialized:
        bucket_open = _utc(str(row["bucket_open_utc"]))
        bucket_close = _utc(str(row["bucket_close_utc"]))
        if bucket_close - bucket_open != timedelta(seconds=1):
            raise ValueError("diagnostic input is not an exact one-second bucket")
        if bucket_close in by_timestamp:
            raise ValueError("duplicate one-second diagnostic bucket")
        available = row.get("last_feature_available_at_utc")
        if available and _utc(str(available)) > bucket_close:
            raise ValueError("future feature timestamp detected")
        by_timestamp[bucket_close] = row

    samples: list[DiagnosticSample] = []
    for timestamp in sorted(by_timestamp):
        row = by_timestamp[timestamp]
        current_mid = _number(row.get("mid_price_last"))
        if current_mid is None or current_mid <= 0:
            continue
        targets: list[float | None] = []
        target_timestamps: list[datetime | None] = []
        for horizon in TARGET_HORIZONS:
            future_timestamp = timestamp + timedelta(seconds=horizon)
            future = by_timestamp.get(future_timestamp)
            future_mid = _number(future.get("mid_price_last")) if future else None
            if future_mid is None or future_mid <= 0:
                targets.append(None)
                target_timestamps.append(None)
            else:
                future_available = future.get("last_feature_available_at_utc")
                if future_available and _utc(str(future_available)) > future_timestamp:
                    raise ValueError("future target-source timestamp detected")
                targets.append(math.log(future_mid / current_mid))
                target_timestamps.append(future_timestamp)
        if targets[TARGET_HORIZONS.index(PRIMARY_HORIZON_SECONDS)] is None:
            continue
        samples.append(
            DiagnosticSample(
                session_id=session_id,
                timestamp=timestamp,
                features=tuple(_number(row.get(column)) for column in _INPUT_COLUMNS),
                targets=tuple(targets),
                target_timestamps=tuple(target_timestamps),
            )
        )
    return tuple(samples)


def build_chronological_folds(
    samples: Sequence[DiagnosticSample],
) -> tuple[ChronologicalFold, ...]:
    if not samples:
        raise ValueError("diagnostic samples are empty")
    width = len(FROZEN_FEATURES)
    if any(len(sample.features) != width for sample in samples):
        raise ValueError("diagnostic samples do not match the frozen feature family")

    session_ranges: dict[str, tuple[datetime, datetime]] = {}
    for sample in samples:
        if sample.timestamp.tzinfo is None or sample.timestamp.utcoffset() is None:
            raise ValueError("sample timestamps must be UTC-aware")
        current = session_ranges.get(sample.session_id)
        session_ranges[sample.session_id] = (
            min(current[0], sample.timestamp) if current else sample.timestamp,
            max(current[1], sample.timestamp) if current else sample.timestamp,
        )
    ordered_sessions = sorted(session_ranges, key=lambda item: session_ranges[item][0])
    for previous, current in zip(ordered_sessions, ordered_sessions[1:]):
        if session_ranges[previous][1] >= session_ranges[current][0]:
            raise ValueError("chronological session blocks overlap")

    features = [sample.features for sample in samples]
    folds: list[ChronologicalFold] = []
    for held_out_position in range(1, len(ordered_sessions)):
        held_out_session = ordered_sessions[held_out_position]
        held_out_start = session_ranges[held_out_session][0]
        training_sessions = set(ordered_sessions[:held_out_position])
        training = [
            index
            for index, sample in enumerate(samples)
            if sample.session_id in training_sessions
            and sample.target(PRIMARY_HORIZON_SECONDS) is not None
            and sample.target_timestamp(PRIMARY_HORIZON_SECONDS) is not None
            and sample.target_timestamp(PRIMARY_HORIZON_SECONDS) <= held_out_start
        ]
        held_out = [
            index
            for index, sample in enumerate(samples)
            if sample.session_id == held_out_session
            and sample.target(PRIMARY_HORIZON_SECONDS) is not None
        ]
        if not training or not held_out:
            raise ValueError("chronological fold has no leakage-safe training or held-out rows")
        if any(samples[index].session_id == held_out_session for index in training):
            raise ValueError("held-out session leaked into training")
        prepared = prepare_fold(
            features,
            training_indices=training,
            held_out_indices=held_out,
            alpha=RIDGE_ALPHA,
        )
        folds.append(ChronologicalFold(held_out_session, held_out_start, prepared))
    if not folds:
        raise ValueError("at least two chronological session blocks are required")
    return tuple(folds)


def classify_l2_information(
    *,
    held_out_spearman: float | None,
    permutation_p_value: float,
    model_mse: float,
    baseline_mse: float,
    positive_direction_sessions: int,
    held_out_session_count: int,
) -> str:
    correlation_positive = held_out_spearman is not None and held_out_spearman > 0.0
    significant = permutation_p_value < SIGNIFICANCE_ALPHA
    beats_baseline = model_mse < baseline_mse
    direction_positive = (
        held_out_session_count > 0
        and positive_direction_sessions / held_out_session_count >= 0.70
    )
    evidence = (correlation_positive, significant, beats_baseline, direction_positive)
    if all(evidence):
        return "STABLE_L2_INFORMATION"
    if any(evidence):
        return "WEAK_OR_UNSTABLE_L2_INFORMATION"
    return "NO_STABLE_L2_INFORMATION"


def _evaluate_targets(
    folds: Sequence[ChronologicalFold],
    targets: Sequence[float],
) -> tuple[list[float], list[float], list[float], int]:
    actual: list[float] = []
    predicted: list[float] = []
    baseline: list[float] = []
    positive_sessions = 0
    for fold in folds:
        fit = fit_prepared_fold(fold.prepared, targets)
        held_actual = [targets[index] for index in fold.prepared.held_out_indices]
        actual.extend(held_actual)
        predicted.extend(fit.predictions)
        baseline.extend(fit.baseline_predictions)
        correlation = spearman(held_actual, fit.predictions)
        positive_sessions += correlation is not None and correlation > 0.0
    return actual, predicted, baseline, positive_sessions


def run_diagnostic(
    samples: Sequence[DiagnosticSample],
    readiness: ReadinessReport,
) -> DiagnosticResult:
    if not readiness.ready:
        raise RuntimeError("V9 predictive diagnostic refused: readiness gate has not passed")
    eligible_ids = {
        session.session_id
        for session in readiness.sessions
        if session.classification == "ELIGIBLE"
    }
    if {sample.session_id for sample in samples} != eligible_ids:
        raise ValueError("diagnostic samples do not exactly match readiness-eligible sessions")

    folds = build_chronological_folds(samples)
    targets = [
        float(sample.target(PRIMARY_HORIZON_SECONDS))
        for sample in samples
    ]
    actual, predictions, baseline, positive_sessions = _evaluate_targets(folds, targets)
    observed = spearman(actual, predictions)
    window_ids = [sample.session_id for sample in samples]
    null_values: list[float] = []
    for permutation in within_window_permutations(
        window_ids,
        seed=PERMUTATION_SEED,
        count=PERMUTATION_COUNT,
    ):
        permuted_targets = [targets[source] for source in permutation]
        null_actual, null_predictions, _, _ = _evaluate_targets(folds, permuted_targets)
        null_correlation = spearman(null_actual, null_predictions)
        if null_correlation is not None:
            null_values.append(null_correlation)
    exceedances = (
        len(null_values)
        if observed is None
        else sum(value >= observed for value in null_values)
    )
    p_value = (exceedances + 1) / (PERMUTATION_COUNT + 1)
    model_mse = mean_squared_error(actual, predictions)
    baseline_mse = mean_squared_error(actual, baseline)
    classification = classify_l2_information(
        held_out_spearman=observed,
        permutation_p_value=p_value,
        model_mse=model_mse,
        baseline_mse=baseline_mse,
        positive_direction_sessions=positive_sessions,
        held_out_session_count=len(folds),
    )
    return DiagnosticResult(
        sample_count=len(samples),
        held_out_sample_count=len(actual),
        held_out_session_count=len(folds),
        held_out_spearman=observed,
        model_mse=model_mse,
        baseline_mse=baseline_mse,
        positive_direction_sessions=positive_sessions,
        permutation_p_value=p_value,
        permutation_null_mean=(sum(null_values) / len(null_values) if null_values else None),
        classification=classification,
    )
