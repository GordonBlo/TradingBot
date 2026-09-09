"""Frozen V2, leakage-safe statistical engine for the prospective V9 L2 diagnostic."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from src.diagnostics.v8_derivatives_discovery import (
    PreparedFold,
    fit_prepared_fold,
    mean_squared_error,
    prepare_fold,
    spearman,
)
from src.research.v9_l2_early_information_preregistration_v2 import (
    CIRCULAR_SHIFT_EXCLUSION_SAMPLES,
    FROZEN_FEATURES,
    MINIMUM_ELIGIBLE_CLOSED_SESSIONS,
    MINIMUM_ELIGIBLE_UTC_DATES,
    MINIMUM_TOTAL_ELIGIBLE_HOURS,
    PERMUTATION_COUNT,
    PERMUTATION_SEED,
    PRIMARY_HORIZON_SECONDS,
    RIDGE_ALPHA,
    SECONDARY_HORIZONS_SECONDS,
    SIGNIFICANCE_ALPHA,
)
from src.research.v9_l2_readiness import (
    ReadinessReport,
    SessionArtifactBinding,
    SessionEligibility,
)


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


class DiagnosticIntegrityError(RuntimeError):
    """An explicit V9 V2 protocol or immutable-input integrity failure."""


@dataclass(frozen=True, slots=True)
class DiagnosticSample:
    session_id: str
    timestamp: datetime
    feature_source_available_at: datetime
    features: tuple[float | None, ...]
    targets: tuple[float | None, ...]
    target_timestamps: tuple[datetime | None, ...]
    target_source_available_at: tuple[datetime | None, ...]
    artifact_binding: SessionArtifactBinding

    def target(self, horizon_seconds: int) -> float | None:
        return self.targets[TARGET_HORIZONS.index(horizon_seconds)]

    def target_timestamp(self, horizon_seconds: int) -> datetime | None:
        return self.target_timestamps[TARGET_HORIZONS.index(horizon_seconds)]

    def target_source_timestamp(self, horizon_seconds: int) -> datetime | None:
        return self.target_source_available_at[TARGET_HORIZONS.index(horizon_seconds)]


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
    held_out_spearman: float
    model_mse: float
    baseline_mse: float
    positive_direction_sessions: int
    permutation_p_value: float
    permutation_null_mean: float
    classification: str


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("V9 diagnostic timestamps must be timezone-aware")
    return parsed.astimezone(UTC)


def _aware_utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise DiagnosticIntegrityError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("V9 diagnostic values must be finite")
    return parsed


def _source_available_at(row: dict[str, Any], timestamp: datetime, *, label: str) -> datetime:
    value = row.get("last_feature_available_at_utc")
    if value in (None, ""):
        raise DiagnosticIntegrityError(f"{label} source-availability provenance is missing")
    available = _utc(str(value))
    if available > timestamp:
        raise DiagnosticIntegrityError(f"future {label} source timestamp detected")
    return available


def _session_metadata(
    session: SessionEligibility,
) -> tuple[datetime, datetime, SessionArtifactBinding]:
    if session.classification != "ELIGIBLE" or session.integrity_passed is not True:
        raise DiagnosticIntegrityError("diagnostic session is not readiness-eligible")
    if session.ended_at_utc is None:
        raise DiagnosticIntegrityError("diagnostic session is not CLOSED")
    if session.artifact_binding is None:
        raise DiagnosticIntegrityError("diagnostic session lacks immutable artifact binding")
    if session.artifact_binding.session_id != session.session_id:
        raise DiagnosticIntegrityError("diagnostic session artifact identity mismatch")
    started = _utc(session.started_at_utc)
    ended = _utc(session.ended_at_utc)
    if ended <= started:
        raise DiagnosticIntegrityError("diagnostic session has invalid authoritative boundaries")
    return started, ended, session.artifact_binding


def samples_from_one_second_rows(
    session: SessionEligibility,
    rows: Iterable[dict[str, Any]],
) -> tuple[DiagnosticSample, ...]:
    """Build exact-time forward targets inside one authoritative CLOSED session."""

    session_start, session_end, binding = _session_metadata(session)
    by_timestamp: dict[datetime, dict[str, Any]] = {}
    for row in rows:
        bucket_open = _utc(str(row["bucket_open_utc"]))
        bucket_close = _utc(str(row["bucket_close_utc"]))
        if bucket_close - bucket_open != timedelta(seconds=1):
            raise ValueError("diagnostic input is not an exact one-second bucket")
        if bucket_close in by_timestamp:
            raise ValueError("duplicate one-second diagnostic bucket")
        by_timestamp[bucket_close] = row

    samples: list[DiagnosticSample] = []
    primary_position = TARGET_HORIZONS.index(PRIMARY_HORIZON_SECONDS)
    for timestamp in sorted(by_timestamp):
        if timestamp < session_start or timestamp > session_end:
            continue
        row = by_timestamp[timestamp]
        current_mid = _number(row.get("mid_price_last"))
        if current_mid is None or current_mid <= 0:
            continue
        feature_values = tuple(_number(row.get(column)) for column in _INPUT_COLUMNS)
        current_available = _source_available_at(row, timestamp, label="feature/target-start")
        targets: list[float | None] = []
        target_timestamps: list[datetime | None] = []
        target_source_timestamps: list[datetime | None] = []
        for horizon in TARGET_HORIZONS:
            future_timestamp = timestamp + timedelta(seconds=horizon)
            future = (
                by_timestamp.get(future_timestamp)
                if future_timestamp <= session_end
                else None
            )
            future_mid = _number(future.get("mid_price_last")) if future else None
            if future_mid is None or future_mid <= 0:
                targets.append(None)
                target_timestamps.append(None)
                target_source_timestamps.append(None)
            else:
                future_available = _source_available_at(
                    future, future_timestamp, label=f"target-{horizon}s"
                )
                targets.append(math.log(future_mid / current_mid))
                target_timestamps.append(future_timestamp)
                target_source_timestamps.append(future_available)
        if targets[primary_position] is None:
            continue
        samples.append(
            DiagnosticSample(
                session_id=session.session_id,
                timestamp=timestamp,
                feature_source_available_at=current_available,
                features=feature_values,
                targets=tuple(targets),
                target_timestamps=tuple(target_timestamps),
                target_source_available_at=tuple(target_source_timestamps),
                artifact_binding=binding,
            )
        )
    return tuple(samples)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_path(value: str, workspace: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def _verify_bound_file(path_value: str, expected_hash: str, workspace: Path, label: str) -> Path:
    path = _resolved_path(path_value, workspace)
    if not path.is_file() or _sha256(path) != expected_hash:
        raise DiagnosticIntegrityError(f"{label} artifact hash mismatch")
    return path


def load_bound_session_samples(
    session: SessionEligibility,
    *,
    workspace: Path = Path("."),
) -> tuple[DiagnosticSample, ...]:
    """Reverify immutable readiness artifacts and load the bound one-second feature file."""

    started, ended, binding = _session_metadata(session)
    raw_path = _verify_bound_file(
        binding.raw_event_log, binding.raw_sha256, workspace, "raw L2 session"
    )
    summary_path = _verify_bound_file(
        binding.closure_summary,
        binding.closure_summary_sha256,
        workspace,
        "closure summary",
    )
    features_path = _verify_bound_file(
        binding.features_1s, binding.features_1s_sha256, workspace, "features_1s"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        str(summary.get("session_id")) != session.session_id
        or summary.get("collector_left_running") is not False
        or summary.get("validation_status") != "PASSED"
        or _utc(str(summary.get("started_at_utc"))) != started
        or _utc(str(summary.get("ended_at_utc"))) != ended
        or _resolved_path(str(summary.get("raw_event_log")), workspace).resolve()
        != raw_path.resolve()
    ):
        raise DiagnosticIntegrityError("closure summary session identity mismatch")
    with features_path.open("r", encoding="utf-8", newline="") as stream:
        return samples_from_one_second_rows(session, csv.DictReader(stream))


def load_readiness_bound_samples(
    readiness: ReadinessReport,
    *,
    workspace: Path = Path("."),
) -> tuple[DiagnosticSample, ...]:
    if not readiness.ready:
        raise DiagnosticIntegrityError("V9 predictive diagnostic refused: readiness gate has not passed")
    eligible = sorted(
        (session for session in readiness.sessions if session.classification == "ELIGIBLE"),
        key=lambda session: (_utc(session.started_at_utc), session.session_id),
    )
    return tuple(
        sample
        for session in eligible
        for sample in load_bound_session_samples(session, workspace=workspace)
    )


def _canonical_samples(
    samples: Sequence[DiagnosticSample], readiness: ReadinessReport
) -> tuple[DiagnosticSample, ...]:
    eligible = {
        session.session_id: session
        for session in readiness.sessions
        if session.classification == "ELIGIBLE"
    }
    session_order = {
        session.session_id: position
        for position, session in enumerate(
            sorted(eligible.values(), key=lambda item: (_utc(item.started_at_utc), item.session_id))
        )
    }
    ordered = tuple(sorted(samples, key=lambda item: (session_order[item.session_id], item.timestamp)))
    seen: set[tuple[str, datetime]] = set()
    for sample in ordered:
        identity = (sample.session_id, sample.timestamp)
        if identity in seen:
            raise DiagnosticIntegrityError("duplicate canonical diagnostic sample")
        seen.add(identity)
    return ordered


def _validate_readiness_and_samples(
    samples: Sequence[DiagnosticSample], readiness: ReadinessReport
) -> tuple[DiagnosticSample, ...]:
    if not readiness.ready:
        raise DiagnosticIntegrityError("V9 predictive diagnostic refused: readiness gate has not passed")
    eligible = [
        session for session in readiness.sessions if session.classification == "ELIGIBLE"
    ]
    if (
        readiness.eligible_closed_sessions != len(eligible)
        or len(eligible) < MINIMUM_ELIGIBLE_CLOSED_SESSIONS
        or readiness.eligible_hours < MINIMUM_TOTAL_ELIGIBLE_HOURS
        or readiness.eligible_utc_dates < MINIMUM_ELIGIBLE_UTC_DATES
        or readiness.integrity_failed_sessions != 0
        or readiness.straddling_sessions != 0
    ):
        raise DiagnosticIntegrityError("V9 readiness report does not satisfy frozen V2 gates")
    expected = {session.session_id: session for session in eligible}
    if len(expected) != len(eligible):
        raise DiagnosticIntegrityError("duplicate readiness-eligible session identity")
    if {sample.session_id for sample in samples} != set(expected):
        raise DiagnosticIntegrityError(
            "diagnostic samples do not exactly match readiness-eligible sessions"
        )
    ordered = _canonical_samples(samples, readiness)
    for sample in ordered:
        session = expected[sample.session_id]
        started, ended, binding = _session_metadata(session)
        timestamp = _aware_utc(sample.timestamp, label="sample timestamp")
        feature_source = _aware_utc(
            sample.feature_source_available_at, label="feature source timestamp"
        )
        target_timestamp = sample.target_timestamp(PRIMARY_HORIZON_SECONDS)
        target_source = sample.target_source_timestamp(PRIMARY_HORIZON_SECONDS)
        if target_timestamp is None or target_source is None:
            raise DiagnosticIntegrityError("primary target timestamp provenance is missing")
        target_timestamp = _aware_utc(target_timestamp, label="primary target timestamp")
        target_source = _aware_utc(target_source, label="primary target source timestamp")
        if (
            sample.artifact_binding != binding
            or timestamp < started
            or timestamp > ended
            or target_timestamp != timestamp + timedelta(seconds=PRIMARY_HORIZON_SECONDS)
            or target_timestamp > ended
            or feature_source > timestamp
            or target_source > target_timestamp
        ):
            raise DiagnosticIntegrityError(
                "diagnostic sample boundary, provenance, or artifact binding mismatch"
            )
    return ordered


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
        timestamp = _aware_utc(sample.timestamp, label="sample timestamp")
        current = session_ranges.get(sample.session_id)
        session_ranges[sample.session_id] = (
            min(current[0], timestamp) if current else timestamp,
            max(current[1], timestamp) if current else timestamp,
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
        for column, feature_name in enumerate(FROZEN_FEATURES):
            if all(features[index][column] is None for index in training):
                raise DiagnosticIntegrityError(
                    f"training fold feature is entirely missing: {feature_name}"
                )
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


def valid_circular_offsets(sample_count: int) -> tuple[int, ...]:
    offsets = tuple(
        offset
        for offset in range(1, sample_count)
        if min(offset, sample_count - offset) > CIRCULAR_SHIFT_EXCLUSION_SAMPLES
    )
    if not offsets:
        raise DiagnosticIntegrityError(
            "session has too few primary samples for a valid nontrivial circular shift"
        )
    return offsets


def session_circular_shift_permutations(
    samples: Sequence[DiagnosticSample],
    *,
    seed: int,
    count: int,
) -> Iterator[tuple[int, ...]]:
    if count < 1:
        raise ValueError("Permutation count must be positive")
    grouped: dict[str, list[int]] = {}
    for index, sample in enumerate(samples):
        grouped.setdefault(sample.session_id, []).append(index)
    ordered_groups: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for session_id, indices in grouped.items():
        canonical = tuple(sorted(indices, key=lambda index: samples[index].timestamp))
        timestamps = [samples[index].timestamp for index in canonical]
        if len(set(timestamps)) != len(timestamps):
            raise DiagnosticIntegrityError("duplicate sample timestamp inside session")
        ordered_groups.append((session_id, canonical, valid_circular_offsets(len(canonical))))
    ordered_groups.sort(key=lambda item: (samples[item[1][0]].timestamp, item[0]))

    randomizer = random.Random(seed)
    for _ in range(count):
        source_indices = list(range(len(samples)))
        for _, group, valid_offsets in ordered_groups:
            offset = randomizer.choice(valid_offsets)
            for position, destination in enumerate(group):
                source_indices[destination] = group[(position + offset) % len(group)]
        yield tuple(source_indices)


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
    ordered_samples = _validate_readiness_and_samples(samples, readiness)
    folds = build_chronological_folds(ordered_samples)
    targets = [float(sample.target(PRIMARY_HORIZON_SECONDS)) for sample in ordered_samples]
    actual, predictions, baseline, positive_sessions = _evaluate_targets(folds, targets)
    observed = spearman(actual, predictions)
    if observed is None:
        raise DiagnosticIntegrityError("observed aggregate Spearman statistic is undefined")

    null_values: list[float] = []
    for permutation in session_circular_shift_permutations(
        ordered_samples,
        seed=PERMUTATION_SEED,
        count=PERMUTATION_COUNT,
    ):
        permuted_targets = [targets[source] for source in permutation]
        null_actual, null_predictions, _, _ = _evaluate_targets(folds, permuted_targets)
        null_correlation = spearman(null_actual, null_predictions)
        if null_correlation is None:
            raise DiagnosticIntegrityError("required permutation statistic is undefined")
        null_values.append(null_correlation)
        if len(null_values) > PERMUTATION_COUNT:
            raise DiagnosticIntegrityError("too many permutation statistics generated")
    if len(null_values) != PERMUTATION_COUNT:
        raise DiagnosticIntegrityError(
            f"expected exactly {PERMUTATION_COUNT} valid permutation statistics"
        )
    exceedances = sum(value >= observed for value in null_values)
    p_value = (1 + exceedances) / (1 + PERMUTATION_COUNT)
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
        sample_count=len(ordered_samples),
        held_out_sample_count=len(actual),
        held_out_session_count=len(folds),
        held_out_spearman=observed,
        model_mse=model_mse,
        baseline_mse=baseline_mse,
        positive_direction_sessions=positive_sessions,
        permutation_p_value=p_value,
        permutation_null_mean=sum(null_values) / len(null_values),
        classification=classification,
    )
