"""Non-repairing derivatives source continuity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class CoverageClassification(str, Enum):
    COMPLETE = "COMPLETE"
    GAPPED = "GAPPED"
    DUPLICATED = "DUPLICATED"
    PARTIAL = "PARTIAL"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class ContinuityIssue:
    source: str
    issue_type: str
    timestamp: datetime | None
    detail: str


@dataclass(frozen=True, slots=True)
class CoverageResult:
    classification: CoverageClassification
    rows: int
    duplicate_timestamps: int
    duplicate_exact_records: int
    missing_intervals: int
    first_timestamp: datetime | None
    last_timestamp: datetime | None
    issues: tuple[ContinuityIssue, ...]


def analyze_coverage(
    records: tuple,
    *,
    source: str,
    timestamp_field: str,
    expected_step: timedelta | None = None,
    expected_count: int | None = None,
) -> CoverageResult:
    if not records:
        return CoverageResult(
            CoverageClassification.UNAVAILABLE, 0, 0, 0, expected_count or 0, None, None, ()
        )
    timestamps = tuple(getattr(record, timestamp_field) for record in records)
    issues = []
    decreases = sum(later < earlier for earlier, later in zip(timestamps, timestamps[1:]))
    if decreases:
        raise ValueError(f"{source} timestamps are non-monotonic.")
    duplicate_timestamps = len(timestamps) - len(set(timestamps))
    duplicate_records = len(records) - len(set(records))
    if duplicate_timestamps:
        issues.append(
            ContinuityIssue(source, "DUPLICATE_TIMESTAMP", None, str(duplicate_timestamps))
        )
    missing = 0
    if expected_step is not None:
        for earlier, later in zip(timestamps, timestamps[1:]):
            delta = later - earlier
            if delta > expected_step:
                count = int(delta / expected_step) - 1
                missing += count
                issues.append(
                    ContinuityIssue(
                        source,
                        "MISSING_INTERVALS",
                        earlier,
                        f"{count} intervals before {later.isoformat()}",
                    )
                )
            elif delta != expected_step and delta != timedelta(0):
                issues.append(
                    ContinuityIssue(source, "IRREGULAR_INTERVAL", later, str(delta))
                )
    if expected_count is not None and len(set(timestamps)) < expected_count:
        missing = max(missing, expected_count - len(set(timestamps)))
    if duplicate_timestamps:
        classification = CoverageClassification.DUPLICATED
    elif missing:
        classification = CoverageClassification.GAPPED
    elif expected_count is not None and len(records) != expected_count:
        classification = CoverageClassification.PARTIAL
    else:
        classification = CoverageClassification.COMPLETE
    return CoverageResult(
        classification,
        len(records),
        duplicate_timestamps,
        duplicate_records,
        missing,
        timestamps[0],
        timestamps[-1],
        tuple(issues),
    )


def validate_archive_boundary(previous: tuple, current: tuple, *, timestamp_field: str) -> tuple[ContinuityIssue, ...]:
    if not previous or not current:
        return ()
    previous_time = getattr(previous[-1], timestamp_field)
    current_time = getattr(current[0], timestamp_field)
    if current_time < previous_time:
        raise ValueError("Archive boundary timestamps decrease.")
    if current_time == previous_time:
        return (
            ContinuityIssue("archive_boundary", "DUPLICATE_TIMESTAMP", current_time, "Boundary timestamps overlap."),
        )
    return ()

