from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SessionEligibility:
    session_id: str
    started_at_utc: str
    ended_at_utc: str | None
    duration_hours: float
    classification: str
    integrity_passed: bool
    reason: str


@dataclass(frozen=True)
class ReadinessReport:
    cutoff_utc: str
    discovered_sessions: int
    engineering_only_sessions: int
    eligible_closed_sessions: int
    eligible_hours: float
    eligible_utc_dates: int
    straddling_sessions: int
    active_sessions: int
    integrity_failed_sessions: int
    ready: bool
    reasons: tuple[str, ...]
    sessions: tuple[SessionEligibility, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        value["sessions"] = [asdict(session) for session in self.sessions]
        return value


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("session timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_path(value: str | Path, workspace: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def _session_integrity(
    summary: dict[str, Any],
    feature_report: Path,
    workspace: Path,
) -> tuple[bool, str]:
    if summary.get("collector_left_running") is not False:
        return False, "session is not CLOSED"
    if summary.get("validation_status") != "PASSED":
        return False, "recorder validation did not pass"
    counters = summary.get("integrity", {})
    for key in ("sequence_gaps", "invalid_events", "crossed_invalid_book_states"):
        if int(counters.get(key, -1)) != 0:
            return False, f"nonzero recorder integrity counter: {key}"
    if not feature_report.is_file():
        return False, "feature reconstruction report missing"

    report = json.loads(feature_report.read_text(encoding="utf-8"))
    deterministic = report.get("deterministic_replay_hash_check", {})
    if deterministic.get("passed") is not True:
        return False, "deterministic feature replay failed"
    feature_counters = report.get("counters", {})
    for key in (
        "sequence_gap_count",
        "invalid_event_count",
        "crossed_book_state_count",
        "unreconstructed_event_count",
    ):
        if int(feature_counters.get(key, -1)) != 0:
            return False, f"nonzero feature integrity counter: {key}"

    raw_path = _resolved_path(summary["raw_event_log"], workspace)
    if not raw_path.is_file():
        return False, "raw event log missing"
    raw_hash = _sha256(raw_path)
    if report.get("raw_sha256") != raw_hash:
        return False, "raw event checksum mismatch"
    files = report.get("files", {})
    file_hashes = report.get("file_sha256", {})
    for name, value in files.items():
        output_path = _resolved_path(value, workspace)
        if not output_path.is_file() or _sha256(output_path) != file_hashes.get(name):
            return False, "derived feature checksum mismatch"
    if set(files) != {"event_features", "features_1s", "features_15m"}:
        return False, "deterministic feature hashes incomplete"
    return True, "all recorder/replay/feature integrity checks passed"


def scan_session_readiness(
    manifest: dict[str, Any],
    *,
    data_root: Path = Path("data/orderbook/v9"),
    feature_root: Path = Path("data/orderbook/v9/features"),
    workspace: Path = Path("."),
) -> ReadinessReport:
    definition = manifest["definition"]
    cutoff_text = definition["prospective_cutoff_utc"]
    cutoff = _parse_utc(cutoff_text)
    sessions: list[SessionEligibility] = []

    for summary_path in sorted(data_root.rglob("*.summary.json")) if data_root.exists() else []:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        session_id = str(summary["session_id"])
        started = _parse_utc(summary["started_at_utc"])
        ended_text = summary.get("ended_at_utc")
        ended = _parse_utc(ended_text) if ended_text else None
        duration_hours = 0.0 if ended is None else max(0.0, (ended - started).total_seconds() / 3600)

        if ended is None or summary.get("collector_left_running") is not False:
            classification, passed, reason = "ACTIVE", False, "session is active or incomplete"
        elif started < cutoff < ended:
            classification, passed, reason = "CUTOFF_STRADDLING", False, "session straddles cutoff"
        elif started < cutoff:
            classification, passed, reason = "ENGINEERING_ONLY", False, "session started before cutoff"
        else:
            report_path = feature_root / session_id / "validation_report.json"
            passed, reason = _session_integrity(summary, report_path, workspace)
            classification = "ELIGIBLE" if passed else "INTEGRITY_FAILED"
        sessions.append(
            SessionEligibility(
                session_id=session_id,
                started_at_utc=summary["started_at_utc"],
                ended_at_utc=ended_text,
                duration_hours=duration_hours,
                classification=classification,
                integrity_passed=passed,
                reason=reason,
            )
        )

    eligible = [session for session in sessions if session.classification == "ELIGIBLE"]
    hours = sum(session.duration_hours for session in eligible)
    dates = {session.started_at_utc[:10] for session in eligible}
    reasons: list[str] = []
    gate = definition["readiness_gate"]
    if len(eligible) < int(gate["minimum_eligible_closed_sessions"]):
        reasons.append("fewer than 8 eligible closed sessions")
    if hours < float(gate["minimum_total_eligible_hours"]):
        reasons.append("fewer than 20 eligible hours")
    if len(dates) < int(gate["minimum_utc_dates"]):
        reasons.append("fewer than 2 eligible UTC dates")
    integrity_failed = sum(session.classification == "INTEGRITY_FAILED" for session in sessions)
    straddling = sum(session.classification == "CUTOFF_STRADDLING" for session in sessions)
    active = sum(session.classification == "ACTIVE" for session in sessions)
    if integrity_failed:
        reasons.append("one or more prospective sessions failed integrity")
    if straddling:
        reasons.append("one or more sessions straddle the cutoff")

    return ReadinessReport(
        cutoff_utc=cutoff_text,
        discovered_sessions=len(sessions),
        engineering_only_sessions=sum(
            session.classification == "ENGINEERING_ONLY" for session in sessions
        ),
        eligible_closed_sessions=len(eligible),
        eligible_hours=hours,
        eligible_utc_dates=len(dates),
        straddling_sessions=straddling,
        active_sessions=active,
        integrity_failed_sessions=integrity_failed,
        ready=not reasons,
        reasons=tuple(reasons),
        sessions=tuple(sessions),
    )
