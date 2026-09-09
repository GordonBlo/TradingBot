from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SessionArtifactBinding:
    session_id: str
    raw_event_log: str
    raw_sha256: str
    closure_summary: str
    closure_summary_sha256: str
    features_1s: str
    features_1s_sha256: str


@dataclass(frozen=True)
class SessionEligibility:
    session_id: str
    started_at_utc: str
    ended_at_utc: str | None
    duration_hours: float
    classification: str
    integrity_passed: bool
    reason: str
    artifact_binding: SessionArtifactBinding | None = None


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
    summary_path: Path,
    feature_report: Path,
    workspace: Path,
) -> tuple[bool, str, SessionArtifactBinding | None]:
    if summary.get("collector_left_running") is not False:
        return False, "session is not CLOSED", None
    if summary.get("validation_status") != "PASSED":
        return False, "recorder validation did not pass", None
    counters = summary.get("integrity", {})
    for key in ("sequence_gaps", "invalid_events", "crossed_invalid_book_states"):
        if int(counters.get(key, -1)) != 0:
            return False, f"nonzero recorder integrity counter: {key}", None
    if not feature_report.is_file():
        return False, "feature reconstruction report missing", None

    report = json.loads(feature_report.read_text(encoding="utf-8"))
    deterministic = report.get("deterministic_replay_hash_check", {})
    if deterministic.get("passed") is not True:
        return False, "deterministic feature replay failed", None
    feature_counters = report.get("counters", {})
    for key in (
        "sequence_gap_count",
        "invalid_event_count",
        "crossed_book_state_count",
        "unreconstructed_event_count",
    ):
        if int(feature_counters.get(key, -1)) != 0:
            return False, f"nonzero feature integrity counter: {key}", None

    raw_path = _resolved_path(summary["raw_event_log"], workspace)
    if not raw_path.is_file():
        return False, "raw event log missing", None
    session_id = str(summary["session_id"])
    if (
        summary_path.name != f"{session_id}.summary.json"
        or raw_path.stem != session_id
        or feature_report.parent.name != session_id
    ):
        return False, "session artifact identity mismatch", None
    raw_hash = _sha256(raw_path)
    if report.get("raw_sha256") != raw_hash:
        return False, "raw event checksum mismatch", None
    report_raw_path = _resolved_path(str(report.get("raw_path")), workspace)
    report_output_dir = _resolved_path(str(report.get("output_dir")), workspace)
    if (
        report_raw_path.resolve() != raw_path.resolve()
        or report_output_dir.resolve() != feature_report.parent.resolve()
    ):
        return False, "feature report session identity mismatch", None
    files = report.get("files", {})
    file_hashes = report.get("file_sha256", {})
    for name, value in files.items():
        output_path = _resolved_path(value, workspace)
        if not output_path.is_file() or _sha256(output_path) != file_hashes.get(name):
            return False, "derived feature checksum mismatch", None
    if set(files) != {"event_features", "features_1s", "features_15m"}:
        return False, "deterministic feature hashes incomplete", None
    expected_feature_paths = {
        "event_features": feature_report.parent / "event_features.csv",
        "features_1s": feature_report.parent / "features_1s.csv",
        "features_15m": feature_report.parent / "features_15m.csv",
    }
    if any(
        _resolved_path(files[name], workspace).resolve() != expected.resolve()
        for name, expected in expected_feature_paths.items()
    ):
        return False, "derived feature session identity mismatch", None
    binding = SessionArtifactBinding(
        session_id=str(summary["session_id"]),
        raw_event_log=str(raw_path),
        raw_sha256=raw_hash,
        closure_summary=str(summary_path),
        closure_summary_sha256=_sha256(summary_path),
        features_1s=str(_resolved_path(files["features_1s"], workspace)),
        features_1s_sha256=str(file_hashes["features_1s"]),
    )
    return True, "all recorder/replay/feature integrity checks passed", binding


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

        binding: SessionArtifactBinding | None = None
        if ended is None or summary.get("collector_left_running") is not False:
            classification, passed, reason = "ACTIVE", False, "session is active or incomplete"
        elif started < cutoff < ended:
            classification, passed, reason = "CUTOFF_STRADDLING", False, "session straddles cutoff"
        elif started < cutoff:
            classification, passed, reason = "ENGINEERING_ONLY", False, "session started before cutoff"
        else:
            report_path = feature_root / session_id / "validation_report.json"
            passed, reason, binding = _session_integrity(
                summary, summary_path, report_path, workspace
            )
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
                artifact_binding=binding,
            )
        )

    sessions.sort(key=lambda session: (_parse_utc(session.started_at_utc), session.session_id))

    eligible = [session for session in sessions if session.classification == "ELIGIBLE"]
    hours = sum(session.duration_hours for session in eligible)
    dates = {_parse_utc(session.started_at_utc).date().isoformat() for session in eligible}
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
