from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.research.v9_l2_early_information_preregistration import build_manifest
from src.research.v9_l2_readiness import scan_session_readiness


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_session(
    root: Path,
    feature_root: Path,
    *,
    session_id: str,
    start: datetime,
    hours: float,
    valid: bool = True,
    active: bool = False,
) -> None:
    raw = root / f"{session_id}.jsonl"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text('{"type":"snapshot"}\n', encoding="utf-8")
    summary = {
        "session_id": session_id,
        "started_at_utc": start.isoformat(),
        "ended_at_utc": None if active else (start + timedelta(hours=hours)).isoformat(),
        "collector_left_running": active,
        "validation_status": "PASSED",
        "raw_event_log": str(raw.relative_to(root.parent)),
        "integrity": {
            "sequence_gaps": 0 if valid else 1,
            "invalid_events": 0,
            "crossed_invalid_book_states": 0,
        },
    }
    (root / f"{session_id}.summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    output_dir = feature_root / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    hashes = {}
    for name in ("event_features", "features_1s", "features_15m"):
        output = output_dir / f"{name}.csv"
        output.write_text(f"synthetic-{session_id}-{name}\n", encoding="utf-8")
        files[name] = str(output.relative_to(root.parent))
        hashes[name] = _sha(output)
    report = {
        "raw_sha256": _sha(raw),
        "deterministic_replay_hash_check": {"passed": True},
        "counters": {
            "sequence_gap_count": 0,
            "invalid_event_count": 0,
            "crossed_book_state_count": 0,
            "unreconstructed_event_count": 0,
        },
        "files": files,
        "file_sha256": hashes,
    }
    (output_dir / "validation_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )


def test_readiness_requires_eight_closed_sessions_twenty_hours_and_two_dates(tmp_path) -> None:
    cutoff_created = datetime(2026, 1, 1, 0, 1, tzinfo=UTC)
    manifest = build_manifest(cutoff_created)
    data_root = tmp_path / "data"
    feature_root = tmp_path / "features"
    starts = [
        datetime(2026, 1, day, hour, tzinfo=UTC)
        for day in (1, 2)
        for hour in (1, 4, 7, 10)
    ]
    for index, start in enumerate(starts):
        _write_session(
            data_root,
            feature_root,
            session_id=f"S{index:02d}",
            start=start,
            hours=2.5,
        )
    report = scan_session_readiness(
        manifest, data_root=data_root, feature_root=feature_root, workspace=tmp_path
    )
    assert report.ready
    assert report.eligible_closed_sessions == 8
    assert report.eligible_hours == 20
    assert report.eligible_utc_dates == 2


def test_pre_cutoff_is_engineering_and_straddling_or_failed_sessions_block(tmp_path) -> None:
    manifest = build_manifest(datetime(2026, 1, 1, 0, 1, tzinfo=UTC))
    data_root = tmp_path / "data"
    feature_root = tmp_path / "features"
    _write_session(
        data_root,
        feature_root,
        session_id="ENGINEERING",
        start=datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        hours=0.1,
    )
    _write_session(
        data_root,
        feature_root,
        session_id="STRADDLE",
        start=datetime(2026, 1, 1, 0, 10, tzinfo=UTC),
        hours=0.25,
    )
    _write_session(
        data_root,
        feature_root,
        session_id="FAILED",
        start=datetime(2026, 1, 1, 1, 0, tzinfo=UTC),
        hours=3,
        valid=False,
    )
    report = scan_session_readiness(
        manifest, data_root=data_root, feature_root=feature_root, workspace=tmp_path
    )
    classifications = {session.session_id: session.classification for session in report.sessions}
    assert classifications == {
        "ENGINEERING": "ENGINEERING_ONLY",
        "FAILED": "INTEGRITY_FAILED",
        "STRADDLE": "CUTOFF_STRADDLING",
    }
    assert not report.ready
    assert report.straddling_sessions == 1
    assert report.integrity_failed_sessions == 1
