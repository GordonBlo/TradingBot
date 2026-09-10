from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

from src.cli.build_v9_l2_features import main as build_features
from src.cli.import_v9_github_artifacts import import_artifacts
from src.cli.validate_v9_l2_collection import (
    ACTIVE_MANIFEST,
    CANONICAL_DATA_ROOT,
    EXPECTED_DERIVED_FILES,
    validate_session,
)
from src.orderbook.storage import RawDepthEventStore


Mutation = Callable[[Path, str], None]
BASE_START = datetime(2026, 9, 10, 1, 2, 3, tzinfo=UTC)
REQUIRED_OUTCOMES = {
    "setup_python": "success",
    "dependencies": "success",
    "preflight": "success",
    "collect": "success",
    "identify": "success",
    "features": "success",
    "validate": "success",
}


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _workspace(path: Path) -> Path:
    manifest = path / ACTIVE_MANIFEST
    manifest.parent.mkdir(parents=True)
    shutil.copyfile(ACTIVE_MANIFEST, manifest)
    return path


def _snapshot() -> dict:
    return {
        "lastUpdateId": 100,
        "bids": [["100", "2"], ["99", "4"]],
        "asks": [["101", "3"], ["102", "5"]],
    }


def _diff(at: datetime) -> dict:
    return {
        "e": "depthUpdate",
        "E": int(at.timestamp() * 1000),
        "s": "BTCUSDC",
        "U": 101,
        "u": 101,
        "b": [["100", "3"]],
        "a": [],
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _build_artifact(
    base: Path,
    *,
    started: datetime = BASE_START,
    run_id: str = "123456",
    mutate: Mutation | None = None,
    omit: str | None = None,
    traversal: bool = False,
) -> tuple[Path, str]:
    session_id = started.strftime("%Y%m%dT%H%M%S%fZ")
    source = _workspace(base / f"source-{run_id}-{session_id}")
    data_root = source / CANONICAL_DATA_ROOT
    raw = (
        data_root
        / "BTCUSDC"
        / f"{started:%Y}"
        / f"{started:%m}"
        / f"{started:%d}"
        / f"{session_id}.jsonl"
    )
    ended = started + timedelta(seconds=10_800)
    with RawDepthEventStore(raw, session_id=session_id, fsync_each_record=False) as store:
        store.append(
            record_type="REST_SNAPSHOT",
            received_at=started + timedelta(milliseconds=100),
            payload=_snapshot(),
        )
        store.append(
            record_type="DIFF_DEPTH",
            received_at=started + timedelta(milliseconds=200),
            payload=_diff(started + timedelta(milliseconds=200)),
        )

    relative_raw = raw.relative_to(source).as_posix()
    summary = raw.with_suffix(".summary.json")
    _write_json(
        summary,
        {
            "version": "V9_DEPTH_FOUNDATION_1",
            "session_id": session_id,
            "symbol": "BTCUSDC",
            "source": "BINANCE_PUBLIC_SPOT",
            "rest_snapshot": "PUBLIC_API_V3_DEPTH",
            "websocket_stream": "BTCUSDC_DIFF_DEPTH_100MS",
            "uses_authentication": False,
            "orders_enabled": False,
            "duration_seconds": 10_800,
            "started_at_utc": started.isoformat(),
            "ended_at_utc": ended.isoformat(),
            "raw_event_log": relative_raw,
            "integrity": {
                "snapshots": 1,
                "diff_events": 1,
                "stale_events": 0,
                "sequence_gaps": 0,
                "resync_count": 0,
                "reconnect_count": 0,
                "reconstructed_updates": 1,
                "crossed_invalid_book_states": 0,
                "invalid_events": 0,
                "first_event_timestamp": (
                    started + timedelta(milliseconds=200)
                ).isoformat(),
                "last_event_timestamp": (
                    started + timedelta(milliseconds=200)
                ).isoformat(),
            },
            "validation_status": "PASSED",
            "collector_left_running": False,
        },
    )
    with _working_directory(source):
        assert (
            build_features(
                [
                    "--raw",
                    relative_raw,
                    "--output-root",
                    "data/orderbook/v9/features",
                    "--max-levels",
                    "5000",
                ]
            )
            == 0
        )
    metadata = validate_session(
        raw_path=raw, summary_path=summary, workspace=source
    )
    artifact_name = f"v9-l2-overnight-{run_id}-attempt-1-{session_id}"
    metadata.update(
        {
            "session_number": 1,
            "github_run_id": run_id,
            "github_run_attempt": "1",
            "github_sha": "a" * 40,
            "artifact_name": artifact_name,
            "required_step_outcomes": REQUIRED_OUTCOMES,
        }
    )
    _write_json(data_root / "validation" / f"{session_id}.json", metadata)

    if mutate is not None:
        mutate(data_root, session_id)

    archive_path = base / "downloads" / f"{artifact_name}.zip"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(data_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(data_root).as_posix()
            if relative == omit:
                continue
            archive.write(path, arcname=relative)
        if traversal:
            archive.writestr("../escaped.txt", "must not be extracted")
    return archive_path, session_id


def _expected_session_files(workspace: Path, session_id: str) -> set[str]:
    root = workspace / CANONICAL_DATA_ROOT
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and session_id in path.as_posix()
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_valid_import_is_discovered_by_canonical_readiness(tmp_path) -> None:
    archive, session_id = _build_artifact(tmp_path / "artifact")
    destination = _workspace(tmp_path / "destination")

    result = import_artifacts([archive], workspace=destination)

    assert result.imported_session_ids == (session_id,)
    assert result.skipped_identical_session_ids == ()
    assert result.rejected_artifacts == ()
    assert len(_expected_session_files(destination, session_id)) == 7
    assert not list((destination / CANONICAL_DATA_ROOT).rglob("*.tmp"))


def test_canonical_readiness_discovers_imported_session(tmp_path) -> None:
    archive, _ = _build_artifact(tmp_path / "artifact")
    destination = _workspace(tmp_path / "destination")

    result = import_artifacts([archive], workspace=destination)

    assert result.readiness.discovered_sessions == 1
    assert result.readiness.eligible_closed_sessions == 1
    assert result.readiness.eligible_hours == 3
    assert result.readiness.sessions[0].classification == "ELIGIBLE"


def test_multiple_artifacts_are_imported(tmp_path) -> None:
    first, first_id = _build_artifact(tmp_path / "first", started=BASE_START)
    second, second_id = _build_artifact(
        tmp_path / "second", started=BASE_START + timedelta(hours=4)
    )
    destination = _workspace(tmp_path / "destination")

    result = import_artifacts([first, second], workspace=destination)

    assert result.imported_session_ids == (first_id, second_id)
    assert result.rejected_artifacts == ()
    assert result.readiness.eligible_closed_sessions == 2
    assert result.readiness.eligible_hours == 6


def test_identical_duplicate_is_skipped(tmp_path) -> None:
    archive, session_id = _build_artifact(tmp_path / "artifact")
    destination = _workspace(tmp_path / "destination")

    first = import_artifacts([archive], workspace=destination)
    second = import_artifacts([archive], workspace=destination)

    assert first.imported_session_ids == (session_id,)
    assert second.imported_session_ids == ()
    assert second.skipped_identical_session_ids == (session_id,)
    assert second.rejected_artifacts == ()


def test_conflicting_duplicate_is_rejected_without_overwrite(tmp_path) -> None:
    first, session_id = _build_artifact(tmp_path / "first", run_id="123456")
    conflict, _ = _build_artifact(tmp_path / "conflict", run_id="999999")
    destination = _workspace(tmp_path / "destination")
    imported = import_artifacts([first], workspace=destination)
    validation = (
        destination / CANONICAL_DATA_ROOT / "validation" / f"{session_id}.json"
    )
    original_hash = _sha256(validation)

    result = import_artifacts([conflict], workspace=destination)

    assert imported.imported_session_ids == (session_id,)
    assert result.imported_session_ids == ()
    assert len(result.rejected_artifacts) == 1
    assert "conflicting local duplicate" in result.rejected_artifacts[0].reason
    assert _sha256(validation) == original_hash


def test_malformed_zip_is_rejected(tmp_path) -> None:
    archive = tmp_path / "v9-l2-overnight-malformed.zip"
    archive.write_bytes(b"not a ZIP")
    destination = _workspace(tmp_path / "destination")

    result = import_artifacts([archive], workspace=destination)

    assert len(result.rejected_artifacts) == 1
    assert "malformed or unreadable ZIP" in result.rejected_artifacts[0].reason


def test_path_traversal_is_rejected_before_extraction(tmp_path) -> None:
    artifact_root = tmp_path / "artifact"
    archive, _ = _build_artifact(artifact_root, traversal=True)
    destination = _workspace(tmp_path / "destination")

    result = import_artifacts([archive], workspace=destination)

    assert len(result.rejected_artifacts) == 1
    assert "unsafe ZIP member path" in result.rejected_artifacts[0].reason
    assert not (tmp_path / "escaped.txt").exists()
    assert not (destination / CANONICAL_DATA_ROOT).exists()


def test_wrong_preregistration_id_is_rejected(tmp_path) -> None:
    def wrong_preregistration(root: Path, session_id: str) -> None:
        path = root / "validation" / f"{session_id}.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["preregistration"]["run_id"] = "wrong"
        _write_json(path, metadata)

    archive, _ = _build_artifact(
        tmp_path / "artifact", mutate=wrong_preregistration
    )
    destination = _workspace(tmp_path / "destination")

    result = import_artifacts([archive], workspace=destination)

    assert len(result.rejected_artifacts) == 1
    assert "preregistration identity mismatch" in result.rejected_artifacts[0].reason


def test_pre_cutoff_session_is_rejected(tmp_path) -> None:
    def move_summary_before_cutoff(root: Path, session_id: str) -> None:
        summary = next(root.rglob(f"{session_id}.summary.json"))
        value = json.loads(summary.read_text(encoding="utf-8"))
        started = datetime(2026, 9, 9, 16, tzinfo=UTC)
        value["started_at_utc"] = started.isoformat()
        value["ended_at_utc"] = (started + timedelta(hours=3)).isoformat()
        _write_json(summary, value)

    archive, _ = _build_artifact(tmp_path / "artifact", mutate=move_summary_before_cutoff)
    destination = _workspace(tmp_path / "destination")

    result = import_artifacts([archive], workspace=destination)

    assert len(result.rejected_artifacts) == 1
    assert "started before the frozen V9 V2 cutoff" in result.rejected_artifacts[0].reason


def test_hash_mismatch_is_rejected(tmp_path) -> None:
    def corrupt_hash(root: Path, session_id: str) -> None:
        path = root / "features" / session_id / "validation_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        report["raw_sha256"] = "0" * 64
        _write_json(path, report)

    archive, _ = _build_artifact(tmp_path / "artifact", mutate=corrupt_hash)
    destination = _workspace(tmp_path / "destination")

    result = import_artifacts([archive], workspace=destination)

    assert len(result.rejected_artifacts) == 1
    assert "raw hash mismatch" in result.rejected_artifacts[0].reason


def test_missing_required_artifact_is_rejected(tmp_path) -> None:
    started = BASE_START
    session_id = started.strftime("%Y%m%dT%H%M%S%fZ")
    missing = f"features/{session_id}/{EXPECTED_DERIVED_FILES['features_15m']}"
    archive, _ = _build_artifact(tmp_path / "artifact", omit=missing)
    destination = _workspace(tmp_path / "destination")

    result = import_artifacts([archive], workspace=destination)

    assert len(result.rejected_artifacts) == 1
    assert "artifact file set mismatch" in result.rejected_artifacts[0].reason
    assert missing in result.rejected_artifacts[0].reason
