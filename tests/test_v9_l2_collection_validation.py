from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.cli.build_v9_l2_features import main as build_features
from src.cli.validate_v9_l2_collection import (
    ACTIVE_MANIFEST,
    ACTIVE_MANIFEST_SHA256,
    FROZEN_FEATURE_COLUMNS,
    CollectionValidationError,
    main as validate_collection_main,
    preflight_collection,
    validate_one_second_provenance,
    validate_session,
)
from src.orderbook.storage import RawDepthEventStore


CUTOFF = datetime(2026, 9, 9, 17, 30, tzinfo=UTC)


def _workspace(tmp_path: Path) -> Path:
    target = tmp_path / ACTIVE_MANIFEST
    target.parent.mkdir(parents=True)
    shutil.copyfile(ACTIVE_MANIFEST, target)
    return tmp_path


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


def _write_valid_session(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    started = datetime(2026, 9, 10, 1, 2, 3, tzinfo=UTC)
    ended = started + timedelta(seconds=10_800)
    session_id = started.strftime("%Y%m%dT%H%M%S%fZ")
    raw = (
        workspace
        / "data/orderbook/v9/BTCUSDC"
        / f"{started:%Y}"
        / f"{started:%m}"
        / f"{started:%d}"
        / f"{session_id}.jsonl"
    )
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

    relative_raw = raw.relative_to(workspace)
    summary = raw.with_suffix(".summary.json")
    summary.write_text(
        json.dumps(
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
                "raw_event_log": str(relative_raw),
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
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(workspace)
    assert (
        build_features(
            [
                "--raw",
                str(relative_raw),
                "--output-root",
                "data/orderbook/v9/features",
                "--max-levels",
                "5000",
            ]
        )
        == 0
    )
    return relative_raw, summary.relative_to(workspace)


def test_active_manifest_hash_and_preflight_cutoff(tmp_path) -> None:
    workspace = _workspace(tmp_path)
    assert hashlib.sha256((workspace / ACTIVE_MANIFEST).read_bytes()).hexdigest() == (
        ACTIVE_MANIFEST_SHA256
    )

    with pytest.raises(CollectionValidationError, match="before cutoff"):
        preflight_collection(workspace=workspace, now=CUTOFF - timedelta(microseconds=1))

    result = preflight_collection(workspace=workspace, now=CUTOFF)
    assert result["manifest_run_id"] == "853870051677af08"
    assert result["data_root"] == "data/orderbook/v9"
    assert result["predictive_outcomes_evaluated"] is False
    assert (workspace / "data/orderbook/v9").is_dir()


def test_valid_session_is_bound_and_collection_only(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    raw, summary = _write_valid_session(workspace, monkeypatch)

    result = validate_session(raw_path=raw, summary_path=summary, workspace=workspace)

    assert result["status"] == "PASS"
    assert result["classification"] == "ELIGIBLE"
    assert result["research_eligible"] is True
    assert result["eligible_hours"] == 3
    assert result["eligible_utc_dates"] == ["2026-09-10"]
    assert result["deterministic_replay_hash_check"]["passed"] is True
    assert result["feature_provenance"]["forward_targets_formed"] == 0
    assert result["predictive_outcomes_evaluated"] is False
    assert result["blind_holdout_accessed"] is False


def test_short_authoritative_session_is_not_eligible(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    raw, summary_path = _write_valid_session(workspace, monkeypatch)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    started = datetime.fromisoformat(summary["started_at_utc"])
    summary["ended_at_utc"] = (started + timedelta(seconds=10_799)).isoformat()
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(CollectionValidationError, match="shorter than"):
        validate_session(raw_path=raw, summary_path=summary_path, workspace=workspace)


def test_future_feature_provenance_is_rejected(tmp_path) -> None:
    path = tmp_path / "features_1s.csv"
    fieldnames = [
        "bucket_open_utc",
        "bucket_close_utc",
        "first_feature_available_at_utc",
        "last_feature_available_at_utc",
        "event_count",
        "mid_price_last",
        *FROZEN_FEATURE_COLUMNS.values(),
    ]
    row = {name: "" for name in fieldnames}
    row.update(
        {
            "bucket_open_utc": "2026-09-10T01:02:03+00:00",
            "bucket_close_utc": "2026-09-10T01:02:04+00:00",
            "first_feature_available_at_utc": "2026-09-10T01:02:03.100000+00:00",
            "last_feature_available_at_utc": "2026-09-10T01:02:05+00:00",
            "event_count": "1",
            "mid_price_last": "100.5",
            "spread_bps_last": "1.0",
        }
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(CollectionValidationError, match="future or out-of-order"):
        validate_one_second_provenance(path)


def test_finalize_never_marks_a_failed_step_research_eligible(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result_path = tmp_path / "validation.json"
    github_output = tmp_path / "github-output.txt"
    result_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "classification": "ELIGIBLE",
                "research_eligible": True,
                "session_id": "SESSION",
                "eligible_hours": 3,
                "eligible_utc_dates": ["2026-09-10"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_SHA", "abc")

    assert (
        validate_collection_main(
            [
                "finalize",
                "--session-number",
                "2",
                "--session-id",
                "SESSION",
                "--result",
                str(result_path),
                "--artifact-name",
                "v9-l2-overnight-123-SESSION",
                "--required-outcome",
                "collect=success",
                "--required-outcome",
                "validate=failure",
                "--github-output",
                str(github_output),
            ]
        )
        == 0
    )
    finalized = json.loads(result_path.read_text(encoding="utf-8"))
    assert finalized["status"] == "FAIL"
    assert finalized["classification"] == "INTEGRITY_FAILED"
    assert finalized["research_eligible"] is False
    assert finalized["eligible_hours"] == 0
    assert "status=FAIL" in github_output.read_text(encoding="utf-8")


def test_overnight_workflow_is_manual_sequential_and_smoke_is_unchanged() -> None:
    smoke = Path(".github/workflows/v9-l2-github-smoke.yml")
    assert hashlib.sha256(smoke.read_bytes()).hexdigest() == (
        "2a11a7234d76FDB37DD6A5C829E882FF1DB4B179213AE9B66886832506554765".lower()
    )
    workflow = Path(".github/workflows/v9-l2-overnight.yml").read_text(encoding="utf-8")
    assert "workflow_dispatch:" in workflow
    assert "schedule:" not in workflow and "push:" not in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "group: v9-l2-overnight-prospective-collection" in workflow
    assert "cancel-in-progress: false" in workflow
    assert workflow.count("timeout-minutes: 240") == 4
    assert "--duration-seconds 10800" in workflow
    assert "--output-root data/orderbook/v9" in workflow
    assert "steps: &session_steps" in workflow
    assert workflow.count("steps: *session_steps") == 3
    assert "needs: session_1" in workflow
    assert "needs: session_2" in workflow
    assert "needs: session_3" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "retention-days: 3" in workflow
    assert "secrets." not in workflow
    assert "--mode EVALUATE" not in workflow
    assert "run_v9_l2_early_information" not in workflow
