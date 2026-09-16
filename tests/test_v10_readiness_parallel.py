"""Non-predictive serial/process equivalence, using real spawned workers."""
from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import timedelta
from decimal import localcontext

import pytest

from src.research import v10_collection_readiness as scanner
from src.research.v10_collection_preregistration import canonical, load_manifest
from tests.test_v10_collection_preregistration import START, SCAN_TIME, edit_json, make_session


def compare(root, manifest):
    serial = scanner.scan_readiness(manifest, data_root=root, now=SCAN_TIME, workers=1)
    for workers in (2, None):
        parallel = scanner.scan_readiness(manifest, data_root=root, now=SCAN_TIME, workers=workers)
        # Whole-result and exact byte equality cover reasons, all artifact/replay/
        # context hashes, hours, IDs, dates, ordering, thresholds and status.
        assert parallel == serial
        assert canonical(parallel) == canonical(serial)
        assert json.dumps(parallel, sort_keys=True, indent=2) == json.dumps(serial, sort_keys=True, indent=2)
    return serial


def test_mixed_classifications_and_canonical_merge(tmp_path):
    manifest = load_manifest()[1]
    sessions = [make_session(tmp_path / "original", manifest, started=START + timedelta(days=i))
                for i in range(7)]
    shutil.copytree(sessions[0].directory, tmp_path / "copy" / sessions[0].session_id)
    conflicting = tmp_path / "conflict" / sessions[1].session_id
    shutil.copytree(sessions[1].directory, conflicting)
    edit_json(conflicting / "closure.summary.json", lambda value: value.update(extra="conflict"))
    make_session(tmp_path, manifest, started=START + timedelta(days=2, hours=1))
    sessions[3].summary.unlink()
    sessions[4].aggtrades_raw.write_text("bad raw\n")
    edit_json(sessions[5].summary, lambda value: value["offline_replay"].update(
        synchronized_timeline_sha256="0" * 64))
    cutoff = scanner.utc(manifest["definition"]["prospective_cutoff_utc"])
    make_session(tmp_path, manifest, started=cutoff)
    make_session(tmp_path, manifest, started=cutoff - timedelta(seconds=1))
    bad = tmp_path / "malformed" / "session.manifest.json"
    bad.parent.mkdir()
    bad.write_text('{"session_id": []}')
    report = compare(tmp_path, manifest)
    assert {s["classification"] for s in report["sessions"]} == {
        "ELIGIBLE", "IDENTICAL_DUPLICATE", "CONFLICTING_DUPLICATE", "OVERLAPPING_SESSION",
        "INTERRUPTED", "INTEGRITY_FAILED", "AT_OR_BEFORE_CUTOFF", "CUTOFF_STRADDLING"}
    assert report["eligible_hours"] == "6"
    assert [s["manifest_path"] for s in report["sessions"]] == sorted(s["manifest_path"] for s in report["sessions"])


@pytest.mark.parametrize("fault", ["future", "sequence_gap", "source", "context", "ci", "raw_decimal"])
def test_failure_classification_matches_with_valid_companion(tmp_path, fault):
    manifest = load_manifest()[1]
    make_session(tmp_path, manifest)
    paths = make_session(tmp_path, manifest, started=START + timedelta(days=1))
    if fault == "source":
        edit_json(paths.manifest, lambda value: value.update(source_commit_sha="0" * 40))
        edit_json(paths.summary, lambda value: value["manifest"].update(sha256=scanner.sha256_file(paths.manifest)))
    elif fault == "sequence_gap":
        edit_json(paths.summary, lambda value: value["live_integrity"]["depth"].update(sequence_gaps=1))
    elif fault == "context":
        edit_json(paths.summary, lambda value: value["offline_replay"].update(causal_trade_count=100))
    elif fault == "ci":
        edit_json(paths.manifest, lambda value: value.update(research_eligibility="CI_ONLY"))
    else:
        rows = [json.loads(line) for line in paths.depth_raw.read_text().splitlines()]
        if fault == "future":
            rows[-1]["payload"]["E"] += 1
            rows[-1]["exchange_event_timestamp_ms"] += 1
        else:
            rows[-1]["payload"]["b"][0][1] = "NaN"
        paths.depth_raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
        edit_json(paths.summary, lambda value: value["raw_artifacts"]["depth"].update(
            sha256=scanner.sha256_file(paths.depth_raw), size_bytes=paths.depth_raw.stat().st_size))
    report = compare(tmp_path, manifest)
    assert report["eligible_sessions"] == 1
    assert report["sessions"][1]["classification"] == "INTEGRITY_FAILED"


def test_parent_decimal_context_preserved_in_spawned_workers(tmp_path):
    manifest = load_manifest()[1]
    for i in range(2):
        make_session(tmp_path, manifest, started=START + timedelta(days=i), duration=10801)
    with localcontext() as context:
        context.prec = 12
        report = compare(tmp_path, manifest)
    assert report["eligible_sessions"] == 2


def test_workers_one_never_creates_pool(tmp_path, monkeypatch):
    manifest = load_manifest()[1]
    for i in range(2):
        make_session(tmp_path, manifest, started=START + timedelta(days=i))
    def forbidden(*args, **kwargs):
        raise AssertionError("serial path created a process pool")
    monkeypatch.setattr(scanner, "ProcessPoolExecutor", forbidden)
    assert scanner.scan_readiness(manifest, data_root=tmp_path, now=SCAN_TIME, workers=1)["eligible_sessions"] == 2


def test_worker_limits(monkeypatch):
    monkeypatch.setattr(scanner.os, "cpu_count", lambda: 16)
    assert scanner.session_worker_count(20) == 4
    assert scanner.session_worker_count(3) == 3
    assert scanner.session_worker_count(20, 6) == 6
    assert scanner.session_worker_count(20, 100) == 16
    assert scanner.session_worker_count(2, 100) == 2
    assert scanner.session_worker_count(0) == 1
    monkeypatch.setattr(scanner.os, "cpu_count", lambda: None)
    assert scanner.session_worker_count(20) == 1


@pytest.mark.parametrize("workers", [0, -1, 1.5, True, "2"])
def test_invalid_worker_count(tmp_path, workers):
    with pytest.raises(ValueError, match="workers must be"):
        scanner.scan_readiness(load_manifest()[1], data_root=tmp_path, workers=workers)


def crash_worker(*args):
    os._exit(7)


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="requires two worker processes")
def test_worker_crash_fails_closed(tmp_path, monkeypatch):
    manifest = load_manifest()[1]
    for i in range(2):
        make_session(tmp_path, manifest, started=START + timedelta(days=i))
    monkeypatch.setattr(scanner, "_scan_session_worker", crash_worker)
    with pytest.raises(ValueError, match="no readiness result published") as exc:
        scanner.scan_readiness(manifest, data_root=tmp_path, now=SCAN_TIME, workers=2)
    assert isinstance(exc.value.__cause__, BrokenProcessPool)


def wait_for_second_worker(path, root, manifest, now):
    # In a spawned child this resolves to the original production worker.
    result = scanner._scan_session_worker(path, root, manifest, now)
    if path.parent.name == (root / "first-id").read_text():
        deadline = time.monotonic() + 30
        while not (root / "release-first").exists():
            if time.monotonic() > deadline:
                raise RuntimeError("second worker never completed")
            time.sleep(0.01)
    return result


@pytest.mark.skipif((os.cpu_count() or 1) < 2, reason="requires two worker processes")
def test_reversed_worker_completion_preserves_exact_json(tmp_path, monkeypatch):
    manifest = load_manifest()[1]
    first = make_session(tmp_path, manifest)
    second = make_session(tmp_path, manifest, started=START + timedelta(days=1))
    (tmp_path / "first-id").write_text(first.session_id)
    serial = scanner.scan_readiness(manifest, data_root=tmp_path, now=SCAN_TIME, workers=1)
    completed = []
    def pool_factory(**kwargs):
        pool = ProcessPoolExecutor(**kwargs)
        submit = pool.submit
        def record_submit(fn, *args, **kw):
            future = submit(fn, *args, **kw)
            def done(future):
                # ProcessPoolExecutor.map submits chunks (default size one).
                for identity, _, _ in future.result():
                    completed.append(identity)
                    if identity == second.session_id:
                        (tmp_path / "release-first").touch()
            future.add_done_callback(done)
            return future
        pool.submit = record_submit
        return pool
    monkeypatch.setattr(scanner, "ProcessPoolExecutor", pool_factory)
    monkeypatch.setattr(scanner, "_scan_session_worker", wait_for_second_worker)
    parallel = scanner.scan_readiness(manifest, data_root=tmp_path, now=SCAN_TIME, workers=2)
    assert completed == [second.session_id, first.session_id]
    assert canonical(parallel) == canonical(serial)


@pytest.mark.parametrize("workers", [1, 2])
def test_final_dataset_binding_still_fails_on_mutation(tmp_path, monkeypatch, workers):
    manifest = load_manifest()[1]
    paths = make_session(tmp_path, manifest)
    make_session(tmp_path, manifest, started=START + timedelta(days=1))
    original = scanner._scan_sessions
    def mutate_after_validation(*args):
        yield from original(*args)
        with paths.depth_raw.open("a") as handle:
            handle.write("\n")
    monkeypatch.setattr(scanner, "_scan_sessions", mutate_after_validation)
    with pytest.raises(ValueError, match="session artifacts changed during dataset readiness"):
        scanner.scan_readiness(manifest, data_root=tmp_path, now=SCAN_TIME, workers=workers)


def test_final_replay_source_verification_is_preserved(tmp_path, monkeypatch):
    manifest = load_manifest()[1]
    for i in range(2):
        make_session(tmp_path, manifest, started=START + timedelta(days=i))
    calls = []
    original = scanner.verify_replay_source
    def verify(value):
        calls.append(value)
        original(value)
        if len(calls) == 2:
            raise ValueError("frozen collector/replay source changed")
    monkeypatch.setattr(scanner, "verify_replay_source", verify)
    with pytest.raises(ValueError, match="source changed"):
        scanner.scan_readiness(manifest, data_root=tmp_path, now=SCAN_TIME, workers=2)
    assert len(calls) == 2
