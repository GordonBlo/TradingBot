from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime, timedelta

import pytest

from src.cli.run_v10_collection import build_parser, main
from src.microstructure.v10 import (
    AggTradeSynchronizer, ProspectiveSession, V10DepthSynchronizer,
    iter_causal_trade_contexts, session_paths, sha256_file,
)
from src.research.v10_collection_preregistration import (
    WORKSPACE, build_manifest, load_manifest, prospective_cutoff,
    source_hashes, verify_manifest,
)
from src.research.v10_collection_readiness import scan_readiness


CREATED = datetime(2026, 9, 12, 10, 7, tzinfo=UTC)
START = datetime(2026, 9, 13, tzinfo=UTC)
SCAN_TIME = START + timedelta(days=10)


@pytest.fixture(scope="module")
def protocol():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=WORKSPACE,
                            check=True, capture_output=True, text=True).stdout.strip()
    return build_manifest(CREATED, commit, source_hashes(commit))


def edit_json(path, edit):
    value = json.loads(path.read_text(encoding="utf-8"))
    edit(value)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def make_session(root, protocol, *, started=START, duration=10800, duplicate=False):
    paths = session_paths(root, started_at=started)
    with ProspectiveSession(
        paths, started_at=started, duration_seconds=duration,
        source_commit_sha=protocol["definition"]["creation_source_commit_sha"],
        snapshot_limit=5000, max_levels_per_side=5000, fsync_each_record=False,
    ) as session:
        depth = V10DepthSynchronizer(session.depth_store)
        trades = AggTradeSynchronizer(session.aggtrade_store)
        depth.install_snapshot({"lastUpdateId": 100, "bids": [["100", "2"]],
                                "asks": [["101", "2"]]}, received_at=started)
        for index, received in enumerate((started, started + timedelta(seconds=duration))):
            ms = int(received.timestamp()) * 1000
            event = depth.record_diff({"e": "depthUpdate", "E": ms, "s": "BTCUSDC",
                                       "U": 101 + index, "u": 101 + index,
                                       "b": [["100", str(index + 3)]], "a": []}, received_at=received)
            depth.apply_recorded(event)
            trade = {"e": "aggTrade", "s": "BTCUSDC", "E": ms, "T": ms,
                     "a": index + 1, "f": index + 1, "l": index + 1,
                     "p": "100.5", "q": "0.1", "m": True, "M": True}
            trades.record(trade, received_at=received)
            if duplicate:
                trades.record(trade, received_at=received)
        session.finalize(depth_counters=depth.counters, aggtrade_counters=trades.counters,
                         ended_at=started + timedelta(seconds=duration))
    return paths


def scan(root, protocol):
    return scan_readiness(protocol, data_root=root, now=SCAN_TIME)


@pytest.mark.parametrize("created,expected", [
    (CREATED, datetime(2026, 9, 12, 10, 15, tzinfo=UTC)),
    (datetime(2026, 9, 12, 10, 15, tzinfo=UTC), datetime(2026, 9, 12, 10, 30, tzinfo=UTC)),
    (datetime(2026, 9, 12, 23, 59, 59, tzinfo=UTC), datetime(2026, 9, 13, tzinfo=UTC)),
])
def test_cutoff_is_first_strictly_later_full_quarter_hour(created, expected):
    assert prospective_cutoff(created) == expected


def test_naive_creation_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        prospective_cutoff(datetime(2026, 9, 12))


def test_preregistration_tamper_detection(protocol):
    changed = json.loads(json.dumps(protocol))
    changed["definition"]["readiness"]["minimum_total_eligible_hours"] = 1
    with pytest.raises(ValueError, match="mismatch"):
        verify_manifest(changed)


@pytest.mark.parametrize("offset,expected", [
    (-timedelta(hours=4), "AT_OR_BEFORE_CUTOFF"),
    (-timedelta(seconds=1), "CUTOFF_STRADDLING"),
    (timedelta(0), "AT_OR_BEFORE_CUTOFF"),
    (timedelta(microseconds=1), "ELIGIBLE"),
])
def test_strict_cutoff_and_straddling(tmp_path, protocol, offset, expected):
    make_session(tmp_path, protocol, started=prospective_cutoff(CREATED) + offset)
    report = scan(tmp_path, protocol)
    assert report["sessions"][0]["classification"] == expected


def test_interrupted_session_never_counts(tmp_path, protocol):
    paths = make_session(tmp_path, protocol)
    paths.summary.unlink()
    report = scan(tmp_path, protocol)
    assert report["sessions"][0]["classification"] == "INTERRUPTED"
    assert report["eligible_sessions"] == 0


@pytest.mark.parametrize("source,key", [
    ("depth", "sequence_gaps"), ("depth", "invalid_events"),
    ("depth", "crossed_invalid_book_states"), ("aggtrades", "id_regressions"),
    ("aggtrades", "invalid_events"), ("aggtrades", "missing_aggregate_trade_ids"),
    ("depth", "reconnect_count"), ("aggtrades", "reconnect_count"),
])
def test_forged_pass_with_bad_live_integrity_is_rejected(tmp_path, protocol, source, key):
    paths = make_session(tmp_path, protocol)
    edit_json(paths.summary, lambda value: value["live_integrity"][source].update({key: 1}))
    report = scan(tmp_path, protocol)
    assert report["eligible_sessions"] == 0
    assert report["sessions"][0]["classification"] == "INTEGRITY_FAILED"


@pytest.mark.parametrize("artifact", ["depth_raw", "aggtrades_raw", "manifest"])
def test_hash_mismatch_rejected(tmp_path, protocol, artifact):
    paths = make_session(tmp_path, protocol)
    path = getattr(paths, artifact)
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    report = scan(tmp_path, protocol)
    assert report["eligible_sessions"] == 0
    assert "hash mismatch" in report["sessions"][0]["reason"]


def test_forged_replay_hash_rejected(tmp_path, protocol):
    paths = make_session(tmp_path, protocol)
    edit_json(paths.summary, lambda value: value["offline_replay"].update(
        synchronized_timeline_sha256="0" * 64))
    report = scan(tmp_path, protocol)
    assert report["eligible_sessions"] == 0
    assert "replay" in report["sessions"][0]["reason"]


def test_duplicate_trade_and_causal_tie_ordering(tmp_path, protocol):
    paths = make_session(tmp_path, protocol, duplicate=True)
    contexts = list(iter_causal_trade_contexts(paths.depth_raw, paths.aggtrades_raw,
                                             session_id=paths.session_id))
    assert len(contexts) == 2
    assert [item.depth.last_update_id for item in contexts] == [101, 102]
    assert contexts[0].depth.bids[0][1] == 3
    assert contexts[1].depth.bids[0][1] == 4
    assert all(item.depth.available_at <= item.observed_trade.received_at for item in contexts)
    first = scan(tmp_path, protocol)
    second = scan(tmp_path, protocol)
    assert first == second
    assert first["eligible_sessions"] == 1


def test_identical_session_copy_counts_once(tmp_path, protocol):
    paths = make_session(tmp_path / "original", protocol)
    shutil.copytree(paths.directory, tmp_path / "copy" / paths.session_id)
    report = scan(tmp_path, protocol)
    assert report["eligible_sessions"] == 1
    assert report["eligible_hours"] == "3"
    assert sum(item["classification"] == "IDENTICAL_DUPLICATE" for item in report["sessions"]) == 1


def test_conflicting_duplicate_invalidates_both_copies(tmp_path, protocol):
    paths = make_session(tmp_path / "original", protocol)
    copied = tmp_path / "copy" / paths.session_id
    shutil.copytree(paths.directory, copied)
    edit_json(copied / "closure.summary.json", lambda value: value.update(extra="conflict"))
    report = scan(tmp_path, protocol)
    assert report["eligible_sessions"] == 0
    assert {item["classification"] for item in report["sessions"]} == {"CONFLICTING_DUPLICATE"}


def test_readiness_requires_eight_sessions_24_hours_three_start_dates(tmp_path, protocol):
    for index in range(8):
        started = START + timedelta(days=index // 3, hours=(index % 3) * 4)
        make_session(tmp_path, protocol, started=started)
        report = scan(tmp_path, protocol)
        assert (report["status"] == "READY") is (index == 7)
    assert report["eligible_sessions"] == 8
    assert report["eligible_hours"] == "24"
    assert len(report["eligible_utc_dates"]) == 3
    assert report["independent_confirmatory_strategy_evidence"] is False


def test_independent_readiness_hours_and_date_gates(tmp_path, protocol):
    for index in range(8):
        make_session(tmp_path, protocol, started=START + timedelta(hours=index * 2), duration=3600)
    report = scan(tmp_path, protocol)
    assert report["checks"] == {"eligible_sessions": True, "eligible_hours": False, "utc_dates": False}


def test_overlapping_sessions_never_double_count(tmp_path, protocol):
    make_session(tmp_path, protocol)
    make_session(tmp_path, protocol, started=START + timedelta(hours=1))
    report = scan(tmp_path, protocol)
    assert report["eligible_sessions"] == 0
    assert {item["classification"] for item in report["sessions"]} == {"OVERLAPPING_SESSION"}


def test_raw_receipt_outside_closure_rejected_even_with_resealed_hash(tmp_path, protocol):
    paths = make_session(tmp_path, protocol)
    lines = [json.loads(line) for line in paths.aggtrades_raw.read_text().splitlines()]
    lines[-1]["received_at_utc"] = (START + timedelta(days=1)).isoformat()
    paths.aggtrades_raw.write_text("".join(json.dumps(line) + "\n" for line in lines))
    edit_json(paths.summary, lambda value: value["raw_artifacts"]["aggtrades"].update(
        sha256=sha256_file(paths.aggtrades_raw), size_bytes=paths.aggtrades_raw.stat().st_size))
    report = scan(tmp_path, protocol)
    assert "outside closed session" in report["sessions"][0]["reason"]


def test_source_commit_required(tmp_path, protocol):
    paths = make_session(tmp_path, protocol)
    edit_json(paths.manifest, lambda value: value.update(source_commit_sha=None))
    edit_json(paths.summary, lambda value: value["manifest"].update(sha256=sha256_file(paths.manifest)))
    assert "source commit" in scan(tmp_path, protocol)["sessions"][0]["reason"]


def test_no_predictive_cli_path():
    parser = build_parser()
    assert parser.parse_args(["--mode", "PLAN"]).mode == "PLAN"
    assert parser.parse_args(["--mode", "READINESS"]).mode == "READINESS"
    with pytest.raises(SystemExit) as raised:
        parser.parse_args(["--mode", "EVALUATE"])
    assert raised.value.code == 2


def test_plan_is_read_only_and_does_not_scan(monkeypatch, protocol, tmp_path, capsys):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(protocol))
    before = path.read_bytes()
    monkeypatch.setattr("src.cli.run_v10_collection.load_manifest", lambda: (path, protocol))
    def forbidden(*args, **kwargs):
        raise AssertionError("PLAN must not scan any market data")
    monkeypatch.setattr("src.cli.run_v10_collection.scan_readiness", forbidden)
    assert main(["--mode", "PLAN"]) == 0
    assert path.read_bytes() == before
    assert "DATA_ACQUISITION_ONLY" in capsys.readouterr().out


def test_empty_root_does_not_create_data(tmp_path, protocol):
    root = tmp_path / "absent"
    report = scan(root, protocol)
    assert report["status"] == "NOT_READY"
    assert report["eligible_sessions"] == 0
    assert not root.exists()


def test_loader_never_creates_or_rewrites_manifest(tmp_path, protocol):
    with pytest.raises(ValueError, match="exactly one"):
        load_manifest(tmp_path)
    path = tmp_path / protocol["preregistration_id"] / "manifest.json"
    path.parent.mkdir()
    path.write_text(json.dumps(protocol))
    before = path.read_bytes()
    assert load_manifest(tmp_path)[1] == protocol
    assert path.read_bytes() == before


@pytest.mark.parametrize("payload_change", [{"q": "0"}, {"m": "true"}, {"a": 0}])
def test_invalid_raw_trade_rejected_despite_updated_file_hash(tmp_path, protocol, payload_change):
    paths = make_session(tmp_path, protocol)
    lines = [json.loads(line) for line in paths.aggtrades_raw.read_text().splitlines()]
    lines[-1]["payload"].update(payload_change)
    paths.aggtrades_raw.write_text("".join(json.dumps(line) + "\n" for line in lines))
    edit_json(paths.summary, lambda value: value["raw_artifacts"]["aggtrades"].update(
        sha256=sha256_file(paths.aggtrades_raw), size_bytes=paths.aggtrades_raw.stat().st_size))
    report = scan(tmp_path, protocol)
    assert report["eligible_sessions"] == 0
    assert report["sessions"][0]["classification"] == "INTEGRITY_FAILED"


def test_malformed_session_does_not_hide_good_sessions(tmp_path, protocol):
    paths = make_session(tmp_path, protocol)
    bad = tmp_path / "bad" / "session.manifest.json"
    bad.parent.mkdir()
    bad.write_text('{"session_id": []}')
    report = scan(tmp_path, protocol)
    assert report["eligible_sessions"] == 1
    assert any(item["classification"] == "INTEGRITY_FAILED" for item in report["sessions"])
    assert any(item["session_id"] == paths.session_id for item in report["sessions"])


def test_missing_early_context_stays_missing_and_is_reported(tmp_path, protocol):
    from src.microstructure.v10 import replay_synchronized_session

    paths = make_session(tmp_path, protocol)
    rows = [json.loads(line) for line in paths.depth_raw.read_text().splitlines()]
    # Snapshot and its first diff become available AFTER the first trade.
    for row in rows[:2]:
        row["received_at_utc"] = (START + timedelta(seconds=1)).isoformat()
    paths.depth_raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    replay = replay_synchronized_session(paths.depth_raw, paths.aggtrades_raw,
                                         session_id=paths.session_id).to_dict()
    def reseal(value):
        value["raw_artifacts"]["depth"].update(
            sha256=sha256_file(paths.depth_raw), size_bytes=paths.depth_raw.stat().st_size)
        value["offline_replay"] = replay
    edit_json(paths.summary, reseal)
    report = scan(tmp_path, protocol)
    assert report["eligible_sessions"] == 1
    assert report["sessions"][0]["trades_without_depth"] == 1
    contexts = list(iter_causal_trade_contexts(paths.depth_raw, paths.aggtrades_raw,
                                             session_id=paths.session_id))
    assert contexts[0].depth is None
    assert contexts[1].depth.last_update_id == 102


def test_active_versioned_manifest_is_frozen_and_source_bound():
    path, manifest = load_manifest()
    verify_manifest(manifest)
    definition = manifest["definition"]
    created = datetime.fromisoformat(definition["created_at_utc"].replace("Z", "+00:00"))
    cutoff = datetime.fromisoformat(definition["prospective_cutoff_utc"].replace("Z", "+00:00"))
    assert cutoff == prospective_cutoff(created)
    assert cutoff > created
    assert source_hashes(definition["creation_source_commit_sha"]) == definition["collector_source_sha256_lf_normalized"]
    assert path.parent.name == manifest["preregistration_id"]
