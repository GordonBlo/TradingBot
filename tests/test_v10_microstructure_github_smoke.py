from __future__ import annotations

import json
import shutil

import pytest

from src.cli.validate_v10_microstructure_smoke import locate, prepare, validate
from src.microstructure.ci import CI_MARKER, SMOKE_ROOT
from src.microstructure.v10 import sha256_file
from src.research.v10_collection_preregistration import WORKSPACE, load_manifest
from src.research.v10_collection_readiness import scan_readiness
from tests.test_v10_collection_preregistration import SCAN_TIME, edit_json, make_session


@pytest.fixture
def smoke(tmp_path):
    _, manifest = load_manifest()
    prepare(tmp_path)
    paths = make_session(tmp_path / SMOKE_ROOT, manifest, duration=120)
    return tmp_path, manifest, paths


def test_valid_smoke_replays_without_research_readiness(smoke, monkeypatch):
    workspace, _, paths = smoke
    before = {path.name: sha256_file(path) for path in paths.directory.iterdir()}
    def forbidden(*args, **kwargs):
        raise AssertionError("research readiness must never run for CI")
    monkeypatch.setattr("src.research.v10_collection_readiness.scan_readiness", forbidden)
    report = validate(workspace, now=SCAN_TIME)
    assert report["research_eligibility"] == "CI_ONLY"
    assert report["depth_count"] == report["aggtrade_count"] == 2
    assert report["trade_contexts"] == report["reconstructed_updates"] == 2
    assert report["missing_early_contexts"] == 0
    assert report["deterministic_dual_stream_replay"] == "PASS"
    assert report["causal_trade_l2_merge"] == "PASS"
    assert report["artifact_sha256"] == before
    assert {path.name: sha256_file(path) for path in paths.directory.iterdir() if path.name in before} == before
    assert json.loads(paths.manifest.read_text())["research_eligibility"] == "UNASSESSED"
    assert (paths.directory / CI_MARKER).is_file()


@pytest.mark.parametrize("subpath", ["", "BTCUSDC", "BTCUSDC/session"])
def test_ci_roots_rejected_before_manifest_or_data_read(tmp_path, subpath):
    # Empty preregistration proves rejection precedes even protocol validation.
    with pytest.raises(ValueError, match="CI data root forbidden"):
        scan_readiness({}, data_root=tmp_path / SMOKE_ROOT / subpath)


def test_copied_ci_session_remains_ineligible(smoke, tmp_path):
    workspace, manifest, paths = smoke
    locate(workspace)
    root = tmp_path / "research_copy"
    shutil.copytree(paths.directory, root / paths.session_id)
    report = scan_readiness(manifest, data_root=root, now=SCAN_TIME)
    assert report["eligible_sessions"] == 0
    assert report["eligible_hours"] == "0"
    assert "CI_ONLY" in report["sessions"][0]["reason"]


def test_locate_rejects_multiple_sessions(smoke):
    workspace, manifest, _ = smoke
    from datetime import timedelta
    from tests.test_v10_collection_preregistration import START
    make_session(workspace / SMOKE_ROOT, manifest, started=START + timedelta(hours=1), duration=120)
    with pytest.raises(ValueError, match="exactly one"):
        locate(workspace)


def test_locate_rejects_extra_files(smoke):
    workspace, _, _ = smoke
    (workspace / SMOKE_ROOT / "unrelated.jsonl").write_text("{}")
    with pytest.raises(ValueError, match="unexpected artifacts"):
        locate(workspace)


def test_prepare_never_overwrites_existing_root(smoke):
    workspace, _, paths = smoke
    before = paths.depth_raw.read_bytes()
    with pytest.raises(ValueError, match="must be fresh"):
        prepare(workspace)
    assert paths.depth_raw.read_bytes() == before


def test_interrupted_session_can_be_archived_but_fails_validation(smoke):
    workspace, _, paths = smoke
    paths.summary.unlink()
    assert locate(workspace) == paths.directory
    assert (paths.directory / CI_MARKER).is_file()
    with pytest.raises(OSError):
        validate(workspace, now=SCAN_TIME)


@pytest.mark.parametrize("source,counter", [
    ("depth", "sequence_gaps"), ("depth", "invalid_events"),
    ("depth", "crossed_invalid_book_states"), ("depth", "reconnect_count"),
    ("aggtrades", "invalid_events"), ("aggtrades", "id_regressions"),
    ("aggtrades", "reconnect_count"),
])
def test_smoke_rejects_bad_integrity_counters(smoke, source, counter):
    workspace, _, paths = smoke
    edit_json(paths.summary, lambda value: value["live_integrity"][source].update({counter: 1}))
    with pytest.raises(ValueError):
        validate(workspace, now=SCAN_TIME)


def test_smoke_rejects_raw_hash_tampering(smoke):
    workspace, _, paths = smoke
    with paths.depth_raw.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate(workspace, now=SCAN_TIME)


def test_smoke_rejects_fake_replay_hash(smoke):
    workspace, _, paths = smoke
    edit_json(paths.summary, lambda value: value["offline_replay"].update(synchronized_timeline_sha256="0" * 64))
    with pytest.raises(ValueError, match="replay"):
        validate(workspace, now=SCAN_TIME)


def test_ci_validation_does_not_require_prospective_cutoff(smoke):
    from datetime import datetime, UTC
    workspace, manifest, paths = smoke
    shutil.rmtree(paths.directory)
    make_session(workspace / SMOKE_ROOT, manifest, started=datetime(2026, 3, 1, tzinfo=UTC), duration=120)
    assert validate(workspace, now=SCAN_TIME)["research_eligibility"] == "CI_ONLY"


def test_smoke_workflow_static_contract():
    workflow = (WORKSPACE / ".github/workflows/v10-microstructure-github-smoke.yml").read_text()
    assert "on:\n  workflow_dispatch:\n" in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "runs-on: ubuntu-latest" in workflow
    assert "timeout-minutes: 10" in workflow
    assert 'python-version: "3.12"' in workflow
    assert "pip install -r requirements.txt" in workflow
    assert "--validate-seconds 120" in workflow
    assert "--output-root data/ci/v10_microstructure_smoke" in workflow
    assert "--snapshot-limit 5000" in workflow and "--max-levels 5000" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "retention-days: 3" in workflow
    assert "path: ${{ steps.session.outputs.session_dir }}" in workflow
    assert "always() && steps.session.outputs.session_dir != ''" in workflow
    for forbidden in ("secrets.", "EVALUATE", "--mode READINESS", "schedule:", "push:", "pull_request:"):
        assert forbidden not in workflow
