from __future__ import annotations

import json
import stat
import subprocess
import zipfile
from datetime import timedelta

import pytest

from src.cli import import_v10_github_artifacts as importer
from src.cli import validate_v10_overnight as overnight
from src.microstructure.v10 import session_paths
from src.research.v10_collection_preregistration import load_manifest
from src.research.v10_collection_readiness import scan_readiness
from tests.test_v10_collection_preregistration import START, SCAN_TIME, make_session


@pytest.fixture
def artifact_factory(tmp_path):
    def create(index=0, *, prefix=""):
        workspace = tmp_path / f"generated_{index}"
        workspace.mkdir()
        started = START + timedelta(days=index)
        proof = overnight.preflight(workspace, now=started - timedelta(seconds=1))
        paths = make_session(workspace / overnight.ROOT,
                             {"definition": {"creation_source_commit_sha": proof["source_commit_sha"]}},
                             started=started)
        metadata = overnight.validate(index % 3 + 1, "123", "1", workspace=workspace, now=SCAN_TIME)
        assert metadata["result"]["status"] == "PASS"
        archive_path = tmp_path / (metadata["result"]["artifact_name"] + ".zip")
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name in importer.FILES:
                archive.write(paths.directory / name, prefix + name)
        return archive_path, paths
    return create


def rewrite(path, transform):
    with zipfile.ZipFile(path) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    transform(files)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def change_json(files, name, change):
    value = json.loads(files[name])
    change(value)
    files[name] = json.dumps(value).encode()


def run_import(tmp_path, archives):
    destination = tmp_path / "local"
    destination.mkdir(exist_ok=True)
    return destination, importer.import_artifacts(archives, workspace=destination, now=SCAN_TIME)


def test_valid_import_preserves_bytes_and_canonical_readiness(artifact_factory, tmp_path):
    archive, original = artifact_factory()
    destination, result = run_import(tmp_path, [archive])
    assert result["imported_session_ids"] == [original.session_id]
    assert result["rejected_artifacts"] == []
    target = session_paths(destination / importer.ROOT, started_at=START).directory
    assert set(item.name for item in target.iterdir()) == set(importer.FILES)
    assert all((target / name).read_bytes() == (original.directory / name).read_bytes() for name in importer.FILES)
    report = scan_readiness(load_manifest()[1], data_root=destination / importer.ROOT, now=SCAN_TIME)
    assert report["eligible_sessions"] == 1
    assert report["eligible_hours"] == "3"
    assert report["eligible_utc_dates"] == [START.date().isoformat()]
    assert report["status"] == "NOT_READY"


def test_multiple_artifacts(artifact_factory, tmp_path):
    one, p1 = artifact_factory(0)
    two, p2 = artifact_factory(1)
    destination, result = run_import(tmp_path, [one, two])
    assert result["imported_session_ids"] == [p1.session_id, p2.session_id]
    assert result["rejected_artifacts"] == []
    report = scan_readiness(load_manifest()[1], data_root=destination / importer.ROOT, now=SCAN_TIME)
    assert report["eligible_sessions"] == 2 and report["eligible_hours"] == "6"


@pytest.mark.parametrize("layout", ["session", "relative", "canonical"])
def test_single_wrapped_session_layout(artifact_factory, tmp_path, layout):
    relative = session_paths(importer.Path(), started_at=START).directory
    prefix = {"session": relative.name, "relative": relative.as_posix(),
              "canonical": (importer.ROOT / relative).as_posix()}[layout] + "/"
    archive, original = artifact_factory(prefix=prefix)
    _, result = run_import(tmp_path, [archive])
    assert result["imported_session_ids"] == [original.session_id]
    assert not result["rejected_artifacts"]


def test_two_sessions_in_one_zip_rejected_before_staging(artifact_factory, tmp_path):
    archive, _ = artifact_factory()
    second, _ = artifact_factory(1)
    with zipfile.ZipFile(second) as handle:
        extra = {"second/" + name: handle.read(name) for name in handle.namelist()}
    rewrite(archive, lambda files: files.update(extra))
    destination, result = run_import(tmp_path, [archive])
    assert "multiple sessions" in result["rejected_artifacts"][0]["reason"]
    assert not (destination / "data").exists()


def test_case_collision_rejected_before_staging(artifact_factory, tmp_path):
    archive, _ = artifact_factory()
    rewrite(archive, lambda files: files.update({"DEPTH.jsonl": files["depth.jsonl"]}))
    destination, result = run_import(tmp_path, [archive])
    assert "case-colliding" in result["rejected_artifacts"][0]["reason"]
    assert not (destination / "data").exists()


def test_overnight_depth_limits_cannot_be_changed(artifact_factory, tmp_path):
    archive, _ = artifact_factory()
    rewrite(archive, lambda files: change_json(files, "session.manifest.json",
        lambda value: value["streams"]["depth"].update(snapshot_limit=100)))
    _, result = run_import(tmp_path, [archive])
    assert "depth limits" in result["rejected_artifacts"][0]["reason"]


def test_identical_duplicate_skips_all_five_files(artifact_factory, tmp_path):
    archive, paths = artifact_factory()
    _, result = run_import(tmp_path, [archive, archive])
    assert result["imported_session_ids"] == [paths.session_id]
    assert result["identical_duplicates_skipped"] == [paths.session_id]
    assert result["rejected_artifacts"] == []


def test_conflicting_duplicate_never_overwrites_local_bytes(artifact_factory, tmp_path):
    archive, paths = artifact_factory()
    destination, _ = run_import(tmp_path, [archive])
    target = session_paths(destination / importer.ROOT, started_at=START).directory
    before = {name: (target / name).read_bytes() for name in importer.FILES}
    # Whitespace alone makes the immutable validation artifact a conflicting copy.
    rewrite(archive, lambda files: files.update({importer.VALIDATION: files[importer.VALIDATION] + b"\n"}))
    _, result = run_import(tmp_path, [archive])
    assert result["imported_session_ids"] == []
    assert "conflicting local duplicate" in result["rejected_artifacts"][0]["reason"]
    assert {name: (target / name).read_bytes() for name in importer.FILES} == before


@pytest.mark.parametrize("unsafe", ["../escape", "/absolute", "C:/escape", "..\\escape", "depth.jsonl:ads"])
def test_path_traversal_rejected_before_staging(artifact_factory, tmp_path, unsafe):
    archive, _ = artifact_factory()
    rewrite(archive, lambda files: files.update({unsafe: b"bad"}))
    destination, result = run_import(tmp_path, [archive])
    assert result["imported_session_ids"] == []
    assert "unsafe ZIP" in result["rejected_artifacts"][0]["reason"]
    assert not (destination / "data").exists()


def test_malformed_zip(tmp_path):
    archive = tmp_path / "v10-microstructure-overnight-broken.zip"
    archive.write_bytes(b"not a ZIP")
    _, result = run_import(tmp_path, [archive])
    assert result["imported_session_ids"] == []
    assert "zip" in result["rejected_artifacts"][0]["reason"].lower()


def test_wrong_preregistration(artifact_factory, tmp_path):
    archive, _ = artifact_factory()
    rewrite(archive, lambda files: change_json(files, importer.VALIDATION,
                                              lambda value: value.update(preregistration_id="wrong")))
    _, result = run_import(tmp_path, [archive])
    assert "wrong preregistration" in result["rejected_artifacts"][0]["reason"]


@pytest.mark.parametrize("ci_form", ["marker", "manifest", "metadata"])
def test_ci_only_rejection(artifact_factory, tmp_path, ci_form):
    archive, _ = artifact_factory()
    def change(files):
        if ci_form == "marker":
            files["ci.provenance.json"] = b'{}'
        elif ci_form == "manifest":
            change_json(files, "session.manifest.json", lambda value: value.update(research_eligibility="CI_ONLY"))
        else:
            change_json(files, importer.VALIDATION, lambda value: value["result"].update(research_eligibility="CI_ONLY"))
    rewrite(archive, change)
    destination, result = run_import(tmp_path, [archive])
    assert "CI_ONLY" in result["rejected_artifacts"][0]["reason"]
    assert not (destination / "data").exists()


def test_raw_hash_mismatch(artifact_factory, tmp_path):
    archive, _ = artifact_factory()
    rewrite(archive, lambda files: files.update({"depth.jsonl": files["depth.jsonl"] + b"\n"}))
    destination, result = run_import(tmp_path, [archive])
    assert "hash mismatch" in result["rejected_artifacts"][0]["reason"]
    assert not (destination / importer.ROOT).exists()


@pytest.mark.parametrize("missing", importer.FILES)
def test_missing_required_files(artifact_factory, tmp_path, missing):
    archive, _ = artifact_factory()
    rewrite(archive, lambda files: files.pop(missing))
    _, result = run_import(tmp_path, [archive])
    assert "missing required files" in result["rejected_artifacts"][0]["reason"]


def test_replay_metadata_mismatch(artifact_factory, tmp_path):
    archive, _ = artifact_factory()
    rewrite(archive, lambda files: change_json(files, importer.VALIDATION,
        lambda value: value["readiness"]["sessions"][0].update(causal_contexts_sha256="0" * 64)))
    _, result = run_import(tmp_path, [archive])
    assert "stored readiness metadata" in result["rejected_artifacts"][0]["reason"]


def test_publication_is_one_atomic_directory_operation(artifact_factory, tmp_path, monkeypatch):
    archive, _ = artifact_factory()
    calls = []
    actual = importer._atomic_publish
    def checked(source, target):
        assert not target.exists()
        assert set(item.name for item in source.iterdir()) == set(importer.FILES)
        calls.append(target)
        actual(source, target)
        assert set(item.name for item in target.iterdir()) == set(importer.FILES)
    monkeypatch.setattr(importer, "_atomic_publish", checked)
    _, result = run_import(tmp_path, [archive])
    assert len(calls) == 1 and not result["rejected_artifacts"]


def test_racing_destination_cannot_be_replaced(artifact_factory, tmp_path, monkeypatch):
    archive, _ = artifact_factory()
    actual = importer._atomic_publish
    targets = []
    def race(source, target):
        target.mkdir()
        (target / "keep.txt").write_text("existing local bytes")
        targets.append(target)
        actual(source, target)
    monkeypatch.setattr(importer, "_atomic_publish", race)
    _, result = run_import(tmp_path, [archive])
    assert result["imported_session_ids"] == []
    assert len(result["rejected_artifacts"]) == 1
    assert (targets[0] / "keep.txt").read_text() == "existing local bytes"


def test_zip_symlink_is_rejected(artifact_factory, tmp_path):
    archive, _ = artifact_factory()
    with zipfile.ZipFile(archive, "a") as handle:
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        handle.writestr(info, "../elsewhere")
    _, result = run_import(tmp_path, [archive])
    assert "links" in result["rejected_artifacts"][0]["reason"]


def test_pre_cutoff_rejected_without_raw_writes(artifact_factory, tmp_path):
    archive, _ = artifact_factory()
    def change(files):
        change_json(files, "session.manifest.json", lambda value: value.update(
            started_at_utc="2026-09-12T08:30:00Z", session_id="20260912T083000000000Z"))
    rewrite(archive, change)
    destination, result = run_import(tmp_path, [archive])
    assert "strictly after" in result["rejected_artifacts"][0]["reason"]
    assert not (destination / "data").exists()


def test_rejection_does_not_prevent_later_valid_import(artifact_factory, tmp_path):
    bad, _ = artifact_factory(0)
    good, paths = artifact_factory(1)
    bad.write_bytes(b"broken")
    _, result = run_import(tmp_path, [bad, good])
    assert result["imported_session_ids"] == [paths.session_id]
    assert len(result["rejected_artifacts"]) == 1


def test_readiness_command_is_exact_and_nonpredictive(monkeypatch):
    calls = []
    def fake(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 2, json.dumps({"status": "NOT_READY"}), "")
    monkeypatch.setattr(importer.subprocess, "run", fake)
    assert importer.run_readiness()["status"] == "NOT_READY"
    assert calls == [[importer.sys.executable, "-m", "src.cli.run_v10_collection", "--mode", "READINESS"]]


def test_cli_automatically_runs_readiness_and_prints_import_results(monkeypatch, capsys):
    order = []
    def fake_import(paths):
        order.append("import")
        return {"imported_session_ids": ["session"], "identical_duplicates_skipped": [], "rejected_artifacts": []}
    def fake_readiness():
        order.append("READINESS")
        return {"eligible_sessions": 1, "eligible_hours": "3", "eligible_utc_dates": ["2026-09-13"], "status": "NOT_READY"}
    monkeypatch.setattr(importer, "import_artifacts", fake_import)
    monkeypatch.setattr(importer, "run_readiness", fake_readiness)
    assert importer.main(["v10-microstructure-overnight-test.zip"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["readiness_status"] == "NOT_READY" and output["eligible_hours"] == "3"
    assert order == ["import", "READINESS"]
