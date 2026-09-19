from __future__ import annotations

import json
import shutil
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest

from src.cli import run_v10_economic_discovery as cli
from src.cli import validate_v10_overnight as overnight
from src.research import v10_economic_discovery_preregistration as discovery
from src.research.v10_collection_preregistration import load_manifest as load_acquisition
from src.research.v10_collection_readiness import scan_readiness
from tests.test_v10_collection_preregistration import START, make_session


BOUND_AT = datetime(2026, 9, 16, 12, tzinfo=UTC)


@pytest.fixture(scope="module")
def bound(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("discovery_binding")
    for name in discovery.REFERENCE_SOURCES:
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(discovery.WORKSPACE / name, target)
    acquisition = load_acquisition()[1]
    for index in range(8):
        generated = workspace / f"generated_{index}"
        generated.mkdir()
        start = START + timedelta(days=index // 3, hours=(index % 3) * 4)
        proof = overnight.preflight(generated, now=start - timedelta(seconds=1))
        paths = make_session(generated / overnight.ROOT,
                             {"definition": {"creation_source_commit_sha": proof["source_commit_sha"]}},
                             started=start)
        result = overnight.validate(index % 3 + 1, "123", "1", workspace=generated, now=BOUND_AT)
        assert result["result"]["status"] == "PASS"
        destination = workspace / paths.directory.relative_to(generated)
        shutil.copytree(paths.directory, destination)
    report = scan_readiness(acquisition, data_root=workspace / discovery.DATA_ROOT, now=BOUND_AT)
    report["manifest_sha256"] = discovery.PREREGISTRATION_SHA256
    dataset = discovery.bind_ready_dataset(report, bound_at=BOUND_AT, workspace=workspace)
    manifest = discovery.build_manifest(BOUND_AT, "a" * 40, discovery.reference_hashes(), dataset)
    return workspace, report, dataset, manifest


def test_deterministic_content_identity_and_exact_dataset(bound):
    _, _, dataset, manifest = bound
    rebuilt = discovery.build_manifest(BOUND_AT, "a" * 40, discovery.reference_hashes(), dataset)
    assert discovery.canonical(manifest) == discovery.canonical(rebuilt)
    assert manifest["definition_sha256"] == discovery.digest(manifest["definition"])
    assert manifest["preregistration_id"] == manifest["definition_sha256"][:16]
    discovery.verify_manifest(manifest)
    assert len(dataset["readiness"]["eligible_session_ids"]) == 8
    assert dataset["readiness"]["eligible_hours"] == "24"
    assert len(dataset["validation_sha256"]) == 8


@pytest.mark.parametrize("change", [
    lambda rules: rules["targets"].update(primary_seconds=30),
    lambda rules: rules["model"].update(alpha="0.1"),
    lambda rules: rules["features"]["combined_order"].append("future_return"),
    lambda rules: rules["economics"]["scenarios"]["base"].update(fee_bps_per_side="0"),
    lambda rules: rules["authorization"].update(executions_permitted_now=1),
    lambda rules: rules["statistics"]["null"].update(valid_permutations=10),
    lambda rules: rules["classification"]["information_gate_all_required"].update(combined_shift_p="<0.50"),
])
def test_rules_cannot_be_rewritten_even_with_new_digest(bound, change):
    value = deepcopy(bound[3])
    change(value["definition"]["rules"])
    value["definition_sha256"] = discovery.digest(value["definition"])
    value["preregistration_id"] = value["definition_sha256"][:16]
    with pytest.raises(ValueError, match="preregistration mismatch"):
        discovery.verify_manifest(value)


@pytest.mark.parametrize("change", [
    lambda report: report.update(status="NOT_READY"),
    lambda report: report.update(eligible_sessions=7),
    lambda report: report.update(eligible_hours="23.9"),
    lambda report: report.update(predictive_outcomes_evaluated=True),
    lambda report: report.update(independent_confirmatory_strategy_evidence=True),
    lambda report: report.update(prospective_cutoff_utc="2026-01-01T00:00:00Z"),
    lambda report: report["sessions"][0].update(classification="CI_ONLY"),
    lambda report: report["sessions"].reverse(),
    lambda report: report["eligible_session_ids"].reverse(),
    lambda report: report["sessions"][0].update(manifest_path="../escape/session.manifest.json"),
])
def test_invalid_dataset_cannot_be_frozen(bound, change):
    dataset = deepcopy(bound[2])
    change(dataset["readiness"])
    dataset["readiness_sha256"] = discovery.digest(dataset["readiness"])
    with pytest.raises(ValueError):
        discovery.build_manifest(BOUND_AT, "a" * 40, discovery.reference_hashes(), dataset)


def test_readiness_digest_is_required(bound):
    dataset = deepcopy(bound[2])
    dataset["readiness"]["sessions"][0]["artifact_sha256"]["depth.jsonl"] = "0" * 64
    with pytest.raises(ValueError, match="readiness digest"):
        discovery.validate_dataset(dataset)


def test_readonly_binding_checks_all_five_files(bound):
    workspace, _, _, manifest = bound
    before = {path: discovery.sha256_file(path) for path in (workspace / discovery.DATA_ROOT).rglob("*") if path.is_file()}
    discovery.verify_dataset_binding(manifest, workspace=workspace)
    assert before == {path: discovery.sha256_file(path) for path in before}
    assert len(before) == 40


@pytest.mark.parametrize("filename", discovery.FILES)
def test_changed_artifacts_fail_binding(bound, tmp_path, filename):
    workspace, _, _, manifest = bound
    shutil.copytree(workspace / "src", tmp_path / "src")
    shutil.copytree(workspace / discovery.DATA_ROOT, tmp_path / discovery.DATA_ROOT)
    path = next((tmp_path / discovery.DATA_ROOT).rglob(filename))
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed"):
        discovery.verify_dataset_binding(manifest, workspace=tmp_path)


def test_no_extra_sessions_during_rebinding(bound, tmp_path):
    workspace, report, _, _ = bound
    shutil.copytree(workspace / discovery.DATA_ROOT, tmp_path / discovery.DATA_ROOT)
    report = deepcopy(report)
    for row in report["sessions"]:
        row["manifest_path"] = str(tmp_path / discovery.Path(row["manifest_path"]).relative_to(workspace))
    extra = tmp_path / discovery.DATA_ROOT / "extra" / "session.manifest.json"
    extra.parent.mkdir()
    extra.write_text("{}")
    with pytest.raises(ValueError, match="session set changed"):
        discovery.bind_ready_dataset(report, bound_at=BOUND_AT, workspace=tmp_path)


def test_changed_reference_semantics_fail_before_data_binding(bound, tmp_path):
    workspace, _, _, manifest = bound
    shutil.copytree(workspace / "src", tmp_path / "src")
    path = tmp_path / discovery.REFERENCE_SOURCES[0]
    path.write_bytes(path.read_bytes() + b"\n# changed reference\n")
    with pytest.raises(ValueError, match="reference source changed"):
        discovery.verify_dataset_binding(manifest, workspace=tmp_path)


def test_loader_is_read_only_and_never_creates_manifest(bound, tmp_path):
    with pytest.raises(ValueError, match="never auto-create"):
        discovery.load_manifest(tmp_path)
    assert list(tmp_path.iterdir()) == []
    manifest = bound[3]
    path = tmp_path / manifest["preregistration_id"] / "manifest.json"
    path.parent.mkdir()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    before = path.read_bytes()
    assert discovery.load_manifest(tmp_path)[1] == manifest
    assert path.read_bytes() == before


@pytest.mark.parametrize("mode", ["EVALUATE", "RUN", "DISCOVER", "EXECUTE"])
def test_no_evaluation_mode_even_before_loading_manifest(monkeypatch, mode):
    def forbidden(*args, **kwargs):
        raise AssertionError("unauthorized mode must stop before any artifact read")
    monkeypatch.setattr(cli, "load_manifest", forbidden)
    with pytest.raises(SystemExit) as raised:
        cli.main(["--mode", mode])
    assert raised.value.code == 2


def test_no_authorization_flag_can_enable_execution(monkeypatch):
    monkeypatch.setattr(cli, "load_manifest", lambda: pytest.fail("must reject before loading"))
    with pytest.raises(SystemExit) as raised:
        cli.main(["--mode", "PLAN", "--authorize-evaluation"])
    assert raised.value.code == 2


def test_plan_never_reads_market_data_or_constructs_targets(bound, monkeypatch, capsys):
    monkeypatch.setattr(cli, "load_manifest", lambda: (discovery.Path("manifest.json"), bound[3]))
    monkeypatch.setattr(cli, "verify_dataset_binding", lambda *args: pytest.fail("PLAN must not read data"))
    assert cli.main(["--mode", "PLAN"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["predictive_outcomes_evaluated"] is False
    assert output["execution_authorized"] is False
    assert output["evidence_role"] == "DISCOVERY_ONLY"


def test_fixed_model_features_costs_and_primary_only_classification():
    rules = discovery.frozen_rules()
    assert len(rules["features"]["l2_only_order"]) == 13
    assert len(rules["features"]["combined_order"]) == len(set(rules["features"]["combined_order"])) == 40
    assert "bid_depth_top_20" not in rules["features"]["combined_order"]
    assert "ask_depth_top_20" not in rules["features"]["combined_order"]
    assert rules["targets"]["horizons_seconds"] == [5, 30, 60, 300]
    assert rules["targets"]["supporting_can_change_classification"] is False
    assert rules["model"]["alpha"] == "1.0"
    assert rules["model"]["folds"] == [
        {"train_session_indices": list(range(1, i)), "test_session_index": i} for i in range(2, 9)]
    assert rules["classification"]["primary_horizon_seconds"] == 300
    assert rules["economics"]["reference_round_trip_bps"] == {"base": "24", "stress": "48"}
    assert rules["authorization"]["executions_permitted_now"] == 0


def test_active_acquisition_remains_frozen():
    assert discovery.sha256_file(load_acquisition()[0]) == discovery.PREREGISTRATION_SHA256


def test_versioned_discovery_manifest_identity_and_nonexecution_state():
    # Reads versioned metadata only, never the real market files it binds.
    path, manifest = discovery.load_manifest()
    assert path.parent.name == manifest["preregistration_id"] == "a296e5ed304640d7"
    assert manifest["definition_sha256"] == "a296e5ed304640d706916cc8e562cedc6febf561b20b0c8a99d59ac30b275a33"
    definition = manifest["definition"]
    assert definition["dataset"]["readiness"]["eligible_session_ids"] == [
        "20260913T192323929783Z", "20260913T223513036401Z", "20260914T015309595012Z",
        "20260914T212646630426Z", "20260915T004347737080Z", "20260915T174845528831Z",
        "20260915T211259081136Z", "20260916T003212326370Z",
    ]
    assert definition["rules"]["evidence"]["role"] == "DISCOVERY_ONLY"
    assert definition["rules"]["authorization"]["executions_permitted_now"] == 0
