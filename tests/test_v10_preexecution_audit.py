"""Synthetic acquisition-only audit tests; never load real predictive outcomes."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.cli import audit_v10_preexecution as audit
from src.research import v10_economic_discovery_preregistration as discovery
from src.research import v10_economic_execution as execution
from tests.test_v10_economic_discovery_preregistration import BOUND_AT, bound  # noqa: F401


@pytest.fixture
def sandbox(bound, tmp_path, monkeypatch):
    workspace, _, _, manifest = bound
    audit.shutil.copytree(workspace / "src", tmp_path / "src")
    audit.shutil.copytree(workspace / discovery.DATA_ROOT, tmp_path / discovery.DATA_ROOT)
    root = tmp_path / "research/discovery"
    path = root / manifest["preregistration_id"] / "manifest.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(discovery.canonical(manifest))
    acquisition_root = tmp_path / "research/acquisition"
    acquisition_root.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(audit, "WORKSPACE", tmp_path)
    monkeypatch.setattr(audit, "EXPECTED_ROOT", tmp_path)
    monkeypatch.setattr(audit, "ROOT", root)
    monkeypatch.setattr(audit, "ACQUISITION_ROOT", acquisition_root)
    monkeypatch.setattr(audit, "AUTHORIZATION", tmp_path / "authorization/authorization.json")
    monkeypatch.setattr(audit, "REPORT_DIRECTORY", tmp_path / "reports/discovery")
    monkeypatch.setattr(audit, "sys", SimpleNamespace(version_info=(3, 12, 14)))
    monkeypatch.setattr(audit.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))
    for module in (audit, execution):
        monkeypatch.setattr(module, "PREREGISTRATION_ID", manifest["preregistration_id"])
        monkeypatch.setattr(module, "DEFINITION_SHA256", manifest["definition_sha256"])
    monkeypatch.setattr(audit, "MANIFEST_SHA256", discovery.sha256_file(path))
    monkeypatch.setattr(audit, "load_manifest", lambda: discovery.load_manifest(root))
    monkeypatch.setattr(audit, "verify_dataset_binding",
                        lambda value: discovery.verify_dataset_binding(value, workspace=tmp_path))
    monkeypatch.setattr(audit, "bind_ready_dataset", lambda report, *, bound_at:
                        discovery.bind_ready_dataset(report, bound_at=bound_at, workspace=tmp_path))
    scanner = audit.scan_readiness
    monkeypatch.setattr(audit, "scan_readiness", lambda value, *, data_root, workers:
                        scanner(value, data_root=data_root, now=BOUND_AT, workers=1))
    identity = {"source_commit": "a" * 40, "sources_sha256": {"synthetic.py": "b" * 64}}
    monkeypatch.setattr(audit, "executor_identity", lambda: deepcopy(identity))
    state = {"branch": audit.EXPECTED_BRANCH, "status": "", "root": str(tmp_path)}

    def git(*args):
        if args[0] == "rev-parse":
            return state["root"]
        return state["branch"] if args[0] == "branch" else state["status"]

    monkeypatch.setattr(audit, "git", git)

    def forbidden(*args, **kwargs):
        pytest.fail("audit reached a predictive or mutating execution operation")

    for name in ("execute_once", "reserve", "exclusive_write", "sample_session", "build_diagnostic", "_merged_records"):
        monkeypatch.setattr(execution, name, forbidden)
    return tmp_path, path, state


def failed(report, reason):
    assert report["status"] == "FAIL"
    assert any(reason in item.get("reason", "") for item in report["checks"])


def test_pass_is_readonly_and_does_not_consume_or_execute(sandbox):
    root = sandbox[0]
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    result = audit.audit()
    assert result["status"] == "PASS", result
    assert result["acquisition"]["eligible_sessions"] == 8
    assert result["acquisition"]["eligible_hours"] == "24"
    assert len(result["acquisition"]["eligible_utc_dates"]) >= 3
    assert result["predictive_outcomes_evaluated"] is False
    assert result["dataset_consumed_by_audit"] is False
    assert result["orders_invoked"] is False
    assert result["blind_holdout"] == "LOCKED_NOT_ACCESSED_BY_AUDIT"
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert not audit.AUTHORIZATION.exists() and not audit.REPORT_DIRECTORY.exists()


def test_wrong_python_fails_before_inputs(sandbox, monkeypatch):
    monkeypatch.setattr(audit, "sys", SimpleNamespace(version_info=(3, 13, 0)))
    monkeypatch.setattr(audit, "load_manifest", lambda: pytest.fail("unexpected input read"))
    failed(audit.audit(), "exactly 3.12")


def test_dirty_worktree_fails_even_with_all_integrity_checks_passed(sandbox):
    sandbox[2]["status"] = "?? untracked.py"
    result = audit.audit()
    failed(result, "dirty worktree")
    assert result["acquisition"]["status"] == "READY"
    assert [row["name"] for row in result["checks"] if row["status"] == "FAIL"] == [
        "clean_worktree_before", "clean_worktree_after"
    ]


@pytest.mark.parametrize("key,value,reason", [
    ("branch", "main", "wrong branch"),
    ("branch", "", "wrong branch"),
    ("root", "C:/another_repository", "wrong Git repository root"),
])
def test_wrong_git_identity(sandbox, key, value, reason):
    sandbox[2][key] = value
    failed(audit.audit(), reason)


@pytest.mark.parametrize("name", ["PREREGISTRATION_ID", "DEFINITION_SHA256"])
def test_wrong_preregistration_identity(sandbox, monkeypatch, name):
    monkeypatch.setattr(execution, name, "0" * len(getattr(execution, name)))
    failed(audit.audit(), "unauthorized preregistration identity")


def test_not_ready(sandbox, monkeypatch):
    original = audit.scan_readiness

    def not_ready(*args, **kwargs):
        return {**original(*args, **kwargs), "status": "NOT_READY"}

    monkeypatch.setattr(audit, "scan_readiness", not_ready)
    failed(audit.audit(), "NOT_READY")


@pytest.mark.parametrize("filename", discovery.FILES)
def test_changed_session_hashes(sandbox, filename):
    path = next((sandbox[0] / discovery.DATA_ROOT).rglob(filename))
    path.write_bytes(path.read_bytes() + b"\n")
    failed(audit.audit(), "changed")


def test_changed_reference_hash(sandbox):
    path = sandbox[0] / discovery.REFERENCE_SOURCES[0]
    path.write_bytes(path.read_bytes() + b"\n# changed\n")
    failed(audit.audit(), "reference source changed")


def test_changed_manifest_bytes(sandbox):
    sandbox[1].write_bytes(sandbox[1].read_bytes() + b"\n")
    failed(audit.audit(), "frozen manifest bytes changed")


def test_authorization_present(sandbox):
    audit.AUTHORIZATION.parent.mkdir()
    audit.AUTHORIZATION.write_text("synthetic invalid receipt; presence alone must fail")
    failed(audit.audit(), "authorization already present")


@pytest.mark.parametrize("kind", ["empty_directory", "result", "file"])
def test_existing_reservation_or_result(sandbox, kind):
    audit.REPORT_DIRECTORY.parent.mkdir()
    if kind == "file":
        audit.REPORT_DIRECTORY.write_text("synthetic")
    else:
        audit.REPORT_DIRECTORY.mkdir()
        if kind == "result":
            (audit.REPORT_DIRECTORY / "report.json").write_text("{}")
    failed(audit.audit(), "reservation/result already present")


def test_insufficient_disk(sandbox, monkeypatch):
    monkeypatch.setattr(audit.shutil, "disk_usage",
                        lambda _: SimpleNamespace(free=audit.MINIMUM_FREE_BYTES - 1))
    result = audit.audit()
    failed(result, "insufficient free disk space")
    assert result["free_bytes_before"] == audit.MINIMUM_FREE_BYTES - 1


def test_session_set_changed(sandbox):
    path = sandbox[0] / discovery.DATA_ROOT / "extra/session.manifest.json"
    path.parent.mkdir()
    path.write_text("{}")
    failed(audit.audit(), "session set changed")


def test_fresh_rebinding_mismatch(sandbox, monkeypatch):
    original = audit.scan_readiness

    def changed(*args, **kwargs):
        value = original(*args, **kwargs)
        value["sessions"][0]["causal_contexts_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(audit, "scan_readiness", changed)
    failed(audit.audit(), "differs from fresh READINESS")


def test_redirect_rejected_before_artifact_read(sandbox, monkeypatch):
    target = next((sandbox[0] / discovery.DATA_ROOT).rglob("depth.jsonl"))
    original = Path.is_junction
    monkeypatch.setattr(Path, "is_junction", lambda path: path == target or original(path))
    monkeypatch.setattr(audit, "load_manifest", lambda: pytest.fail("unexpected input read"))
    failed(audit.audit(), "redirected audit input")


def test_late_executor_change(sandbox, monkeypatch):
    original = audit.executor_identity
    calls = []

    def changed():
        calls.append(True)
        return {**original(), "generation": len(calls)}

    monkeypatch.setattr(audit, "executor_identity", changed)
    failed(audit.audit(), "executor/runtime changed")


def test_cli_exit_status_and_no_bypass_flags(sandbox, monkeypatch, capsys):
    monkeypatch.setattr(audit, "audit", lambda: {"status": "PASS"})
    assert audit.main([]) == 0
    assert '"PASS"' in capsys.readouterr().out
    monkeypatch.setattr(audit, "audit", lambda: {"status": "FAIL"})
    assert audit.main([]) == 1
    for flag in ("--allow-dirty", "--authorize", "--execute", "--data-root", "--output"):
        with pytest.raises(SystemExit) as error:
            audit.main([flag])
        assert error.value.code == 2
