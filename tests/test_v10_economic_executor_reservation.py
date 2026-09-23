from __future__ import annotations

import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace

import pytest

from src.research import v10_economic_execution as execution
from src.research.v10_economic_discovery_preregistration import load_manifest


def test_exact_authorized_preregistration_and_tampering_rejection():
    path, manifest = load_manifest()  # Versioned metadata only.
    execution.verify_identity(manifest)
    assert execution.sha256_file(path) == execution.MANIFEST_SHA256
    for field in ("preregistration_id", "definition_sha256"):
        other = deepcopy(manifest)
        other[field] = "0" * len(other[field])
        with pytest.raises(ValueError):
            execution.verify_identity(other)


def receipt(identity):
    return {
        "preregistration_id": execution.PREREGISTRATION_ID,
        "definition_sha256": execution.DEFINITION_SHA256,
        "scope": "ONE_TIME_REAL_DISCOVERY",
        "authorized_by": "USER",
        "authorization_reference": "SYNTHETIC TEST ONLY",
        "executor_identity": identity,
        "executor_identity_sha256": execution.digest(identity),
    }


@pytest.fixture
def synthetic_executor(tmp_path, monkeypatch):
    """Patch dependencies only in tests; never open a bound real market artifact."""
    path, manifest = load_manifest()  # Frozen metadata, not outcomes.
    copied = tmp_path / "frozen_manifest.json"
    copied.write_bytes(path.read_bytes())
    identity = {"source": "SYNTHETIC", "runtime": "SYNTHETIC"}
    authorization = tmp_path / "synthetic_authorization.json"
    authorization.write_bytes(execution.encoded(receipt(identity)))
    monkeypatch.setattr(execution, "WORKSPACE", tmp_path)
    monkeypatch.setattr(execution, "AUTHORIZATION", authorization)
    monkeypatch.setattr(execution, "REPORT_DIRECTORY", tmp_path / "reserved")
    monkeypatch.setattr(execution, "executor_identity", lambda: identity)
    monkeypatch.setattr(execution, "load_manifest", lambda: (copied, manifest))
    calls = []
    monkeypatch.setattr(
        execution, "verify_dataset_binding", lambda _: calls.append("VERIFY")
    )

    def readiness(*args, **kwargs):
        calls.append("READINESS")
        return {"status": "READY"}

    def rebound(*args, **kwargs):
        calls.append("REBIND")
        return manifest["definition"]["dataset"]

    monkeypatch.setattr(execution, "scan_readiness", readiness)
    monkeypatch.setattr(execution, "bind_ready_dataset", rebound)
    for bound in manifest["definition"]["dataset"]["readiness"]["sessions"]:
        target = tmp_path / bound["manifest_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"streams":{"depth":{"max_levels_per_side":5000}}}')

    def records(depth, trades, **kwargs):
        assert depth.is_relative_to(tmp_path) and trades.is_relative_to(tmp_path)
        assert (execution.REPORT_DIRECTORY / "reservation.json").is_file()
        calls.append("LOAD")
        return ()

    monkeypatch.setattr(execution, "_merged_records", records)
    monkeypatch.setattr(
        execution,
        "sample_session",
        lambda _, **kwargs: SimpleNamespace(session_id=kwargs["session_id"], rows=()),
    )

    def diagnostic(_):
        calls.append("DIAGNOSTIC")
        return {"synthetic": True, "undefined_descriptive": None}, {}, {}

    monkeypatch.setattr(execution, "build_diagnostic", diagnostic)
    return SimpleNamespace(
        calls=calls, manifest=manifest, copied=copied, identity=identity
    )


def test_authorization_binds_executor_runtime_and_exact_scope():
    identity = {"source": "synthetic", "runtime": "synthetic"}
    value = receipt(identity)
    execution.validate_authorization(value, identity)
    with pytest.raises(ValueError, match="identity changed"):
        execution.validate_authorization(value, {**identity, "runtime": "changed"})
    for field, replacement in (
        ("scope", "IMPLEMENTATION_ONLY"),
        ("preregistration_id", "wrong"),
        ("authorized_by", "ASSISTANT"),
        ("authorization_reference", ""),
    ):
        with pytest.raises(ValueError, match="authorization"):
            execution.validate_authorization({**value, field: replacement}, identity)
    with pytest.raises(ValueError, match="schema"):
        execution.validate_authorization({**value, "force": True}, identity)


def test_no_real_authorization_no_raw_read_no_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(execution, "REPORT_DIRECTORY", tmp_path / "reserved")
    monkeypatch.setattr(
        execution, "AUTHORIZATION", tmp_path / "absent_authorization.json"
    )
    monkeypatch.setattr(
        execution,
        "executor_identity",
        lambda: pytest.fail("must stop before runtime/data inspection"),
    )
    monkeypatch.setattr(
        execution,
        "load_manifest",
        lambda: pytest.fail("must stop before manifest/data loading"),
    )
    with pytest.raises(ValueError, match="authorization is absent"):
        execution.execute_once()
    assert not execution.REPORT_DIRECTORY.exists()


@pytest.mark.parametrize(
    "failure", ["authorization", "runtime", "source", "manifest", "acquisition"]
)
def test_invalid_preflight_never_loads_or_reserves(
    synthetic_executor, monkeypatch, failure
):
    def forbidden(*args, **kwargs):
        pytest.fail("predictive data touched before valid preflight")

    monkeypatch.setattr(execution, "_merged_records", forbidden)
    monkeypatch.setattr(execution, "sample_session", forbidden)
    monkeypatch.setattr(execution, "build_diagnostic", forbidden)
    if failure == "authorization":
        value = receipt(synthetic_executor.identity)
        value["scope"] = "IMPLEMENTATION_ONLY"
        execution.AUTHORIZATION.write_bytes(execution.encoded(value))
    elif failure in ("runtime", "source"):
        changed = {**synthetic_executor.identity, failure: "CHANGED"}
        monkeypatch.setattr(execution, "executor_identity", lambda: changed)
    elif failure == "manifest":
        # Semantically unchanged JSON still must match the frozen byte hash.
        with synthetic_executor.copied.open("ab") as stream:
            stream.write(b"\n")
    else:

        def invalid_protocol():
            raise ValueError("acquisition hash mismatch")

        monkeypatch.setattr(execution, "protocol", invalid_protocol)
    with pytest.raises(ValueError):
        execution.execute_once()
    assert not execution.REPORT_DIRECTORY.exists()


@pytest.mark.parametrize(
    "existing", [None, "reservation.json", "report.json", "seal.json"]
)
def test_canonical_existing_state_blocks_before_loading(
    synthetic_executor, monkeypatch, existing
):
    execution.REPORT_DIRECTORY.mkdir()
    if existing:
        (execution.REPORT_DIRECTORY / existing).write_text("partial synthetic state")
    monkeypatch.setattr(
        execution, "_merged_records", lambda *a, **kw: pytest.fail("no raw read")
    )
    with pytest.raises(ValueError, match="permanently refused"):
        execution.execute_once()
    assert not synthetic_executor.calls


def test_reservation_is_atomic_under_competing_claims(tmp_path):
    directory = tmp_path / "one_time"

    def claim(_):
        try:
            execution.reserve(directory, {"synthetic": True})
            return True
        except (ValueError, FileExistsError):
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(claim, range(16))) == 1
    assert json.loads((directory / "reservation.json").read_text()) == {
        "synthetic": True
    }


@pytest.mark.parametrize(
    "existing", ["reservation.json", "report.json", "seal.json", None]
)
def test_existing_reservation_result_or_empty_crash_remnant_refused(tmp_path, existing):
    directory = tmp_path / "reserved"
    directory.mkdir()
    if existing:
        (directory / existing).write_text("partial", encoding="utf-8")
    with pytest.raises(ValueError, match="permanently refused"):
        execution.reserve(directory, {})


def test_process_crash_after_reservation_never_allows_restart(tmp_path):
    directory = tmp_path / "crash"
    code = (
        "import os,sys; from pathlib import Path; "
        "from src.research.v10_economic_execution import reserve; "
        "reserve(Path(sys.argv[1]), {'synthetic':True}); os._exit(17)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(directory)],
        cwd=execution.WORKSPACE,
        check=False,
    )
    assert result.returncode == 17
    assert (directory / "reservation.json").is_file()
    with pytest.raises(ValueError, match="permanently refused"):
        execution.reserve(directory, {})


@pytest.mark.parametrize("phase", ["precheck", "load", "diagnostic", "postcheck"])
def test_failure_at_each_reserved_phase_leaves_permanent_lock(
    synthetic_executor, monkeypatch, phase
):
    directory = execution.REPORT_DIRECTORY
    calls = []

    def check(_):
        calls.append("check")
        if (phase == "precheck" and len(calls) == 2) or (
            phase == "postcheck" and len(calls) == 3
        ):
            raise ValueError("synthetic integrity failure")

    def load(*args, **kwargs):
        assert (directory / "reservation.json").is_file()
        if phase == "load":
            raise KeyboardInterrupt("synthetic crash")
        return ()

    def diagnostic(_):
        if phase == "diagnostic":
            raise ValueError("undefined required statistic")
        return {"synthetic": True}, {}, {}

    monkeypatch.setattr(execution, "verify_dataset_binding", check)
    monkeypatch.setattr(execution, "_merged_records", load)
    monkeypatch.setattr(execution, "build_diagnostic", diagnostic)
    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        execution.execute_once()
    assert not (directory / "report.json").exists()
    with pytest.raises(ValueError, match="permanently refused"):
        execution.execute_once()


def test_partial_ledger_write_still_locks_and_never_opens_outcomes(
    tmp_path, monkeypatch
):
    directory = tmp_path / "partial_ledger"
    monkeypatch.setattr(
        execution,
        "exclusive_write",
        lambda *args: (_ for _ in ()).throw(OSError("disk failure")),
    )
    with pytest.raises(OSError):
        execution.reserve(directory, {})
    assert directory.is_dir()
    with pytest.raises(ValueError):
        execution.refuse_existing(directory)


def test_exclusive_deterministic_reporting_and_seal(
    synthetic_executor, tmp_path, monkeypatch
):
    def diagnostic(_):
        return (
            {
                "evidence_role": "DISCOVERY ONLY",
                "warning": "Positive classification does not establish executable profitability.",
                "synthetic": True,
                "undefined_descriptive": None,
            },
            {"predictions": [-2.0, 1.0]},
            {"statistics": [0.0]},
        )

    results = []
    monkeypatch.setattr(execution, "build_diagnostic", diagnostic)
    for name in ("a", "b"):
        directory = tmp_path / name
        monkeypatch.setattr(execution, "REPORT_DIRECTORY", directory)
        results.append(execution.execute_once())
        report = json.loads((directory / "report.json").read_text())
        for file, checksum in report["artifact_sha256"].items():
            assert execution.sha256_file(directory / file) == checksum
        assert (
            execution.sha256_file(directory / "report.json")
            == results[-1]["report_sha256"]
        )
        assert report["undefined_descriptive"] is None
        with pytest.raises(FileExistsError):
            execution.exclusive_write(directory / "report.json", [b"replacement"])
    assert results[0] == results[1]
    assert {p.name: p.read_bytes() for p in (tmp_path / "a").iterdir()} == {
        p.name: p.read_bytes() for p in (tmp_path / "b").iterdir()
    }


def test_nonfinite_report_cannot_be_serialized():
    with pytest.raises(ValueError, match="nonfinite"):
        execution.encoded({"statistic": float("nan")})


def test_no_cli_execution_surface_or_authorization_artifact_created(tmp_path, monkeypatch):
    # Existing real authorization/results are historical evidence, not test fixtures.
    monkeypatch.setattr(execution, "AUTHORIZATION", tmp_path / "authorization.json")
    monkeypatch.setattr(execution, "REPORT_DIRECTORY", tmp_path / "discovery")
    from src.cli.run_v10_economic_discovery import build_parser

    for mode in ("RUN", "EXECUTE", "EVALUATE"):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--mode", mode])
    assert not execution.AUTHORIZATION.exists()
    assert not execution.REPORT_DIRECTORY.exists()


def test_executor_identity_covers_current_source_and_runtime():
    value = execution.executor_identity()
    assert value["python_version"].startswith("3.12.")
    assert "src/diagnostics/v10_economic_features.py" in value["sources_sha256"]
    assert "src/research/v10_economic_execution.py" in value["sources_sha256"]
    assert len(value["python_executable_sha256"]) == 64


@pytest.mark.parametrize("failure", [None, "binding", "readiness", "rebound"])
def test_dormant_entry_requires_fresh_preflight_before_reservation(
    synthetic_executor, monkeypatch, failure
):
    manifest = synthetic_executor.manifest
    calls = []

    def binding(_):
        calls.append("VERIFY")
        if failure == "binding":
            raise ValueError("synthetic hash mismatch")

    def readiness(*args, **kwargs):
        calls.append("READINESS")
        return {"status": "NOT_READY" if failure == "readiness" else "READY"}

    def rebound(*args, **kwargs):
        calls.append("REBIND")
        return {} if failure == "rebound" else manifest["definition"]["dataset"]

    def reserved(directory, ledger):
        calls.append("RESERVE")
        assert ledger["authorization"] == receipt(synthetic_executor.identity)
        # Sentinel stops at the filesystem seam before sampling.
        raise OSError("synthetic stop at reservation")

    monkeypatch.setattr(execution, "verify_dataset_binding", binding)
    monkeypatch.setattr(execution, "scan_readiness", readiness)
    monkeypatch.setattr(execution, "bind_ready_dataset", rebound)
    monkeypatch.setattr(execution, "reserve", reserved)
    monkeypatch.setattr(
        execution,
        "_merged_records",
        lambda *a, **kw: pytest.fail("no market data read"),
    )
    if failure:
        with pytest.raises(ValueError):
            execution.execute_once()
        assert "RESERVE" not in calls
    else:
        with pytest.raises(RuntimeError, match="synthetic stop at reservation"):
            execution.execute_once()
        assert calls == ["VERIFY", "READINESS", "REBIND", "RESERVE"]
    assert not execution.REPORT_DIRECTORY.exists()
