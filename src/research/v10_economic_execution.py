"""Dormant one-time runner. No CLI; a separate user authorization artifact is mandatory.

This module never creates authorization, deletes reservations or retries execution.
An exclusive directory is the permanent lock, including an empty/partially written one.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from src.diagnostics.v10_economic_features import sample_session
from src.diagnostics.v10_economic_reporting import build_diagnostic
from src.microstructure.v10 import _merged_records, sha256_file
from src.research.v10_collection_preregistration import (
    WORKSPACE,
    canonical,
    strict_json,
    timestamp,
    utc,
)
from src.research.v10_collection_readiness import require, scan_readiness
from src.research.v10_economic_discovery_preregistration import (
    DATA_ROOT,
    bind_ready_dataset,
    digest,
    load_manifest,
    protocol,
    verify_dataset_binding,
    verify_manifest,
)

PREREGISTRATION_ID = "a296e5ed304640d7"
DEFINITION_SHA256 = "a296e5ed304640d706916cc8e562cedc6febf561b20b0c8a99d59ac30b275a33"
MANIFEST_SHA256 = "3b68b0e18a60a7fdda2c4c5baa70c0e7951e10c575f65a9cd3e6000a75f0c277"
AUTHORIZATION = (
    WORKSPACE
    / "research/v10_economic_execution_authorization"
    / PREREGISTRATION_ID
    / "authorization.json"
)
REPORT_DIRECTORY = (
    WORKSPACE / "reports/v10_microstructure_economic_discovery" / PREREGISTRATION_ID
)


def serializable(value):
    if isinstance(value, Decimal):
        require(value.is_finite(), "nonfinite financial report value")
        return str(value)
    if isinstance(value, datetime):
        return timestamp(value)
    if isinstance(value, float):
        require(math.isfinite(value), "nonfinite report value")
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    return value


def encoded(value) -> bytes:
    return canonical(serializable(value)) + b"\n"


def verify_identity(manifest: dict) -> None:
    verify_manifest(manifest)
    require(
        manifest["preregistration_id"] == PREREGISTRATION_ID
        and manifest["definition_sha256"] == DEFINITION_SHA256,
        "unauthorized preregistration identity",
    )


def executor_identity(workspace: Path = WORKSPACE) -> dict:
    """Freeze all project Python dependencies and the runtime, not only the entry-point file."""
    require(sys.version_info[:2] == (3, 12), "executor requires Python 3.12")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    sources = {
        p.relative_to(workspace).as_posix(): sha256_file(p)
        for p in sorted((workspace / "src").rglob("*.py"))
    }
    require(bool(sources), "missing executor source identity")
    import decimal
    import random

    return {
        "source_commit": commit,
        "sources_sha256": sources,
        "python_version": sys.version,
        "platform": platform.platform(),
        "python_executable_sha256": sha256_file(sys.executable),
        "random_source_sha256": sha256_file(random.__file__),
        "decimal_version": decimal.__version__,
        "libmpdec_version": decimal.__libmpdec_version__,
        "ambient_decimal_context": {
            name: getattr(decimal.getcontext(), name)
            for name in ("prec", "rounding", "Emin", "Emax", "capitals", "clamp")
        },
        "decimal_traps": {
            signal.__name__: enabled
            for signal, enabled in decimal.getcontext().traps.items()
        },
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("binance-sdk-spot", "python-dotenv", "websockets")
        },
    }


def validate_authorization(receipt: dict, identity: dict) -> None:
    require(
        set(receipt)
        == {
            "preregistration_id",
            "definition_sha256",
            "scope",
            "authorized_by",
            "authorization_reference",
            "executor_identity",
            "executor_identity_sha256",
        },
        "invalid separate execution authorization schema",
    )
    require(
        receipt["preregistration_id"] == PREREGISTRATION_ID
        and receipt["definition_sha256"] == DEFINITION_SHA256
        and receipt["scope"] == "ONE_TIME_REAL_DISCOVERY"
        and receipt["authorized_by"] == "USER"
        and isinstance(receipt["authorization_reference"], str)
        and bool(receipt["authorization_reference"].strip()),
        "separate explicit authorization required",
    )
    require(
        receipt["executor_identity"] == identity
        and receipt["executor_identity_sha256"] == digest(identity),
        "executor/runtime identity changed",
    )


def refuse_existing(directory: Path) -> None:
    # Any entry, including an empty crash remnant, a symlink or a lone result, consumes the opportunity.
    require(
        not directory.exists() and not directory.is_symlink(),
        "existing reservation/result; execution permanently refused",
    )


def _sync_directory(directory: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def exclusive_write(path: Path, chunks: Iterable[bytes]) -> str:
    checksum = hashlib.sha256()
    # Never unlink a partial file on failure: the lock and partial bytes must survive a crash.
    with path.open("xb") as stream:
        for chunk in chunks:
            stream.write(chunk)
            checksum.update(chunk)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(path.parent)
    return checksum.hexdigest()


def reserve(directory: Path, ledger: dict) -> None:
    refuse_existing(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    # mkdir is atomic across competing processes. A crash after mkdir but before ledger write stays locked.
    directory.mkdir(exist_ok=False)
    _sync_directory(directory.parent)
    exclusive_write(directory / "reservation.json", [encoded(ledger)])


def _safe_paths() -> None:
    for path in (REPORT_DIRECTORY, AUTHORIZATION):
        require(path.is_relative_to(WORKSPACE), "noncanonical execution location")
        for component in (path, *path.parents):
            if component == WORKSPACE:
                break
            require(
                not component.is_symlink() and not component.is_junction(),
                "redirected execution path",
            )
    refuse_existing(REPORT_DIRECTORY)


def execute_once() -> dict:
    """Dormant: fails before raw access unless a later user task supplies the separate receipt.

    There is deliberately no command-line mode, receipt writer, authorization flag,
    configurable research root or output location. Implementation-only authorization
    does not create the separate receipt required here.
    """
    _safe_paths()
    require(
        AUTHORIZATION.is_file(),
        "separate explicit real-execution authorization is absent",
    )
    authorization_sha256 = sha256_file(AUTHORIZATION)
    receipt = strict_json(AUTHORIZATION.read_text(encoding="utf-8"))
    identity = executor_identity()
    validate_authorization(receipt, identity)
    manifest_path, manifest = load_manifest()
    verify_identity(manifest)
    require(
        sha256_file(manifest_path) == MANIFEST_SHA256, "frozen manifest bytes changed"
    )
    verify_dataset_binding(manifest)
    acquisition_path, acquisition = protocol()
    fresh = scan_readiness(acquisition, data_root=WORKSPACE / DATA_ROOT)
    require(fresh["status"] == "READY", "fresh acquisition READINESS must be READY")
    fresh["manifest_sha256"] = sha256_file(acquisition_path)
    rebound = bind_ready_dataset(
        fresh, bound_at=utc(manifest["definition"]["dataset"]["bound_at_utc"])
    )
    require(
        canonical(rebound) == canonical(manifest["definition"]["dataset"]),
        "fresh acquisition binding differs",
    )

    def integrity_check() -> None:
        require(
            sha256_file(manifest_path) == MANIFEST_SHA256, "frozen manifest changed"
        )
        verify_dataset_binding(manifest)
        require(
            sha256_file(AUTHORIZATION) == authorization_sha256,
            "authorization changed during execution",
        )
        validate_authorization(
            strict_json(AUTHORIZATION.read_text(encoding="utf-8")), identity
        )
        require(
            executor_identity() == identity, "executor/runtime changed during execution"
        )

    def load_samples():
        output = []
        for bound in manifest["definition"]["dataset"]["readiness"]["sessions"]:
            directory = (WORKSPACE / bound["manifest_path"]).parent
            session = strict_json(
                (directory / "session.manifest.json").read_text(encoding="utf-8")
            )
            records = _merged_records(
                directory / "depth.jsonl",
                directory / "aggtrades.jsonl",
                session_id=bound["session_id"],
            )
            output.append(
                sample_session(
                    records,
                    session_id=bound["session_id"],
                    started_at=utc(bound["started_at_utc"]),
                    ended_at=utc(bound["ended_at_utc"]),
                    max_levels=session["streams"]["depth"]["max_levels_per_side"],
                )
            )
        return tuple(output)

    ledger = {
        "state": "RESERVED_IRREVOCABLY",
        "dataset_status": "CONSUMED_DISCOVERY_EVIDENCE",
        "authorization": receipt,
        "manifest_file_sha256": MANIFEST_SHA256,
        "dataset_binding": manifest["definition"]["dataset"],
        "executor_identity": identity,
    }
    try:
        # Sole production orchestration path: no caller-selected roots or callbacks.
        # mkdir permanently consumes this opportunity before any sample/target construction.
        reserve(REPORT_DIRECTORY, ledger)
        integrity_check()
        sessions = load_samples()
        report, predictions, null = build_diagnostic(sessions)
        integrity_check()
        # Do not publish a classification before post-computation binding checks pass.
        audit = (
            encoded({"session_id": session.session_id, **asdict(row)})
            for session in sessions
            for row in session.rows
        )
        hashes = {
            "samples.jsonl": exclusive_write(REPORT_DIRECTORY / "samples.jsonl", audit),
            "predictions.json": exclusive_write(
                REPORT_DIRECTORY / "predictions.json", [encoded(predictions)]
            ),
            "null.json": exclusive_write(
                REPORT_DIRECTORY / "null.json", [encoded(null)]
            ),
        }
        report = {
            **report,
            "preregistration_id": PREREGISTRATION_ID,
            "definition_sha256": DEFINITION_SHA256,
            "execution_binding": ledger,
            "artifact_sha256": hashes,
        }
        report_hash = exclusive_write(
            REPORT_DIRECTORY / "report.json", [encoded(report)]
        )
        exclusive_write(
            REPORT_DIRECTORY / "seal.json",
            [
                encoded(
                    {
                        "report_sha256": report_hash,
                        "artifact_sha256": hashes,
                        "status": "SEALED",
                    }
                )
            ],
        )
        return {"report_sha256": report_hash, "status": "SEALED"}
    except Exception as exc:
        # No result/classification is returned and no lock is removed, including on an integrity failure.
        raise RuntimeError(
            f"INTEGRITY_FAILURE or interrupted execution; reservation retained: {exc}"
        ) from exc
