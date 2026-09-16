"""Inspect, replay and atomically import downloaded V10 overnight session ZIPs."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from src.cli.import_v9_github_artifacts import _safe_parts
from src.cli.validate_v10_overnight import (
    DURATION_SECONDS, PREREGISTRATION_ID, PREREGISTRATION_SHA256, ROOT, VALIDATION,
    artifact_name, protocol,
)
from src.microstructure.ci import reject_ci_research_path
from src.microstructure.v10 import session_paths
from src.research.v10_collection_preregistration import WORKSPACE, canonical, strict_json, utc
from src.research.v10_collection_readiness import ARTIFACTS, require, scan_readiness


FILES = (*ARTIFACTS, VALIDATION)
MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_BYTES = 64 * 1024**3


@dataclass(frozen=True)
class InspectedArchive:
    session_id: str
    session: dict
    closure: dict
    validation: dict
    members: dict[str, str]


def _read_json(archive: zipfile.ZipFile, member: str) -> dict:
    require(archive.getinfo(member).file_size <= MAX_METADATA_BYTES, "oversized JSON metadata")
    return strict_json(archive.read(member).decode("utf-8"))


def inspect_archive(archive: zipfile.ZipFile, manifest: dict) -> InspectedArchive:
    """Validate all member paths and small metadata before creating any staging files."""
    infos = archive.infolist()
    require(len(infos) <= 64, "too many ZIP members for one session")
    seen = set()
    files = {}
    parents = set()
    directories = []
    total_bytes = 0
    for info in infos:
        require(info.orig_filename == info.filename, "unsafe truncated ZIP filename")
        parts = _safe_parts(info.filename, directory=info.is_dir())
        require(all(not part.endswith((".", " ")) for part in parts), "unsafe Windows ZIP path")
        folded = "/".join(parts).casefold()
        require(folded not in seen, "duplicate/case-colliding ZIP member")
        seen.add(folded)
        mode = stat.S_IFMT(info.external_attr >> 16)
        require(not info.flag_bits & 1, "encrypted ZIP entries are forbidden")
        require(mode in ((0, stat.S_IFDIR) if info.is_dir() else (0, stat.S_IFREG)),
                "ZIP links and non-regular entries are forbidden")
        require("ci.provenance.json" not in parts and not any(
            tuple(part.lower() for part in parts[index:index + 2]) == ("data", "ci")
            for index in range(len(parts) - 1)), "CI_ONLY artifact is forbidden")
        if info.is_dir():
            directories.append(parts)
            continue
        require(parts[-1] in FILES, f"unexpected ZIP file: {info.filename}")
        require(parts[-1] not in files, "artifact contains multiple sessions or duplicate required files")
        files[parts[-1]] = info.filename
        parents.add(parts[:-1])
        total_bytes += info.file_size
    require(set(files) == set(FILES), f"missing required files: {sorted(set(FILES) - set(files))}")
    require(len(parents) == 1, "required files must belong to exactly one session directory")
    require(total_bytes <= MAX_ARCHIVE_BYTES, "ZIP exceeds 64 GiB safety limit")
    session = _read_json(archive, files["session.manifest.json"])
    closure = _read_json(archive, files["closure.summary.json"])
    validation = _read_json(archive, files[VALIDATION])
    for value in (session, closure):
        require(value.get("research_eligibility") == "UNASSESSED", "CI_ONLY or modified session eligibility is forbidden")
    session_id = session["session_id"]
    started = utc(session["started_at_utc"])
    require(session_id == started.strftime("%Y%m%dT%H%M%S%fZ"), "session identity/start mismatch")
    require(started > utc(manifest["definition"]["prospective_cutoff_utc"]), "session must start strictly after frozen cutoff")
    relative = session_paths(Path(), started_at=started).directory.parts
    parent = next(iter(parents))
    require(parent in ((), (session_id,), relative, (*ROOT.parts, *relative)), "noncanonical ZIP session layout")
    require(all(parts == parent[:len(parts)] for parts in directories), "unrelated ZIP directory")
    require(closure.get("session_id") == session_id, "closure session identity mismatch")
    require(validation.get("schema_version") == "V10_OVERNIGHT_ACQUISITION_VALIDATION_1", "invalid validation schema")
    require(validation.get("preregistration_id") == PREREGISTRATION_ID
            and validation.get("preregistration_sha256") == PREREGISTRATION_SHA256,
            "wrong preregistration ID/hash in validation metadata")
    result = validation["result"]
    require(result.get("session_id") == session_id, "validation session identity mismatch")
    require(result.get("status") == "PASS" and result.get("research_eligibility") == "ELIGIBLE",
            "CI_ONLY or failed acquisition validation is ineligible")
    require(type(result.get("session_number")) is int and result["session_number"] in (1, 2, 3),
            "invalid overnight session number")
    require(result.get("artifact_name") == artifact_name(
        validation["github_run_id"], validation["github_run_attempt"], session_id),
        "GitHub artifact run/attempt/session identity mismatch")
    proof = validation["preflight"]
    require(proof.get("preregistration_id") == PREREGISTRATION_ID
            and proof.get("preregistration_sha256") == PREREGISTRATION_SHA256
            and proof.get("prospective_cutoff_utc") == manifest["definition"]["prospective_cutoff_utc"],
            "preflight preregistration identity mismatch")
    require(utc(proof["prospective_cutoff_utc"]) < utc(proof["checked_at_utc"]) <= started,
            "invalid preflight cutoff/session chronology")
    require(proof.get("source_commit_sha") == session.get("source_commit_sha"), "source commit identity mismatch")
    require(type(session.get("requested_duration_seconds")) is int
            and session["requested_duration_seconds"] == DURATION_SECONDS
            and type(proof.get("duration_seconds")) is int
            and proof["duration_seconds"] == DURATION_SECONDS, "not a 10800-second overnight session")
    depth = session["streams"]["depth"]
    require(type(depth.get("snapshot_limit")) is int and depth["snapshot_limit"] == 5000
            and type(depth.get("max_levels_per_side")) is int and depth["max_levels_per_side"] == 5000,
            "overnight depth limits must both be 5000")
    for value in (closure, result, proof):
        require(value.get("predictive_outcomes_evaluated") is False, "predictive evaluation restriction missing")
    return InspectedArchive(session_id, session, closure, validation, files)


def _directory(path: Path, *, create: bool = False) -> None:
    """Reject redirecting local path components before any file publication."""
    for current in reversed((path, *path.parents)):
        require(not current.is_symlink() and not current.is_junction(), f"local path redirects: {current}")
        if current.exists():
            require(current.is_dir(), f"local directory path occupied by a file: {current}")
        elif create:
            current.mkdir()


def _extract(archive: zipfile.ZipFile, inspected: InspectedArchive, staging: Path) -> Path:
    total = sum(archive.getinfo(member).file_size for member in inspected.members.values())
    require(total <= shutil.disk_usage(staging).free, "insufficient staging disk space")
    directory = staging / inspected.session_id
    directory.mkdir()
    for name in FILES:
        info = archive.getinfo(inspected.members[name])
        with archive.open(info) as source, (directory / name).open("xb") as target:
            copied = 0
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                copied += len(chunk)
                require(copied <= info.file_size, "ZIP member exceeded declared size")
                target.write(chunk)
            require(copied == info.file_size, "truncated ZIP member")
            target.flush()
            os.fsync(target.fileno())
    return directory


def _portable_readiness(report: dict) -> dict:
    result = json.loads(json.dumps(report))
    # Only the machine-local manifest path differs after transport. No stored path is followed.
    for session in result["sessions"]:
        session.pop("manifest_path", None)
    return result


def validate_staged(staging: Path, inspected: InspectedArchive, manifest: dict, *, now=None) -> dict:
    directory = staging / inspected.session_id
    for name, expected in (("session.manifest.json", inspected.session),
                           ("closure.summary.json", inspected.closure), (VALIDATION, inspected.validation)):
        require(canonical(strict_json((directory / name).read_text(encoding="utf-8"))) == canonical(expected),
                f"ZIP metadata changed after inspection: {name}")
    report = scan_readiness(manifest, data_root=staging, now=now)
    require(report["eligible_sessions"] == 1 and len(report["sessions"]) == 1,
            "frozen integrity/replay rejection: " + json.dumps(report["sessions"]))
    eligible = report["sessions"][0]
    require(eligible["session_id"] == inspected.session_id
            and eligible["eligible_microseconds"] == DURATION_SECONDS * 1_000_000, "session identity/duration mismatch")
    metadata = inspected.validation
    require(canonical(_portable_readiness(report)) == canonical(_portable_readiness(metadata["readiness"])),
            "stored readiness metadata differs from fresh deterministic replay")
    closure = inspected.closure
    expected = {
        "started_at_utc": inspected.session["started_at_utc"], "ended_at_utc": closure["ended_at_utc"],
        "eligible_hours": "3", "utc_start_date": utc(inspected.session["started_at_utc"]).date().isoformat(),
        "depth_count": closure["live_integrity"]["depth"]["diff_events"],
        "aggtrade_count": closure["live_integrity"]["aggtrades"]["raw_events"],
        "causal_context_count": closure["offline_replay"]["causal_trade_count"],
        "artifact_hashes": eligible["artifact_sha256"],
        "deterministic_dual_stream_replay": "PASS", "causal_trade_l2_merge": "PASS",
        "synchronized_timeline_sha256": eligible["synchronized_timeline_sha256"],
        "causal_contexts_sha256": eligible["causal_contexts_sha256"],
    }
    for key, value in expected.items():
        require(canonical(metadata["result"].get(key)) == canonical(value), f"validation metadata mismatch: {key}")
    return eligible


def _same_bytes(first: Path, second: Path) -> bool:
    if not second.is_file() or second.is_symlink() or first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        while True:
            chunk = left.read(1024 * 1024)
            if chunk != right.read(1024 * 1024):
                return False
            if not chunk:
                return True


def _atomic_publish(source: Path, destination: Path) -> None:
    """Publish the complete directory at once, atomically refusing any existing destination."""
    if os.name == "nt":
        os.rename(source, destination)  # Windows rename never replaces an existing destination.
        return
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        rename = getattr(libc, "renameat2", None)
        require(rename is not None, "atomic no-replace directory rename is unavailable")
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
        if result != 0:
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code), str(destination))
        return
    raise ValueError("atomic no-replace publication requires Windows or Linux")


@contextmanager
def _lock(parent: Path):
    path = parent / ".v10-import.lock"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError("V10 importer lock exists; another import or interrupted import requires inspection") from exc
    try:
        os.close(descriptor)
        yield
    finally:
        path.unlink()


def _existing(root: Path, session_id: str) -> list[Path]:
    found = set()
    if root.exists():
        for path in root.rglob("session.manifest.json"):
            _directory(path.parent)
            require(not path.is_symlink(), "local session manifest is a symlink")
            value = strict_json(path.read_text(encoding="utf-8"))
            if value.get("session_id") == session_id or path.parent.name == session_id:
                found.add(path.parent)
    return sorted(found)


def import_artifacts(zip_paths, *, workspace: Path = WORKSPACE, now=None, progress=None) -> dict:
    """Import each artifact independently; the CLI performs the final canonical READINESS run."""
    _, manifest = protocol()
    workspace = workspace.absolute()
    root = workspace / ROOT
    reject_ci_research_path(root)
    _directory(root)
    result = {"imported_session_ids": [], "identical_duplicates_skipped": [], "rejected_artifacts": []}
    total = len(zip_paths) if hasattr(zip_paths, "__len__") else "?"
    for index, value in enumerate(zip_paths, 1):
        path = Path(value).absolute()
        def announce(stage):
            if progress is not None:
                progress(f"[{index}/{total}] {stage}")
        announce("inspecting")
        try:
            require(path.name.startswith("v10-microstructure-overnight-") and path.suffix.lower() == ".zip",
                    "filename must match v10-microstructure-overnight-*.zip")
            with zipfile.ZipFile(path) as archive:
                inspected = inspect_archive(archive, manifest)
                # Inspect first. Staging is on the destination filesystem, outside readiness discovery.
                _directory(root.parent, create=True)
                with _lock(root.parent), tempfile.TemporaryDirectory(prefix=".v10-import-", dir=root.parent) as temporary:
                    staging = Path(temporary)
                    directory = _extract(archive, inspected, staging)
                    announce("replaying")
                    validate_staged(staging, inspected, manifest, now=now)
                    announce("validated")
                    protocol()  # Fail closed if the frozen protocol/source changed during replay.
                    target = session_paths(root, started_at=utc(inspected.session["started_at_utc"])).directory
                    existing = _existing(root, inspected.session_id)
                    if existing or target.exists():
                        require(existing == [target] and {item.name for item in target.iterdir()} == set(FILES)
                                and all(_same_bytes(directory / name, target / name) for name in FILES),
                                f"conflicting local duplicate session ID: {inspected.session_id}")
                        result["identical_duplicates_skipped"].append(inspected.session_id)
                        continue
                    _directory(target.parent, create=True)
                    # Staging owns all five files; no partially closed session is ever published.
                    _atomic_publish(directory, target)
                    result["imported_session_ids"].append(inspected.session_id)
        except (OSError, ValueError, KeyError, TypeError, ArithmeticError, AttributeError, RuntimeError,
                zipfile.BadZipFile, zipfile.LargeZipFile, zlib.error) as exc:
            result["rejected_artifacts"].append({"path": str(path), "reason": str(exc)})
            announce("rejected")
    return result


def run_readiness(*, workers: int | None = None) -> dict:
    command = [sys.executable, "-m", "src.cli.run_v10_collection", "--mode", "READINESS"]
    if workers is not None:
        command.extend(("--workers", str(workers)))
    completed = subprocess.run(command, cwd=WORKSPACE, capture_output=True, text=True, check=False)
    require(completed.returncode in (0, 2), "READINESS failed: " + (completed.stdout + completed.stderr).strip())
    report = strict_json(completed.stdout)
    require(report.get("status") == ("READY" if completed.returncode == 0 else "NOT_READY"),
            "READINESS status/exit code mismatch")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_paths", type=Path, nargs="+")
    parser.add_argument("--workers", type=int, default=None,
                        help="final readiness processes (default: up to 4 CPUs; 1: serial)")
    args = parser.parse_args(argv)
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be a positive integer")
    def progress(message):
        print(message, file=sys.stderr, flush=True)
    try:
        result = import_artifacts(args.zip_paths, progress=progress)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"V10 IMPORT INTEGRITY_FAILURE: {exc}")
        return 1
    try:
        progress("FINAL READINESS validating")
        readiness = run_readiness(workers=args.workers)
        result.update(eligible_sessions=readiness["eligible_sessions"], eligible_hours=readiness["eligible_hours"],
                      eligible_utc_dates=readiness["eligible_utc_dates"], readiness_status=readiness["status"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result.update(readiness_status="INTEGRITY_FAILURE", readiness_error=str(exc))
    progress(f"FINAL READINESS {result['readiness_status']}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["rejected_artifacts"] or result["readiness_status"] == "INTEGRITY_FAILURE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
