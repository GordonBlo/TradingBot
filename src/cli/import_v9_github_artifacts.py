"""Safely validate and import GitHub V9 V2 prospective session artifacts."""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from src.cli.validate_v9_l2_collection import (
    ACTIVE_MANIFEST,
    ACTIVE_MANIFEST_RUN_ID,
    ACTIVE_MANIFEST_SHA256,
    CANONICAL_DATA_ROOT,
    CANONICAL_FEATURE_ROOT,
    EXPECTED_DERIVED_FILES,
    CollectionValidationError,
    load_active_manifest,
    validate_session,
)
from src.research.v9_l2_readiness import ReadinessReport, scan_session_readiness


ARTIFACT_NAME_PATTERN = "v9-l2-overnight-*.zip"
SESSION_ID_PATTERN = re.compile(r"\d{8}T\d{12}Z")
OPTIONAL_ARCHIVE_PREFIX = ("data", "orderbook", "v9")
REQUIRED_STEP_OUTCOMES = {
    "setup_python",
    "dependencies",
    "preflight",
    "collect",
    "identify",
    "features",
    "validate",
}


class ArtifactImportError(RuntimeError):
    """Raised when a GitHub artifact cannot be safely imported."""


@dataclass(frozen=True)
class ArtifactLayout:
    session_id: str
    relative_files: tuple[PurePosixPath, ...]
    raw_relative: PurePosixPath
    summary_relative: PurePosixPath
    validation_relative: PurePosixPath
    feature_directory: PurePosixPath


@dataclass(frozen=True)
class RejectedArtifact:
    path: str
    reason: str


@dataclass(frozen=True)
class ImportResult:
    imported_session_ids: tuple[str, ...]
    skipped_identical_session_ids: tuple[str, ...]
    rejected_artifacts: tuple[RejectedArtifact, ...]
    readiness: ReadinessReport


def _safe_parts(name: str, *, directory: bool) -> tuple[str, ...]:
    if (
        not name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
        or "\\" in name
        or name.startswith("/")
    ):
        raise ArtifactImportError(f"unsafe ZIP member path: {name!r}")
    raw_parts = name.split("/")
    if directory and raw_parts[-1] == "":
        raw_parts = raw_parts[:-1]
    if not raw_parts or any(part in ("", ".", "..") for part in raw_parts):
        raise ArtifactImportError(f"unsafe ZIP member path: {name!r}")
    if any(":" in part for part in raw_parts):
        raise ArtifactImportError(f"unsafe ZIP member path: {name!r}")
    path = PurePosixPath(*raw_parts)
    if path.is_absolute() or tuple(path.parts) != tuple(raw_parts):
        raise ArtifactImportError(f"unsafe ZIP member path: {name!r}")
    return tuple(raw_parts)


def _inspect_members(archive: zipfile.ZipFile) -> dict[PurePosixPath, zipfile.ZipInfo]:
    raw_files: list[tuple[tuple[str, ...], zipfile.ZipInfo]] = []
    seen_names: set[str] = set()
    for info in archive.infolist():
        if info.filename in seen_names:
            raise ArtifactImportError(f"duplicate ZIP member: {info.filename}")
        seen_names.add(info.filename)
        parts = _safe_parts(info.filename, directory=info.is_dir())
        if info.is_dir():
            continue
        if info.flag_bits & 0x1:
            raise ArtifactImportError(f"encrypted ZIP member is not allowed: {info.filename}")
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(unix_mode)
        if file_type not in (0, stat.S_IFREG):
            raise ArtifactImportError(f"non-regular ZIP member is not allowed: {info.filename}")
        raw_files.append((parts, info))
    if not raw_files:
        raise ArtifactImportError("ZIP contains no files")

    prefix_flags = {
        parts[: len(OPTIONAL_ARCHIVE_PREFIX)] == OPTIONAL_ARCHIVE_PREFIX
        for parts, _ in raw_files
    }
    if len(prefix_flags) != 1:
        raise ArtifactImportError("ZIP mixes prefixed and unprefixed research paths")
    strip_prefix = prefix_flags == {True}

    members: dict[PurePosixPath, zipfile.ZipInfo] = {}
    casefolded: set[str] = set()
    for parts, info in raw_files:
        relative_parts = parts[len(OPTIONAL_ARCHIVE_PREFIX) :] if strip_prefix else parts
        if not relative_parts:
            raise ArtifactImportError(f"ZIP member has no research-relative path: {info.filename}")
        relative = PurePosixPath(*relative_parts)
        folded = relative.as_posix().casefold()
        if relative in members or folded in casefolded:
            raise ArtifactImportError(f"colliding ZIP member path: {relative.as_posix()}")
        members[relative] = info
        casefolded.add(folded)
    return members


def _layout(members: dict[PurePosixPath, zipfile.ZipInfo]) -> ArtifactLayout:
    raw_candidates: list[tuple[str, PurePosixPath]] = []
    for relative in members:
        parts = relative.parts
        if len(parts) != 5 or parts[0] != "BTCUSDC" or relative.suffix != ".jsonl":
            continue
        if not all(part.isdigit() for part in parts[1:4]):
            continue
        session_id = relative.stem
        if SESSION_ID_PATTERN.fullmatch(session_id):
            raw_candidates.append((session_id, relative))
    if len(raw_candidates) != 1:
        raise ArtifactImportError(
            f"artifact must contain exactly one raw session; found {len(raw_candidates)}"
        )

    session_id, raw_relative = raw_candidates[0]
    summary_relative = raw_relative.with_suffix(".summary.json")
    feature_directory = PurePosixPath("features") / session_id
    validation_relative = PurePosixPath("validation") / f"{session_id}.json"
    required = {
        raw_relative,
        summary_relative,
        validation_relative,
        feature_directory / "validation_report.json",
        *(
            feature_directory / filename
            for filename in EXPECTED_DERIVED_FILES.values()
        ),
    }
    actual = set(members)
    if actual != required:
        missing = sorted(path.as_posix() for path in required - actual)
        unexpected = sorted(path.as_posix() for path in actual - required)
        raise ArtifactImportError(
            f"artifact file set mismatch; missing={missing}, unexpected={unexpected}"
        )
    return ArtifactLayout(
        session_id=session_id,
        relative_files=tuple(sorted(required, key=lambda path: path.as_posix())),
        raw_relative=raw_relative,
        summary_relative=summary_relative,
        validation_relative=validation_relative,
        feature_directory=feature_directory,
    )


def _extract_to_staging(
    archive: zipfile.ZipFile,
    members: dict[PurePosixPath, zipfile.ZipInfo],
    layout: ArtifactLayout,
    staging_workspace: Path,
) -> Path:
    staging_root = staging_workspace / CANONICAL_DATA_ROOT
    staging_root.mkdir(parents=True, exist_ok=False)
    required_bytes = sum(members[path].file_size for path in layout.relative_files)
    if required_bytes > shutil.disk_usage(staging_workspace).free:
        raise ArtifactImportError("insufficient temporary disk space for artifact inspection")
    for relative in layout.relative_files:
        destination = staging_root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.resolve().is_relative_to(staging_root.resolve()):
            raise ArtifactImportError("staging path escaped the canonical staging root")
        try:
            with archive.open(members[relative], "r") as source, destination.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
            raise ArtifactImportError(
                f"could not safely read ZIP member {relative.as_posix()}: {exc}"
            ) from exc
    return staging_root


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactImportError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ArtifactImportError(f"{label} must be a JSON object")
    return value


def _canonical_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ArtifactImportError(f"{label} path is missing")
    try:
        parts = _safe_parts(value, directory=False)
    except ArtifactImportError as exc:
        raise ArtifactImportError(f"{label} path is invalid") from exc
    return PurePosixPath(*parts).as_posix()


def _decimal(value: Any, *, label: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ArtifactImportError(f"{label} is not numeric") from exc
    if not parsed.is_finite():
        raise ArtifactImportError(f"{label} must be finite")
    return parsed


def _validate_github_metadata(
    metadata: dict[str, Any],
    validated: dict[str, Any],
    layout: ArtifactLayout,
    *,
    archive_stem: str,
) -> None:
    session_id = layout.session_id
    expected_scalars = {
        "schema_version": 1,
        "purpose": "V9_V2_PROSPECTIVE_COLLECTION_ONLY",
        "status": "PASS",
        "classification": "ELIGIBLE",
        "research_eligible": True,
        "session_id": session_id,
        "requested_duration_seconds": 10_800,
        "predictive_outcomes_evaluated": False,
        "forward_targets_formed": 0,
        "blind_holdout_accessed": False,
    }
    for key, expected in expected_scalars.items():
        if metadata.get(key) != expected:
            raise ArtifactImportError(f"validation metadata field mismatch: {key}")

    preregistration = metadata.get("preregistration")
    expected_preregistration = validated["preregistration"]
    if not isinstance(preregistration, dict) or (
        preregistration.get("run_id") != ACTIVE_MANIFEST_RUN_ID
        or preregistration.get("manifest_sha256") != ACTIVE_MANIFEST_SHA256
        or preregistration != expected_preregistration
    ):
        raise ArtifactImportError("validation metadata preregistration identity mismatch")

    binding = metadata.get("artifact_binding")
    expected_binding = validated["artifact_binding"]
    if not isinstance(binding, dict) or binding != expected_binding:
        raise ArtifactImportError("validation metadata artifact binding mismatch")
    expected_binding_paths = {
        "raw_event_log": (CANONICAL_DATA_ROOT / Path(*layout.raw_relative.parts)).as_posix(),
        "closure_summary": (
            CANONICAL_DATA_ROOT / Path(*layout.summary_relative.parts)
        ).as_posix(),
        "features_1s": (
            CANONICAL_DATA_ROOT
            / Path(*layout.feature_directory.parts)
            / EXPECTED_DERIVED_FILES["features_1s"]
        ).as_posix(),
    }
    for key, expected in expected_binding_paths.items():
        if _canonical_text(binding.get(key), label=f"artifact binding {key}") != expected:
            raise ArtifactImportError(f"validation metadata path mismatch: {key}")

    for key in (
        "collector_integrity",
        "feature_integrity",
        "derived_file_sha256",
        "deterministic_replay_hash_check",
        "feature_provenance",
        "public_data_only",
    ):
        if metadata.get(key) != validated.get(key):
            raise ArtifactImportError(f"validation metadata mismatch: {key}")
    if metadata.get("started_at_utc") != validated.get("started_at_utc") or metadata.get(
        "ended_at_utc"
    ) != validated.get("ended_at_utc"):
        raise ArtifactImportError("validation metadata session boundary mismatch")
    if _decimal(metadata.get("eligible_hours"), label="eligible hours") != _decimal(
        validated.get("eligible_hours"), label="validated eligible hours"
    ) or metadata.get("eligible_utc_dates") != validated.get("eligible_utc_dates"):
        raise ArtifactImportError("validation metadata eligibility summary mismatch")

    outcomes = metadata.get("required_step_outcomes")
    if (
        not isinstance(outcomes, dict)
        or set(outcomes) != REQUIRED_STEP_OUTCOMES
        or any(value != "success" for value in outcomes.values())
    ):
        raise ArtifactImportError("GitHub session did not pass every required workflow step")
    run_id = str(metadata.get("github_run_id", ""))
    run_attempt = str(metadata.get("github_run_attempt", ""))
    github_sha = str(metadata.get("github_sha", ""))
    session_number = metadata.get("session_number")
    artifact_name = metadata.get("artifact_name")
    expected_artifact_name = (
        f"v9-l2-overnight-{run_id}-attempt-{run_attempt}-{session_id}"
    )
    if (
        type(session_number) is not int
        or not 1 <= session_number <= 4
        or not run_id.isdigit()
        or int(run_id) < 1
        or not run_attempt.isdigit()
        or int(run_attempt) < 1
        or not re.fullmatch(r"[0-9a-f]{40}", github_sha)
        or artifact_name != expected_artifact_name
        or archive_stem != expected_artifact_name
    ):
        raise ArtifactImportError("GitHub workflow provenance metadata is invalid")


def _stage_and_validate(
    zip_path: Path, staging_workspace: Path, manifest_source: Path
) -> ArtifactLayout:
    manifest_target = staging_workspace / ACTIVE_MANIFEST
    manifest_target.parent.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(manifest_source, manifest_target)
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            members = _inspect_members(archive)
            layout = _layout(members)
            staging_root = _extract_to_staging(
                archive, members, layout, staging_workspace
            )
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ArtifactImportError(f"malformed or unreadable ZIP: {exc}") from exc

    raw_path = staging_root.joinpath(*layout.raw_relative.parts)
    summary_path = staging_root.joinpath(*layout.summary_relative.parts)
    try:
        validated = validate_session(
            raw_path=raw_path,
            summary_path=summary_path,
            workspace=staging_workspace,
        )
    except (CollectionValidationError, OSError, ValueError, KeyError, TypeError) as exc:
        raise ArtifactImportError(f"V9 V2 session validation failed: {exc}") from exc
    metadata_path = staging_root.joinpath(*layout.validation_relative.parts)
    metadata = _json_object(metadata_path, label="GitHub validation metadata")
    _validate_github_metadata(
        metadata, validated, layout, archive_stem=zip_path.stem
    )
    return layout


def _files_equal(first: Path, second: Path) -> bool:
    return first.is_file() and second.is_file() and filecmp.cmp(first, second, shallow=False)


def _ensure_canonical_root(workspace: Path) -> Path:
    current = workspace
    resolved_workspace = workspace.resolve()
    for part in CANONICAL_DATA_ROOT.parts:
        candidate = current / part
        if candidate.is_symlink():
            raise ArtifactImportError(
                f"canonical research path contains a symbolic link: {candidate}"
            )
        if candidate.exists():
            if not candidate.is_dir():
                raise ArtifactImportError(
                    f"canonical research path is not a directory: {candidate}"
                )
        else:
            candidate.mkdir()
        if not candidate.resolve().is_relative_to(resolved_workspace):
            raise ArtifactImportError("canonical research directory escaped the workspace")
        current = candidate
    return current


def _ensure_relative_directory(root: Path, relative: PurePosixPath) -> Path:
    current = root
    resolved_root = root.resolve()
    for part in relative.parts:
        candidate = current / part
        if candidate.is_symlink():
            raise ArtifactImportError(
                f"canonical session path contains a symbolic link: {candidate}"
            )
        if candidate.exists():
            if not candidate.is_dir():
                raise ArtifactImportError(
                    f"canonical session path is not a directory: {candidate}"
                )
        else:
            candidate.mkdir()
        if not candidate.resolve().is_relative_to(resolved_root):
            raise ArtifactImportError("canonical session directory escaped the research root")
        current = candidate
    return current


def _existing_session_files(root: Path, layout: ArtifactLayout) -> set[PurePosixPath]:
    session_id = layout.session_id
    found: set[PurePosixPath] = set()
    if root.exists():
        for path in (root / "BTCUSDC").rglob(f"{session_id}.jsonl") if (
            root / "BTCUSDC"
        ).exists() else ():
            if path.is_file():
                found.add(PurePosixPath(path.relative_to(root).as_posix()))
        for path in (root / "BTCUSDC").rglob(f"{session_id}.summary.json") if (
            root / "BTCUSDC"
        ).exists() else ():
            if path.is_file():
                found.add(PurePosixPath(path.relative_to(root).as_posix()))
        feature_directory = root / "features" / session_id
        if feature_directory.exists():
            found.update(
                PurePosixPath(path.relative_to(root).as_posix())
                for path in feature_directory.rglob("*")
                if path.is_file()
            )
        validation_path = root / "validation" / f"{session_id}.json"
        if validation_path.is_file():
            found.add(PurePosixPath(validation_path.relative_to(root).as_posix()))
    return found


def _publish_new_session(
    staging_root: Path, canonical_root: Path, layout: ArtifactLayout
) -> None:
    required_bytes = sum(
        staging_root.joinpath(*relative.parts).stat().st_size
        for relative in layout.relative_files
    )
    if required_bytes > shutil.disk_usage(canonical_root).free:
        raise ArtifactImportError("insufficient disk space for atomic import")

    ordered = [
        relative
        for relative in layout.relative_files
        if relative != layout.summary_relative
    ] + [layout.summary_relative]
    prepared: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for relative in ordered:
            source = staging_root.joinpath(*relative.parts)
            destination = canonical_root.joinpath(*relative.parts)
            _ensure_relative_directory(
                canonical_root, PurePosixPath(*relative.parts[:-1])
            )
            temporary = destination.parent / (
                f".{destination.name}.import-{uuid.uuid4().hex}.tmp"
            )
            with source.open("rb") as source_stream, temporary.open("xb") as target_stream:
                shutil.copyfileobj(source_stream, target_stream, length=1024 * 1024)
                target_stream.flush()
                os.fsync(target_stream.fileno())
            if not _files_equal(source, temporary):
                raise ArtifactImportError("atomic staging copy differs from validated artifact")
            prepared.append((temporary, destination))

        for temporary, destination in prepared:
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise ArtifactImportError(
                    f"destination appeared during atomic import: {destination}"
                ) from exc
            published.append(destination)
            temporary.unlink()
    except Exception:
        for destination in reversed(published):
            destination.unlink(missing_ok=True)
        raise
    finally:
        for temporary, _ in prepared:
            temporary.unlink(missing_ok=True)


def _import_staged(
    staging_workspace: Path, workspace: Path, layout: ArtifactLayout
) -> str:
    staging_root = staging_workspace / CANONICAL_DATA_ROOT
    canonical_root = _ensure_canonical_root(workspace)
    existing = _existing_session_files(canonical_root, layout)
    expected = set(layout.relative_files)
    if existing:
        if existing != expected or any(
            not _files_equal(
                staging_root.joinpath(*relative.parts),
                canonical_root.joinpath(*relative.parts),
            )
            for relative in layout.relative_files
        ):
            raise ArtifactImportError(
                f"conflicting local duplicate session ID: {layout.session_id}"
            )
        return "SKIPPED_IDENTICAL"
    if any(canonical_root.joinpath(*relative.parts).exists() for relative in expected):
        raise ArtifactImportError(
            f"canonical destination conflict for session ID: {layout.session_id}"
        )
    _publish_new_session(staging_root, canonical_root, layout)
    return "IMPORTED"


def import_artifacts(
    zip_paths: Sequence[str | Path], *, workspace: Path = Path(".")
) -> ImportResult:
    workspace = workspace.resolve()
    manifest_path, manifest, _ = load_active_manifest(workspace)
    imported: list[str] = []
    skipped: list[str] = []
    rejected: list[RejectedArtifact] = []

    for path_value in zip_paths:
        zip_path = Path(path_value).resolve()
        try:
            if not zip_path.is_file():
                raise ArtifactImportError("artifact ZIP does not exist")
            if not zip_path.match(ARTIFACT_NAME_PATTERN):
                raise ArtifactImportError(
                    f"artifact filename must match {ARTIFACT_NAME_PATTERN}"
                )
            with tempfile.TemporaryDirectory(prefix="v9-l2-import-") as temporary:
                staging_workspace = Path(temporary).resolve()
                layout = _stage_and_validate(
                    zip_path, staging_workspace, manifest_path
                )
                outcome = _import_staged(staging_workspace, workspace, layout)
            if outcome == "IMPORTED":
                imported.append(layout.session_id)
            else:
                skipped.append(layout.session_id)
        except (ArtifactImportError, OSError, ValueError, TypeError) as exc:
            rejected.append(RejectedArtifact(str(zip_path), str(exc)))

    readiness = scan_session_readiness(
        manifest,
        data_root=workspace / CANONICAL_DATA_ROOT,
        feature_root=workspace / CANONICAL_FEATURE_ROOT,
        workspace=workspace,
    )
    return ImportResult(
        imported_session_ids=tuple(imported),
        skipped_identical_session_ids=tuple(skipped),
        rejected_artifacts=tuple(rejected),
        readiness=readiness,
    )


def _utc_dates(readiness: ReadinessReport) -> list[str]:
    dates = {
        datetime.fromisoformat(session.started_at_utc.replace("Z", "+00:00"))
        .date()
        .isoformat()
        for session in readiness.sessions
        if session.classification == "ELIGIBLE"
    }
    return sorted(dates)


def _print_result(result: ImportResult) -> None:
    print("IMPORTED SESSION IDS")
    for session_id in result.imported_session_ids:
        print(f"- {session_id}")
    if not result.imported_session_ids:
        print("- none")
    print("SKIPPED IDENTICAL DUPLICATES")
    for session_id in result.skipped_identical_session_ids:
        print(f"- {session_id}")
    if not result.skipped_identical_session_ids:
        print("- none")
    print("REJECTED ARTIFACTS")
    for rejected in result.rejected_artifacts:
        print(f"- {rejected.path}: {rejected.reason}")
    if not result.rejected_artifacts:
        print("- none")

    readiness = result.readiness
    eligible_ids = [
        session.session_id
        for session in readiness.sessions
        if session.classification == "ELIGIBLE"
    ]
    print("V9 V2 READINESS")
    print(f"Eligible sessions: {readiness.eligible_closed_sessions}")
    print(f"Eligible session IDs: {json.dumps(eligible_ids)}")
    print(f"Eligible hours: {readiness.eligible_hours}")
    print(f"UTC dates: {json.dumps(_utc_dates(readiness))}")
    print(f"Readiness status: {'READY' if readiness.ready else 'NOT_READY'}")
    if readiness.reasons:
        print(f"Readiness reasons: {json.dumps(list(readiness.reasons))}")
    print("NO PREDICTIVE OUTCOMES EVALUATED")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_paths", nargs="+", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = import_artifacts(args.zip_paths)
    except (CollectionValidationError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"V9 GITHUB ARTIFACT IMPORT FAILED: {exc}")
        return 2
    _print_result(result)
    return 1 if result.rejected_artifacts else 0


if __name__ == "__main__":
    raise SystemExit(main())
