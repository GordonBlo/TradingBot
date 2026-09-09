"""Validate one V9 V2 prospective L2 collection without evaluating outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Sequence

from src.research.v9_l2_early_information_preregistration_v2 import (
    FROZEN_FEATURES,
    PROTOCOL_VERSION,
    verify_manifest,
)
from src.research.v9_l2_readiness import scan_session_readiness


ACTIVE_MANIFEST = Path(
    "research/v9_l2_early_information_v2/853870051677af08/manifest.json"
)
ACTIVE_MANIFEST_RUN_ID = "853870051677af08"
ACTIVE_MANIFEST_SHA256 = (
    "f6dd3d95ec595c8d2c9d6f9aca104a963341de21c910af358009fd27fffa6db1"
)
CANONICAL_DATA_ROOT = Path("data/orderbook/v9")
CANONICAL_FEATURE_ROOT = CANONICAL_DATA_ROOT / "features"
EXPECTED_DURATION_SECONDS = 10_800
EXPECTED_DERIVED_FILES = {
    "event_features": "event_features.csv",
    "features_1s": "features_1s.csv",
    "features_15m": "features_15m.csv",
}
FROZEN_FEATURE_COLUMNS = {
    "spread_bps": "spread_bps_last",
    "microprice_minus_mid_bps": "microprice_minus_mid_bps_last",
    "depth_imbalance_1": "depth_imbalance_1_last",
    "depth_imbalance_5": "depth_imbalance_5_last",
    "depth_imbalance_10": "depth_imbalance_10_last",
    "depth_imbalance_20": "depth_imbalance_20_last",
    "bid_depth_top_20": "bid_depth_top_20_last",
    "ask_depth_top_20": "ask_depth_top_20_last",
    "bid_depth_concentration": "bid_depth_concentration_top1_over_top20_last",
    "ask_depth_concentration": "ask_depth_concentration_top1_over_top20_last",
    "bid_depth_added": "bid_depth_added",
    "bid_depth_removed": "bid_depth_removed",
    "ask_depth_added": "ask_depth_added",
    "ask_depth_removed": "ask_depth_removed",
    "update_intensity": "update_intensity_per_second",
}


class CollectionValidationError(RuntimeError):
    """Raised when a prospective collection artifact violates frozen V2 rules."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_utc(value: str, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise CollectionValidationError(f"{label} is not a valid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CollectionValidationError(f"{label} must be timezone-aware")
    return parsed.astimezone(UTC)


def _resolved(path: str | Path, workspace: Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (workspace / value).resolve()


def _require_sha256(value: Any, *, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise CollectionValidationError(f"{label} is not a lowercase SHA-256 digest")
    return text


def _write_github_outputs(path: str | Path | None, values: dict[str, Any]) -> None:
    if path is None:
        return
    output_path = Path(path)
    with output_path.open("a", encoding="utf-8", newline="\n") as stream:
        for key, value in values.items():
            if isinstance(value, bool):
                rendered = "true" if value else "false"
            elif isinstance(value, (list, dict)):
                rendered = json.dumps(value, sort_keys=True, separators=(",", ":"))
            else:
                rendered = str(value)
            if "\n" in rendered or "\r" in rendered:
                raise CollectionValidationError(f"GitHub output {key} must be one line")
            stream.write(f"{key}={rendered}\n")


def load_active_manifest(workspace: Path = Path(".")) -> tuple[Path, dict[str, Any], str]:
    workspace = workspace.resolve()
    expected_path = (workspace / ACTIVE_MANIFEST).resolve()
    manifest_root = expected_path.parents[1]
    discovered = sorted(path.resolve() for path in manifest_root.glob("*/manifest.json"))
    if discovered != [expected_path]:
        raise CollectionValidationError(
            f"expected exactly the active frozen V9 V2 manifest, found {discovered}"
        )
    manifest_hash = _sha256(expected_path)
    if manifest_hash != ACTIVE_MANIFEST_SHA256:
        raise CollectionValidationError("active frozen V9 V2 manifest hash mismatch")
    try:
        manifest = json.loads(expected_path.read_text(encoding="utf-8"))
        verify_manifest(manifest)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise CollectionValidationError("active frozen V9 V2 manifest is invalid") from exc

    definition = manifest["definition"]
    if (
        manifest.get("run_id") != ACTIVE_MANIFEST_RUN_ID
        or expected_path.parent.name != ACTIVE_MANIFEST_RUN_ID
        or definition.get("protocol_version") != PROTOCOL_VERSION
        or definition.get("market")
        != {
            "symbol": "BTCUSDC",
            "venue": "Binance PUBLIC Spot",
            "source": "V9 persisted raw L2 events",
        }
    ):
        raise CollectionValidationError("active V9 V2 manifest identity mismatch")
    restrictions = definition.get("restrictions", {})
    if restrictions.get("blind_holdout") != {
        "status": "LOCKED",
        "loaded": False,
        "revealed": False,
        "consumed": False,
        "evaluated": False,
    } or restrictions.get("real_predictive_evaluation_during_preregistration") is not False:
        raise CollectionValidationError("frozen V9 V2 restrictions mismatch")
    if set(FROZEN_FEATURE_COLUMNS) != set(FROZEN_FEATURES):
        raise CollectionValidationError("frozen V9 V2 feature/provenance mapping mismatch")
    return expected_path, manifest, manifest_hash


def preflight_collection(
    *, workspace: Path = Path("."), now: datetime | None = None
) -> dict[str, Any]:
    workspace = workspace.resolve()
    manifest_path, manifest, manifest_hash = load_active_manifest(workspace)
    cutoff_text = str(manifest["definition"]["prospective_cutoff_utc"])
    cutoff = _parse_utc(cutoff_text, label="prospective cutoff")
    checked_at = (now or datetime.now(UTC)).astimezone(UTC)
    if checked_at < cutoff:
        raise CollectionValidationError(
            f"prospective collection cannot start before cutoff {cutoff_text}"
        )

    data_root = (workspace / CANONICAL_DATA_ROOT).resolve()
    if data_root.exists():
        raise CollectionValidationError(
            f"fresh runner expected; canonical research root already exists: {data_root}"
        )
    data_root.mkdir(parents=True, exist_ok=False)
    return {
        "manifest_path": manifest_path.relative_to(workspace).as_posix(),
        "manifest_run_id": str(manifest["run_id"]),
        "manifest_sha256": manifest_hash,
        "prospective_cutoff_utc": cutoff_text,
        "preflight_checked_at_utc": checked_at.isoformat().replace("+00:00", "Z"),
        "data_root": CANONICAL_DATA_ROOT.as_posix(),
        "predictive_outcomes_evaluated": False,
        "blind_holdout_accessed": False,
    }


def identify_raw_session(
    *, workspace: Path = Path("."), data_root: Path = CANONICAL_DATA_ROOT
) -> dict[str, Any]:
    workspace = workspace.resolve()
    root = _resolved(data_root, workspace)
    if root != (workspace / CANONICAL_DATA_ROOT).resolve():
        raise CollectionValidationError("collection did not use the canonical V9 research root")
    raw_paths = sorted(root.rglob("*.jsonl")) if root.exists() else []
    if len(raw_paths) != 1:
        raise CollectionValidationError(
            f"expected exactly one raw V9 session, found {len(raw_paths)}"
        )
    raw_path = raw_paths[0]
    summary_path = raw_path.with_suffix(".summary.json")
    return {
        "has_session": True,
        "session_id": raw_path.stem,
        "raw_path": raw_path.relative_to(workspace).as_posix(),
        "summary_path": summary_path.relative_to(workspace).as_posix(),
        "summary_exists": summary_path.is_file(),
    }


def _validate_decimal(value: str, *, label: str, positive: bool = False) -> None:
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise CollectionValidationError(f"{label} is not numeric") from exc
    if not parsed.is_finite() or (positive and parsed <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise CollectionValidationError(f"{label} must be {qualifier}")


def validate_one_second_provenance(path: Path) -> dict[str, int]:
    required_columns = {
        "bucket_open_utc",
        "bucket_close_utc",
        "first_feature_available_at_utc",
        "last_feature_available_at_utc",
        "event_count",
        "mid_price_last",
        *FROZEN_FEATURE_COLUMNS.values(),
    }
    row_count = 0
    observed_rows = 0
    possible_endpoint_rows = 0
    previous_close: datetime | None = None
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        missing_columns = required_columns.difference(reader.fieldnames or ())
        if missing_columns:
            raise CollectionValidationError(
                f"features_1s provenance columns missing: {sorted(missing_columns)}"
            )
        for row_count, row in enumerate(reader, start=1):
            bucket_open = _parse_utc(
                str(row["bucket_open_utc"]), label=f"row {row_count} bucket open"
            )
            bucket_close = _parse_utc(
                str(row["bucket_close_utc"]), label=f"row {row_count} bucket close"
            )
            if bucket_close - bucket_open != timedelta(seconds=1):
                raise CollectionValidationError("features_1s contains a non-1s bucket")
            if previous_close is not None and bucket_open != previous_close:
                raise CollectionValidationError(
                    "features_1s buckets are not unique, chronological, and contiguous"
                )
            previous_close = bucket_close

            try:
                event_count = int(row["event_count"])
            except (TypeError, ValueError) as exc:
                raise CollectionValidationError("features_1s event_count is invalid") from exc
            if event_count < 0:
                raise CollectionValidationError("features_1s event_count is negative")

            populated_features = [
                column
                for column in FROZEN_FEATURE_COLUMNS.values()
                if row.get(column) not in (None, "")
            ]
            mid_value = row.get("mid_price_last")
            needs_provenance = event_count > 0 and bool(populated_features)
            if mid_value not in (None, ""):
                _validate_decimal(
                    str(mid_value), label=f"row {row_count} mid-price", positive=True
                )
                possible_endpoint_rows += 1
                needs_provenance = True
            for column in populated_features:
                _validate_decimal(str(row[column]), label=f"row {row_count} {column}")

            if needs_provenance:
                first_text = row.get("first_feature_available_at_utc")
                last_text = row.get("last_feature_available_at_utc")
                if first_text in (None, "") or last_text in (None, ""):
                    raise CollectionValidationError(
                        "non-missing prospective feature/endpoint lacks source provenance"
                    )
                first_available = _parse_utc(
                    str(first_text), label=f"row {row_count} first feature availability"
                )
                last_available = _parse_utc(
                    str(last_text), label=f"row {row_count} last feature availability"
                )
                if not bucket_open <= first_available <= last_available <= bucket_close:
                    raise CollectionValidationError(
                        "future or out-of-order feature source provenance detected"
                    )
                observed_rows += 1
    if row_count == 0 or observed_rows == 0 or possible_endpoint_rows == 0:
        raise CollectionValidationError("features_1s lacks valid prospective observations")
    return {
        "one_second_rows": row_count,
        "rows_with_observed_provenance": observed_rows,
        "possible_endpoint_rows_with_provenance": possible_endpoint_rows,
        "forward_targets_formed": 0,
    }


def validate_session(
    *,
    raw_path: Path,
    summary_path: Path,
    workspace: Path = Path("."),
    expected_duration_seconds: int = EXPECTED_DURATION_SECONDS,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    canonical_root = (workspace / CANONICAL_DATA_ROOT).resolve()
    canonical_feature_root = (workspace / CANONICAL_FEATURE_ROOT).resolve()
    raw_path = _resolved(raw_path, workspace)
    summary_path = _resolved(summary_path, workspace)
    manifest_path, manifest, manifest_hash = load_active_manifest(workspace)
    cutoff_text = str(manifest["definition"]["prospective_cutoff_utc"])
    cutoff = _parse_utc(cutoff_text, label="prospective cutoff")

    if not raw_path.is_relative_to(canonical_root) or not summary_path.is_relative_to(
        canonical_root
    ):
        raise CollectionValidationError("session artifacts escaped the canonical research root")
    discovered_raw = sorted(canonical_root.rglob("*.jsonl"))
    if discovered_raw != [raw_path]:
        raise CollectionValidationError("canonical research root does not contain one raw session")
    if not raw_path.is_file() or not summary_path.is_file():
        raise CollectionValidationError("raw session or closure summary is missing")
    if summary_path != raw_path.with_suffix(".summary.json"):
        raise CollectionValidationError("raw session and closure summary identities differ")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_summary = {
        "session_id": raw_path.stem,
        "symbol": "BTCUSDC",
        "source": "BINANCE_PUBLIC_SPOT",
        "rest_snapshot": "PUBLIC_API_V3_DEPTH",
        "websocket_stream": "BTCUSDC_DIFF_DEPTH_100MS",
        "uses_authentication": False,
        "orders_enabled": False,
        "duration_seconds": expected_duration_seconds,
        "validation_status": "PASSED",
        "collector_left_running": False,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise CollectionValidationError(
                f"unexpected closure summary field {key}: {summary.get(key)!r}"
            )
    if _resolved(str(summary.get("raw_event_log", "")), workspace) != raw_path:
        raise CollectionValidationError("closure summary raw path mismatch")
    started = _parse_utc(str(summary.get("started_at_utc")), label="session start")
    ended = _parse_utc(str(summary.get("ended_at_utc")), label="session end")
    if started < cutoff:
        raise CollectionValidationError("session started before the frozen V9 V2 cutoff")
    if ended <= started:
        raise CollectionValidationError("session did not close after it started")
    if ended - started < timedelta(seconds=expected_duration_seconds):
        raise CollectionValidationError(
            "authoritative session interval is shorter than the requested collection duration"
        )
    expected_raw_parent = (
        canonical_root
        / "BTCUSDC"
        / f"{started:%Y}"
        / f"{started:%m}"
        / f"{started:%d}"
    )
    if raw_path.parent != expected_raw_parent:
        raise CollectionValidationError("raw session path does not match its UTC start date")

    # Import the network adapter only in full artifact validation. Preflight/finalization
    # remain usable with the standard library if dependency installation failed.
    from src.exchange.public_market_client import PublicMarketDataClient
    from src.orderbook.recorder import V9DepthRecorder

    public_endpoints = {
        "rest": PublicMarketDataClient.REST_BASE_URL,
        "websocket": V9DepthRecorder.WEBSOCKET_URL,
    }
    if public_endpoints != {
        "rest": "https://data-api.binance.vision",
        "websocket": "wss://data-stream.binance.vision/ws/btcusdc@depth@100ms",
    }:
        raise CollectionValidationError("collector is not bound to public Binance endpoints")

    collector_integrity = summary.get("integrity", {})
    for key in ("sequence_gaps", "invalid_events", "crossed_invalid_book_states"):
        if collector_integrity.get(key) != 0:
            raise CollectionValidationError(
                f"nonzero collector integrity counter: {key}={collector_integrity.get(key)!r}"
            )

    record_types: set[str] = set()
    previous_received_at: datetime | None = None
    with raw_path.open("r", encoding="utf-8") as raw_stream:
        for expected_index, line in enumerate(raw_stream):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CollectionValidationError("raw session contains invalid JSON") from exc
            if record.get("schema_version") != "V9_DEPTH_RAW_1":
                raise CollectionValidationError("raw session schema mismatch")
            if record.get("session_id") != raw_path.stem:
                raise CollectionValidationError("raw record session identity mismatch")
            if record.get("record_index") != expected_index:
                raise CollectionValidationError("raw record indexes are not contiguous")
            received_at = _parse_utc(
                str(record.get("received_at_utc")), label="raw record receipt timestamp"
            )
            if not started <= received_at <= ended:
                raise CollectionValidationError("raw record falls outside session boundaries")
            if previous_received_at is not None and received_at < previous_received_at:
                raise CollectionValidationError("raw receipt timestamps are not chronological")
            previous_received_at = received_at
            record_types.add(str(record.get("record_type")))
    if record_types != {"REST_SNAPSHOT", "DIFF_DEPTH"}:
        raise CollectionValidationError(f"unexpected raw record types: {sorted(record_types)}")

    report_path = canonical_feature_root / raw_path.stem / "validation_report.json"
    if not report_path.is_file():
        raise CollectionValidationError("feature validation report is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if _resolved(str(report.get("raw_path", "")), workspace) != raw_path:
        raise CollectionValidationError("feature report raw path mismatch")
    expected_output_dir = report_path.parent
    if _resolved(str(report.get("output_dir", "")), workspace) != expected_output_dir:
        raise CollectionValidationError("feature report output directory mismatch")

    raw_hash = _sha256(raw_path)
    if report.get("raw_sha256") != raw_hash:
        raise CollectionValidationError("feature report raw hash mismatch")
    files = report.get("files", {})
    file_hashes = report.get("file_sha256", {})
    if set(files) != set(EXPECTED_DERIVED_FILES) or set(file_hashes) != set(
        EXPECTED_DERIVED_FILES
    ):
        raise CollectionValidationError("deterministic feature hash set is incomplete")
    for name, filename in EXPECTED_DERIVED_FILES.items():
        output_path = _resolved(str(files[name]), workspace)
        if output_path != expected_output_dir / filename or not output_path.is_file():
            raise CollectionValidationError(f"derived feature identity mismatch: {name}")
        expected_hash = _require_sha256(file_hashes[name], label=f"{name} hash")
        if _sha256(output_path) != expected_hash:
            raise CollectionValidationError(f"derived feature hash mismatch: {name}")

    deterministic = report.get("deterministic_replay_hash_check", {})
    primary_hashes = deterministic.get("primary_file_sha256")
    replay_hashes = deterministic.get("replay_file_sha256")
    if (
        deterministic.get("passed") is not True
        or primary_hashes != file_hashes
        or replay_hashes != file_hashes
    ):
        raise CollectionValidationError("deterministic replay hash check failed")

    feature_counters = report.get("counters", {})
    for key in (
        "sequence_gap_count",
        "invalid_event_count",
        "crossed_book_state_count",
        "unreconstructed_event_count",
    ):
        if feature_counters.get(key) != 0:
            raise CollectionValidationError(
                f"nonzero feature integrity counter: {key}={feature_counters.get(key)!r}"
            )
    if feature_counters.get("reconstructed_update_count", 0) < 1:
        raise CollectionValidationError("feature reconstruction produced no valid updates")

    provenance = validate_one_second_provenance(
        expected_output_dir / EXPECTED_DERIVED_FILES["features_1s"]
    )

    readiness = scan_session_readiness(
        manifest,
        data_root=canonical_root,
        feature_root=canonical_feature_root,
        workspace=workspace,
    )
    if readiness.discovered_sessions != 1 or len(readiness.sessions) != 1:
        raise CollectionValidationError("one-session readiness classification is ambiguous")
    session = readiness.sessions[0]
    if (
        session.session_id != raw_path.stem
        or session.classification != "ELIGIBLE"
        or session.integrity_passed is not True
        or session.artifact_binding is None
    ):
        raise CollectionValidationError(
            f"session is not V9 V2 research-eligible: {session.reason}"
        )

    closure_hash = _sha256(summary_path)
    binding = session.artifact_binding
    if (
        binding.raw_sha256 != raw_hash
        or binding.closure_summary_sha256 != closure_hash
        or binding.features_1s_sha256 != file_hashes["features_1s"]
    ):
        raise CollectionValidationError("V9 V2 immutable artifact binding mismatch")

    return {
        "schema_version": 1,
        "purpose": "V9_V2_PROSPECTIVE_COLLECTION_ONLY",
        "status": "PASS",
        "classification": "ELIGIBLE",
        "research_eligible": True,
        "session_id": session.session_id,
        "started_at_utc": session.started_at_utc,
        "ended_at_utc": session.ended_at_utc,
        "requested_duration_seconds": expected_duration_seconds,
        "eligible_hours": session.duration_hours,
        "eligible_utc_dates": [started.date().isoformat()],
        "preregistration": {
            "run_id": manifest["run_id"],
            "protocol_version": PROTOCOL_VERSION,
            "manifest_path": manifest_path.relative_to(workspace).as_posix(),
            "manifest_sha256": manifest_hash,
            "prospective_cutoff_utc": cutoff_text,
        },
        "artifact_binding": {
            **asdict(binding),
            "raw_event_log": raw_path.relative_to(workspace).as_posix(),
            "closure_summary": summary_path.relative_to(workspace).as_posix(),
            "features_1s": (
                expected_output_dir / EXPECTED_DERIVED_FILES["features_1s"]
            ).relative_to(workspace).as_posix(),
        },
        "collector_integrity": collector_integrity,
        "feature_integrity": feature_counters,
        "derived_file_sha256": file_hashes,
        "deterministic_replay_hash_check": deterministic,
        "feature_provenance": provenance,
        "public_data_only": {
            "endpoints": public_endpoints,
            "uses_authentication": False,
            "orders_enabled": False,
            "raw_record_types": sorted(record_types),
        },
        "predictive_outcomes_evaluated": False,
        "forward_targets_formed": 0,
        "blind_holdout_accessed": False,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--github-output", type=Path)

    identify = subparsers.add_parser("identify")
    identify.add_argument("--github-output", type=Path)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--raw", type=Path, required=True)
    validate.add_argument("--summary", type=Path, required=True)
    validate.add_argument("--result", type=Path, required=True)
    validate.add_argument("--duration-seconds", type=int, default=EXPECTED_DURATION_SECONDS)
    validate.add_argument("--github-output", type=Path)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--session-number", required=True)
    finalize.add_argument("--session-id", required=True)
    finalize.add_argument("--result", type=Path, required=True)
    finalize.add_argument("--artifact-name", required=True)
    finalize.add_argument("--github-output", type=Path)
    finalize.add_argument(
        "--required-outcome",
        action="append",
        default=[],
        help="NAME=OUTCOME; every required outcome must equal success",
    )
    return parser


def _failed_result(session_id: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "V9_V2_PROSPECTIVE_COLLECTION_ONLY",
        "status": "FAIL",
        "classification": "INTEGRITY_FAILED",
        "research_eligible": False,
        "session_id": session_id,
        "eligible_hours": 0,
        "eligible_utc_dates": [],
        "failure_reason": reason,
        "predictive_outcomes_evaluated": False,
        "forward_targets_formed": 0,
        "blind_holdout_accessed": False,
    }


def _main_preflight(args: argparse.Namespace) -> int:
    result = preflight_collection()
    _write_github_outputs(args.github_output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _main_identify(args: argparse.Namespace) -> int:
    try:
        result = identify_raw_session()
    except CollectionValidationError as exc:
        result = {
            "has_session": False,
            "session_id": "NOT_CREATED",
            "raw_path": "",
            "summary_path": "",
            "summary_exists": False,
            "identification_error": str(exc),
        }
        _write_github_outputs(args.github_output, result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    _write_github_outputs(args.github_output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _main_validate(args: argparse.Namespace) -> int:
    try:
        result = validate_session(
            raw_path=args.raw,
            summary_path=args.summary,
            expected_duration_seconds=args.duration_seconds,
        )
        exit_code = 0
    except (CollectionValidationError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        result = _failed_result(args.raw.stem, str(exc))
        exit_code = 1
    _write_json(args.result, result)
    _write_github_outputs(
        args.github_output,
        {
            "status": result["status"],
            "research_eligible": result["research_eligible"],
            "eligible_hours": result["eligible_hours"],
            "eligible_utc_dates": result["eligible_utc_dates"],
            "result_path": str(args.result),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return exit_code


def _main_finalize(args: argparse.Namespace) -> int:
    outcomes: dict[str, str] = {}
    for item in args.required_outcome:
        name, separator, outcome = item.partition("=")
        if not separator or not name:
            raise CollectionValidationError(f"invalid required outcome: {item}")
        outcomes[name] = outcome
    failures = sorted(name for name, outcome in outcomes.items() if outcome != "success")
    try:
        result = json.loads(args.result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        result = _failed_result(args.session_id, "session validation did not produce a result")
    if not isinstance(result, dict):
        result = _failed_result(args.session_id, "session validation result is not an object")
    if result.get("research_eligible") is not True or failures:
        reasons = list(failures)
        if result.get("failure_reason"):
            reasons.append(str(result["failure_reason"]))
        result.update(
            {
                "status": "FAIL",
                "classification": "INTEGRITY_FAILED",
                "research_eligible": False,
                "eligible_hours": 0,
                "eligible_utc_dates": [],
                "failure_reason": "; ".join(reasons) or "session validation failed",
            }
        )
    result.update(
        {
            "session_number": int(args.session_number),
            "github_run_id": os.environ.get("GITHUB_RUN_ID", "LOCAL"),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "LOCAL"),
            "github_sha": os.environ.get("GITHUB_SHA", "LOCAL"),
            "artifact_name": args.artifact_name,
            "required_step_outcomes": outcomes,
            "predictive_outcomes_evaluated": False,
            "forward_targets_formed": 0,
            "blind_holdout_accessed": False,
        }
    )
    _write_json(args.result, result)
    _write_github_outputs(
        args.github_output,
        {
            "status": result["status"],
            "has_session": args.session_id != "NOT_CREATED",
            "session_id": args.session_id,
            "eligible_hours": result["eligible_hours"],
            "eligible_utc_dates": result["eligible_utc_dates"],
            "artifact_name": args.artifact_name,
            "result_path": str(args.result),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            return _main_preflight(args)
        if args.command == "identify":
            return _main_identify(args)
        if args.command == "validate":
            return _main_validate(args)
        if args.command == "finalize":
            return _main_finalize(args)
    except (CollectionValidationError, OSError, ValueError, KeyError) as exc:
        print(f"V9 L2 COLLECTION VALIDATION FAILED: {exc}")
        return 1
    raise RuntimeError("unreachable collection validation command")


if __name__ == "__main__":
    raise SystemExit(main())
