"""CI-only preparation, session discovery and artifact validation; no collection/evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from src.microstructure.ci import CI_MARKER, SMOKE_ROOT
from src.microstructure.v10 import _write_immutable_json, sha256_file
from src.research.v10_collection_preregistration import (
    WORKSPACE, load_manifest, strict_json, verify_replay_source,
)
from src.research.v10_collection_readiness import (
    ARTIFACTS, artifact_hashes, require, validate_closed_session_integrity,
)


REPORT = "ci.validation.json"
CI_PROVENANCE = {
    "schema_version": "V10_MICROSTRUCTURE_CI_PROVENANCE_1",
    "research_eligibility": "CI_ONLY",
    "original_root": SMOKE_ROOT.as_posix(),
    "purpose": "GITHUB_ACTIONS_ENGINEERING_SMOKE",
    "prospective_research_evidence": False,
}


def smoke_root(workspace: Path) -> Path:
    root = workspace / SMOKE_ROOT
    require(root.resolve() == workspace.resolve() / SMOKE_ROOT, "CI root symlink/redirect forbidden")
    return root


def prepare(workspace: Path = WORKSPACE) -> None:
    _, manifest = load_manifest()
    verify_replay_source(manifest)
    root = smoke_root(workspace)
    require(not root.exists(), "smoke root must be fresh; no existing files are removed")
    root.mkdir(parents=True)
    _write_immutable_json(root / CI_MARKER, CI_PROVENANCE)


def locate(workspace: Path = WORKSPACE) -> Path:
    root = smoke_root(workspace)
    require(strict_json((root / CI_MARKER).read_text(encoding="utf-8")) == CI_PROVENANCE,
            "CI root provenance missing or invalid")
    paths = sorted(root.rglob("session.manifest.json"))
    require(len(paths) == 1, f"expected exactly one smoke session; found {len(paths)}")
    path = paths[0]
    for item in root.rglob("*"):
        require(not item.is_symlink() and not item.is_junction()
                and item.resolve().is_relative_to(root.resolve()), "CI artifact symlink/escape forbidden")
    session = strict_json(path.read_text(encoding="utf-8"))
    session_id = session.get("session_id")
    require(isinstance(session_id, str) and re.fullmatch(r"\d{8}T\d{12}Z", session_id) is not None
            and path.parent.name == session_id, "invalid smoke session identity")
    allowed = {root / CI_MARKER} | {path.parent / name for name in (*ARTIFACTS, CI_MARKER, REPORT)}
    require(all(item in allowed for item in root.rglob("*") if item.is_file()),
            "unexpected artifacts outside the single smoke session")
    marker = path.parent / CI_MARKER
    expected = {**CI_PROVENANCE, "session_id": session_id}
    if marker.exists():
        require(strict_json(marker.read_text(encoding="utf-8")) == expected, "session CI provenance mismatch")
    else:
        _write_immutable_json(marker, expected)
    return path.parent


def validate(workspace: Path = WORKSPACE, *, now: datetime | None = None) -> dict:
    directory = locate(workspace)
    protocol_path, manifest = load_manifest()
    manifest_hash = sha256_file(protocol_path)
    verify_replay_source(manifest)
    session = strict_json((directory / "session.manifest.json").read_text(encoding="utf-8"))
    closure = strict_json((directory / "closure.summary.json").read_text(encoding="utf-8"))
    for item in (session, closure):
        require(item.get("research_eligibility") in {"UNASSESSED", "CI_ONLY"},
                "smoke session must remain UNASSESSED / CI_ONLY")
    require(session.get("requested_duration_seconds") == 120, "smoke duration must be 120 seconds")
    require(session["streams"]["depth"].get("snapshot_limit") == 5000
            and session["streams"]["depth"].get("max_levels_per_side") == 5000,
            "smoke depth limits must both be 5000")
    # Share low-level artifact checks, never invoke research readiness or cutoff eligibility.
    checked = validate_closed_session_integrity(
        directory / "session.manifest.json", manifest, {}, now=now or datetime.now(UTC),
    )
    require(checked["classification"] == "INTEGRITY_PASSED", "normal closed session required")
    depth, trades = closure["live_integrity"]["depth"], closure["live_integrity"]["aggtrades"]
    require(checked["validated_duration_microseconds"] == 120_000_000,
            "smoke session did not complete its full 120-second interval")
    require(sha256_file(protocol_path) == manifest_hash, "frozen acquisition manifest changed")
    report = {
        "session_id": session["session_id"],
        "research_eligibility": "CI_ONLY",
        "depth_count": depth["diff_events"],
        "aggtrade_count": trades["raw_events"],
        "reconstructed_updates": depth["reconstructed_updates"],
        "trade_contexts": closure["offline_replay"]["causal_trade_count"],
        "missing_early_contexts": checked["trades_without_depth"],
        "integrity_counters": closure["live_integrity"],
        "deterministic_dual_stream_replay": "PASS",
        "causal_trade_l2_merge": "PASS",
        "artifact_sha256": checked["artifact_sha256"],
        "synchronized_timeline_sha256": checked["synchronized_timeline_sha256"],
        "causal_contexts_sha256": checked["causal_contexts_sha256"],
        "preregistration_id": manifest["preregistration_id"],
        "preregistration_sha256": manifest_hash,
        "source_commit_sha": checked["source_commit_sha"],
        "uses_authentication": False, "orders_enabled": False,
        "predictive_outcomes_evaluated": False,
    }
    require(artifact_hashes(directory) == checked["artifact_sha256"], "smoke artifacts changed")
    _write_immutable_json(directory / REPORT, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("PREPARE", "LOCATE", "VALIDATE"))
    args = parser.parse_args(argv)
    try:
        if args.mode == "PREPARE":
            prepare()
        elif args.mode == "LOCATE":
            directory = locate()
            print(f"SMOKE SESSION: {directory.name}")
            if os.environ.get("GITHUB_OUTPUT"):
                with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
                    stream.write(f"session_dir={directory.as_posix()}\n")
        else:
            report = validate()
            rendered = json.dumps(report, indent=2, sort_keys=True)
            print(rendered)
            if os.environ.get("GITHUB_STEP_SUMMARY"):
                with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as stream:
                    stream.write("V10 CI smoke validation passed. Research eligibility: CI_ONLY.\n\n```json\n"
                                 + rendered + "\n```\n")
        return 0
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError, AttributeError) as exc:
        print(f"V10 CI SMOKE INTEGRITY_FAILURE: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
