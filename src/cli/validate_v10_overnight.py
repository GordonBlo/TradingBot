"""Frozen V10 acquisition preflight, artifact validation and campaign accounting."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from src.microstructure.ci import reject_ci_research_path
from src.microstructure.v10 import _write_immutable_json, sha256_file, source_commit
from src.research.v10_collection_preregistration import (
    WORKSPACE, load_manifest, source_hashes, strict_json, timestamp, utc, verify_replay_source,
)
from src.research.v10_collection_readiness import ARTIFACTS, artifact_hashes, require, scan_readiness


PREREGISTRATION_ID = "8326791b411c27b5"
PREREGISTRATION_SHA256 = "8368b421bb1fa1af0823fc5139facb36274b18d7151935b48d71a45b2c3c5c10"
ROOT = Path("data/microstructure/v10")
PREFLIGHT = "overnight.preflight.json"
VALIDATION = "acquisition.validation.json"
DURATION_SECONDS = 10800
REQUIRED_STEPS = ("checkout", "python", "dependencies", "preflight", "collect", "identify", "validate", "upload")


def protocol() -> tuple[Path, dict]:
    path, manifest = load_manifest()
    require(manifest["preregistration_id"] == PREREGISTRATION_ID, "wrong V10 preregistration ID")
    require(sha256_file(path) == PREREGISTRATION_SHA256, "frozen V10 manifest bytes changed")
    verify_replay_source(manifest)
    return path, manifest


def root_path(workspace: Path) -> Path:
    root = workspace / ROOT
    reject_ci_research_path(root)
    require(root.resolve() == workspace.resolve() / ROOT, "canonical root was redirected")
    return root


def preflight(workspace: Path = WORKSPACE, *, now: datetime | None = None) -> dict:
    _, manifest = protocol()
    current = utc(timestamp(now or datetime.now(UTC)))
    cutoff = utc(manifest["definition"]["prospective_cutoff_utc"])
    print(f"Preregistration ID: {PREREGISTRATION_ID}")
    print(f"Frozen cutoff: {timestamp(cutoff)}")
    require(current > cutoff, "current UTC start must be strictly after frozen cutoff")
    commit = source_commit()
    require(source_hashes(commit) == manifest["definition"]["collector_source_sha256_lf_normalized"],
            "current source commit does not contain frozen collector semantics")
    if os.environ.get("GITHUB_SHA"):
        require(commit == os.environ["GITHUB_SHA"], "checkout/source commit mismatch")
    root = root_path(workspace)
    require(not root.exists(), "fresh runner requires an absent canonical collection root")
    proof = {
        "preregistration_id": PREREGISTRATION_ID,
        "preregistration_sha256": PREREGISTRATION_SHA256,
        "prospective_cutoff_utc": timestamp(cutoff), "checked_at_utc": timestamp(current),
        "source_commit_sha": commit, "duration_seconds": DURATION_SECONDS,
        "predictive_outcomes_evaluated": False,
    }
    _write_immutable_json(root / PREFLIGHT, proof)
    return proof


def identify(workspace: Path = WORKSPACE) -> Path:
    root = root_path(workspace)
    paths = sorted(root.rglob("session.manifest.json"))
    require(len(paths) == 1, f"expected exactly one newly created session; found {len(paths)}")
    directory = paths[0].parent
    for item in root.rglob("*"):
        require(not item.is_symlink() and not item.is_junction()
                and item.resolve().is_relative_to(root.resolve()), "session path redirect/escape")
    reject_ci_research_path(directory)
    session = strict_json(paths[0].read_text(encoding="utf-8"))
    require(isinstance(session.get("session_id"), str)
            and re.fullmatch(r"\d{8}T\d{12}Z", session["session_id"]) is not None
            and directory.name == session["session_id"], "session artifact identity mismatch")
    allowed = {root / PREFLIGHT} | {directory / name for name in (*ARTIFACTS, VALIDATION)}
    require(all(item in allowed for item in root.rglob("*") if item.is_file()), "unexpected session artifacts")
    return directory


def artifact_name(run_id: str, attempt: str, session_id: str) -> str:
    require(re.fullmatch(r"[1-9]\d*", run_id) is not None
            and re.fullmatch(r"[1-9]\d*", attempt) is not None, "invalid GitHub run identity")
    require(re.fullmatch(r"\d{8}T\d{12}Z", session_id) is not None, "invalid session ID")
    return f"v10-microstructure-overnight-{run_id}-attempt-{attempt}-{session_id}"


def empty_result(number: int) -> dict:
    return {
        "session_number": number, "status": "FAIL", "session_id": "NOT_CREATED",
        "started_at_utc": None, "ended_at_utc": None, "eligible_hours": "0",
        "utc_start_date": None, "depth_count": None, "aggtrade_count": None,
        "causal_context_count": None, "artifact_hashes": {}, "artifact_name": None,
        "research_eligibility": "INELIGIBLE", "predictive_outcomes_evaluated": False,
        "reason": "session did not complete validation and artifact upload",
    }


def validate(
    number: int, run_id: str, attempt: str, *, workspace: Path = WORKSPACE,
    now: datetime | None = None, collection_outcome: str = "success",
) -> dict:
    directory = identify(workspace)
    root = root_path(workspace)
    result = empty_result(number)
    result.update(session_id=directory.name, artifact_name=artifact_name(run_id, attempt, directory.name))
    result["artifact_hashes"] = {
        name: sha256_file(directory / name) for name in ARTIFACTS if (directory / name).is_file()
    }
    metadata = {
        "schema_version": "V10_OVERNIGHT_ACQUISITION_VALIDATION_1",
        "preregistration_id": PREREGISTRATION_ID, "preregistration_sha256": PREREGISTRATION_SHA256,
        "github_run_id": run_id, "github_run_attempt": attempt, "result": result,
        "readiness": None, "preflight": None,
    }
    try:
        _, manifest = protocol()
        proof = strict_json((root / PREFLIGHT).read_text(encoding="utf-8"))
        metadata["preflight"] = proof
        require(proof.get("preregistration_id") == PREREGISTRATION_ID
                and proof.get("preregistration_sha256") == PREREGISTRATION_SHA256
                and proof.get("prospective_cutoff_utc") == manifest["definition"]["prospective_cutoff_utc"]
                and proof.get("duration_seconds") == DURATION_SECONDS, "preflight identity mismatch")
        require(utc(proof["checked_at_utc"]) > utc(proof["prospective_cutoff_utc"]), "preflight cutoff failed")
        session = strict_json((directory / "session.manifest.json").read_text(encoding="utf-8"))
        result.update(started_at_utc=session["started_at_utc"],
                      utc_start_date=utc(session["started_at_utc"]).date().isoformat())
        require(utc(session["started_at_utc"]) >= utc(proof["checked_at_utc"]), "session predates preflight")
        require(session.get("source_commit_sha") == proof.get("source_commit_sha") == source_commit(),
                "session/preflight/checkout source identity mismatch")
        require(type(session.get("requested_duration_seconds")) is int
                and session["requested_duration_seconds"] == DURATION_SECONDS, "session duration must be 10800 seconds")
        depth_stream = session["streams"]["depth"]
        require(depth_stream.get("snapshot_limit") == 5000 and depth_stream.get("max_levels_per_side") == 5000,
                "overnight depth limits must both be 5000")
        closure = strict_json((directory / "closure.summary.json").read_text(encoding="utf-8"))
        result.update(
            ended_at_utc=closure.get("ended_at_utc"),
            depth_count=closure.get("live_integrity", {}).get("depth", {}).get("diff_events"),
            aggtrade_count=closure.get("live_integrity", {}).get("aggtrades", {}).get("raw_events"),
            causal_context_count=closure.get("offline_replay", {}).get("causal_trade_count"),
            artifact_hashes=artifact_hashes(directory),
        )
        require(collection_outcome == "success", "collector step failed")
        readiness = scan_readiness(manifest, data_root=root, now=now)
        metadata["readiness"] = readiness
        require(readiness["eligible_sessions"] == 1 and len(readiness["sessions"]) == 1,
                "frozen acquisition scanner rejected session: " + json.dumps(readiness["sessions"]))
        eligible = readiness["sessions"][0]
        require(eligible["classification"] == "ELIGIBLE" and eligible["session_id"] == directory.name,
                "scanner session identity mismatch")
        require(eligible["eligible_microseconds"] == DURATION_SECONDS * 1_000_000,
                "session did not complete the full 10800 seconds")
        require(eligible["artifact_sha256"] == result["artifact_hashes"] == artifact_hashes(directory),
                "session artifacts changed during validation")
        require(sha256_file(load_manifest()[0]) == PREREGISTRATION_SHA256, "frozen manifest changed")
        result.update(status="PASS", eligible_hours="3", research_eligibility="ELIGIBLE",
                      deterministic_dual_stream_replay="PASS", causal_trade_l2_merge="PASS",
                      synchronized_timeline_sha256=eligible["synchronized_timeline_sha256"],
                      causal_contexts_sha256=eligible["causal_contexts_sha256"],
                      reason="frozen acquisition eligibility passed; artifact upload pending")
    except (ValueError, OSError, KeyError, TypeError, ArithmeticError, AttributeError) as exc:
        result.update(status="FAIL", eligible_hours="0", research_eligibility="INELIGIBLE", reason=str(exc))
    _write_immutable_json(directory / VALIDATION, metadata)
    return metadata


def publish(number: int, outcomes: dict, *, workspace: Path = WORKSPACE) -> dict:
    result = empty_result(number)
    validation_hash = None
    try:
        directory = identify(workspace)
        result["session_id"] = directory.name
        metadata = strict_json((directory / VALIDATION).read_text(encoding="utf-8"))
        validation_hash = sha256_file(directory / VALIDATION)
        result.update(metadata["result"])
        require(result["session_number"] == number and result["session_id"] == directory.name,
                "publication identity mismatch")
        require(all(outcomes.get(key) == "success" for key in REQUIRED_STEPS), "one or more required job steps failed")
        require(result["status"] == "PASS" and result["research_eligibility"] == "ELIGIBLE",
                "validation did not pass")
        require(result["artifact_hashes"] == artifact_hashes(directory), "artifacts changed before publication")
        result["reason"] = "acquisition eligibility and artifact upload passed"
    except (ValueError, OSError, KeyError, TypeError, ArithmeticError, AttributeError) as exc:
        result.update(status="FAIL", research_eligibility="INELIGIBLE", eligible_hours="0", reason=str(exc))
    if validation_hash is not None:
        result["artifact_hashes"][VALIDATION] = validation_hash
    result["step_outcomes"] = outcomes
    return result


def campaign_summary(needs: dict) -> dict:
    rows = []
    seen = set()
    previous_end = None
    for number in range(1, 4):
        row = empty_result(number)
        try:
            job = needs.get(f"session_{number}", {})
            encoded = job.get("outputs", {}).get("report", "")
            if encoded:
                row.update(strict_json(encoded))
            require(job.get("result") == "success" and row["status"] == "PASS", "job failed or missing result")
            require(row["session_number"] == number and row["research_eligibility"] == "ELIGIBLE"
                    and row["eligible_hours"] == "3"
                    and row["predictive_outcomes_evaluated"] is False, "invalid eligibility accounting")
            require(row["session_id"] not in seen, "duplicate campaign session ID")
            start, end = utc(row["started_at_utc"]), utc(row["ended_at_utc"])
            require(end > start and (previous_end is None or start >= previous_end), "nonsequential campaign interval")
            require(row["utc_start_date"] == start.date().isoformat(), "UTC date accounting mismatch")
            require(set(row["artifact_hashes"]) == {*ARTIFACTS, VALIDATION}
                    and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in row["artifact_hashes"].values()),
                    "incomplete artifact hashes")
            require(isinstance(row["artifact_name"], str) and row["artifact_name"].endswith(row["session_id"]),
                    "artifact identity missing")
            seen.add(row["session_id"])
            previous_end = end
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            row.update(status="FAIL", eligible_hours="0", research_eligibility="INELIGIBLE", reason=str(exc))
        rows.append(row)
    eligible = [row for row in rows if row["status"] == "PASS"]
    return {
        "preregistration_id": PREREGISTRATION_ID, "sessions": rows,
        "eligible_sessions": len(eligible), "eligible_hours": str(3 * len(eligible)),
        "utc_start_dates": sorted({row["utc_start_date"] for row in eligible}),
        "predictive_outcomes_evaluated": False, "data_sufficiency_only": True,
        "independent_confirmatory_strategy_evidence": False,
    }


def emit_output(key: str, value: str) -> None:
    require("\n" not in value and "\r" not in value, "unsafe GitHub output")
    if os.environ.get("GITHUB_OUTPUT"):
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
            stream.write(f"{key}={value}\n")


def print_report(report: dict) -> None:
    rendered = json.dumps(report, sort_keys=True, indent=2)
    print(rendered)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a", encoding="utf-8") as stream:
            stream.write("V10 prospective acquisition — no predictive outcomes evaluated.\n\n```json\n"
                         + rendered + "\n```\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("PREFLIGHT", "IDENTIFY", "VALIDATE", "PUBLISH", "SUMMARY"))
    args = parser.parse_args(argv)
    try:
        number = int(os.environ.get("SESSION_NUMBER", "1"))
        require(number in (1, 2, 3), "invalid session number")
        if args.mode == "PREFLIGHT":
            preflight()
        elif args.mode == "IDENTIFY":
            directory = identify()
            name = artifact_name(os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"], directory.name)
            emit_output("session_dir", directory.as_posix())
            emit_output("artifact_name", name)
        elif args.mode == "VALIDATE":
            metadata = validate(number, os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_RUN_ATTEMPT"],
                                collection_outcome=os.environ.get("COLLECT_OUTCOME", "missing"))
            print_report(metadata["result"])
            return 0 if metadata["result"]["status"] == "PASS" else 1
        elif args.mode == "PUBLISH":
            result = publish(number, {key: os.environ.get(key.upper() + "_OUTCOME", "missing") for key in REQUIRED_STEPS})
            emit_output("report", json.dumps(result, sort_keys=True, separators=(",", ":")))
            print_report(result)
            return 0 if result["status"] == "PASS" else 1
        else:
            result = campaign_summary(strict_json(os.environ.get("CAMPAIGN_NEEDS", "{}")))
            print_report(result)
            return 0 if result["eligible_sessions"] == 3 else 1
        return 0
    except (ValueError, OSError, KeyError, TypeError, ArithmeticError, AttributeError) as exc:
        print(f"V10 OVERNIGHT INTEGRITY_FAILURE: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
