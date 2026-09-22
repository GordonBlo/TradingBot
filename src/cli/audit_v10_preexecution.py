"""Read-only V10 pre-authorization audit; stdout JSON, no research execution."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from src.microstructure.v10 import sha256_file
from src.research.v10_collection_preregistration import WORKSPACE, canonical, utc
from src.research.v10_collection_readiness import require, scan_readiness
from src.research.v10_economic_discovery_preregistration import (
    DATA_ROOT,
    ROOT,
    bind_ready_dataset,
    digest,
    load_manifest,
    protocol,
    verify_dataset_binding,
)
from src.research.v10_economic_execution import (
    AUTHORIZATION,
    DEFINITION_SHA256,
    MANIFEST_SHA256,
    PREREGISTRATION_ID,
    REPORT_DIRECTORY,
    executor_identity,
    verify_identity,
)

EXPECTED_ROOT = Path("C:/Coding/TradingBot")
EXPECTED_BRANCH = "v3.3-manual"
# Operational headroom, not a change to the frozen experiment: <=86,400 sample
# rows, 40 features, four horizons and 1,000 null statistics, plus report overhead.
# 10 GiB is a conservative minimum, not a guarantee of future storage availability.
MINIMUM_FREE_BYTES = 10 * 1024**3
ACQUISITION_ROOT = WORKSPACE / "research/v10_public_microstructure_collection"


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "--no-optional-locks", *arguments], cwd=WORKSPACE,
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout.strip()


def unredirected(path: Path) -> None:
    """Check path components before any input read, including dangling links."""
    require(path.is_relative_to(WORKSPACE), "path outside expected workspace")
    for component in (path, *path.parents):
        require(not component.is_symlink() and not component.is_junction(),
                "redirected audit input or execution path")
        if component == WORKSPACE:
            break


def input_tree(root: Path) -> None:
    unredirected(root)
    require(root.is_dir(), "missing audit input directory")

    def walk_error(error):
        raise error

    # Never traverse links into V9, historical/holdout data or another workspace.
    for directory, directories, files in os.walk(root, followlinks=False, onerror=walk_error):
        for name in (*directories, *files):
            unredirected(Path(directory) / name)


def absent_execution() -> None:
    for path in (AUTHORIZATION, REPORT_DIRECTORY):
        unredirected(path)
    require(not AUTHORIZATION.exists(), "execution authorization already present")
    require(not REPORT_DIRECTORY.exists(), "discovery reservation/result already present")


def audit() -> dict:
    """Check the fixed local repository without authorizing, reserving or evaluating.

    A dirty tree is recorded as FAIL but does not suppress read-only integrity
    checks. Every other prerequisite failure stops dependent checks. No bypass.
    """
    report = {
        "status": "FAIL", "checks": [], "preregistration_id": PREREGISTRATION_ID,
        "definition_sha256": DEFINITION_SHA256,
        "predictive_outcomes_evaluated": False, "dataset_consumed_by_audit": False,
        "blind_holdout": "LOCKED_NOT_ACCESSED_BY_AUDIT",
        "orders_invoked": False, "execution_authorized_by_audit": False,
        "minimum_free_bytes": MINIMUM_FREE_BYTES,
    }

    def passed(name: str) -> None:
        report["checks"].append({"name": name, "status": "PASS"})

    def clean_tree(name: str) -> None:
        clean = not git("status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none")
        report["checks"].append({"name": name, "status": "PASS" if clean else "FAIL",
                                 "reason": "clean" if clean else "dirty worktree"})

    stage = "python"
    try:
        report["python_version"] = ".".join(map(str, sys.version_info[:3]))
        require(sys.version_info[:2] == (3, 12), "Python major/minor must be exactly 3.12")
        passed(stage)
        stage = "repository"
        require(WORKSPACE.resolve() == EXPECTED_ROOT.resolve()
                and Path.cwd().resolve() == EXPECTED_ROOT.resolve(), "wrong repository root")
        unredirected(WORKSPACE)
        require(Path(git("rev-parse", "--show-toplevel")).resolve() == EXPECTED_ROOT.resolve(),
                "wrong Git repository root")
        require(git("branch", "--show-current") == EXPECTED_BRANCH, "wrong branch")
        passed(stage)
        stage = "clean_worktree_before"
        clean_tree(stage)
        stage = "execution_absent_before"
        absent_execution()
        passed(stage)
        stage = "disk_space_before"
        report["free_bytes_before"] = shutil.disk_usage(WORKSPACE).free
        require(report["free_bytes_before"] >= MINIMUM_FREE_BYTES, "insufficient free disk space")
        passed(stage)
        stage = "input_paths"
        for root in (ROOT, ACQUISITION_ROOT, WORKSPACE / "src", WORKSPACE / DATA_ROOT):
            input_tree(root)
        passed(stage)
        stage = "frozen_preregistrations"
        manifest_path, manifest = load_manifest()
        verify_identity(manifest)
        require(sha256_file(manifest_path) == MANIFEST_SHA256, "frozen manifest bytes changed")
        acquisition_path, acquisition = protocol()
        acquisition_hash = sha256_file(acquisition_path)
        passed(stage)
        stage = "frozen_binding_and_reference_hashes"
        verify_dataset_binding(manifest)
        passed(stage)
        stage = "executor_identity_before"
        identity = executor_identity()
        report["executor_identity_sha256"] = digest(identity)
        report["source_commit"] = identity["source_commit"]
        report["executor_source_count"] = len(identity["sources_sha256"])
        passed(stage)
        stage = "fresh_acquisition_readiness"
        # Acquisition replay only: no predictive features/targets, fits or permutations.
        fresh = scan_readiness(acquisition, data_root=WORKSPACE / DATA_ROOT, workers=None)
        report["acquisition"] = {key: fresh[key] for key in (
            "status", "eligible_sessions", "eligible_hours", "eligible_utc_dates", "eligible_session_ids"
        )}
        require(fresh["status"] == "READY", "fresh acquisition READINESS is NOT_READY")
        fresh["manifest_sha256"] = acquisition_hash
        dataset = manifest["definition"]["dataset"]
        rebound = bind_ready_dataset(fresh, bound_at=utc(dataset["bound_at_utc"]))
        require(canonical(rebound) == canonical(dataset), "fresh acquisition binding differs")
        # bind_ready_dataset validates exact 8 sessions / 24 hours / >=3 UTC dates,
        # all five artifact hashes per session, and public-only/orders-disabled seals.
        passed(stage)
        stage = "final_integrity"
        require(sha256_file(manifest_path) == MANIFEST_SHA256, "frozen manifest changed during audit")
        require(sha256_file(protocol()[0]) == acquisition_hash, "acquisition manifest changed during audit")
        verify_dataset_binding(manifest)
        require(executor_identity() == identity, "executor/runtime changed during audit")
        absent_execution()
        require(git("branch", "--show-current") == EXPECTED_BRANCH, "branch changed during audit")
        report["free_bytes_after"] = shutil.disk_usage(WORKSPACE).free
        require(report["free_bytes_after"] >= MINIMUM_FREE_BYTES, "insufficient free disk space")
        passed(stage)
        stage = "clean_worktree_after"
        clean_tree(stage)
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError, AttributeError,
            subprocess.SubprocessError) as exc:
        report["checks"].append({"name": stage, "status": "FAIL", "reason": str(exc)})
    report["status"] = "PASS" if all(row["status"] == "PASS" for row in report["checks"]) else "FAIL"
    return report


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    report = audit()
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
