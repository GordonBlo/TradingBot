"""Guarded H9 research runner.

Only integrity validation is allowed at this stage.
No H9 historical replay is executed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal
from pathlib import Path

from src.cli.run_multiregime import _require_locked_holdout
from src.hypotheses.h9_early_failure import H9EarlyFailureExit
from src.hypotheses.manifest import (
    HoldoutStatus,
    ResearchManifestStore,
)


EXPECTED_H9_RUN_ID = "eb7a43ccaad63dfc"
EXPECTED_V33_RUN_ID = "b104ea78b4c88bcf"
EXPECTED_V331_RUN_ID = "ef593be9a0bb2697"

EXPECTED_REFERENCE_TRADES = 197
EXPECTED_EARLY_FAILURE_TRADES = 85
EXPECTED_WINDOWS = 11


def _load_json(path: str | Path) -> dict:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Required research artifact missing: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def validate_h9_manifest(payload: dict) -> None:
    if payload.get("run_id") != EXPECTED_H9_RUN_ID:
        raise ValueError("Unexpected H9 preregistration run.")

    if payload.get("source_v33_run_id") != EXPECTED_V33_RUN_ID:
        raise ValueError("Unexpected H9 V3.3 source.")

    if payload.get("source_v331_run_id") != EXPECTED_V331_RUN_ID:
        raise ValueError("Unexpected H9 V3.3.1 source.")

    if payload.get("dataset_status") != "CONSUMED_RESEARCH_DATA":
        raise ValueError("H9 dataset status changed.")

    if payload.get("reference_candidate") != "H5_Q25":
        raise ValueError("H9 reference candidate changed.")

    if payload.get("reference_trade_count") != EXPECTED_REFERENCE_TRADES:
        raise ValueError("H9 reference must remain 197 trades.")

    definition = payload.get("definition", {})

    if str(definition.get("adverse_trigger_r")) not in {
        "0.5",
        "0.50",
    }:
        raise ValueError("H9 trigger changed.")

    if definition.get("observation_bars") != 4:
        raise ValueError("H9 observation window changed.")

    if (
        definition.get("trigger_semantics")
        != "CLOSED_CANDLE_LOW_REACHES_ENTRY_MINUS_0.5R"
    ):
        raise ValueError("H9 trigger semantics changed.")

    if definition.get("execution_timing") != "NEXT_BAR_OPEN":
        raise ValueError("H9 execution timing changed.")

    if definition.get("preserve_h5_q25_entries") is not True:
        raise ValueError("H9 entry semantics changed.")

    if (
        payload.get("blind_holdout_status")
        != "LOCKED_BLIND_HOLDOUT"
    ):
        raise ValueError("H9 blind holdout is not locked.")

    for field in (
        "blind_holdout_revealed",
        "blind_holdout_consumed",
        "blind_holdout_evaluated",
    ):
        if payload.get(field) is not False:
            raise ValueError(
                f"H9 holdout integrity failed: {field}."
            )


def validate_early_failure_report(
    payload: dict,
    *,
    raw: bytes,
    h9_manifest: dict,
) -> None:
    actual_sha = hashlib.sha256(raw).hexdigest()

    if (
        actual_sha
        != h9_manifest.get("source_early_failure_sha256")
    ):
        raise ValueError(
            "Early Failure diagnostic hash differs "
            "from H9 preregistration."
        )

    if payload.get("source_v331_run_id") != EXPECTED_V331_RUN_ID:
        raise ValueError(
            "Unexpected Early Failure V3.3.1 source."
        )

    if payload.get("source_v33_run_id") != EXPECTED_V33_RUN_ID:
        raise ValueError(
            "Unexpected Early Failure V3.3 source."
        )

    if payload.get("candidate") != "H5_Q25":
        raise ValueError(
            "Early Failure candidate must remain H5_Q25."
        )

    if payload.get("trade_count") != EXPECTED_REFERENCE_TRADES:
        raise ValueError(
            "Early Failure source must contain 197 trades."
        )

    summary = payload.get("summary", {})

    if summary.get("trades") != EXPECTED_EARLY_FAILURE_TRADES:
        raise ValueError(
            "Expected exactly 85 Early Failure trades."
        )

    if summary.get("same_bar_positive_two_r_ambiguous") != 0:
        raise ValueError(
            "Unexpected +2R same-bar ambiguity."
        )

    holdout = payload.get("blind_holdout", {})

    if holdout.get("status") != "LOCKED_BLIND_HOLDOUT":
        raise ValueError(
            "Early Failure holdout is not locked."
        )

    for field in (
        "loaded",
        "revealed",
        "consumed",
        "evaluated",
    ):
        if holdout.get(field) is not False:
            raise ValueError(
                f"Early Failure holdout integrity failed: {field}."
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Guarded H9 research runner."
    )

    parser.add_argument(
        "--h9-manifest",
        default=(
            "research/h9/"
            "eb7a43ccaad63dfc/manifest.json"
        ),
    )

    parser.add_argument(
        "--early-failure-report",
        default=(
            "reports/diagnostics/v331_h5/"
            "ef593be9a0bb2697/"
            "early_failure_summary.json"
        ),
    )

    parser.add_argument(
        "--root-manifest",
        default="research/hypothesis_manifest.json",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.dry_run:
        raise SystemExit(
            "H9 REPLAY DISABLED. "
            "Run with --dry-run only until integrity guard is committed."
        )

    h9 = _load_json(args.h9_manifest)

    report_path = Path(args.early_failure_report)

    if not report_path.is_file():
        raise FileNotFoundError(
            f"Early Failure report missing: {report_path}"
        )

    raw_report = report_path.read_bytes()

    early_failure = json.loads(
        raw_report.decode("utf-8")
    )

    validate_h9_manifest(h9)

    validate_early_failure_report(
        early_failure,
        raw=raw_report,
        h9_manifest=h9,
    )

    # Verify implementation still matches preregistration.
    if H9EarlyFailureExit.ADVERSE_TRIGGER_R != Decimal("0.5"):
        raise ValueError(
            "H9 implementation trigger differs from preregistration."
        )

    if H9EarlyFailureExit.OBSERVATION_BARS != 4:
        raise ValueError(
            "H9 implementation observation window changed."
        )

    root_manifest = ResearchManifestStore(
        args.root_manifest
    ).load()

    _require_locked_holdout(root_manifest)

    if (
        root_manifest.holdout_status
        is not HoldoutStatus.LOCKED_BLIND_HOLDOUT
    ):
        raise ValueError(
            "Root blind holdout is not locked."
        )

    holdout = root_manifest.blind_holdout

    if (
        holdout.reveal_timestamp is not None
        or holdout.consumed_timestamp is not None
    ):
        raise ValueError(
            "Blind holdout contamination detected."
        )

    print()
    print("H9 DRY-RUN INTEGRITY CHECK")
    print(f"H9 preregistration: {EXPECTED_H9_RUN_ID} VERIFIED")
    print(f"Source V3.3: {EXPECTED_V33_RUN_ID} VERIFIED")
    print(f"Source V3.3.1: {EXPECTED_V331_RUN_ID} VERIFIED")
    print()
    print("Frozen mechanism:")
    print("  Reference: H5_Q25")
    print("  Trigger: -0.5R")
    print("  Observation: first 4 held bars")
    print("  Trigger source: CLOSED candle LOW")
    print("  Execution: NEXT BAR OPEN")
    print("  Existing STOP/TP precedence: preserved")
    print()
    print("Dataset: CONSUMED_RESEARCH_DATA")
    print(f"Eligible windows: {EXPECTED_WINDOWS}")
    print(f"Frozen H5_Q25 trades: {EXPECTED_REFERENCE_TRADES}")
    print(
        f"Diagnostic Early Failure trades: "
        f"{EXPECTED_EARLY_FAILURE_TRADES}"
    )
    print("Diagnostic artifact SHA-256: VERIFIED")
    print()
    print(
        "Blind holdout: LOCKED | NOT LOADED | "
        "NOT REVEALED | NOT CONSUMED | NOT EVALUATED"
    )
    print()
    print("DRY RUN PASSED — NO H9 REPLAY EXECUTED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())