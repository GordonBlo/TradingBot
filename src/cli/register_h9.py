"""Register immutable H9 research before implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.hypotheses.h9_preregistration import (
    H9PreregistrationStore,
)


EXPECTED_V33_RUN_ID = "b104ea78b4c88bcf"
EXPECTED_V331_RUN_ID = "ef593be9a0bb2697"
EXPECTED_REFERENCE_TRADES = 197

EARLY_FAILURE_REPORT = Path(
    "reports/diagnostics/v331_h5/"
    "ef593be9a0bb2697/"
    "early_failure_summary.json"
)


def main() -> int:
    if not EARLY_FAILURE_REPORT.is_file():
        raise FileNotFoundError(
            f"Missing diagnostic: {EARLY_FAILURE_REPORT}"
        )

    raw = EARLY_FAILURE_REPORT.read_bytes()

    diagnostic_sha256 = hashlib.sha256(raw).hexdigest()

    payload = json.loads(
        raw.decode("utf-8")
    )

    if payload.get("source_v331_run_id") != EXPECTED_V331_RUN_ID:
        raise ValueError(
            "Unexpected Early Failure source run."
        )

    if payload.get("source_v33_run_id") != EXPECTED_V33_RUN_ID:
        raise ValueError(
            "Unexpected V3.3 source run."
        )

    if payload.get("candidate") != "H5_Q25":
        raise ValueError(
            "Early Failure source must be H5_Q25."
        )

    if payload.get("trade_count") != EXPECTED_REFERENCE_TRADES:
        raise ValueError(
            "Frozen H5_Q25 source must contain 197 trades."
        )

    if payload.get("dataset_status") != "CONSUMED_RESEARCH_DATA":
        raise ValueError(
            "Unexpected dataset status."
        )

    definition = payload.get("definition", {})

    if str(
        definition.get("early_failure_threshold_r")
    ) not in {"0.5", "0.50"}:
        raise ValueError(
            "Early Failure threshold changed."
        )

    if definition.get("early_failure_window_bars") != 4:
        raise ValueError(
            "Early Failure observation window changed."
        )

    if (
        definition.get(
            "recovery_requires_strictly_later_bar"
        )
        is not True
    ):
        raise ValueError(
            "Recovery ordering semantics changed."
        )

    summary = payload.get("summary", {})

    if summary.get("trades") != 85:
        raise ValueError(
            "Expected exactly 85 Early Failure trades."
        )

    if (
        summary.get("same_bar_positive_two_r_ambiguous")
        != 0
    ):
        raise ValueError(
            "Unexpected +2R same-bar ambiguity."
        )

    holdout = payload.get("blind_holdout", {})

    if holdout.get("status") != "LOCKED_BLIND_HOLDOUT":
        raise ValueError(
            "Blind holdout is not locked."
        )

    for field in (
        "loaded",
        "revealed",
        "consumed",
        "evaluated",
    ):
        if holdout.get(field) is not False:
            raise ValueError(
                f"Blind holdout integrity failed: {field}."
            )

    store = H9PreregistrationStore()

    run_id, path, manifest = store.prepare(
        source_v33_run_id=EXPECTED_V33_RUN_ID,
        source_v331_run_id=EXPECTED_V331_RUN_ID,
        source_early_failure_sha256=diagnostic_sha256,
    )

    print()
    print("H9 PREREGISTRATION COMPLETE")
    print(f"Run ID: {run_id}")
    print(
        f"SHA-256: {manifest['configuration_sha256']}"
    )
    print(f"Manifest: {path}")
    print()
    print("Frozen mechanism:")
    print("  Reference: H5_Q25")
    print("  Trigger: -0.5R")
    print("  Observation window: first 4 held bars")
    print("  Trigger source: CLOSED candle")
    print("  Execution: NEXT BAR OPEN")
    print("  Entry logic: unchanged")
    print("  Original stop/target: unchanged before exit")
    print()
    print("Parameter search: NO")
    print("H9 replay executed: NO")
    print(
        "Blind holdout: LOCKED | NOT LOADED | "
        "NOT REVEALED | NOT CONSUMED | NOT EVALUATED"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())