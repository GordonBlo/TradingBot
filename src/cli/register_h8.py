"""Register the immutable H8 research hypothesis before implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.hypotheses.h8_preregistration import H8PreregistrationStore


EXPECTED_V33_RUN_ID = "b104ea78b4c88bcf"
EXPECTED_V331_RUN_ID = "ef593be9a0bb2697"
EXPECTED_REFERENCE_TRADES = 197


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register immutable H8 +1R break-even research hypothesis."
    )

    parser.add_argument(
        "--v33-summary",
        default=(
            "reports/v3_3_h5/"
            "b104ea78b4c88bcf/summary.json"
        ),
    )

    parser.add_argument(
        "--v331-summary",
        default=(
            "reports/diagnostics/v331_h5/"
            "ef593be9a0bb2697/summary.json"
        ),
    )

    parser.add_argument(
        "--output-root",
        default="research/h8",
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    v33 = _load_json(Path(args.v33_summary))
    v331 = _load_json(Path(args.v331_summary))

    # --------------------------------------------------------
    # Frozen V3.3 source verification
    # --------------------------------------------------------

    if v33.get("run_id") != EXPECTED_V33_RUN_ID:
        raise ValueError(
            "Unexpected V3.3 source run. "
            f"Expected {EXPECTED_V33_RUN_ID}, "
            f"got {v33.get('run_id')}."
        )

    holdout = v33.get("blind_holdout", {})

    if holdout.get("status") != "LOCKED_BLIND_HOLDOUT":
        raise ValueError("V3.3 blind holdout is not locked.")

    for field in ("revealed", "consumed", "evaluated"):
        if holdout.get(field) is not False:
            raise ValueError(
                f"V3.3 blind holdout integrity failed: {field}."
            )

    # --------------------------------------------------------
    # Frozen V3.3.1 diagnostic verification
    # --------------------------------------------------------

    if v331.get("run_id") != EXPECTED_V331_RUN_ID:
        raise ValueError(
            "Unexpected V3.3.1 diagnostic run. "
            f"Expected {EXPECTED_V331_RUN_ID}, "
            f"got {v331.get('run_id')}."
        )

    if v331.get("candidate") != "H5_Q25":
        raise ValueError("V3.3.1 source candidate must be H5_Q25.")

    if v331.get("dataset_status") != "CONSUMED_RESEARCH_DATA":
        raise ValueError(
            "H8 may only be preregistered from consumed research data."
        )

    if v331.get("eligible_windows") != 11:
        raise ValueError("Expected exactly 11 eligible windows.")

    if v331.get("trade_count") != EXPECTED_REFERENCE_TRADES:
        raise ValueError(
            "Frozen H5_Q25 reference must reproduce 197 trades."
        )

    if v331.get("frozen_reproduction_verified") is not True:
        raise ValueError("Frozen H5_Q25 reproduction is not verified.")

    accounting = v331.get("r_accounting", {})

    if accounting.get("verified") is not True:
        raise ValueError("V3.3.1 R accounting is not verified.")

    diagnostic_holdout = v331.get("blind_holdout", {})

    if (
        diagnostic_holdout.get("status")
        != "LOCKED_BLIND_HOLDOUT"
    ):
        raise ValueError(
            "V3.3.1 blind holdout is not locked."
        )

    for field in ("loaded", "revealed", "consumed", "evaluated"):
        if diagnostic_holdout.get(field) is not False:
            raise ValueError(
                f"V3.3.1 blind holdout integrity failed: {field}."
            )

    # --------------------------------------------------------
    # Immutable preregistration
    # --------------------------------------------------------

    store = H8PreregistrationStore(args.output_root)

    run_id, path, payload = store.prepare(
        source_v33_run_id=EXPECTED_V33_RUN_ID,
        source_v331_diagnostic_run_id=EXPECTED_V331_RUN_ID,
    )

    print()
    print("H8 PREREGISTRATION COMPLETE")
    print(f"Run ID: {run_id}")
    print(f"SHA-256: {payload['configuration_sha256']}")
    print(f"Manifest: {path}")
    print()
    print("Frozen mechanism:")
    print("  H5_Q25 entries: unchanged")
    print("  Trigger: +1R on CLOSED candle")
    print("  Activation: NEXT BAR")
    print("  Protective stop: ENTRY PRICE")
    print("  Target: +2R unchanged")
    print("  STOP_FIRST unchanged")
    print()
    print("Parameter search: NO")
    print("H8 replay executed: NO")
    print(
        "Blind holdout: LOCKED | NOT LOADED | "
        "NOT REVEALED | NOT CONSUMED | NOT EVALUATED"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())