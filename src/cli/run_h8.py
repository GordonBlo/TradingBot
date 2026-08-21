"""Guarded H8 research runner.

At this stage only dry-run integrity validation is allowed.
No H8 historical replay is implemented here yet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_H8_RUN_ID = "369586c25375c0ef"
EXPECTED_V33_RUN_ID = "b104ea78b4c88bcf"
EXPECTED_V331_RUN_ID = "ef593be9a0bb2697"

EXPECTED_REFERENCE_CANDIDATE = "H5_Q25"
EXPECTED_REFERENCE_TRADES = 197
EXPECTED_ELIGIBLE_WINDOWS = 11


def _load_json(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Required research artifact not found: {path}"
        )

    return json.loads(path.read_text(encoding="utf-8"))


def _require_false(
    payload: dict,
    fields: tuple[str, ...],
    *,
    label: str,
) -> None:
    for field in fields:
        if payload.get(field) is not False:
            raise ValueError(
                f"{label} integrity failed: {field} must be false."
            )


def validate_h8_manifest(payload: dict) -> None:
    if payload.get("run_id") != EXPECTED_H8_RUN_ID:
        raise ValueError("Unexpected H8 preregistration run ID.")

    if payload.get("source_v33_run_id") != EXPECTED_V33_RUN_ID:
        raise ValueError("Unexpected H8 V3.3 source run.")

    if (
        payload.get("source_v331_diagnostic_run_id")
        != EXPECTED_V331_RUN_ID
    ):
        raise ValueError("Unexpected H8 V3.3.1 source run.")

    if payload.get("dataset_status") != "CONSUMED_RESEARCH_DATA":
        raise ValueError("H8 dataset status is invalid.")

    if (
        payload.get("reference_candidate")
        != EXPECTED_REFERENCE_CANDIDATE
    ):
        raise ValueError("H8 reference candidate is not H5_Q25.")

    if payload.get("reference_trade_count") != EXPECTED_REFERENCE_TRADES:
        raise ValueError("H8 reference trade count is not 197.")

    if payload.get("blind_holdout_status") != "LOCKED_BLIND_HOLDOUT":
        raise ValueError("H8 blind holdout is not locked.")

    _require_false(
        payload,
        (
            "blind_holdout_revealed",
            "blind_holdout_consumed",
            "blind_holdout_evaluated",
        ),
        label="H8 blind holdout",
    )

    definition = payload.get("definition", {})

    expected_definition = {
        "hypothesis_id": "H8",
        "activation_timing": "NEXT_BAR_AFTER_CLOSED_TRIGGER",
        "protective_stop": "ENTRY_PRICE",
        "protective_stop_semantics": (
            "PRICE_BREAK_EVEN_NOT_NET_BREAK_EVEN"
        ),
        "preserve_h5_q25_entries": True,
        "preserve_original_atr_stop_until_activation": True,
        "preserve_take_profit": True,
        "partial_exit": False,
        "ambiguous_bar_policy": "STOP_FIRST",
    }

    for field, expected in expected_definition.items():
        if definition.get(field) != expected:
            raise ValueError(
                f"H8 definition changed: {field}."
            )

    if str(definition.get("trigger_r")) not in {"1", "1.0"}:
        raise ValueError("H8 trigger is not frozen at +1R.")

    if str(definition.get("take_profit_r")) not in {"2", "2.0"}:
        raise ValueError("H8 target is not frozen at +2R.")

    integrity = payload.get("research_integrity", {})

    forbidden_true = (
        "parameter_search",
        "alternative_trigger_R_values",
        "partial_exit_research",
        "trailing_stop_research",
        "entry_logic_changes",
        "target_changes",
        "ATR_stop_changes_before_activation",
        "blind_holdout_access",
    )

    _require_false(
        integrity,
        forbidden_true,
        label="H8 research policy",
    )


def validate_v33_summary(payload: dict) -> None:
    if payload.get("run_id") != EXPECTED_V33_RUN_ID:
        raise ValueError("Unexpected V3.3 frozen source run.")

    holdout = payload.get("blind_holdout", {})

    if holdout.get("status") != "LOCKED_BLIND_HOLDOUT":
        raise ValueError("V3.3 blind holdout is not locked.")

    _require_false(
        holdout,
        ("revealed", "consumed", "evaluated"),
        label="V3.3 blind holdout",
    )


def validate_v331_summary(payload: dict) -> None:
    if payload.get("run_id") != EXPECTED_V331_RUN_ID:
        raise ValueError("Unexpected V3.3.1 diagnostic run.")

    if payload.get("source_v33_run_id") != EXPECTED_V33_RUN_ID:
        raise ValueError("V3.3.1 source V3.3 run changed.")

    if payload.get("candidate") != EXPECTED_REFERENCE_CANDIDATE:
        raise ValueError("V3.3.1 candidate is not H5_Q25.")

    if payload.get("dataset_status") != "CONSUMED_RESEARCH_DATA":
        raise ValueError("V3.3.1 dataset status changed.")

    if payload.get("eligible_windows") != EXPECTED_ELIGIBLE_WINDOWS:
        raise ValueError(
            "V3.3.1 must contain exactly 11 eligible windows."
        )

    if payload.get("trade_count") != EXPECTED_REFERENCE_TRADES:
        raise ValueError(
            "Frozen H5_Q25 must reproduce exactly 197 trades."
        )

    if payload.get("frozen_reproduction_verified") is not True:
        raise ValueError("H5_Q25 frozen reproduction is not verified.")

    accounting = payload.get("r_accounting", {})

    if accounting.get("verified") is not True:
        raise ValueError("V3.3.1 R accounting is not verified.")

    warmup = payload.get("warmup_parity", {})

    if warmup.get("continuous_state_trade_count") != 197:
        raise ValueError("Continuous-state trade count changed.")

    if warmup.get("continuous_state_entry_signal_differences") != 0:
        raise ValueError("Warm-up causes entry-signal differences.")

    if warmup.get("changes_frozen_results") is not False:
        raise ValueError("Warm-up changes frozen results.")

    holdout = payload.get("blind_holdout", {})

    if holdout.get("status") != "LOCKED_BLIND_HOLDOUT":
        raise ValueError("V3.3.1 blind holdout is not locked.")

    _require_false(
        holdout,
        ("loaded", "revealed", "consumed", "evaluated"),
        label="V3.3.1 blind holdout",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Guarded H8 research runner."
    )

    parser.add_argument(
        "--h8-manifest",
        default=(
            "research/h8/"
            "369586c25375c0ef/manifest.json"
        ),
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
        "--dry-run",
        action="store_true",
        help=(
            "Validate frozen H8 research state without "
            "executing any historical replay."
        ),
    )

    return parser


def main() -> int:
    args = build_parser().parse_args()

    if not args.dry_run:
        raise SystemExit(
            "H8 REPLAY DISABLED. "
            "Run with --dry-run only until integrity guard is committed."
        )

    h8 = _load_json(args.h8_manifest)
    v33 = _load_json(args.v33_summary)
    v331 = _load_json(args.v331_summary)

    validate_h8_manifest(h8)
    validate_v33_summary(v33)
    validate_v331_summary(v331)

    print()
    print("H8 DRY-RUN INTEGRITY CHECK")
    print(f"H8 preregistration: {EXPECTED_H8_RUN_ID} VERIFIED")
    print(f"Source V3.3: {EXPECTED_V33_RUN_ID} VERIFIED")
    print(f"Source V3.3.1: {EXPECTED_V331_RUN_ID} VERIFIED")
    print()
    print("Frozen mechanism:")
    print("  Candidate: H5_Q25")
    print("  Trigger: +1R on CLOSED candle")
    print("  Activation: NEXT BAR")
    print("  Protective stop: ENTRY PRICE")
    print("  Target: +2R unchanged")
    print("  STOP_FIRST unchanged")
    print()
    print("Dataset: CONSUMED_RESEARCH_DATA")
    print("Eligible windows: 11")
    print("Frozen H5_Q25 reference trades: 197")
    print("Warm-up entry-signal differences: 0")
    print()
    print(
        "Blind holdout: LOCKED | NOT LOADED | "
        "NOT REVEALED | NOT CONSUMED | NOT EVALUATED"
    )
    print()
    print("DRY RUN PASSED — NO H8 REPLAY EXECUTED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())