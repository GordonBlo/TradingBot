from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = 1
MANIFEST_ROOT = Path("research/v9_l2_early_information")
SYMBOL = "BTCUSDC"
INTERVAL = "1s"
RIDGE_ALPHA = 1.0
PERMUTATION_COUNT = 1_000
PERMUTATION_SEED = 20260825
SIGNIFICANCE_ALPHA = 0.05
PRIMARY_HORIZON_SECONDS = 30
SECONDARY_HORIZONS_SECONDS = (1, 5, 60)
FROZEN_FEATURES = (
    "spread_bps",
    "microprice_minus_mid_bps",
    "depth_imbalance_1",
    "depth_imbalance_5",
    "depth_imbalance_10",
    "depth_imbalance_20",
    "bid_depth_top_20",
    "ask_depth_top_20",
    "bid_depth_concentration",
    "ask_depth_concentration",
    "bid_depth_added",
    "bid_depth_removed",
    "ask_depth_added",
    "ask_depth_removed",
    "update_intensity",
)


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def prospective_cutoff(created_at: datetime) -> datetime:
    created = created_at.astimezone(UTC)
    floored = created.replace(second=0, microsecond=0)
    return floored.replace(minute=(floored.minute // 15) * 15) + timedelta(minutes=15)


def frozen_definition(created_at: datetime) -> dict[str, Any]:
    cutoff = prospective_cutoff(created_at)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "V9_EARLY_INFORMATION_DIAGNOSTIC",
        "created_at_utc": _utc_text(created_at),
        "prospective_cutoff_utc": _utc_text(cutoff),
        "market": {
            "symbol": SYMBOL,
            "venue": "Binance PUBLIC Spot",
            "source": "V9 persisted raw L2 events",
        },
        "session_policy": {
            "engineering_only": "session_start < prospective_cutoff",
            "research_eligible": "session_start >= prospective_cutoff",
            "cutoff_straddling_sessions": "REJECT",
            "active_sessions": "REJECT",
            "required_state": "CLOSED_IMMUTABLE",
            "required_integrity": {
                "deterministic_replay": "PASS",
                "sequence_gaps": 0,
                "invalid_events": 0,
                "crossed_or_invalid_book_states": 0,
                "feature_reconstruction": "PASS",
                "deterministic_hashes": "VERIFIED",
            },
        },
        "sampling": {
            "frequency": INTERVAL,
            "timestamp": "exact UTC one-second bucket close",
            "causality": "feature_source_timestamp <= sample_timestamp",
            "missing_values": "preserved; train-fold-only median preprocessing",
            "partial_buckets": "explicit and retained only when required values are valid",
        },
        "features": list(FROZEN_FEATURES),
        "targets": {
            "definition": "log(mid_price[T+h] / mid_price[T])",
            "alignment": "exact timestamps only; no fill or interpolation",
            "primary_horizon_seconds": PRIMARY_HORIZON_SECONDS,
            "secondary_descriptive_horizons_seconds": list(SECONDARY_HORIZONS_SECONDS),
        },
        "readiness_gate": {
            "minimum_eligible_closed_sessions": 8,
            "minimum_total_eligible_hours": 20,
            "minimum_utc_dates": 2,
            "all_sessions_pass_integrity": True,
        },
        "evaluation": {
            "folds": "chronological expanding session-blocked out-of-sample",
            "preprocessing": "fit on training sessions only",
            "input_imputation": "training-fold median only",
            "model": "Ridge",
            "ridge_alpha": RIDGE_ALPHA,
            "baseline": "constant training-target mean",
            "primary_metric": "held-out Spearman correlation",
            "null": "within-session-block target permutation with full fold refit",
            "permutations": PERMUTATION_COUNT,
            "permutation_seed": PERMUTATION_SEED,
            "permutation_tail": "one-sided positive held-out Spearman",
            "significance_alpha": SIGNIFICANCE_ALPHA,
        },
        "classification": {
            "STABLE_L2_INFORMATION": {
                "all_required": True,
                "aggregate_primary_spearman": "> 0",
                "permutation_p": "< 0.05",
                "model_mse": "< constant_training_baseline_mse",
                "positive_direction_session_fraction": ">= 0.70",
            },
            "WEAK_OR_UNSTABLE_L2_INFORMATION": (
                "at least one stable gate has positive evidence, but not all gates pass"
            ),
            "NO_STABLE_L2_INFORMATION": "none of the stable gates has positive evidence",
        },
        "restrictions": {
            "additional_features": False,
            "threshold_search": False,
            "real_predictive_evaluation_during_preregistration": False,
            "blind_holdout": {
                "status": "LOCKED",
                "loaded": False,
                "revealed": False,
                "consumed": False,
                "evaluated": False,
            },
        },
    }


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_manifest(created_at: datetime) -> dict[str, Any]:
    definition = frozen_definition(created_at)
    run_id = hashlib.sha256(_canonical_bytes(definition)).hexdigest()[:16]
    return {"run_id": run_id, "definition": definition}


def verify_manifest(manifest: dict[str, Any]) -> None:
    definition = manifest.get("definition")
    if not isinstance(definition, dict):
        raise ValueError("invalid V9 diagnostic preregistration definition")
    created = datetime.fromisoformat(str(definition["created_at_utc"]).replace("Z", "+00:00"))
    expected = build_manifest(created)
    if manifest != expected:
        raise ValueError("V9 diagnostic preregistration manifest mismatch")


def write_or_verify_manifest(
    root: Path = MANIFEST_ROOT,
    *,
    now: Callable[[], datetime] | None = None,
) -> tuple[Path, dict[str, Any]]:
    paths = sorted(root.glob("*/manifest.json")) if root.exists() else []
    if len(paths) > 1:
        raise ValueError("multiple V9 diagnostic preregistration manifests found")
    if paths:
        manifest = json.loads(paths[0].read_text(encoding="utf-8"))
        verify_manifest(manifest)
        return paths[0], manifest

    created_at = (now or (lambda: datetime.now(UTC)))()
    manifest = build_manifest(created_at)
    path = root / manifest["run_id"] / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=False)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path, manifest


def main() -> int:
    path, manifest = write_or_verify_manifest()
    definition = manifest["definition"]
    print(f"Preregistration ID: {manifest['run_id']}")
    print(f"Prospective cutoff: {definition['prospective_cutoff_utc']}")
    print(f"Manifest: {path}")
    print("PREREGISTRATION COMPLETE — NO PREDICTIVE DIAGNOSTIC EXECUTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
