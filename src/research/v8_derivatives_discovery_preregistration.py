"""Immutable V8 derivatives discovery diagnostic preregistration only."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.research.v8_funding_context_preregistration import (
    V6_RUN_ID,
    V8_DERIVATIVES_DATASET_DEFINITION_SHA256,
    V8_DERIVATIVES_DATASET_ID,
)


VERSION = "8.2"
BASELINE_ID = "V8_DERIVATIVES_DISCOVERY_DIAGNOSTIC"
V6_CANDIDATE_COUNT = 443
ELIGIBLE_WINDOWS = (
    "W002",
    "W003",
    "W004",
    "W005",
    "W006",
    "W007",
    "W008",
    "W009",
    "W010",
    "W012",
    "W013",
)
PERMUTATION_SEED = 20260824
PERMUTATION_COUNT = 1000
RIDGE_ALPHA = "1.0"
SIGNIFICANCE_ALPHA = "0.05"
PREDICTIVE_DIRECTION_MINIMUM_WINDOWS = 7
STABLE_COEFFICIENT_MINIMUM_FOLDS = 7
HOLDOUT_START = "2025-08-01T00:00:00+00:00"
HOLDOUT_END = "2026-02-01T00:00:00+00:00"


def causal_timestamp_allowed(
    source_timestamp: datetime | None,
    bucket_close: datetime,
) -> bool:
    """Reject missing, naive, or future observations without substituting data."""
    if source_timestamp is None:
        return False
    if (
        source_timestamp.tzinfo is None
        or source_timestamp.utcoffset() is None
        or bucket_close.tzinfo is None
        or bucket_close.utcoffset() is None
    ):
        raise ValueError("Causal timestamps must be timezone-aware.")
    return source_timestamp.astimezone(timezone.utc) <= bucket_close.astimezone(
        timezone.utc
    )


def leave_one_window_out_folds(
    window_ids: tuple[str, ...] = ELIGIBLE_WINDOWS,
) -> tuple[tuple[tuple[str, ...], str], ...]:
    """Build the fixed 11-fold split without reading features or outcomes."""
    if window_ids != ELIGIBLE_WINDOWS:
        raise ValueError("V8 discovery requires the exact 11 frozen windows.")
    return tuple(
        (tuple(item for item in window_ids if item != held_out), held_out)
        for held_out in window_ids
    )


def _feature_definitions() -> list[dict]:
    return [
        {
            "name": "FUNDING_RATE_LEVEL",
            "source": "LATEST_KNOWN_FUNDING_RATE",
            "timestamp": "FUNDING_TIMESTAMP",
            "formula": "FUNDING_RATE_AT_T",
        },
        {
            "name": "FUNDING_AGE_MINUTES",
            "source": "FUNDING_TIMESTAMP",
            "timestamp": "FUNDING_TIMESTAMP",
            "formula": "MINUTES_BETWEEN_T_AND_FUNDING_TIMESTAMP",
        },
        {
            "name": "OI_LEVEL",
            "source": "SUM_OPEN_INTEREST",
            "timestamp": "OPEN_INTEREST_TIMESTAMP",
            "formula": "SUM_OPEN_INTEREST_AT_T",
        },
        *[
            {
                "name": f"OI_CHANGE_{horizon}_MINUTES",
                "source": "SUM_OPEN_INTEREST",
                "timestamp": "OPEN_INTEREST_TIMESTAMP",
                "horizon_minutes": horizon,
                "formula": "OI_AT_T_MINUS_OI_AT_T_MINUS_H",
            }
            for horizon in (15, 60, 240)
        ],
        *[
            {
                "name": f"OI_FRACTIONAL_CHANGE_{horizon}_MINUTES",
                "source": "SUM_OPEN_INTEREST",
                "timestamp": "OPEN_INTEREST_TIMESTAMP",
                "horizon_minutes": horizon,
                "formula": "(OI_AT_T_MINUS_OI_AT_T_MINUS_H)_DIVIDED_BY_OI_AT_T_MINUS_H",
                "zero_or_missing_denominator": "MISSING_FEATURE",
            }
            for horizon in (15, 60, 240)
        ],
        {
            "name": "TAKER_LONG_SHORT_VOLUME_RATIO_LEVEL",
            "source": "SUM_TAKER_LONG_SHORT_VOL_RATIO",
            "timestamp": "METRICS_TIMESTAMP",
            "formula": "SUM_TAKER_LONG_SHORT_VOL_RATIO_AT_T",
        },
        *[
            {
                "name": f"TAKER_LONG_SHORT_VOLUME_RATIO_CHANGE_{horizon}_MINUTES",
                "source": "SUM_TAKER_LONG_SHORT_VOL_RATIO",
                "timestamp": "METRICS_TIMESTAMP",
                "horizon_minutes": horizon,
                "formula": "RATIO_AT_T_MINUS_RATIO_AT_T_MINUS_H",
            }
            for horizon in (15, 60)
        ],
        *[
            {
                "name": f"{field.upper()}_{suffix}",
                "source": field.upper(),
                "timestamp": "METRICS_TIMESTAMP",
                "horizon_minutes": 60 if suffix == "CHANGE_60_MINUTES" else None,
                "formula": (
                    "RATIO_AT_T_MINUS_RATIO_AT_T_MINUS_60_MINUTES"
                    if suffix == "CHANGE_60_MINUTES"
                    else "RATIO_AT_T"
                ),
            }
            for field in (
                "count_toptrader_long_short_ratio",
                "sum_toptrader_long_short_ratio",
                "count_long_short_ratio",
            )
            for suffix in ("LEVEL", "CHANGE_60_MINUTES")
        ],
        {
            "name": "PREMIUM_INDEX_CLOSE_LEVEL",
            "source": "PREMIUM_INDEX_CLOSE",
            "timestamp": "PREMIUM_INDEX_TIMESTAMP",
            "formula": "PREMIUM_INDEX_CLOSE_AT_T",
        },
        *[
            {
                "name": f"PREMIUM_INDEX_CLOSE_CHANGE_{horizon}_MINUTES",
                "source": "PREMIUM_INDEX_CLOSE",
                "timestamp": "PREMIUM_INDEX_TIMESTAMP",
                "horizon_minutes": horizon,
                "formula": "PREMIUM_INDEX_CLOSE_AT_T_MINUS_PREMIUM_INDEX_CLOSE_AT_T_MINUS_H",
            }
            for horizon in (15, 60)
        ],
        {
            "name": "DERIVED_MARK_INDEX_PREMIUM_LEVEL",
            "source": "MARK_AND_INDEX_PRICE_CLOSES",
            "timestamp": "MARK_PRICE_TIMESTAMP_AND_INDEX_PRICE_TIMESTAMP",
            "formula": "(MARK_PRICE_MINUS_INDEX_PRICE)_DIVIDED_BY_INDEX_PRICE",
        },
        *[
            {
                "name": f"DERIVED_MARK_INDEX_PREMIUM_CHANGE_{horizon}_MINUTES",
                "source": "MARK_AND_INDEX_PRICE_CLOSES",
                "timestamp": "MARK_PRICE_TIMESTAMP_AND_INDEX_PRICE_TIMESTAMP",
                "horizon_minutes": horizon,
                "formula": "DERIVED_PREMIUM_AT_T_MINUS_DERIVED_PREMIUM_AT_T_MINUS_H",
            }
            for horizon in (15, 60)
        ],
        {
            "name": "SPOT_RETURN_60M_X_OI_CHANGE_60M",
            "source": "BTCUSDC_SPOT_CLOSE_AND_SUM_OPEN_INTEREST",
            "timestamp": "SPOT_CLOSE_AND_OPEN_INTEREST_TIMESTAMP",
            "horizon_minutes": 60,
            "formula": "SPOT_RETURN_60_MINUTES_TIMES_OI_CHANGE_60_MINUTES",
        },
        {
            "name": "SPOT_RETURN_60M_X_DERIVED_PREMIUM_CHANGE_60M",
            "source": "BTCUSDC_SPOT_CLOSE_AND_MARK_INDEX_PRICES",
            "timestamp": "SPOT_CLOSE_MARK_PRICE_TIMESTAMP_AND_INDEX_PRICE_TIMESTAMP",
            "horizon_minutes": 60,
            "formula": "SPOT_RETURN_60_MINUTES_TIMES_DERIVED_MARK_INDEX_PREMIUM_CHANGE_60_MINUTES",
        },
        {
            "name": "SPOT_RETURN_60M_X_TAKER_FLOW_DEVIATION_FROM_1",
            "source": "BTCUSDC_SPOT_CLOSE_AND_SUM_TAKER_LONG_SHORT_VOL_RATIO",
            "timestamp": "SPOT_CLOSE_AND_METRICS_TIMESTAMP",
            "horizon_minutes": 60,
            "formula": "SPOT_RETURN_60_MINUTES_TIMES_(TAKER_RATIO_AT_T_MINUS_1)",
        },
    ]


def _definition() -> dict:
    return {
        "version": VERSION,
        "baseline_id": BASELINE_ID,
        "purpose": (
            "DETERMINE_WHETHER_BTCUSDT_DERIVATIVES_CONTEXT_CONTAINS_STABLE_"
            "OUT_OF_WINDOW_INFORMATION_ABOUT_FROZEN_V6_H0_TRADE_QUALITY"
        ),
        "universe": {
            "frozen_v6_h0_run_id": V6_RUN_ID,
            "candidate_schedule": "FROZEN_V6_H0_CLOSED_SIGNAL_CANDIDATES_DIRECT",
            "candidate_count": V6_CANDIDATE_COUNT,
            "eligible_consumed_windows": list(ELIGIBLE_WINDOWS),
            "window_count": len(ELIGIBLE_WINDOWS),
        },
        "market": {
            "traded_market": "BTCUSDC_BINANCE_PUBLIC_SPOT",
            "execution_interval": "15m",
            "direction": "LONG_ONLY",
            "derivatives_context_market": "BTCUSDT_BINANCE_PUBLIC_USDM_PERPETUAL",
        },
        "derivatives_context_dataset": {
            "dataset_id": V8_DERIVATIVES_DATASET_ID,
            "definition_sha256": V8_DERIVATIVES_DATASET_DEFINITION_SHA256,
            "only_checksum_verified_official_binance_sources": True,
            "metrics_ratio_source": "PARSED_FUTURESMETRICSRECORD_FROM_IMMUTABLE_METRICS_ARCHIVES",
            "metrics_ratio_fields": [
                "COUNT_TOPTRADER_LONG_SHORT_RATIO",
                "SUM_TOPTRADER_LONG_SHORT_RATIO",
                "COUNT_LONG_SHORT_RATIO",
                "SUM_TAKER_LONG_SHORT_VOL_RATIO",
            ],
        },
        "targets": {
            "primary": "FROZEN_V6_TRADE_FRICTIONLESS_GROSS_R",
            "secondary_descriptive": "FROZEN_V6_TRADE_NET_R",
            "target_is_not_used_for_entry_filter_or_threshold_selection": True,
        },
        "causal_alignment": {
            "decision_time": "BTCUSDC_SPOT_SIGNAL_CANDLE_CLOSE_T",
            "all_source_timestamps_must_be_less_than_or_equal_to_bucket_close": True,
            "level_feature": "SOURCE_TIMESTAMP_LESS_THAN_OR_EQUAL_TO_T",
            "horizon_change_feature": (
                "CURRENT_SOURCE_TIMESTAMP_LESS_THAN_OR_EQUAL_TO_T_AND_"
                "HISTORICAL_SOURCE_TIMESTAMP_LESS_THAN_OR_EQUAL_TO_T_MINUS_H"
            ),
            "metrics_maximum_source_age_minutes": 5,
            "missing_or_future_observation": "PRESERVE_AS_MISSING_NO_INTERPOLATION_NO_FUTURE_FILL",
            "spot_return_60_minutes": "(BTCUSDC_CLOSE_AT_T_DIVIDED_BY_BTCUSDC_CLOSE_AT_T_MINUS_60_MINUTES)_MINUS_1",
        },
        "fixed_features": _feature_definitions(),
        "feature_family_constraints": {
            "only_fixed_interactions": [
                "SPOT_RETURN_60M_X_OI_CHANGE_60M",
                "SPOT_RETURN_60M_X_DERIVED_PREMIUM_CHANGE_60M",
                "SPOT_RETURN_60M_X_TAKER_FLOW_DEVIATION_FROM_1",
            ],
            "no_arbitrary_polynomial_combinations": True,
            "no_alternative_horizons": True,
            "no_oi_value_variant": True,
            "no_feature_threshold_mining": True,
        },
        "univariate_diagnostics": {
            "per_feature_outputs": [
                "VALID_SAMPLE_COUNT",
                "SPEARMAN_CORRELATION_WITH_GROSS_R",
                "WINNER_LOSER_COHENS_D_EFFECT_SIZE",
                "SIGN_CONSISTENCY_ACROSS_11_WINDOWS",
            ],
            "winner_definition": "GROSS_R_STRICTLY_GREATER_THAN_0",
            "loser_definition": "GROSS_R_LESS_THAN_OR_EQUAL_TO_0",
            "effect_direction": "WINNER_MEAN_FEATURE_MINUS_LOSER_MEAN_FEATURE",
            "sign_consistency": "WINDOW_SPEARMAN_SIGN_MATCHES_AGGREGATE_SPEARMAN_SIGN",
        },
        "leave_one_window_out": {
            "fold_count": len(ELIGIBLE_WINDOWS),
            "held_out_unit": "ENTIRE_RESEARCH_WINDOW",
            "training_windows_per_fold": len(ELIGIBLE_WINDOWS) - 1,
            "preprocessing": {
                "feature_median_imputation": "FIT_ON_TRAINING_FOLD_ONLY",
                "standardization": "FIT_ON_TRAINING_FOLD_ONLY",
                "zero_variance_feature": "TRANSFORM_TO_ZERO_WITH_COEFFICIENT_REPORTED_AS_ZERO",
                "raw_missingness": "PRESERVED_AND_REPORTED",
            },
            "primary_model": {
                "family": "RIDGE_LINEAR_REGRESSION",
                "regularization_alpha": RIDGE_ALPHA,
                "fit_intercept": True,
                "hyperparameter_search": False,
                "feature_selection_using_held_out_outcomes": False,
            },
            "constant_baseline": "TRAINING_FOLD_MEAN_GROSS_R",
        },
        "null_comparison": {
            "scheme": "DETERMINISTIC_WITHIN_WINDOW_GROSS_R_PERMUTATION",
            "permutation_count": PERMUTATION_COUNT,
            "seed": PERMUTATION_SEED,
            "statistic": "AGGREGATE_HELD_OUT_PREDICTION_VS_GROSS_R_SPEARMAN",
            "p_value": "(1_PLUS_PERMUTED_STATISTICS_GREATER_THAN_OR_EQUAL_TO_OBSERVED)_DIVIDED_BY_(1_PLUS_PERMUTATION_COUNT)",
            "alternative": "ONE_SIDED_POSITIVE",
            "alpha": SIGNIFICANCE_ALPHA,
        },
        "stability_outputs": [
            "AGGREGATE_HELD_OUT_PREDICTION_VS_GROSS_R_SPEARMAN",
            "HELD_OUT_PREDICTION_ERROR_VS_CONSTANT_TRAINING_BASELINE",
            "COEFFICIENT_SIGN_CONSISTENCY_ACROSS_FOLDS",
            "PER_WINDOW_PREDICTIVE_DIRECTION",
            "FEATURE_MISSINGNESS",
        ],
        "classification": {
            "stable_derivatives_signal": {
                "all_required": True,
                "conditions": [
                    "AGGREGATE_HELD_OUT_PREDICTION_GROSS_R_SPEARMAN_STRICTLY_POSITIVE",
                    "ONE_SIDED_PERMUTATION_P_VALUE_LESS_THAN_OR_EQUAL_TO_0_05",
                    "HELD_OUT_MODEL_MSE_STRICTLY_LESS_THAN_CONSTANT_TRAINING_BASELINE_MSE",
                    "POSITIVE_PREDICTIVE_DIRECTION_IN_AT_LEAST_7_OF_11_WINDOWS",
                    "AT_LEAST_ONE_FROZEN_ECONOMIC_FEATURE_OR_INTERACTION_HAS_MODAL_NONZERO_COEFFICIENT_SIGN_IN_AT_LEAST_7_OF_11_FOLDS",
                ],
            },
            "weak_or_unstable_signal": {
                "definition": "SOME_AGGREGATE_EVIDENCE_EXISTS_BUT_STABLE_DERIVATIVES_SIGNAL_CONDITIONS_FAIL",
                "aggregate_evidence": [
                    "AGGREGATE_HELD_OUT_PREDICTION_GROSS_R_SPEARMAN_STRICTLY_POSITIVE",
                    "HELD_OUT_MODEL_MSE_STRICTLY_LESS_THAN_CONSTANT_TRAINING_BASELINE_MSE",
                ],
            },
            "no_stable_derivatives_signal": {
                "definition": "NO_AGGREGATE_EVIDENCE_ABOVE_BASELINE_OR_PERMUTATION_NULL",
            },
        },
        "prohibitions": {
            "v8_trading_replay": False,
            "entry_filter": False,
            "profitability_optimization": False,
            "neural_network": False,
            "tree_model_search": False,
            "hyperparameter_sweep": False,
            "post_result_alternative_horizons": False,
        },
        "dataset_policy": {
            "research_data_status": "CONSUMED_RESEARCH_DATA_ONLY",
            "eligible_windows": list(ELIGIBLE_WINDOWS),
            "blind_holdout": {
                "status": "LOCKED_BLIND_HOLDOUT",
                "start": HOLDOUT_START,
                "end": HOLDOUT_END,
                "downloaded": False,
                "loaded": False,
                "revealed": False,
                "consumed": False,
                "evaluated": False,
            },
        },
        "research_rule": "PREREGISTRATION_ONLY_NO_DIAGNOSTIC_EXECUTION_OR_OUTCOME_INSPECTION",
    }


def _canonical_json(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def build_manifest() -> dict:
    definition = _definition()
    digest = hashlib.sha256(_canonical_json(definition).encode("utf-8")).hexdigest()
    return {"run_id": digest[:16], "definition_sha256": digest, **definition}


def write_manifest(
    root: str | Path = "research/v8_derivatives_discovery",
) -> Path:
    manifest = build_manifest()
    path = Path(root) / manifest["run_id"] / "manifest.json"
    serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError("Existing V8 discovery manifest differs from frozen definition.")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")
    return path


def main() -> int:
    manifest = build_manifest()
    path = write_manifest()
    print("V8 DERIVATIVES DISCOVERY DIAGNOSTIC PREREGISTRATION")
    print(f"Run ID: {manifest['run_id']}")
    print("Universe: 443 frozen V6-H0 candidates across 11 consumed windows")
    print("Protocol: fixed causal features, LOOWO ridge, 1,000 within-window permutations")
    print(f"Manifest: {path}")
    print("PREREGISTRATION COMPLETE — NO DIAGNOSTIC EXECUTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
