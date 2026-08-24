from datetime import datetime, timedelta, timezone

import pytest

from src.research.v8_derivatives_discovery_preregistration import (
    ELIGIBLE_WINDOWS,
    PERMUTATION_COUNT,
    PERMUTATION_SEED,
    RIDGE_ALPHA,
    SIGNIFICANCE_ALPHA,
    V6_CANDIDATE_COUNT,
    build_manifest,
    causal_timestamp_allowed,
    leave_one_window_out_folds,
    write_manifest,
)


T = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)


def test_universe_target_and_market_are_frozen() -> None:
    manifest = build_manifest()
    assert V6_CANDIDATE_COUNT == 443
    assert manifest["run_id"] == "d15c2c2b5bafefd2"
    assert manifest["universe"] == {
        "frozen_v6_h0_run_id": "e1eef7bdd0c37ad4",
        "candidate_schedule": "FROZEN_V6_H0_CLOSED_SIGNAL_CANDIDATES_DIRECT",
        "candidate_count": 443,
        "eligible_consumed_windows": list(ELIGIBLE_WINDOWS),
        "window_count": 11,
    }
    assert manifest["targets"] == {
        "primary": "FROZEN_V6_TRADE_FRICTIONLESS_GROSS_R",
        "secondary_descriptive": "FROZEN_V6_TRADE_NET_R",
        "target_is_not_used_for_entry_filter_or_threshold_selection": True,
    }
    assert manifest["market"] == {
        "traded_market": "BTCUSDC_BINANCE_PUBLIC_SPOT",
        "execution_interval": "15m",
        "direction": "LONG_ONLY",
        "derivatives_context_market": "BTCUSDT_BINANCE_PUBLIC_USDM_PERPETUAL",
    }


def test_fixed_feature_family_uses_only_declared_sources_horizons_and_interactions() -> None:
    manifest = build_manifest()
    features = manifest["fixed_features"]
    names = {row["name"] for row in features}
    assert {
        "FUNDING_RATE_LEVEL",
        "FUNDING_AGE_MINUTES",
        "OI_LEVEL",
        "OI_CHANGE_15_MINUTES",
        "OI_CHANGE_60_MINUTES",
        "OI_CHANGE_240_MINUTES",
        "OI_FRACTIONAL_CHANGE_15_MINUTES",
        "OI_FRACTIONAL_CHANGE_60_MINUTES",
        "OI_FRACTIONAL_CHANGE_240_MINUTES",
        "TAKER_LONG_SHORT_VOLUME_RATIO_LEVEL",
        "TAKER_LONG_SHORT_VOLUME_RATIO_CHANGE_15_MINUTES",
        "TAKER_LONG_SHORT_VOLUME_RATIO_CHANGE_60_MINUTES",
        "COUNT_TOPTRADER_LONG_SHORT_RATIO_LEVEL",
        "SUM_TOPTRADER_LONG_SHORT_RATIO_CHANGE_60_MINUTES",
        "COUNT_LONG_SHORT_RATIO_LEVEL",
        "PREMIUM_INDEX_CLOSE_LEVEL",
        "DERIVED_MARK_INDEX_PREMIUM_LEVEL",
    } <= names
    assert manifest["derivatives_context_dataset"]["metrics_ratio_fields"] == [
        "COUNT_TOPTRADER_LONG_SHORT_RATIO",
        "SUM_TOPTRADER_LONG_SHORT_RATIO",
        "COUNT_LONG_SHORT_RATIO",
        "SUM_TAKER_LONG_SHORT_VOL_RATIO",
    ]
    assert manifest["feature_family_constraints"]["only_fixed_interactions"] == [
        "SPOT_RETURN_60M_X_OI_CHANGE_60M",
        "SPOT_RETURN_60M_X_DERIVED_PREMIUM_CHANGE_60M",
        "SPOT_RETURN_60M_X_TAKER_FLOW_DEVIATION_FROM_1",
    ]
    assert manifest["feature_family_constraints"] == {
        "only_fixed_interactions": manifest["feature_family_constraints"]["only_fixed_interactions"],
        "no_arbitrary_polynomial_combinations": True,
        "no_alternative_horizons": True,
        "no_oi_value_variant": True,
        "no_feature_threshold_mining": True,
    }


def test_causal_timestamp_rule_rejects_missing_and_future_values() -> None:
    assert causal_timestamp_allowed(T, T)
    assert causal_timestamp_allowed(T - timedelta(minutes=60), T)
    assert not causal_timestamp_allowed(None, T)
    assert not causal_timestamp_allowed(T + timedelta(microseconds=1), T)
    with pytest.raises(ValueError, match="timezone-aware"):
        causal_timestamp_allowed(datetime(2024, 1, 1, 12), T)


def test_leave_one_window_out_folds_are_strictly_isolated() -> None:
    folds = leave_one_window_out_folds()
    assert len(folds) == 11
    assert tuple(held_out for _, held_out in folds) == ELIGIBLE_WINDOWS
    for training, held_out in folds:
        assert len(training) == 10
        assert held_out not in training
        assert set(training).isdisjoint({held_out})
        assert set(training) | {held_out} == set(ELIGIBLE_WINDOWS)
    with pytest.raises(ValueError, match="exact 11"):
        leave_one_window_out_folds(ELIGIBLE_WINDOWS[:-1])


def test_fold_only_preprocessing_model_and_null_protocol_are_frozen() -> None:
    manifest = build_manifest()
    assert RIDGE_ALPHA == "1.0"
    assert PERMUTATION_COUNT == 1000
    assert PERMUTATION_SEED == 20260824
    assert SIGNIFICANCE_ALPHA == "0.05"
    loo = manifest["leave_one_window_out"]
    assert loo["fold_count"] == 11
    assert loo["training_windows_per_fold"] == 10
    assert loo["preprocessing"] == {
        "feature_median_imputation": "FIT_ON_TRAINING_FOLD_ONLY",
        "standardization": "FIT_ON_TRAINING_FOLD_ONLY",
        "zero_variance_feature": "TRANSFORM_TO_ZERO_WITH_COEFFICIENT_REPORTED_AS_ZERO",
        "raw_missingness": "PRESERVED_AND_REPORTED",
    }
    assert loo["primary_model"] == {
        "family": "RIDGE_LINEAR_REGRESSION",
        "regularization_alpha": "1.0",
        "fit_intercept": True,
        "hyperparameter_search": False,
        "feature_selection_using_held_out_outcomes": False,
    }
    assert loo["constant_baseline"] == "TRAINING_FOLD_MEAN_GROSS_R"
    assert manifest["null_comparison"] == {
        "scheme": "DETERMINISTIC_WITHIN_WINDOW_GROSS_R_PERMUTATION",
        "permutation_count": 1000,
        "seed": 20260824,
        "statistic": "AGGREGATE_HELD_OUT_PREDICTION_VS_GROSS_R_SPEARMAN",
        "p_value": "(1_PLUS_PERMUTED_STATISTICS_GREATER_THAN_OR_EQUAL_TO_OBSERVED)_DIVIDED_BY_(1_PLUS_PERMUTATION_COUNT)",
        "alternative": "ONE_SIDED_POSITIVE",
        "alpha": "0.05",
    }


def test_univariate_stability_and_classification_are_frozen_before_results() -> None:
    manifest = build_manifest()
    assert manifest["univariate_diagnostics"]["per_feature_outputs"] == [
        "VALID_SAMPLE_COUNT",
        "SPEARMAN_CORRELATION_WITH_GROSS_R",
        "WINNER_LOSER_COHENS_D_EFFECT_SIZE",
        "SIGN_CONSISTENCY_ACROSS_11_WINDOWS",
    ]
    assert manifest["stability_outputs"] == [
        "AGGREGATE_HELD_OUT_PREDICTION_VS_GROSS_R_SPEARMAN",
        "HELD_OUT_PREDICTION_ERROR_VS_CONSTANT_TRAINING_BASELINE",
        "COEFFICIENT_SIGN_CONSISTENCY_ACROSS_FOLDS",
        "PER_WINDOW_PREDICTIVE_DIRECTION",
        "FEATURE_MISSINGNESS",
    ]
    stable = manifest["classification"]["stable_derivatives_signal"]
    assert stable["all_required"] is True
    assert stable["conditions"] == [
        "AGGREGATE_HELD_OUT_PREDICTION_GROSS_R_SPEARMAN_STRICTLY_POSITIVE",
        "ONE_SIDED_PERMUTATION_P_VALUE_LESS_THAN_OR_EQUAL_TO_0_05",
        "HELD_OUT_MODEL_MSE_STRICTLY_LESS_THAN_CONSTANT_TRAINING_BASELINE_MSE",
        "POSITIVE_PREDICTIVE_DIRECTION_IN_AT_LEAST_7_OF_11_WINDOWS",
        "AT_LEAST_ONE_FROZEN_ECONOMIC_FEATURE_OR_INTERACTION_HAS_MODAL_NONZERO_COEFFICIENT_SIGN_IN_AT_LEAST_7_OF_11_FOLDS",
    ]


def test_holdout_and_non_trading_prohibitions_are_explicit() -> None:
    manifest = build_manifest()
    assert manifest["dataset_policy"]["research_data_status"] == "CONSUMED_RESEARCH_DATA_ONLY"
    assert manifest["dataset_policy"]["eligible_windows"] == list(ELIGIBLE_WINDOWS)
    assert manifest["dataset_policy"]["blind_holdout"] == {
        "status": "LOCKED_BLIND_HOLDOUT",
        "start": "2025-08-01T00:00:00+00:00",
        "end": "2026-02-01T00:00:00+00:00",
        "downloaded": False,
        "loaded": False,
        "revealed": False,
        "consumed": False,
        "evaluated": False,
    }
    assert all(value is False for value in manifest["prohibitions"].values())
    assert manifest["research_rule"] == (
        "PREREGISTRATION_ONLY_NO_DIAGNOSTIC_EXECUTION_OR_OUTCOME_INSPECTION"
    )


def test_manifest_writing_is_deterministic_and_immutable(tmp_path) -> None:
    assert build_manifest() == build_manifest()
    path = write_manifest(tmp_path)
    assert path == tmp_path / "d15c2c2b5bafefd2" / "manifest.json"
    original = path.read_text(encoding="utf-8")
    assert write_manifest(tmp_path) == path
    assert path.read_text(encoding="utf-8") == original
    path.write_text("different\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs"):
        write_manifest(tmp_path)
