from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

import src.cli.run_v8_derivatives_discovery as runner
from src.cli.run_v8_derivatives_discovery import (
    load_context_rows,
    main,
    validate_manifest,
)
from src.diagnostics.v8_derivatives_discovery import (
    classify_discovery,
    fit_prepared_fold,
    prepare_fold,
    within_window_permutations,
)
from src.research.v8_derivatives_discovery_preregistration import build_manifest


T = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)


def test_execute_flag_is_required_before_preflight(monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "_load_json",
        lambda path: pytest.fail("preflight must not run without --execute"),
    )
    assert main([]) == 1


def test_manifest_mismatch_is_rejected() -> None:
    manifest = deepcopy(build_manifest())
    manifest["run_id"] = "wrong"
    with pytest.raises(ValueError, match="preregistration ID"):
        validate_manifest(manifest)


def test_fold_preprocessing_and_fit_are_isolated_from_held_out_targets() -> None:
    features = ((1.0, None), (3.0, 2.0), (1000.0, 9000.0))
    fold = prepare_fold(
        features,
        training_indices=(0, 1),
        held_out_indices=(2,),
        alpha=1.0,
    )
    assert fold.medians == (2.0, 2.0)
    assert fold.means == (2.0, 2.0)
    assert fold.training_indices == (0, 1)
    assert fold.held_out_indices == (2,)

    original = fit_prepared_fold(fold, (1.0, 3.0, -999999.0))
    changed_held_out = fit_prepared_fold(fold, (1.0, 3.0, 999999.0))
    assert original.intercept == changed_held_out.intercept
    assert original.coefficients == changed_held_out.coefficients
    assert original.predictions == changed_held_out.predictions


def _context_csv(*, mark_timestamp: datetime) -> str:
    header = (
        "bucket_open_time,bucket_close_time,latest_known_funding_rate,"
        "funding_timestamp,open_interest,open_interest_value,"
        "open_interest_timestamp,mark_price,mark_price_timestamp,index_price,"
        "index_price_timestamp,premium_index,premium_index_timestamp,"
        "premium_fraction\n"
    )
    row = (
        "2024-01-01T11:45:00+00:00,2024-01-01T12:00:00+00:00,0.0001,"
        "2024-01-01T08:00:00+00:00,100,100000,"
        "2024-01-01T12:00:00+00:00,42000,"
        f"{mark_timestamp.isoformat()},41990,2024-01-01T11:59:59+00:00,"
        "0.0002,2024-01-01T11:59:59+00:00,0.000238\n"
    )
    return header + row


def test_context_loader_accepts_causal_timestamps_and_rejects_future(tmp_path) -> None:
    causal = tmp_path / "causal.csv"
    causal.write_text(_context_csv(mark_timestamp=T), encoding="utf-8")
    loaded = load_context_rows(
        {"partition_paths": [str(causal)], "context_buckets": 1},
        required_bucket_closes=frozenset((T,)),
    )
    assert loaded[T].mark_timestamp == T

    future = tmp_path / "future.csv"
    future.write_text(
        _context_csv(mark_timestamp=T + timedelta(microseconds=1)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="future observation"):
        load_context_rows(
            {"partition_paths": [str(future)], "context_buckets": 1},
            required_bucket_closes=frozenset((T,)),
        )


def test_permutations_are_seeded_deterministic_and_within_window_only() -> None:
    windows = ("A", "A", "A", "B", "B", "B")
    first = tuple(within_window_permutations(windows, seed=42, count=5))
    second = tuple(within_window_permutations(windows, seed=42, count=5))
    assert first == second
    assert len(first) == 5
    for source_indices in first:
        assert set(source_indices[:3]) == {0, 1, 2}
        assert set(source_indices[3:]) == {3, 4, 5}


@pytest.mark.parametrize(
    "inputs,expected",
    [
        (
            dict(
                held_out_spearman=0.1,
                permutation_p_value=0.05,
                model_mse=0.9,
                baseline_mse=1.0,
                positive_direction_windows=7,
                has_stable_economic_feature=True,
            ),
            "STABLE_DERIVATIVES_SIGNAL",
        ),
        (
            dict(
                held_out_spearman=0.1,
                permutation_p_value=0.051,
                model_mse=0.9,
                baseline_mse=1.0,
                positive_direction_windows=7,
                has_stable_economic_feature=True,
            ),
            "WEAK_OR_UNSTABLE_SIGNAL",
        ),
        (
            dict(
                held_out_spearman=-0.1,
                permutation_p_value=0.01,
                model_mse=0.9,
                baseline_mse=1.0,
                positive_direction_windows=7,
                has_stable_economic_feature=True,
            ),
            "WEAK_OR_UNSTABLE_SIGNAL",
        ),
        (
            dict(
                held_out_spearman=0.1,
                permutation_p_value=0.01,
                model_mse=1.0,
                baseline_mse=1.0,
                positive_direction_windows=7,
                has_stable_economic_feature=True,
            ),
            "WEAK_OR_UNSTABLE_SIGNAL",
        ),
        (
            dict(
                held_out_spearman=0.1,
                permutation_p_value=0.01,
                model_mse=0.9,
                baseline_mse=1.0,
                positive_direction_windows=6,
                has_stable_economic_feature=True,
            ),
            "WEAK_OR_UNSTABLE_SIGNAL",
        ),
        (
            dict(
                held_out_spearman=0.1,
                permutation_p_value=0.01,
                model_mse=0.9,
                baseline_mse=1.0,
                positive_direction_windows=7,
                has_stable_economic_feature=False,
            ),
            "WEAK_OR_UNSTABLE_SIGNAL",
        ),
        (
            dict(
                held_out_spearman=-0.1,
                permutation_p_value=1.0,
                model_mse=1.0,
                baseline_mse=1.0,
                positive_direction_windows=0,
                has_stable_economic_feature=False,
            ),
            "NO_STABLE_DERIVATIVES_SIGNAL",
        ),
    ],
)
def test_frozen_three_way_classification_gates(inputs, expected) -> None:
    assert classify_discovery(**inputs) == expected


def test_implemented_feature_names_match_exact_frozen_27() -> None:
    manifest = build_manifest()
    names = tuple(row["name"] for row in manifest["fixed_features"])
    assert len(names) == len(set(names)) == 27
    source = __import__(
        "inspect"
    ).getsource(runner.build_feature_rows)
    assert "open_interest_value" not in source
    assert "HORIZONS" in source
