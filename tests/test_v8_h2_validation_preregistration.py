from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.research.v8_h2_validation_preregistration import (
    CHRONOLOGICAL_BLOCK_COUNT,
    FROZEN_V6_CANDIDATE_COUNT,
    PERMUTATION_COUNT,
    SIGNIFICANCE_ALPHA,
    SupportMetrics,
    build_manifest,
    chronological_candidate_blocks,
    derived_mark_index_premium,
    h2_premium_filter_accepts,
    progression_gate_passes,
    require_h2_subset_of_frozen_v6,
    support_gate_passes,
    within_block_permutations,
    write_manifest,
)


T = datetime(2022, 1, 1, 12, tzinfo=timezone.utc)


def _inputs(**overrides):
    values = {
        "mark_price": Decimal("99"), "mark_timestamp": T,
        "index_price": Decimal("100"), "index_timestamp": T,
        "signal_close": T,
    }
    values.update(overrides)
    return values


def _metrics(**overrides) -> SupportMetrics:
    values = {
        "subset_integrity": True, "retained_trades": 50,
        "h2_gross_expectancy_r": Decimal("0.2"), "v6_gross_expectancy_r": Decimal("0.1"),
        "h2_net_expectancy_r": Decimal("0.1"), "v6_net_expectancy_r": Decimal("0"),
        "h2_profit_factor_r": Decimal("1.2"), "v6_profit_factor_r": Decimal("1.1"),
        "gross_improved_blocks": 5, "selection_advantage_permutation_p_value": Decimal("0.0499"),
    }
    values.update(overrides)
    return SupportMetrics(**values)


def test_negative_premium_is_accepted_and_formula_is_exact() -> None:
    assert derived_mark_index_premium(**_inputs()) == Decimal("-0.01")
    assert h2_premium_filter_accepts(**_inputs())


@pytest.mark.parametrize("mark", [Decimal("100"), Decimal("101")])
def test_zero_or_positive_premium_is_rejected(mark: Decimal) -> None:
    assert not h2_premium_filter_accepts(**_inputs(mark_price=mark))


@pytest.mark.parametrize(
    "overrides",
    [
        {"mark_price": None}, {"index_price": None}, {"mark_timestamp": None}, {"index_timestamp": None},
        {"mark_timestamp": T + timedelta(microseconds=1)},
        {"index_timestamp": T + timedelta(microseconds=1)},
    ],
)
def test_missing_or_future_mark_or_index_is_rejected(overrides: dict) -> None:
    assert not h2_premium_filter_accepts(**_inputs(**overrides))


def test_h2_must_be_a_frozen_v6_subset() -> None:
    require_h2_subset_of_frozen_v6(["a", "b"], ["b"])
    with pytest.raises(ValueError, match="non-V6"):
        require_h2_subset_of_frozen_v6(["a"], ["a", "h2-only"])


def test_seven_chronological_blocks_are_deterministic_and_balanced() -> None:
    ordered = tuple(T + timedelta(minutes=15 * index) for index in range(FROZEN_V6_CANDIDATE_COUNT))
    first = chronological_candidate_blocks(ordered)
    assert first == chronological_candidate_blocks(ordered)
    assert len(first) == CHRONOLOGICAL_BLOCK_COUNT
    assert tuple(map(len, first)) == (42, 42, 42, 42, 41, 41, 41)
    assert first[0][0] == ordered[0] and first[-1][-1] == ordered[-1]
    with pytest.raises(ValueError, match="strictly chronological"):
        chronological_candidate_blocks(tuple(reversed(ordered)))


def test_within_block_permutations_are_seeded_and_never_cross_blocks() -> None:
    blocks = (("a", "b"), ("c",), ("d",), ("e",), ("f",), ("g",), ("h",))
    first = tuple(within_block_permutations(blocks, count=3, seed=42))
    assert first == tuple(within_block_permutations(blocks, count=3, seed=42))
    assert all(set(permutation[0]) == {"a", "b"} for permutation in first)
    assert all(permutation[1:] == blocks[1:] for permutation in first)


def test_support_and_progression_gates_are_strict_and_simultaneous() -> None:
    assert support_gate_passes(_metrics())
    assert progression_gate_passes(_metrics())
    assert not support_gate_passes(_metrics(retained_trades=49))
    assert not support_gate_passes(_metrics(gross_improved_blocks=4))
    assert not support_gate_passes(_metrics(selection_advantage_permutation_p_value=SIGNIFICANCE_ALPHA))
    assert not progression_gate_passes(_metrics(h2_net_expectancy_r=Decimal("0")))
    assert not progression_gate_passes(_metrics(h2_profit_factor_r=Decimal("1")))


def test_manifest_freezes_dataset_rule_gates_and_holdout_exclusion() -> None:
    manifest = build_manifest()
    assert manifest["run_id"] == "b334dac73e746982"
    assert manifest["validation_dataset"]["dataset_id"] == "52302ca4385f5240"
    assert manifest["validation_dataset"]["candidate_count"] == 291
    assert manifest["h2_rule"]["acceptance_rule"] == "DERIVED_MARK_INDEX_PREMIUM_STRICTLY_LESS_THAN_ZERO"
    assert manifest["chronological_blocks"]["block_sizes"] == [42, 42, 42, 42, 41, 41, 41]
    assert manifest["permutation_test"]["permutation_count"] == PERMUTATION_COUNT == 10_000
    assert manifest["data_policy"]["blind_holdout"]["status"] == "LOCKED_BLIND_HOLDOUT"
    assert all(not manifest["data_policy"]["blind_holdout"][name] for name in ("loaded", "revealed", "consumed", "evaluated"))


def test_manifest_writing_is_deterministic_and_immutable(tmp_path) -> None:
    path = write_manifest(tmp_path)
    assert path == tmp_path / "b334dac73e746982" / "manifest.json"
    assert write_manifest(tmp_path) == path
    path.write_text("different\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs"):
        write_manifest(tmp_path)
