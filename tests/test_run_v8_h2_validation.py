from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.cli.run_v8_h2_validation import (
    CandidateDecision,
    FrozenValidationCandidate,
    PremiumObservation,
    _gate_details,
    build_block_results,
    filter_candidates,
    permutation_p_value,
    permutation_test_audit,
    validate_dataset_manifest,
    validate_preregistration,
    write_reports,
)
from src.diagnostics.r_normalized import RNormalizedTrade
from src.research.v8_h2_validation_preregistration import SupportMetrics, build_manifest


UTC = timezone.utc
T = datetime(2022, 1, 1, 12, tzinfo=UTC)


def candidate(number: int, *, block: int = 1) -> FrozenValidationCandidate:
    close = T + timedelta(minutes=15 * number)
    return FrozenValidationCandidate(
        candidate_id=f"C{number:04d}", signal_close=close,
        signal_candle_timestamp=close - timedelta(minutes=15),
        atr14_4h=Decimal("100"), block_number=block,
    )


def record(number: int, value: Decimal) -> RNormalizedTrade:
    return RNormalizedTrade(
        trade_id=f"C{number:04d}", initial_risk=Decimal("1"),
        frictionless_r=value, gross_after_slippage_r=value - Decimal("0.01"),
        fee_cost_r=Decimal("0.02"), slippage_cost_r=Decimal("0.01"),
        total_friction_r=Decimal("0.03"), net_r=value - Decimal("0.03"),
    )


def dataset_manifest() -> dict:
    coverage = {
        source: {
            "classification": "COMPLETE", "rows": 20124,
            "missing_intervals": 0, "duplicate_timestamps": 0,
            "duplicate_exact_records": 0,
        }
        for source in ("spot", "mark", "index")
    }
    return {
        "dataset_id": "52302ca4385f5240", "classification": "VALIDATION_DATA_READY",
        "selected_interval": {
            "start": "2021-09-29T09:00:00+00:00",
            "end": "2022-04-27T00:00:00+00:00",
        },
        "candidate_count": 291, "future_data_violations": 0,
        "discovery_overlap": 0, "blind_holdout_overlap": 0,
        "derived_mark_index_premium_available": 20124,
        "coverage": coverage,
        "blind_holdout_exclusion": {
            "start": "2025-08-01T00:00:00+00:00",
            "end": "2026-02-01T00:00:00+00:00",
        },
    }


def test_frozen_preregistration_and_dataset_identity_validate() -> None:
    validate_preregistration(build_manifest())
    validate_dataset_manifest(dataset_manifest())
    changed = build_manifest() | {"run_id": "wrong"}
    with pytest.raises(ValueError, match="preregistration"):
        validate_preregistration(changed)


def test_dataset_mismatch_or_holdout_contamination_is_rejected() -> None:
    wrong_count = dataset_manifest() | {"candidate_count": 290}
    with pytest.raises(ValueError, match="identity"):
        validate_dataset_manifest(wrong_count)
    contaminated = dataset_manifest() | {"blind_holdout_overlap": 1}
    with pytest.raises(ValueError, match="identity"):
        validate_dataset_manifest(contaminated)


def test_negative_premium_retains_only_frozen_v6_candidate() -> None:
    candidates = (candidate(1), candidate(2))
    context = {
        candidates[0].signal_close: PremiumObservation(
            candidates[0].signal_close, Decimal("99"), candidates[0].signal_close,
            Decimal("100"), candidates[0].signal_close, Decimal("-0.01"),
        ),
        candidates[1].signal_close: PremiumObservation(
            candidates[1].signal_close, Decimal("100"), candidates[1].signal_close,
            Decimal("100"), candidates[1].signal_close, Decimal("0"),
        ),
    }
    retained, decisions = filter_candidates(candidates, context)
    assert retained == (candidates[0],)
    assert tuple(item.status for item in decisions) == (
        "RETAINED_NEGATIVE_PREMIUM",
        "FILTERED_NONNEGATIVE_OR_MISSING_PREMIUM",
    )
    assert {item.candidate_id for item in retained} <= {item.candidate_id for item in candidates}


def test_missing_or_future_premium_source_rejects_candidate() -> None:
    row = candidate(1)
    future = PremiumObservation(
        row.signal_close, Decimal("99"), row.signal_close + timedelta(microseconds=1),
        Decimal("100"), row.signal_close, Decimal("-0.01"),
    )
    assert filter_candidates((row,), {})[0] == ()
    assert filter_candidates((row,), {row.signal_close: future})[0] == ()


def test_frozen_blocks_and_permutation_are_deterministic() -> None:
    candidates = tuple(candidate(index, block=min(7, (index - 1) // 42 + 1)) for index in range(1, 292))
    records = {item.candidate_id: record(index, Decimal(index % 5) - Decimal("2")) for index, item in enumerate(candidates, start=1)}
    retained = frozenset(item.candidate_id for item in candidates if int(item.candidate_id[1:]) % 2 == 0)
    blocks = build_block_results(candidates, retained, records)
    assert len(blocks) == 7
    first = permutation_p_value(candidates=candidates, retained_ids=retained, records=records)
    second = permutation_p_value(candidates=candidates, retained_ids=retained, records=records)
    assert first == second
    assert Decimal("0") < first <= Decimal("1")


def test_permutation_audit_distinguishes_conditional_effect_from_raw_advantage() -> None:
    candidates = tuple(
        candidate(index, block=(index - 1) // 10 + 1)
        for index in range(1, 71)
    )
    records = {}
    retained = set()
    for index, item in enumerate(candidates, start=1):
        within_block = (index - 1) % 10
        value = Decimal("100") if item.block_number == 1 else Decimal(within_block - 5)
        records[item.candidate_id] = record(index, value)
        if item.block_number > 1 and within_block >= 5:
            retained.add(item.candidate_id)

    audit = permutation_test_audit(
        candidates=candidates, retained_ids=frozenset(retained), records=records
    )

    assert audit.observed_statistic < 0
    assert audit.centered_observed_statistic > 0
    assert audit.exact_conditional_null_mean < audit.observed_statistic
    assert audit.p_value == Decimal(1 + audit.upper_tail_extreme_count) / Decimal(10001)
    assert audit.p_value < Decimal("0.05")


def test_frozen_gate_classification_requires_every_condition() -> None:
    passing = SupportMetrics(
        subset_integrity=True, retained_trades=50,
        h2_gross_expectancy_r=Decimal("0.2"), v6_gross_expectancy_r=Decimal("0.1"),
        h2_net_expectancy_r=Decimal("0.1"), v6_net_expectancy_r=Decimal("0"),
        h2_profit_factor_r=Decimal("1.2"), v6_profit_factor_r=Decimal("1.1"),
        gross_improved_blocks=5, selection_advantage_permutation_p_value=Decimal("0.049"),
    )
    support, progression, classification = _gate_details(passing)
    assert support["all_conditions_met"] and progression["all_conditions_met"]
    assert classification == "PROGRESSION_ELIGIBLE"
    failed = replace(passing, retained_trades=49)
    assert _gate_details(failed)[2] == "NOT_SUPPORTED"


def test_report_serialization_refuses_rerun(tmp_path) -> None:
    decision = CandidateDecision("C0001", T.isoformat(), 1, "RETAINED_NEGATIVE_PREMIUM", Decimal("-0.01"))
    from src.cli.run_v8_h2_validation import BlockResult, CandidateResult

    result = CandidateResult("C0001", T.isoformat(), 1, True, Decimal("1"), Decimal("0.9"), Decimal("0.08"), Decimal("0.02"), Decimal("0.1"), "TAKE_PROFIT", 2)
    block = BlockResult(1, 1, 1, Decimal("1"), Decimal("1"), False, Decimal("0.9"), True)
    write_reports(output=tmp_path, summary={"run_id": "test"}, decisions=(decision,), results=(result,), blocks=(block,))
    with pytest.raises(ValueError, match="already exists"):
        write_reports(output=tmp_path, summary={}, decisions=(decision,), results=(result,), blocks=(block,))
