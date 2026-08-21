from decimal import Decimal

from src.hypotheses.h5_parameter_manifest import (
    H5ParameterSupportCriteria,
)
from src.hypotheses.h5_parameter_research import (
    H5ParameterCandidateId,
)
from src.research.h5_parameter_stability import (
    H5ParameterClassification,
    H5ParameterMetrics,
    H5WindowComparison,
    classify_h5_parameter_candidate,
    evaluate_h5_parameter_gate,
)


def metrics(
    *,
    trades: int = 100,
    gross: str = "0.20",
    net: str = "0.05",
    pf: str = "1.10",
    dd: str = "0.50",
    stress: str = "-0.10",
) -> H5ParameterMetrics:
    return H5ParameterMetrics(
        trades=trades,
        frictionless_expectancy_r=Decimal(gross),
        net_expectancy_r=Decimal(net),
        profit_factor_r=Decimal(pf),
        maximum_drawdown_percent=Decimal(dd),
        stress_net_expectancy_r=Decimal(stress),
    )


def good_windows() -> tuple[H5WindowComparison, ...]:
    return tuple(
        H5WindowComparison(
            candidate_frictionless_r=Decimal("0.20"),
            reference_frictionless_r=Decimal("0.10"),
            candidate_net_r=Decimal("0.05"),
            reference_net_r=Decimal("-0.10"),
            candidate_net_positive=True,
        )
        for _ in range(6)
    )


def test_all_support_gates_can_pass() -> None:
    reference = metrics(
        trades=100,
        gross="0.10",
        net="-0.10",
        pf="0.90",
        dd="0.50",
        stress="-0.20",
    )

    candidate = metrics()

    gate = evaluate_h5_parameter_gate(
        candidate=candidate,
        reference=reference,
        windows=good_windows(),
        criteria=H5ParameterSupportCriteria(),
    )

    assert gate.all_support_gates_met is True
    assert gate.frictionless_better_windows == 6
    assert gate.net_better_windows == 6
    assert gate.trade_count_ratio == Decimal("1")


def test_sixty_percent_consistency_is_required() -> None:
    reference = metrics(
        gross="0.10",
        net="-0.10",
        pf="0.90",
        stress="-0.20",
    )

    candidate = metrics()

    windows = (
        *good_windows()[:5],
        H5WindowComparison(
            candidate_frictionless_r=Decimal("0"),
            reference_frictionless_r=Decimal("0.10"),
            candidate_net_r=Decimal("-0.20"),
            reference_net_r=Decimal("-0.10"),
            candidate_net_positive=False,
        ),
        H5WindowComparison(
            candidate_frictionless_r=Decimal("0"),
            reference_frictionless_r=Decimal("0.10"),
            candidate_net_r=Decimal("-0.20"),
            reference_net_r=Decimal("-0.10"),
            candidate_net_positive=False,
        ),
        H5WindowComparison(
            candidate_frictionless_r=Decimal("0"),
            reference_frictionless_r=Decimal("0.10"),
            candidate_net_r=Decimal("-0.20"),
            reference_net_r=Decimal("-0.10"),
            candidate_net_positive=False,
        ),
        H5WindowComparison(
            candidate_frictionless_r=Decimal("0"),
            reference_frictionless_r=Decimal("0.10"),
            candidate_net_r=Decimal("-0.20"),
            reference_net_r=Decimal("-0.10"),
            candidate_net_positive=False,
        ),
    )

    gate = evaluate_h5_parameter_gate(
        candidate=candidate,
        reference=reference,
        windows=windows,
        criteria=H5ParameterSupportCriteria(),
    )

    assert gate.frictionless_better_percent < Decimal("60")
    assert gate.net_better_percent < Decimal("60")
    assert gate.all_support_gates_met is False


def test_trade_count_floor_is_fifty_percent() -> None:
    reference = metrics(
        trades=100,
        gross="0.10",
        net="-0.10",
        pf="0.90",
        stress="-0.20",
    )

    candidate = metrics(
        trades=49,
    )

    gate = evaluate_h5_parameter_gate(
        candidate=candidate,
        reference=reference,
        windows=good_windows(),
        criteria=H5ParameterSupportCriteria(),
    )

    assert gate.trade_count_ratio == Decimal("0.49")
    assert gate.trade_count_ratio_met is False


def test_stress_gate_is_relative_to_q25() -> None:
    reference = metrics(
        gross="0.10",
        net="-0.10",
        pf="0.90",
        stress="-0.20",
    )

    candidate = metrics(
        stress="-0.21",
    )

    gate = evaluate_h5_parameter_gate(
        candidate=candidate,
        reference=reference,
        windows=good_windows(),
        criteria=H5ParameterSupportCriteria(),
    )

    assert gate.stress_expectancy_not_worse is False


def test_q25_is_always_frozen_reference() -> None:
    value = metrics()

    gate = evaluate_h5_parameter_gate(
        candidate=value,
        reference=value,
        windows=good_windows(),
        criteria=H5ParameterSupportCriteria(),
    )

    classification = classify_h5_parameter_candidate(
        candidate_id=H5ParameterCandidateId.H5_Q25,
        candidate=value,
        reference=value,
        gate=gate,
    )

    assert (
        classification
        is H5ParameterClassification.FROZEN_REFERENCE
    )


def test_v34_requires_positive_net_pf_and_windows() -> None:
    reference = metrics(
        trades=100,
        gross="0.10",
        net="-0.10",
        pf="0.90",
        dd="0.50",
        stress="-0.20",
    )

    candidate = metrics(
        trades=80,
        gross="0.30",
        net="0.10",
        pf="1.20",
        dd="0.50",
        stress="-0.10",
    )

    gate = evaluate_h5_parameter_gate(
        candidate=candidate,
        reference=reference,
        windows=good_windows(),
        criteria=H5ParameterSupportCriteria(),
    )

    classification = classify_h5_parameter_candidate(
        candidate_id=H5ParameterCandidateId.H5_Q35,
        candidate=candidate,
        reference=reference,
        gate=gate,
    )

    assert (
        classification
        is H5ParameterClassification.V3_4_ELIGIBLE
    )


def test_supported_but_not_profitable_is_not_v34_eligible() -> None:
    reference = metrics(
        gross="0.10",
        net="-0.20",
        pf="0.80",
        stress="-0.30",
    )

    candidate = metrics(
        gross="0.20",
        net="-0.05",
        pf="0.90",
        stress="-0.20",
    )

    gate = evaluate_h5_parameter_gate(
        candidate=candidate,
        reference=reference,
        windows=good_windows(),
        criteria=H5ParameterSupportCriteria(),
    )

    classification = classify_h5_parameter_candidate(
        candidate_id=H5ParameterCandidateId.H5_Q35,
        candidate=candidate,
        reference=reference,
        gate=gate,
    )

    assert (
        classification
        is H5ParameterClassification.MECHANISM_SUPPORTED
    )
