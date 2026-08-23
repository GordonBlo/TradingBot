from decimal import Decimal

import pytest

from src.research.v6_mtf_continuation_stability import (
    REQUIRED_ELIGIBLE_WINDOWS,
    V6H0Metrics,
    evaluate_v6_h0_gate,
)


def gate(*, net: str = "0.01", pf: str = "1.01", positive: int = 7):
    return evaluate_v6_h0_gate(
        metrics=V6H0Metrics(Decimal(net), Decimal(pf)),
        window_net_expectancies_r=tuple(
            Decimal("0.01") if index < positive else Decimal("-0.01")
            for index in range(REQUIRED_ELIGIBLE_WINDOWS)
        ),
    )


def test_all_preregistered_v6_h0_gates_pass() -> None:
    assert gate().all_conditions_met is True


@pytest.mark.parametrize("net", ["0", "-0.01"])
def test_non_positive_net_expectancy_fails(net: str) -> None:
    assert gate(net=net).all_conditions_met is False


@pytest.mark.parametrize("pf", ["1", "0.99"])
def test_profit_factor_at_or_below_one_fails(pf: str) -> None:
    assert gate(pf=pf).all_conditions_met is False


def test_six_of_eleven_positive_net_windows_fails() -> None:
    assert gate(positive=6).all_conditions_met is False


def test_seven_of_eleven_positive_net_windows_passes() -> None:
    assert gate(positive=7).positive_net_window_ratio_passed is True


def test_incorrect_eligible_window_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly 11"):
        evaluate_v6_h0_gate(
            metrics=V6H0Metrics(Decimal("0.01"), Decimal("1.01")),
            window_net_expectancies_r=(Decimal("0.01"),) * 10,
        )


def test_progression_requires_all_conditions_simultaneously() -> None:
    result = gate(net="0", pf="1", positive=6)

    assert result.net_expectancy_passed is False
    assert result.profit_factor_passed is False
    assert result.positive_net_window_ratio_passed is False
    assert result.all_conditions_met is False
