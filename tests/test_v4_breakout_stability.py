from decimal import Decimal

import pytest

from src.research.v4_breakout_stability import (
    REQUIRED_ELIGIBLE_WINDOWS,
    V4H0Metrics,
    evaluate_v4_h0_gate,
)


def gate(*, net: str = "0.01", pf: str = "1.01", positive: int = 7):
    return evaluate_v4_h0_gate(
        metrics=V4H0Metrics(
            net_expectancy_r=Decimal(net),
            profit_factor_r=Decimal(pf),
        ),
        window_net_expectancies_r=tuple(
            Decimal("0.01") if index < positive else Decimal("-0.01")
            for index in range(REQUIRED_ELIGIBLE_WINDOWS)
        ),
    )


def test_all_preregistered_v4_h0_gates_can_pass() -> None:
    assert gate().all_conditions_met is True


def test_non_positive_net_expectancy_fails() -> None:
    assert gate(net="0").all_conditions_met is False


def test_profit_factor_at_or_below_one_fails() -> None:
    assert gate(pf="1").all_conditions_met is False


def test_six_of_eleven_positive_net_windows_fails() -> None:
    result = gate(positive=6)

    assert result.positive_net_window_ratio_passed is False
    assert result.all_conditions_met is False


def test_seven_of_eleven_positive_net_windows_passes_sixty_percent() -> None:
    result = gate(positive=7)

    assert result.positive_net_window_ratio_passed is True


def test_incorrect_eligible_window_count_fails() -> None:
    with pytest.raises(ValueError, match="exactly 11"):
        evaluate_v4_h0_gate(
            metrics=V4H0Metrics(Decimal("0.01"), Decimal("1.01")),
            window_net_expectancies_r=(Decimal("0.01"),) * 10,
        )


def test_progression_requires_all_conditions_simultaneously() -> None:
    assert gate(net="0", pf="1", positive=6).all_conditions_met is False
