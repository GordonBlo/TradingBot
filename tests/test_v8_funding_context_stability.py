from decimal import Decimal

import pytest

from src.research.v8_funding_context_stability import (
    V6_GROSS_EXPECTANCY_R,
    V6_NET_EXPECTANCY_R,
    V6_PROFIT_FACTOR_R,
    V8H0Metrics,
    V8H0WindowMetrics,
    evaluate_v8_h0_progression_gate,
    evaluate_v8_h0_support_gate,
)


def metrics(
    *,
    gross: Decimal = Decimal("0.06"),
    net: Decimal = Decimal("0.01"),
    pf: Decimal = Decimal("1.1"),
    non_v6: int = 0,
) -> V8H0Metrics:
    return V8H0Metrics(gross, net, pf, non_v6)


def windows(*, better: int = 7, positive: int = 7):
    output = []
    for index in range(11):
        result = Decimal("1") if index < positive else Decimal("-0.2")
        baseline = result - Decimal("1") if index < better else result + Decimal("1")
        output.append(V8H0WindowMetrics(1, result, baseline))
    return tuple(output)


def test_all_frozen_support_conditions_pass() -> None:
    assert evaluate_v8_h0_support_gate(
        metrics=metrics(), windows=windows()
    ).all_conditions_met


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("gross", V6_GROSS_EXPECTANCY_R),
        ("net", V6_NET_EXPECTANCY_R),
        ("pf", V6_PROFIT_FACTOR_R),
    ),
)
def test_combined_metrics_must_strictly_beat_frozen_v6(name, value) -> None:
    arguments = {name: value}
    assert not evaluate_v8_h0_support_gate(
        metrics=metrics(**arguments), windows=windows()
    ).all_conditions_met


def test_six_net_better_windows_fail_and_seven_pass() -> None:
    assert not evaluate_v8_h0_support_gate(
        metrics=metrics(), windows=windows(better=6)
    ).window_improvement_passed
    assert evaluate_v8_h0_support_gate(
        metrics=metrics(), windows=windows(better=7)
    ).window_improvement_passed


def test_non_v6_entry_and_wrong_window_count_are_rejected() -> None:
    assert not evaluate_v8_h0_support_gate(
        metrics=metrics(non_v6=1), windows=windows()
    ).all_conditions_met
    with pytest.raises(ValueError, match="exactly 11"):
        evaluate_v8_h0_support_gate(metrics=metrics(), windows=windows()[:-1])


def test_progression_requires_support_profitability_and_seven_positive_windows() -> None:
    assert evaluate_v8_h0_progression_gate(
        metrics=metrics(), windows=windows()
    ).all_conditions_met
    assert not evaluate_v8_h0_progression_gate(
        metrics=metrics(gross=V6_GROSS_EXPECTANCY_R), windows=windows()
    ).all_conditions_met
    assert not evaluate_v8_h0_progression_gate(
        metrics=metrics(net=Decimal("0")), windows=windows()
    ).all_conditions_met
    assert not evaluate_v8_h0_progression_gate(
        metrics=metrics(pf=Decimal("1")), windows=windows()
    ).all_conditions_met
    assert not evaluate_v8_h0_progression_gate(
        metrics=metrics(), windows=windows(better=7, positive=6)
    ).all_conditions_met


def test_zero_trade_window_is_not_positive_or_better() -> None:
    window = V8H0WindowMetrics(0, None, Decimal("-1"))
    assert not window.positive_net
    assert not window.net_result_better_than_v6
