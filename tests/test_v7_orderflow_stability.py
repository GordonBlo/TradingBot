from decimal import Decimal

import pytest

from src.research.v7_orderflow_stability import (
    REQUIRED_ELIGIBLE_WINDOWS,
    V6_FRICTIONLESS_EXPECTANCY_R,
    V6_NET_EXPECTANCY_R,
    V6_PROFIT_FACTOR_R,
    V7H0Metrics,
    V7H0WindowMetrics,
    ZERO_TRADES_STATUS,
    evaluate_v7_h0_progression_gate,
    evaluate_v7_h0_support_gate,
)


def metrics(
    *,
    gross: Decimal = Decimal("0.06"),
    net: Decimal = Decimal("0.01"),
    pf: Decimal = Decimal("1.1"),
    non_v6: int = 0,
) -> V7H0Metrics:
    return V7H0Metrics(gross, net, pf, non_v6)


def windows(*, better: int = 7, positive: int | None = None):
    positive = better if positive is None else positive
    output = []
    for index in range(REQUIRED_ELIGIBLE_WINDOWS):
        if index < positive:
            net = Decimal("0.01")
        elif index < better:
            net = Decimal("-0.05")
        else:
            net = Decimal("-0.20")
        output.append(V7H0WindowMetrics(1, net, Decimal("-0.10")))
    return tuple(output)


def support(**changes):
    gate_metrics = metrics(
        gross=changes.get("gross", Decimal("0.06")),
        net=changes.get("net", Decimal("0.01")),
        pf=changes.get("pf", Decimal("1.1")),
        non_v6=changes.get("non_v6", 0),
    )
    return evaluate_v7_h0_support_gate(
        metrics=gate_metrics,
        windows=windows(better=changes.get("better", 7)),
    )


def test_all_v7_support_conditions_pass() -> None:
    assert support().all_conditions_met is True


@pytest.mark.parametrize(
    ("change", "value", "field"),
    (
        ("gross", V6_FRICTIONLESS_EXPECTANCY_R, "frictionless_expectancy_better"),
        ("net", V6_NET_EXPECTANCY_R, "net_expectancy_better"),
        ("pf", V6_PROFIT_FACTOR_R, "profit_factor_better"),
    ),
)
def test_v7_must_strictly_beat_each_frozen_v6_combined_metric(
    change: str, value: Decimal, field: str
) -> None:
    result = support(**{change: value})

    assert getattr(result, field) is False
    assert result.all_conditions_met is False


def test_six_of_eleven_net_better_windows_fails_support() -> None:
    assert support(better=6).all_conditions_met is False


def test_seven_of_eleven_net_better_windows_passes_support() -> None:
    assert support(better=7).window_improvement_passed is True


def test_one_non_v6_entry_fails_support() -> None:
    assert support(non_v6=1).all_conditions_met is False


def test_wrong_eligible_window_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly 11"):
        evaluate_v7_h0_support_gate(
            metrics=metrics(),
            windows=windows()[:-1],
        )


def progression(
    *,
    gate_metrics: V7H0Metrics | None = None,
    gate_windows: tuple[V7H0WindowMetrics, ...] | None = None,
):
    return evaluate_v7_h0_progression_gate(
        metrics=gate_metrics or metrics(),
        windows=gate_windows or windows(),
    )


def test_all_v7_progression_conditions_pass() -> None:
    assert progression().all_conditions_met is True


def test_progression_fails_when_support_fails() -> None:
    result = progression(
        gate_metrics=metrics(gross=V6_FRICTIONLESS_EXPECTANCY_R)
    )

    assert result.support.all_conditions_met is False
    assert result.all_conditions_met is False


def test_progression_net_expectancy_must_be_strictly_positive() -> None:
    result = progression(gate_metrics=metrics(net=Decimal("0")))

    assert result.support.all_conditions_met is True
    assert result.net_expectancy_passed is False
    assert result.all_conditions_met is False


def test_progression_profit_factor_must_be_strictly_above_one() -> None:
    result = progression(gate_metrics=metrics(pf=Decimal("1")))

    assert result.support.all_conditions_met is True
    assert result.profit_factor_passed is False
    assert result.all_conditions_met is False


def test_six_positive_windows_fails_progression_even_with_support() -> None:
    result = progression(gate_windows=windows(better=7, positive=6))

    assert result.support.all_conditions_met is True
    assert result.positive_net_windows_passed is False
    assert result.all_conditions_met is False


def test_zero_trade_window_has_explicit_non_improving_semantics() -> None:
    window = V7H0WindowMetrics(
        trades=0,
        net_expectancy_r=Decimal("0"),
        v6_net_expectancy_r=Decimal("-1"),
    )

    assert window.status == ZERO_TRADES_STATUS
    assert window.positive_net is False
    assert window.net_better_than_v6 is False
