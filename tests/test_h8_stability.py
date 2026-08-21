from decimal import Decimal

from src.research.h8_stability import (
    H8Metrics,
    H8WindowComparison,
    evaluate_h8_gate,
    h8_v34_eligible,
)


def metrics(
    *,
    trades: int = 197,
    gross: str = "0.20",
    net: str = "0.05",
    pf: str = "1.10",
    dd: str = "1.0",
    stress: str = "-0.20",
) -> H8Metrics:
    return H8Metrics(
        trades=trades,
        frictionless_expectancy_r=Decimal(gross),
        net_expectancy_r=Decimal(net),
        profit_factor_r=Decimal(pf),
        maximum_drawdown_percent=Decimal(dd),
        stress_net_expectancy_r=Decimal(stress),
    )


def reference() -> H8Metrics:
    return metrics(
        trades=197,
        gross="0.160461",
        net="-0.331631",
        pf="0.620551",
        dd="1.0",
        stress="-0.852389",
    )


def windows(
    *,
    gross_better: int = 7,
    net_better: int = 7,
    positive: int = 7,
) -> tuple[H8WindowComparison, ...]:
    rows = []

    for index in range(11):
        rows.append(
            H8WindowComparison(
                h8_frictionless_r=(
                    Decimal("0.2")
                    if index < gross_better
                    else Decimal("0")
                ),
                reference_frictionless_r=Decimal("0.1"),
                h8_net_r=(
                    Decimal("0.1")
                    if index < net_better
                    else Decimal("-0.2")
                ),
                reference_net_r=Decimal("-0.1"),
                h8_net_positive=index < positive,
            )
        )

    return tuple(rows)


def test_h8_all_support_gates_can_pass() -> None:
    candidate = metrics()

    gate = evaluate_h8_gate(
        h8=candidate,
        reference=reference(),
        windows=windows(),
    )

    assert gate.all_support_gates_met is True


def test_h8_v34_requires_positive_real_edge() -> None:
    candidate = metrics()

    gate = evaluate_h8_gate(
        h8=candidate,
        reference=reference(),
        windows=windows(),
    )

    assert h8_v34_eligible(
        h8=candidate,
        gate=gate,
    ) is True


def test_six_of_eleven_does_not_pass_sixty_percent() -> None:
    gate = evaluate_h8_gate(
        h8=metrics(),
        reference=reference(),
        windows=windows(
            gross_better=6,
            net_better=6,
        ),
    )

    assert gate.frictionless_consistency_passed is False
    assert gate.net_consistency_passed is False


def test_trade_ratio_below_eighty_percent_fails() -> None:
    gate = evaluate_h8_gate(
        h8=metrics(trades=157),
        reference=reference(),
        windows=windows(),
    )

    assert gate.trade_count_passed is False


def test_drawdown_more_than_twenty_five_percent_worse_fails() -> None:
    gate = evaluate_h8_gate(
        h8=metrics(dd="1.251"),
        reference=reference(),
        windows=windows(),
    )

    assert gate.drawdown_passed is False


def test_worse_cost_stress_fails() -> None:
    gate = evaluate_h8_gate(
        h8=metrics(stress="-0.90"),
        reference=reference(),
        windows=windows(),
    )

    assert gate.cost_stress_passed is False