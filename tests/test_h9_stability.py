from decimal import Decimal

from src.research.h9_stability import (
    H9Metrics,
    H9WindowComparison,
    evaluate_h9_gate,
    h9_v34_eligible,
)


def metrics(
    *,
    trades: int = 197,
    gross: str = "0.25",
    net: str = "0.05",
    pf: str = "1.10",
    dd: str = "1.0",
    stress: str = "-0.20",
) -> H9Metrics:
    return H9Metrics(
        trades=trades,
        frictionless_expectancy_r=Decimal(gross),
        net_expectancy_r=Decimal(net),
        profit_factor_r=Decimal(pf),
        maximum_drawdown_percent=Decimal(dd),
        stress_net_expectancy_r=Decimal(stress),
    )


def reference() -> H9Metrics:
    return H9Metrics(
        trades=197,
        frictionless_expectancy_r=Decimal("0.160461"),
        net_expectancy_r=Decimal("-0.331631"),
        profit_factor_r=Decimal("0.620551"),
        maximum_drawdown_percent=Decimal("1.0"),
        stress_net_expectancy_r=Decimal("-0.852389"),
    )


def windows(
    *,
    gross_better: int = 7,
    net_better: int = 7,
    positive: int = 7,
) -> tuple[H9WindowComparison, ...]:
    rows = []

    for index in range(11):
        rows.append(
            H9WindowComparison(
                h9_frictionless_r=(
                    Decimal("0.2")
                    if index < gross_better
                    else Decimal("0")
                ),
                reference_frictionless_r=Decimal("0.1"),

                h9_net_r=(
                    Decimal("0.1")
                    if index < net_better
                    else Decimal("-0.2")
                ),
                reference_net_r=Decimal("-0.1"),

                h9_net_positive=(
                    index < positive
                ),
            )
        )

    return tuple(rows)


def test_h9_all_support_gates_can_pass() -> None:
    gate = evaluate_h9_gate(
        h9=metrics(),
        reference=reference(),
        windows=windows(),
        matched_reference_entries=180,
        reference_entry_count=197,
    )

    assert gate.all_support_gates_met is True


def test_h9_v34_requires_positive_real_edge() -> None:
    candidate = metrics()

    gate = evaluate_h9_gate(
        h9=candidate,
        reference=reference(),
        windows=windows(),
        matched_reference_entries=180,
        reference_entry_count=197,
    )

    assert h9_v34_eligible(
        h9=candidate,
        gate=gate,
    ) is True


def test_six_of_eleven_fails_sixty_percent() -> None:
    gate = evaluate_h9_gate(
        h9=metrics(),
        reference=reference(),
        windows=windows(
            gross_better=6,
            net_better=6,
        ),
        matched_reference_entries=180,
        reference_entry_count=197,
    )

    assert gate.frictionless_consistency_passed is False
    assert gate.net_consistency_passed is False


def test_trade_ratio_below_eighty_percent_fails() -> None:
    gate = evaluate_h9_gate(
        h9=metrics(trades=157),
        reference=reference(),
        windows=windows(),
        matched_reference_entries=180,
        reference_entry_count=197,
    )

    assert gate.trade_count_passed is False


def test_entry_match_below_eighty_percent_fails() -> None:
    gate = evaluate_h9_gate(
        h9=metrics(),
        reference=reference(),
        windows=windows(),
        matched_reference_entries=157,
        reference_entry_count=197,
    )

    assert gate.baseline_entry_match_passed is False


def test_drawdown_more_than_twenty_five_percent_worse_fails() -> None:
    gate = evaluate_h9_gate(
        h9=metrics(dd="1.251"),
        reference=reference(),
        windows=windows(),
        matched_reference_entries=180,
        reference_entry_count=197,
    )

    assert gate.drawdown_passed is False


def test_worse_cost_stress_fails() -> None:
    gate = evaluate_h9_gate(
        h9=metrics(stress="-0.90"),
        reference=reference(),
        windows=windows(),
        matched_reference_entries=180,
        reference_entry_count=197,
    )

    assert gate.cost_stress_passed is False