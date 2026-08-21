"""Frozen support gates for preregistered H8 research."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class H8Metrics:
    trades: int
    frictionless_expectancy_r: Decimal
    net_expectancy_r: Decimal
    profit_factor_r: Decimal | None
    maximum_drawdown_percent: Decimal
    stress_net_expectancy_r: Decimal


@dataclass(frozen=True, slots=True)
class H8WindowComparison:
    h8_frictionless_r: Decimal
    reference_frictionless_r: Decimal
    h8_net_r: Decimal
    reference_net_r: Decimal
    h8_net_positive: bool


@dataclass(frozen=True, slots=True)
class H8GateResult:
    eligible_windows: int

    frictionless_better_windows: int
    net_better_windows: int
    positive_net_windows: int

    trade_count_ratio: Decimal

    combined_frictionless_improved: bool
    combined_net_improved: bool
    profit_factor_preserved_or_improved: bool
    frictionless_consistency_passed: bool
    net_consistency_passed: bool
    trade_count_passed: bool
    drawdown_passed: bool
    cost_stress_passed: bool

    @property
    def all_support_gates_met(self) -> bool:
        return all(
            (
                self.combined_frictionless_improved,
                self.combined_net_improved,
                self.profit_factor_preserved_or_improved,
                self.frictionless_consistency_passed,
                self.net_consistency_passed,
                self.trade_count_passed,
                self.drawdown_passed,
                self.cost_stress_passed,
            )
        )

    @property
    def positive_net_window_ratio(self) -> Decimal:
        if self.eligible_windows == 0:
            return Decimal("0")

        return (
            Decimal(self.positive_net_windows)
            / Decimal(self.eligible_windows)
        )


def evaluate_h8_gate(
    *,
    h8: H8Metrics,
    reference: H8Metrics,
    windows: tuple[H8WindowComparison, ...],
) -> H8GateResult:
    if len(windows) != 11:
        raise ValueError(
            f"H8 requires exactly 11 eligible windows, got {len(windows)}."
        )

    if reference.trades <= 0:
        raise ValueError("H8 reference trade count must be positive.")

    frictionless_better = sum(
        item.h8_frictionless_r > item.reference_frictionless_r
        for item in windows
    )

    net_better = sum(
        item.h8_net_r > item.reference_net_r
        for item in windows
    )

    positive_net = sum(
        item.h8_net_positive
        for item in windows
    )

    trade_ratio = (
        Decimal(h8.trades)
        / Decimal(reference.trades)
    )

    required_consistency = Decimal("0.60")

    frictionless_consistency = (
        Decimal(frictionless_better)
        / Decimal(len(windows))
        >= required_consistency
    )

    net_consistency = (
        Decimal(net_better)
        / Decimal(len(windows))
        >= required_consistency
    )

    if reference.profit_factor_r is None:
        pf_passed = h8.profit_factor_r is not None
    elif h8.profit_factor_r is None:
        pf_passed = False
    else:
        pf_passed = (
            h8.profit_factor_r
            >= reference.profit_factor_r
        )

    drawdown_limit = (
        reference.maximum_drawdown_percent
        * Decimal("1.25")
    )

    return H8GateResult(
        eligible_windows=len(windows),
        frictionless_better_windows=frictionless_better,
        net_better_windows=net_better,
        positive_net_windows=positive_net,
        trade_count_ratio=trade_ratio,

        combined_frictionless_improved=(
            h8.frictionless_expectancy_r
            > reference.frictionless_expectancy_r
        ),
        combined_net_improved=(
            h8.net_expectancy_r
            > reference.net_expectancy_r
        ),
        profit_factor_preserved_or_improved=pf_passed,
        frictionless_consistency_passed=(
            frictionless_consistency
        ),
        net_consistency_passed=net_consistency,
        trade_count_passed=(
            trade_ratio >= Decimal("0.80")
        ),
        drawdown_passed=(
            h8.maximum_drawdown_percent
            <= drawdown_limit
        ),
        cost_stress_passed=(
            h8.stress_net_expectancy_r
            >= reference.stress_net_expectancy_r
        ),
    )


def h8_v34_eligible(
    *,
    h8: H8Metrics,
    gate: H8GateResult,
) -> bool:
    return (
        gate.all_support_gates_met
        and h8.net_expectancy_r > 0
        and h8.profit_factor_r is not None
        and h8.profit_factor_r > 1
        and gate.positive_net_window_ratio
        >= Decimal("0.60")
    )