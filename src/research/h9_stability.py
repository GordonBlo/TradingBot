"""Frozen support gates for preregistered H9 Early Failure research."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class H9Metrics:
    trades: int
    frictionless_expectancy_r: Decimal
    net_expectancy_r: Decimal
    profit_factor_r: Decimal | None
    maximum_drawdown_percent: Decimal
    stress_net_expectancy_r: Decimal


@dataclass(frozen=True, slots=True)
class H9WindowComparison:
    h9_frictionless_r: Decimal
    reference_frictionless_r: Decimal
    h9_net_r: Decimal
    reference_net_r: Decimal
    h9_net_positive: bool


@dataclass(frozen=True, slots=True)
class H9GateResult:
    eligible_windows: int

    frictionless_better_windows: int
    net_better_windows: int
    positive_net_windows: int

    trade_count_ratio: Decimal
    baseline_entry_match_ratio: Decimal

    combined_frictionless_improved: bool
    combined_net_improved: bool
    profit_factor_preserved_or_improved: bool

    frictionless_consistency_passed: bool
    net_consistency_passed: bool

    trade_count_passed: bool
    baseline_entry_match_passed: bool

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
                self.baseline_entry_match_passed,
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


def evaluate_h9_gate(
    *,
    h9: H9Metrics,
    reference: H9Metrics,
    windows: tuple[H9WindowComparison, ...],
    matched_reference_entries: int,
    reference_entry_count: int,
) -> H9GateResult:
    if len(windows) != 11:
        raise ValueError(
            f"H9 requires exactly 11 eligible windows, got {len(windows)}."
        )

    if reference.trades <= 0:
        raise ValueError(
            "H9 reference trade count must be positive."
        )

    if reference_entry_count <= 0:
        raise ValueError(
            "H9 reference entry count must be positive."
        )

    if not 0 <= matched_reference_entries <= reference_entry_count:
        raise ValueError(
            "Matched H9 reference entry count is invalid."
        )

    frictionless_better = sum(
        row.h9_frictionless_r
        > row.reference_frictionless_r
        for row in windows
    )

    net_better = sum(
        row.h9_net_r > row.reference_net_r
        for row in windows
    )

    positive_net = sum(
        row.h9_net_positive
        for row in windows
    )

    trade_ratio = (
        Decimal(h9.trades)
        / Decimal(reference.trades)
    )

    entry_match_ratio = (
        Decimal(matched_reference_entries)
        / Decimal(reference_entry_count)
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
        pf_passed = h9.profit_factor_r is not None

    elif h9.profit_factor_r is None:
        pf_passed = False

    else:
        pf_passed = (
            h9.profit_factor_r
            >= reference.profit_factor_r
        )

    drawdown_limit = (
        reference.maximum_drawdown_percent
        * Decimal("1.25")
    )

    return H9GateResult(
        eligible_windows=len(windows),

        frictionless_better_windows=frictionless_better,
        net_better_windows=net_better,
        positive_net_windows=positive_net,

        trade_count_ratio=trade_ratio,
        baseline_entry_match_ratio=entry_match_ratio,

        combined_frictionless_improved=(
            h9.frictionless_expectancy_r
            > reference.frictionless_expectancy_r
        ),

        combined_net_improved=(
            h9.net_expectancy_r
            > reference.net_expectancy_r
        ),

        profit_factor_preserved_or_improved=pf_passed,

        frictionless_consistency_passed=(
            frictionless_consistency
        ),

        net_consistency_passed=(
            net_consistency
        ),

        trade_count_passed=(
            trade_ratio >= Decimal("0.80")
        ),

        baseline_entry_match_passed=(
            entry_match_ratio >= Decimal("0.80")
        ),

        drawdown_passed=(
            h9.maximum_drawdown_percent
            <= drawdown_limit
        ),

        cost_stress_passed=(
            h9.stress_net_expectancy_r
            >= reference.stress_net_expectancy_r
        ),
    )


def h9_v34_eligible(
    *,
    h9: H9Metrics,
    gate: H9GateResult,
) -> bool:
    return (
        gate.all_support_gates_met
        and h9.net_expectancy_r > 0
        and h9.profit_factor_r is not None
        and h9.profit_factor_r > 1
        and gate.positive_net_window_ratio
        >= Decimal("0.60")
    )