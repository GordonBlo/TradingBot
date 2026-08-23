"""Frozen V7-H0 support and progression gates."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


REQUIRED_ELIGIBLE_WINDOWS = 11
MINIMUM_REQUIRED_WINDOWS = 7
V6_FRICTIONLESS_EXPECTANCY_R = Decimal("0.04722009245553381286395292467")
V6_NET_EXPECTANCY_R = Decimal("-0.1366917232474779417819851766")
V6_PROFIT_FACTOR_R = Decimal("0.7695960517596068371318908235")
ZERO_TRADES_STATUS = "ZERO_TRADES"


@dataclass(frozen=True, slots=True)
class V7H0Metrics:
    frictionless_expectancy_r: Decimal
    net_expectancy_r: Decimal
    profit_factor_r: Decimal | None
    non_v6_entry_count: int = 0


@dataclass(frozen=True, slots=True)
class V7H0WindowMetrics:
    trades: int
    net_expectancy_r: Decimal | None
    v6_net_expectancy_r: Decimal | None

    def __post_init__(self) -> None:
        if self.trades < 0:
            raise ValueError("V7-H0 window trade count cannot be negative.")
        if self.trades > 0 and self.net_expectancy_r is None:
            raise ValueError("A traded V7-H0 window requires net expectancy R.")

    @property
    def status(self) -> str:
        return ZERO_TRADES_STATUS if self.trades == 0 else "TRADES"

    @property
    def positive_net(self) -> bool:
        return (
            self.trades > 0
            and self.net_expectancy_r is not None
            and self.net_expectancy_r > 0
        )

    @property
    def net_better_than_v6(self) -> bool:
        return (
            self.trades > 0
            and self.net_expectancy_r is not None
            and self.v6_net_expectancy_r is not None
            and self.net_expectancy_r > self.v6_net_expectancy_r
        )


@dataclass(frozen=True, slots=True)
class V7H0SupportGateResult:
    eligible_windows: int
    windows_net_better_than_v6: int
    frictionless_expectancy_better: bool
    net_expectancy_better: bool
    profit_factor_better: bool
    window_improvement_passed: bool
    subset_invariant_passed: bool

    @property
    def all_conditions_met(self) -> bool:
        return all(
            (
                self.frictionless_expectancy_better,
                self.net_expectancy_better,
                self.profit_factor_better,
                self.window_improvement_passed,
                self.subset_invariant_passed,
            )
        )


@dataclass(frozen=True, slots=True)
class V7H0ProgressionGateResult:
    support: V7H0SupportGateResult
    positive_net_windows: int
    net_expectancy_passed: bool
    profit_factor_passed: bool
    positive_net_windows_passed: bool

    @property
    def all_conditions_met(self) -> bool:
        return all(
            (
                self.support.all_conditions_met,
                self.net_expectancy_passed,
                self.profit_factor_passed,
                self.positive_net_windows_passed,
            )
        )


def _validate_windows(
    windows: tuple[V7H0WindowMetrics, ...],
) -> None:
    if len(windows) != REQUIRED_ELIGIBLE_WINDOWS:
        raise ValueError(
            "V7-H0 requires exactly 11 eligible windows, got "
            f"{len(windows)}."
        )


def evaluate_v7_h0_support_gate(
    *,
    metrics: V7H0Metrics,
    windows: tuple[V7H0WindowMetrics, ...],
) -> V7H0SupportGateResult:
    """Apply only the preregistered V7-H0 information support gate."""

    _validate_windows(windows)
    net_better_windows = sum(window.net_better_than_v6 for window in windows)
    return V7H0SupportGateResult(
        eligible_windows=REQUIRED_ELIGIBLE_WINDOWS,
        windows_net_better_than_v6=net_better_windows,
        frictionless_expectancy_better=(
            metrics.frictionless_expectancy_r > V6_FRICTIONLESS_EXPECTANCY_R
        ),
        net_expectancy_better=metrics.net_expectancy_r > V6_NET_EXPECTANCY_R,
        profit_factor_better=(
            metrics.profit_factor_r is not None
            and metrics.profit_factor_r > V6_PROFIT_FACTOR_R
        ),
        window_improvement_passed=(
            net_better_windows >= MINIMUM_REQUIRED_WINDOWS
        ),
        subset_invariant_passed=metrics.non_v6_entry_count == 0,
    )


def evaluate_v7_h0_progression_gate(
    *,
    metrics: V7H0Metrics,
    windows: tuple[V7H0WindowMetrics, ...],
) -> V7H0ProgressionGateResult:
    """Apply the frozen V7 progression gate after evaluating support."""

    support = evaluate_v7_h0_support_gate(metrics=metrics, windows=windows)
    positive_windows = sum(window.positive_net for window in windows)
    return V7H0ProgressionGateResult(
        support=support,
        positive_net_windows=positive_windows,
        net_expectancy_passed=metrics.net_expectancy_r > 0,
        profit_factor_passed=(
            metrics.profit_factor_r is not None
            and metrics.profit_factor_r > 1
        ),
        positive_net_windows_passed=(
            positive_windows >= MINIMUM_REQUIRED_WINDOWS
        ),
    )
