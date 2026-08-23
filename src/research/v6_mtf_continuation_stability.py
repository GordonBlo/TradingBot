"""Frozen V6-H0 progression gate."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


REQUIRED_ELIGIBLE_WINDOWS = 11
REQUIRED_POSITIVE_NET_WINDOW_RATIO = Decimal("0.60")


@dataclass(frozen=True, slots=True)
class V6H0Metrics:
    net_expectancy_r: Decimal
    profit_factor_r: Decimal | None


@dataclass(frozen=True, slots=True)
class V6H0GateResult:
    eligible_windows: int
    positive_net_windows: int
    net_expectancy_passed: bool
    profit_factor_passed: bool
    positive_net_window_ratio_passed: bool

    @property
    def positive_net_window_ratio(self) -> Decimal:
        return Decimal(self.positive_net_windows) / Decimal(self.eligible_windows)

    @property
    def all_conditions_met(self) -> bool:
        return all(
            (
                self.net_expectancy_passed,
                self.profit_factor_passed,
                self.positive_net_window_ratio_passed,
            )
        )


def evaluate_v6_h0_gate(
    *,
    metrics: V6H0Metrics,
    window_net_expectancies_r: tuple[Decimal, ...],
) -> V6H0GateResult:
    """Apply the preregistered V6-H0 gate without extra thresholds."""

    if len(window_net_expectancies_r) != REQUIRED_ELIGIBLE_WINDOWS:
        raise ValueError(
            "V6-H0 requires exactly 11 eligible windows, got "
            f"{len(window_net_expectancies_r)}."
        )

    positive_net_windows = sum(
        value > 0 for value in window_net_expectancies_r
    )
    ratio_passed = (
        Decimal(positive_net_windows)
        / Decimal(REQUIRED_ELIGIBLE_WINDOWS)
        >= REQUIRED_POSITIVE_NET_WINDOW_RATIO
    )
    return V6H0GateResult(
        eligible_windows=REQUIRED_ELIGIBLE_WINDOWS,
        positive_net_windows=positive_net_windows,
        net_expectancy_passed=metrics.net_expectancy_r > 0,
        profit_factor_passed=(
            metrics.profit_factor_r is not None
            and metrics.profit_factor_r > 1
        ),
        positive_net_window_ratio_passed=ratio_passed,
    )
