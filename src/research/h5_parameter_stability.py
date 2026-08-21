"""V3.3 support gate for controlled H5 parameter research."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from src.hypotheses.h5_parameter_manifest import (
    H5ParameterSupportCriteria,
)
from src.hypotheses.h5_parameter_research import (
    H5ParameterCandidateId,
)


class H5ParameterClassification(str, Enum):
    FROZEN_REFERENCE = "FROZEN_REFERENCE"
    V3_4_ELIGIBLE = "V3.4_ELIGIBLE"
    MECHANISM_SUPPORTED = "MECHANISM_SUPPORTED_ON_RESEARCH_DATA"
    MIXED = "MIXED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True, slots=True)
class H5ParameterMetrics:
    trades: int
    frictionless_expectancy_r: Decimal
    net_expectancy_r: Decimal
    profit_factor_r: Decimal | None
    maximum_drawdown_percent: Decimal
    stress_net_expectancy_r: Decimal | None


@dataclass(frozen=True, slots=True)
class H5WindowComparison:
    candidate_frictionless_r: Decimal
    reference_frictionless_r: Decimal

    candidate_net_r: Decimal
    reference_net_r: Decimal

    candidate_net_positive: bool


@dataclass(frozen=True, slots=True)
class H5ParameterGateResult:
    enough_eligible_windows: bool
    frictionless_expectancy_better: bool
    net_expectancy_better: bool
    profit_factor_not_worse: bool

    frictionless_consistency_met: bool
    net_consistency_met: bool

    trade_count_ratio_met: bool
    drawdown_limit_met: bool
    stress_expectancy_not_worse: bool

    eligible_windows: int
    frictionless_better_windows: int
    net_better_windows: int
    positive_net_windows: int

    frictionless_better_percent: Decimal
    net_better_percent: Decimal
    positive_net_windows_percent: Decimal

    trade_count_ratio: Decimal

    @property
    def all_support_gates_met(self) -> bool:
        return all(
            (
                self.enough_eligible_windows,
                self.frictionless_expectancy_better,
                self.net_expectancy_better,
                self.profit_factor_not_worse,
                self.frictionless_consistency_met,
                self.net_consistency_met,
                self.trade_count_ratio_met,
                self.drawdown_limit_met,
                self.stress_expectancy_not_worse,
            )
        )


def _percent(count: int, total: int) -> Decimal:
    if total <= 0:
        return Decimal("0")

    return (
        Decimal(count)
        / Decimal(total)
        * Decimal("100")
    )


def evaluate_h5_parameter_gate(
    *,
    candidate: H5ParameterMetrics,
    reference: H5ParameterMetrics,
    windows: tuple[H5WindowComparison, ...],
    criteria: H5ParameterSupportCriteria,
) -> H5ParameterGateResult:
    """Evaluate only the V3.3 rules fixed before replay."""

    eligible_windows = len(windows)

    frictionless_better = sum(
        row.candidate_frictionless_r
        > row.reference_frictionless_r
        for row in windows
    )

    net_better = sum(
        row.candidate_net_r
        > row.reference_net_r
        for row in windows
    )

    positive_net = sum(
        row.candidate_net_positive
        for row in windows
    )

    frictionless_percent = _percent(
        frictionless_better,
        eligible_windows,
    )

    net_percent = _percent(
        net_better,
        eligible_windows,
    )

    positive_percent = _percent(
        positive_net,
        eligible_windows,
    )

    trade_ratio = (
        Decimal(candidate.trades)
        / Decimal(reference.trades)
        if reference.trades > 0
        else Decimal("0")
    )

    pf_not_worse = (
        candidate.profit_factor_r is not None
        and reference.profit_factor_r is not None
        and candidate.profit_factor_r
        >= reference.profit_factor_r
    )

    stress_not_worse = (
        candidate.stress_net_expectancy_r is not None
        and reference.stress_net_expectancy_r is not None
        and candidate.stress_net_expectancy_r
        >= reference.stress_net_expectancy_r
    )

    return H5ParameterGateResult(
        enough_eligible_windows=(
            eligible_windows
            >= criteria.minimum_eligible_windows
        ),

        frictionless_expectancy_better=(
            candidate.frictionless_expectancy_r
            > reference.frictionless_expectancy_r
        ),

        net_expectancy_better=(
            candidate.net_expectancy_r
            > reference.net_expectancy_r
        ),

        profit_factor_not_worse=pf_not_worse,

        frictionless_consistency_met=(
            frictionless_percent
            >= criteria.consistency_percent_required
        ),

        net_consistency_met=(
            net_percent
            >= criteria.consistency_percent_required
        ),

        trade_count_ratio_met=(
            trade_ratio
            >= criteria.trade_count_ratio_required
        ),

        drawdown_limit_met=(
            candidate.maximum_drawdown_percent
            <= (
                reference.maximum_drawdown_percent
                * criteria.maximum_drawdown_worse_ratio
            )
        ),

        stress_expectancy_not_worse=stress_not_worse,

        eligible_windows=eligible_windows,
        frictionless_better_windows=frictionless_better,
        net_better_windows=net_better,
        positive_net_windows=positive_net,

        frictionless_better_percent=frictionless_percent,
        net_better_percent=net_percent,
        positive_net_windows_percent=positive_percent,

        trade_count_ratio=trade_ratio,
    )


def classify_h5_parameter_candidate(
    *,
    candidate_id: H5ParameterCandidateId,
    candidate: H5ParameterMetrics,
    reference: H5ParameterMetrics,
    gate: H5ParameterGateResult,
) -> H5ParameterClassification:
    """Classify without selecting an automatic 'best' parameter."""

    if candidate_id is H5ParameterCandidateId.H5_Q25:
        return H5ParameterClassification.FROZEN_REFERENCE

    if not gate.enough_eligible_windows:
        return H5ParameterClassification.INSUFFICIENT_EVIDENCE

    if gate.all_support_gates_met:
        v34_ready = (
            candidate.net_expectancy_r > 0
            and candidate.profit_factor_r is not None
            and candidate.profit_factor_r > 1
            and gate.positive_net_windows_percent
            >= Decimal("60")
        )

        if v34_ready:
            return H5ParameterClassification.V3_4_ELIGIBLE

        return H5ParameterClassification.MECHANISM_SUPPORTED

    meaningful_improvement = any(
        (
            candidate.frictionless_expectancy_r
            > reference.frictionless_expectancy_r,

            candidate.net_expectancy_r
            > reference.net_expectancy_r,

            gate.frictionless_better_percent
            >= Decimal("50"),

            gate.net_better_percent
            >= Decimal("50"),
        )
    )

    return (
        H5ParameterClassification.MIXED
        if meaningful_improvement
        else H5ParameterClassification.NOT_SUPPORTED
    )