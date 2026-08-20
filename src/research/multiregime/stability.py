"""Pre-registered V3.2.1 support gate and window distributions."""

from __future__ import annotations

from decimal import Decimal

from src.diagnostics.distributions import percentile
from src.hypotheses.models import HypothesisId
from src.hypotheses.runner import HypothesisMetrics
from src.research.multiregime.models import (
    MultiRegimeClassification,
    MultiRegimeConfig,
    SupportGateResult,
    WindowDistribution,
)
from src.hypotheses.mechanisms import (
    MechanismClassification,
    MechanismHypothesisId,
)


def distribution(values: tuple[Decimal, ...]) -> WindowDistribution:
    return WindowDistribution(
        percentile_25=percentile(values, Decimal("0.25")),
        median=percentile(values, Decimal("0.50")),
        percentile_75=percentile(values, Decimal("0.75")),
    )


def support_gate(
    *,
    candidate: HypothesisMetrics,
    baseline: HypothesisMetrics,
    candidate_stress: HypothesisMetrics | None,
    baseline_stress: HypothesisMetrics | None,
    eligible_windows: int,
    frictionless_better_windows: int,
    net_better_windows: int,
    config: MultiRegimeConfig,
) -> SupportGateResult:
    """Evaluate exactly the rules registered before expanded results are viewed."""

    required_fraction = config.consistency_percent_required / Decimal("100")
    frictionless_fraction = (
        Decimal(frictionless_better_windows) / Decimal(eligible_windows)
        if eligible_windows
        else Decimal("0")
    )
    net_fraction = (
        Decimal(net_better_windows) / Decimal(eligible_windows)
        if eligible_windows
        else Decimal("0")
    )
    profit_factor_not_worse = (
        candidate.profit_factor is not None
        and baseline.profit_factor is not None
        and candidate.profit_factor >= baseline.profit_factor
    )
    trade_ratio = (
        Decimal(candidate.trades) / Decimal(baseline.trades)
        if baseline.trades
        else Decimal("0")
    )
    stress_not_worse = (
        candidate_stress is not None
        and baseline_stress is not None
        and candidate_stress.expectancy >= baseline_stress.expectancy
    )
    return SupportGateResult(
        enough_eligible_windows=eligible_windows >= config.minimum_eligible_windows,
        frictionless_expectancy_better=(
            candidate.average_frictionless_pnl_per_trade
            > baseline.average_frictionless_pnl_per_trade
        ),
        net_expectancy_better=candidate.expectancy > baseline.expectancy,
        profit_factor_not_worse=profit_factor_not_worse,
        frictionless_consistency_met=frictionless_fraction >= required_fraction,
        net_consistency_met=net_fraction >= required_fraction,
        trade_count_ratio_met=trade_ratio >= config.trade_count_ratio_required,
        drawdown_limit_met=(
            candidate.maximum_drawdown_percent
            <= baseline.maximum_drawdown_percent * config.maximum_drawdown_worse_ratio
        ),
        stress_expectancy_not_worse=stress_not_worse,
    )


def classify_candidate(
    *,
    hypothesis_id: HypothesisId,
    gate: SupportGateResult,
    combined: HypothesisMetrics,
    baseline: HypothesisMetrics,
    eligible_windows: int,
    minimum_trades_warning: int,
    frictionless_better_windows: int,
    net_better_windows: int,
    config: MultiRegimeConfig,
) -> MultiRegimeClassification:
    if hypothesis_id is HypothesisId.H0:
        return MultiRegimeClassification.CONTROL
    if eligible_windows < config.minimum_eligible_windows or combined.trades < minimum_trades_warning:
        return MultiRegimeClassification.INSUFFICIENT_EVIDENCE
    if gate.all_met:
        if combined.average_frictionless_pnl_per_trade > 0:
            return MultiRegimeClassification.V3_3_ELIGIBLE
        return MultiRegimeClassification.MECHANISM_SUPPORTED
    meaningful_improvement = any(
        (
            combined.average_frictionless_pnl_per_trade
            > baseline.average_frictionless_pnl_per_trade,
            combined.expectancy > baseline.expectancy,
            frictionless_better_windows * 2 >= eligible_windows,
            net_better_windows * 2 >= eligible_windows,
        )
    )
    return (
        MultiRegimeClassification.MIXED
        if meaningful_improvement
        else MultiRegimeClassification.NOT_SUPPORTED
    )


def classify_mechanism_candidate(
    *,
    hypothesis_id: MechanismHypothesisId,
    gate: SupportGateResult,
    combined: HypothesisMetrics,
    baseline: HypothesisMetrics,
    eligible_windows: int,
    minimum_trades_warning: int,
    frictionless_better_windows: int,
    net_better_windows: int,
    config: MultiRegimeConfig,
) -> MechanismClassification:
    """Apply the preregistered V3.2.2 labels without changing the V3.2.1 gate."""

    if hypothesis_id is MechanismHypothesisId.H0:
        return MechanismClassification.CONTROL
    if hypothesis_id is MechanismHypothesisId.H1:
        return MechanismClassification.FROZEN_REFERENCE
    if (
        eligible_windows < config.minimum_eligible_windows
        or combined.trades < minimum_trades_warning
    ):
        return MechanismClassification.INSUFFICIENT_EVIDENCE
    if gate.all_met:
        if combined.average_frictionless_pnl_per_trade > 0:
            return MechanismClassification.NEXT_STAGE_ELIGIBLE
        return MechanismClassification.MECHANISM_SUPPORTED
    meaningful_improvement = any(
        (
            combined.average_frictionless_pnl_per_trade
            > baseline.average_frictionless_pnl_per_trade,
            combined.expectancy > baseline.expectancy,
            frictionless_better_windows * 2 >= eligible_windows,
            net_better_windows * 2 >= eligible_windows,
        )
    )
    return (
        MechanismClassification.MIXED
        if meaningful_improvement
        else MechanismClassification.NOT_SUPPORTED
    )
