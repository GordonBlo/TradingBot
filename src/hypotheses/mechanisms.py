"""Frozen V3.2.2 H0/H1 controls and one-change H5-H7 mechanisms."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Mapping

from src.backtest.models import BacktestConfig
from src.hypotheses.candidates import build_candidate
from src.hypotheses.causal import CausalRollingPercentile
from src.hypotheses.models import HypothesisId, HypothesisSuiteConfig, JournalEvent
from src.strategy.base import BaseStrategy
from src.strategy.baseline_trend import TrendMomentumBaselineStrategy
from src.strategy.context import StrategyContext
from src.strategy.models import (
    DecisionReason,
    StrategyAction,
    StrategyDecision,
    TrendMomentumConfig,
)


class MechanismHypothesisId(str, Enum):
    H0 = "H0"
    H1 = "H1"
    H5 = "H5"
    H6 = "H6"
    H7 = "H7"


class MechanismClassification(str, Enum):
    CONTROL = "CONTROL"
    FROZEN_REFERENCE = "FROZEN_REFERENCE"
    NEXT_STAGE_ELIGIBLE = "NEXT_STAGE_ELIGIBLE"
    MECHANISM_SUPPORTED = "MECHANISM_SUPPORTED_ON_RESEARCH_DATA"
    MIXED = "MIXED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True, slots=True)
class MechanismSuiteConfig:
    h5_lookback: int = 100
    h5_min_percentile: Decimal = Decimal("25")
    h5_max_percentile: Decimal = Decimal("75")
    h6_confirmation_bars: int = 1
    h7_cost_opportunity_multiplier: Decimal = Decimal("5")
    trade_count_ratio_warning: Decimal = Decimal("0.40")

    def __post_init__(self) -> None:
        for name in (
            "h5_min_percentile",
            "h5_max_percentile",
            "h7_cost_opportunity_multiplier",
            "trade_count_ratio_warning",
        ):
            value = Decimal(str(getattr(self, name)))
            if not value.is_finite():
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        fixed = {
            "h5_lookback": 100,
            "h5_min_percentile": Decimal("25"),
            "h5_max_percentile": Decimal("75"),
            "h6_confirmation_bars": 1,
            "h7_cost_opportunity_multiplier": Decimal("5"),
            "trade_count_ratio_warning": Decimal("0.40"),
        }
        for name, expected in fixed.items():
            if getattr(self, name) != expected:
                raise ValueError(
                    f"{name} is frozen at {expected} for V3.2.2; tuning is refused."
                )


@dataclass(frozen=True, slots=True)
class MechanismDefinition:
    hypothesis_id: MechanismHypothesisId
    name: str
    rationale: str
    mechanism_changed: str
    fixed_parameters: Mapping[str, str | int]
    candidate_behavior: str
    creation_version: str = "3.2.2"


@dataclass(frozen=True, slots=True)
class MechanismJournalRecord:
    timestamp: datetime
    hypothesis_id: MechanismHypothesisId
    event: JournalEvent
    reason: str
    values: Mapping[str, Decimal | str | int | None] = field(default_factory=dict)


def mechanism_registry(
    config: MechanismSuiteConfig | None = None,
) -> tuple[MechanismDefinition, ...]:
    fixed = config or MechanismSuiteConfig()
    return (
        MechanismDefinition(
            MechanismHypothesisId.H0,
            "Frozen Baseline",
            "Exact V3.2.1 control.",
            "None",
            {},
            "Exact frozen H0 behavior.",
        ),
        MechanismDefinition(
            MechanismHypothesisId.H1,
            "Frozen High-Volatility Guard",
            "Exact V3.2.1 reference mechanism.",
            "None relative to frozen H1.",
            {"H1_LOOKBACK": 100, "H1_MAX_PERCENTILE": "75"},
            "ATR% <= causal Q75 of the previous 100 valid observations.",
        ),
        MechanismDefinition(
            MechanismHypothesisId.H5,
            "Volatility Band",
            "Reject both unusually low- and high-volatility baseline entries.",
            "Adds only a causal two-sided ATR% entry band.",
            {
                "H5_LOOKBACK": fixed.h5_lookback,
                "H5_MIN_PERCENTILE": str(fixed.h5_min_percentile),
                "H5_MAX_PERCENTILE": str(fixed.h5_max_percentile),
            },
            "Require prior-window causal Q25 <= ATR% <= causal Q75.",
        ),
        MechanismDefinition(
            MechanismHypothesisId.H6,
            "One-Bar Trend Persistence",
            "Require a baseline crossover to persist for one more closed bar.",
            "Changes only entry timing by exactly one confirmation bar.",
            {"H6_CONFIRMATION_BARS": fixed.h6_confirmation_bars},
            "Arm on H0 crossover; enter only if the next closed bar remains baseline-valid.",
        ),
        MechanismDefinition(
            MechanismHypothesisId.H7,
            "Cost-to-Opportunity Guard",
            "Reject baseline entries whose theoretical target is small relative to costs.",
            "Adds only a signal-time cost/opportunity entry guard.",
            {
                "H7_COST_OPPORTUNITY_MULTIPLIER": str(
                    fixed.h7_cost_opportunity_multiplier
                )
            },
            "Require (4*ATR/close)*10000 >= 5*(2*fee_bps + 2*slippage_bps).",
        ),
    )


def _hold(reason: str) -> StrategyDecision:
    return StrategyDecision(StrategyAction.HOLD, DecisionReason.NO_ENTRY, reason)


def _atr_percent(context: StrategyContext) -> Decimal | None:
    atr = context.indicators.atr
    close = context.indicators.close
    return atr / close * Decimal("100") if atr is not None and close > 0 else None


class VolatilityBandStrategy(BaseStrategy):
    """H5: bounded causal Q25-Q75 ATR% band around unchanged H0 entries."""

    def __init__(
        self,
        baseline_config: TrendMomentumConfig,
        *,
        lookback: int,
        minimum_percentile: Decimal,
        maximum_percentile: Decimal,
    ) -> None:
        self.baseline_config = baseline_config
        self._baseline = TrendMomentumBaselineStrategy(baseline_config)
        self._lower = CausalRollingPercentile(
            lookback=lookback, percentile_rank=minimum_percentile
        )
        self._upper = CausalRollingPercentile(
            lookback=lookback, percentile_rank=maximum_percentile
        )
        self.last_thresholds: tuple[Decimal | None, Decimal | None] = (None, None)
        self.journal: deque[MechanismJournalRecord] = deque(maxlen=lookback)

    @property
    def observation_count(self) -> int:
        return self._lower.observation_count

    def evaluate(self, context: StrategyContext) -> StrategyDecision:
        current_value = _atr_percent(context)
        lower = self._lower.threshold()
        upper = self._upper.threshold()
        self.last_thresholds = (lower, upper)
        self._lower.observe(current_value)
        self._upper.observe(current_value)
        baseline = self._baseline.evaluate(context)
        if baseline.action is not StrategyAction.ENTER_LONG:
            return baseline
        if (
            current_value is not None
            and lower is not None
            and upper is not None
            and lower <= current_value <= upper
        ):
            return baseline
        reason = (
            "H5 lacks 100 prior valid ATR% observations"
            if lower is None or upper is None
            else f"H5 ATR% {current_value} is outside causal band [{lower}, {upper}]"
        )
        self.journal.append(
            MechanismJournalRecord(
                timestamp=context.timestamp,
                hypothesis_id=MechanismHypothesisId.H5,
                event=JournalEvent.SKIPPED_ENTRY,
                reason=reason,
                values={
                    "atr_percent": current_value,
                    "causal_q25": lower,
                    "causal_q75": upper,
                },
            )
        )
        return _hold(reason)


class OneBarPersistenceStrategy(BaseStrategy):
    """H6: one bounded pending flag and exactly one confirmation candle."""

    def __init__(self, baseline_config: TrendMomentumConfig) -> None:
        self.baseline_config = baseline_config
        self._baseline = TrendMomentumBaselineStrategy(baseline_config)
        self._pending = False
        self.journal: deque[MechanismJournalRecord] = deque(maxlen=100)

    @property
    def is_pending(self) -> bool:
        return self._pending

    def evaluate(self, context: StrategyContext) -> StrategyDecision:
        baseline = self._baseline.evaluate(context)
        if context.has_position:
            self._pending = False
            return baseline
        if self._pending:
            self._pending = False
            if self._confirmation_valid(context):
                decision = _entry_from_context(
                    context,
                    self.baseline_config,
                    "H6 one-bar trend persistence confirmed; execute at next bar open",
                    {"confirmation_bars": 1},
                )
                self._record(context, JournalEvent.CONFIRMED, decision.reason)
                return decision
            self._record(
                context,
                JournalEvent.INVALIDATED,
                "H6 one-bar confirmation failed; a new crossover is required",
            )
            return _hold("H6 confirmation failed; setup cancelled")
        if baseline.action is StrategyAction.ENTER_LONG:
            self._pending = True
            self._record(context, JournalEvent.ARMED, "H6 pending exactly one closed bar")
            return _hold("H6 crossover detected; immediate entry delayed one closed bar")
        return baseline

    def _confirmation_valid(self, context: StrategyContext) -> bool:
        current = context.indicators
        if not current.is_ready:
            return False
        assert current.ema_fast is not None
        assert current.ema_slow is not None
        assert current.rsi is not None
        assert current.volume_ratio is not None
        return (
            current.ema_fast > current.ema_slow
            and current.close > current.ema_slow
            and self.baseline_config.rsi_min <= current.rsi <= self.baseline_config.rsi_max
            and current.volume_ratio >= self.baseline_config.minimum_volume_ratio
        )

    def _record(
        self, context: StrategyContext, event: JournalEvent, reason: str
    ) -> None:
        self.journal.append(
            MechanismJournalRecord(
                timestamp=context.timestamp,
                hypothesis_id=MechanismHypothesisId.H6,
                event=event,
                reason=reason,
                values={"confirmation_bars": 1},
            )
        )


def target_distance_bps(atr: Decimal, close: Decimal) -> Decimal:
    if close <= 0 or atr < 0:
        raise ValueError("H7 ATR must be non-negative and close must be positive.")
    return Decimal("4") * atr / close * Decimal("10000")


def round_trip_cost_bps(fee_bps: Decimal, slippage_bps: Decimal) -> Decimal:
    fee = Decimal(str(fee_bps))
    slippage = Decimal(str(slippage_bps))
    if fee < 0 or slippage < 0:
        raise ValueError("H7 configured costs must be non-negative.")
    return Decimal("2") * fee + Decimal("2") * slippage


class CostOpportunityGuardStrategy(BaseStrategy):
    """H7: compare causal theoretical target distance with frozen base costs."""

    def __init__(
        self,
        baseline_config: TrendMomentumConfig,
        *,
        fee_bps: Decimal,
        slippage_bps: Decimal,
        multiplier: Decimal,
    ) -> None:
        self.baseline_config = baseline_config
        self._baseline = TrendMomentumBaselineStrategy(baseline_config)
        self.round_trip_cost_bps = round_trip_cost_bps(fee_bps, slippage_bps)
        self.multiplier = Decimal(str(multiplier))
        self.required_target_distance_bps = self.multiplier * self.round_trip_cost_bps
        self.journal: deque[MechanismJournalRecord] = deque(maxlen=100)

    def evaluate(self, context: StrategyContext) -> StrategyDecision:
        baseline = self._baseline.evaluate(context)
        if baseline.action is not StrategyAction.ENTER_LONG:
            return baseline
        atr = context.indicators.atr
        assert atr is not None
        opportunity = target_distance_bps(atr, context.indicators.close)
        if opportunity >= self.required_target_distance_bps:
            return baseline
        reason = (
            f"H7 target distance {opportunity} bps is below fixed cost threshold "
            f"{self.required_target_distance_bps} bps"
        )
        self.journal.append(
            MechanismJournalRecord(
                timestamp=context.timestamp,
                hypothesis_id=MechanismHypothesisId.H7,
                event=JournalEvent.SKIPPED_ENTRY,
                reason=reason,
                values={
                    "target_distance_bps": opportunity,
                    "round_trip_cost_bps": self.round_trip_cost_bps,
                    "required_target_distance_bps": self.required_target_distance_bps,
                },
            )
        )
        return _hold(reason)


def _entry_from_context(
    context: StrategyContext,
    config: TrendMomentumConfig,
    reason: str,
    metadata: Mapping[str, str | int | Decimal | bool],
) -> StrategyDecision:
    current = context.indicators
    assert current.atr is not None
    stop_distance = current.atr * config.atr_stop_multiplier
    risk_budget = context.equity * config.risk_per_trade_percent / Decimal("100")
    affordable = context.cash_usdc / (Decimal("1") + context.entry_fee_rate)
    max_quote = min(affordable, config.maximum_position_notional_usdc)
    if risk_budget <= 0 or stop_distance <= 0 or max_quote <= 0:
        return StrategyDecision(
            StrategyAction.HOLD,
            DecisionReason.INSUFFICIENT_CAPITAL,
            "No positive risk budget or affordable Spot notional",
        )
    return StrategyDecision(
        StrategyAction.ENTER_LONG,
        DecisionReason.EMA_CROSSOVER_ENTRY,
        reason,
        risk_budget=risk_budget,
        stop_distance=stop_distance,
        reward_risk_ratio=config.reward_risk_ratio,
        max_quote_amount=max_quote,
        metadata=metadata,
    )


def build_mechanism_candidate(
    hypothesis_id: MechanismHypothesisId,
    baseline_config: TrendMomentumConfig,
    mechanism_config: MechanismSuiteConfig,
    v32_config: HypothesisSuiteConfig,
    base_backtest_config: BacktestConfig,
) -> BaseStrategy:
    candidate = MechanismHypothesisId(hypothesis_id)
    if candidate is MechanismHypothesisId.H0:
        return build_candidate(HypothesisId.H0, baseline_config, v32_config)
    if candidate is MechanismHypothesisId.H1:
        return build_candidate(HypothesisId.H1, baseline_config, v32_config)
    if candidate is MechanismHypothesisId.H5:
        return VolatilityBandStrategy(
            baseline_config,
            lookback=mechanism_config.h5_lookback,
            minimum_percentile=mechanism_config.h5_min_percentile,
            maximum_percentile=mechanism_config.h5_max_percentile,
        )
    if candidate is MechanismHypothesisId.H6:
        return OneBarPersistenceStrategy(baseline_config)
    if candidate is MechanismHypothesisId.H7:
        return CostOpportunityGuardStrategy(
            baseline_config,
            fee_bps=base_backtest_config.fee_bps,
            slippage_bps=base_backtest_config.slippage_bps,
            multiplier=mechanism_config.h7_cost_opportunity_multiplier,
        )
    raise AssertionError("Unreachable V3.2.2 candidate.")
