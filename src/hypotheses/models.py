"""Typed V3.2 hypothesis definitions and research journal records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Mapping


class HypothesisId(str, Enum):
    H0 = "H0"
    H1 = "H1"
    H2 = "H2"
    H3 = "H3"
    H4 = "H4"


class HypothesisStatus(str, Enum):
    REGISTERED = "REGISTERED"
    TESTED_RESEARCH_DATA = "TESTED_RESEARCH_DATA"
    ELIGIBLE_FOR_BLIND_HOLDOUT = "ELIGIBLE_FOR_BLIND_HOLDOUT"
    REJECTED = "REJECTED"
    CONSUMED = "CONSUMED"


class DatasetStatus(str, Enum):
    CONSUMED_RESEARCH_DATA = "CONSUMED_RESEARCH_DATA"
    BLIND_HOLDOUT = "BLIND_HOLDOUT"


class SupportClassification(str, Enum):
    CONTROL = "CONTROL"
    SUPPORTED_ON_CONSUMED_DATA = "SUPPORTED ON CONSUMED DATA"
    MIXED = "MIXED"
    NOT_SUPPORTED = "NOT SUPPORTED"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT SAMPLE"


class JournalEvent(str, Enum):
    SKIPPED_ENTRY = "SKIPPED_ENTRY"
    CROSSOVER_DETECTED = "CROSSOVER_DETECTED"
    ARMED = "ARMED"
    CONFIRMED = "CONFIRMED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True, slots=True)
class HypothesisSuiteConfig:
    """The immutable, pre-registered V3.2 candidate parameters."""

    h1_atr_percentile_lookback: int = 100
    h1_max_percentile: Decimal = Decimal("75")
    h2_ema_spread_lookback: int = 100
    h2_max_percentile: Decimal = Decimal("75")
    h3_confirmation_window_bars: int = 8
    h4_reward_risk_ratio: Decimal = Decimal("1.0")
    trade_count_ratio_warning: Decimal = Decimal("0.40")

    def __post_init__(self) -> None:
        for name in (
            "h1_max_percentile",
            "h2_max_percentile",
            "h4_reward_risk_ratio",
            "trade_count_ratio_warning",
        ):
            value = Decimal(str(getattr(self, name)))
            if not value.is_finite():
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if self.h1_atr_percentile_lookback != 100:
            raise ValueError("H1 lookback is pre-registered at exactly 100.")
        if self.h1_max_percentile != Decimal("75"):
            raise ValueError("H1 percentile is pre-registered at exactly 75.")
        if self.h2_ema_spread_lookback != 100:
            raise ValueError("H2 lookback is pre-registered at exactly 100.")
        if self.h2_max_percentile != Decimal("75"):
            raise ValueError("H2 percentile is pre-registered at exactly 75.")
        if self.h3_confirmation_window_bars != 8:
            raise ValueError("H3 confirmation window is pre-registered at exactly 8 bars.")
        if self.h4_reward_risk_ratio != Decimal("1.0"):
            raise ValueError("H4 reward/risk is pre-registered at exactly 1.0.")
        if not Decimal("0") < self.trade_count_ratio_warning <= Decimal("1"):
            raise ValueError("Trade-count warning ratio must be in (0, 1].")


@dataclass(frozen=True, slots=True)
class StrategyHypothesis:
    hypothesis_id: HypothesisId
    name: str
    rationale: str
    diagnostic_evidence: str
    mechanism_changed: str
    baseline_behavior: str
    candidate_behavior: str
    fixed_parameters: Mapping[str, str | int]
    creation_version: str = "3.2"
    status: HypothesisStatus = HypothesisStatus.REGISTERED


@dataclass(frozen=True, slots=True)
class HypothesisJournalRecord:
    timestamp: datetime
    hypothesis_id: HypothesisId
    event: JournalEvent
    reason: str
    values: Mapping[str, Decimal | str | int | None] = field(default_factory=dict)


def hypothesis_registry(
    config: HypothesisSuiteConfig | None = None,
) -> tuple[StrategyHypothesis, ...]:
    """Return exactly H0-H4 in stable order; no generated candidates exist."""

    fixed = config or HypothesisSuiteConfig()
    held_constant = (
        "EMA20/EMA50, RSI14 52-68, volume ratio >=0.80, ATR14 2x stop, "
        "0.50% risk, 96-bar maximum hold, 4-bar cooldown, V2 next-open/STOP_FIRST"
    )
    return (
        StrategyHypothesis(
            HypothesisId.H0,
            "Frozen Baseline",
            "Control group for all V3.2 comparisons.",
            "The exact V3 TrendMomentumBaselineStrategy is frozen.",
            "None",
            held_constant,
            held_constant,
            {},
        ),
        StrategyHypothesis(
            HypothesisId.H1,
            "High Volatility Entry Guard",
            "Unusually high short-term volatility may reduce entry expectancy.",
            "Winner ATR% was lower than loser ATR% in all inspected periods.",
            "Adds only a causal ATR% entry guard.",
            held_constant,
            "Baseline entry is accepted only when current ATR% is <= the prior-window Q75.",
            {
                "H1_ATR_PERCENTILE_LOOKBACK": fixed.h1_atr_percentile_lookback,
                "H1_MAX_PERCENTILE": str(fixed.h1_max_percentile),
            },
        ),
        StrategyHypothesis(
            HypothesisId.H2,
            "EMA Extension Entry Guard",
            "More extended EMA separation may indicate late entries.",
            "Winner EMA spread % was lower than loser spread % in all inspected periods.",
            "Adds only a causal EMA-spread entry guard.",
            held_constant,
            "Baseline entry is accepted only when current EMA spread% is <= the prior-window Q75.",
            {
                "H2_EMA_SPREAD_LOOKBACK": fixed.h2_ema_spread_lookback,
                "H2_MAX_PERCENTILE": str(fixed.h2_max_percentile),
            },
        ),
        StrategyHypothesis(
            HypothesisId.H3,
            "Crossover Pullback Confirmation",
            "A causal pullback/reclaim may improve immediate crossover timing.",
            "Many losing baseline trades moved adverse early.",
            "Changes only entry timing after a baseline-valid crossover.",
            held_constant,
            "Arm after the crossover, wait at most eight closed bars for EMA20 pullback/reclaim, then enter next open.",
            {"H3_CONFIRMATION_WINDOW_BARS": fixed.h3_confirmation_window_bars},
        ),
        StrategyHypothesis(
            HypothesisId.H4,
            "1R Profit Target",
            "The 2R target may exceed the observed favorable-excursion distribution.",
            "Median MFE was below 1R and only about 24-35% of trades reached 2R.",
            "Changes only reward/risk target from 2R to 1R.",
            held_constant,
            "Exact baseline entry/exit behavior with a 1R take-profit bracket.",
            {"H4_REWARD_RISK_RATIO": str(fixed.h4_reward_risk_ratio)},
        ),
    )
