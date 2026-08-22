"""Configuration and decision models for V3 strategy research."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Mapping


def _decimal(name: str, value: Decimal) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a valid decimal.") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be finite.")
    return parsed


class StrategyAction(str, Enum):
    ENTER_LONG = "ENTER_LONG"
    EXIT_LONG = "EXIT_LONG"
    HOLD = "HOLD"


class DecisionReason(str, Enum):
    WARMUP = "WARMUP"
    NO_ENTRY = "NO_ENTRY"
    EMA_CROSSOVER_ENTRY = "EMA_CROSSOVER_ENTRY"
    PRICE_BREAKOUT_ENTRY = "PRICE_BREAKOUT_ENTRY"
    EXHAUSTION_RECLAIM_ENTRY = "EXHAUSTION_RECLAIM_ENTRY"
    TREND_EXIT = "TREND_EXIT"
    TIME_EXIT = "TIME_EXIT"
    COOLDOWN = "COOLDOWN"
    INSUFFICIENT_CAPITAL = "INSUFFICIENT_CAPITAL"


@dataclass(frozen=True, slots=True)
class StrategyDecision:
    action: StrategyAction
    reason_code: DecisionReason
    reason: str
    risk_budget: Decimal | None = None
    stop_distance: Decimal | None = None
    reward_risk_ratio: Decimal | None = None
    max_quote_amount: Decimal | None = None
    metadata: Mapping[str, str | int | Decimal | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.action, StrategyAction):
            object.__setattr__(self, "action", StrategyAction(self.action))
        if not isinstance(self.reason_code, DecisionReason):
            object.__setattr__(
                self, "reason_code", DecisionReason(self.reason_code)
            )
        reason = self.reason.strip()
        if not reason:
            raise ValueError("Strategy decision reason must not be empty.")
        object.__setattr__(self, "reason", reason)
        sizing = (
            self.risk_budget,
            self.stop_distance,
            self.reward_risk_ratio,
            self.max_quote_amount,
        )
        for name in (
            "risk_budget",
            "stop_distance",
            "reward_risk_ratio",
            "max_quote_amount",
        ):
            value = getattr(self, name)
            if value is not None:
                parsed = _decimal(name, value)
                if parsed <= 0:
                    raise ValueError(f"{name} must be greater than zero.")
                object.__setattr__(self, name, parsed)
        if self.action is StrategyAction.ENTER_LONG and not all(
            value is not None for value in sizing
        ):
            raise ValueError("ENTER_LONG requires all risk-sizing fields.")
        if self.action is not StrategyAction.ENTER_LONG and any(
            value is not None for value in sizing
        ):
            raise ValueError("Only ENTER_LONG can contain risk-sizing fields.")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class TrendMomentumConfig:
    fast_ema_period: int = 20
    slow_ema_period: int = 50
    rsi_period: int = 14
    rsi_min: Decimal = Decimal("52")
    rsi_max: Decimal = Decimal("68")
    volume_sma_period: int = 20
    minimum_volume_ratio: Decimal = Decimal("0.80")
    atr_period: int = 14
    atr_stop_multiplier: Decimal = Decimal("2.0")
    reward_risk_ratio: Decimal = Decimal("2.0")
    risk_per_trade_percent: Decimal = Decimal("0.50")
    maximum_bars_in_position: int = 96
    cooldown_bars: int = 4
    maximum_position_notional_usdc: Decimal = Decimal("50")

    def __post_init__(self) -> None:
        for name in (
            "fast_ema_period",
            "slow_ema_period",
            "rsi_period",
            "volume_sma_period",
            "atr_period",
            "maximum_bars_in_position",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1.")
        if self.cooldown_bars < 0:
            raise ValueError("cooldown_bars must not be negative.")
        if self.fast_ema_period >= self.slow_ema_period:
            raise ValueError("Fast EMA period must be less than slow EMA period.")
        for name in (
            "rsi_min",
            "rsi_max",
            "minimum_volume_ratio",
            "atr_stop_multiplier",
            "reward_risk_ratio",
            "risk_per_trade_percent",
            "maximum_position_notional_usdc",
        ):
            parsed = _decimal(name, getattr(self, name))
            object.__setattr__(self, name, parsed)
        if not Decimal("0") <= self.rsi_min <= self.rsi_max <= Decimal("100"):
            raise ValueError("RSI limits must satisfy 0 <= min <= max <= 100.")
        if self.minimum_volume_ratio < 0:
            raise ValueError("Minimum volume ratio must not be negative.")
        if self.atr_stop_multiplier <= 0 or self.reward_risk_ratio <= 0:
            raise ValueError("ATR multiplier and reward/risk ratio must be positive.")
        if not Decimal("0") < self.risk_per_trade_percent <= Decimal("100"):
            raise ValueError("Risk per trade percent must be in (0, 100].")
        if self.maximum_position_notional_usdc <= 0:
            raise ValueError("Maximum position notional must be positive.")
