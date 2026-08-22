"""Frozen V5-H0 exhaustion-reclaim mean-reversion strategy."""

from __future__ import annotations

from decimal import Decimal

from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext
from src.strategy.models import (
    DecisionReason,
    StrategyAction,
    StrategyDecision,
    TrendMomentumConfig,
)


class V5MeanReversionStrategy(BaseStrategy):
    """Buy a strict lower-band reclaim while price remains below its mean."""

    NAME = "V5MeanReversionStrategy"
    VERSION = "5.0"
    REFERENCE_LOOKBACK_BARS = 20
    STANDARD_DEVIATION_DDOF = 0
    LOWER_BAND_STANDARD_DEVIATIONS = Decimal("2")
    ATR_PERIOD = 14
    ATR_STOP_MULTIPLIER = Decimal("2")
    REWARD_RISK_RATIO = Decimal("2")
    MAXIMUM_HOLD_BARS = 96
    COOLDOWN_BARS = 4

    def __init__(self, config: TrendMomentumConfig) -> None:
        frozen = {
            "atr_period": self.ATR_PERIOD,
            "atr_stop_multiplier": self.ATR_STOP_MULTIPLIER,
            "reward_risk_ratio": self.REWARD_RISK_RATIO,
            "maximum_bars_in_position": self.MAXIMUM_HOLD_BARS,
            "cooldown_bars": self.COOLDOWN_BARS,
        }
        for name, expected in frozen.items():
            if getattr(config, name) != expected:
                raise ValueError(f"V5-H0 {name} is frozen at {expected}.")
        self.config = config

    def evaluate(self, context: StrategyContext) -> StrategyDecision:
        if context.has_position:
            if context.bars_in_position >= self.MAXIMUM_HOLD_BARS:
                return StrategyDecision(
                    StrategyAction.EXIT_LONG,
                    DecisionReason.TIME_EXIT,
                    f"Maximum holding period reached ({context.bars_in_position} bars)",
                )
            return self._hold(DecisionReason.NO_ENTRY, "Long position remains valid")

        if (
            context.bars_since_exit is not None
            and context.bars_since_exit <= self.COOLDOWN_BARS
        ):
            return self._hold(
                DecisionReason.COOLDOWN,
                f"Cooldown active ({context.bars_since_exit}/{self.COOLDOWN_BARS} bars)",
            )

        required_history = self.REFERENCE_LOOKBACK_BARS + 1
        if len(context.recent_history) < required_history:
            return self._hold(
                DecisionReason.WARMUP,
                "Previous 20 closed candles unavailable",
            )

        reference_candles = context.recent_history[-required_history:-1]
        if not all(candle.is_closed for candle in reference_candles):
            return self._hold(
                DecisionReason.WARMUP,
                "Previous 20 fully closed candles unavailable",
            )

        atr = context.indicators.atr
        if atr is None or atr <= 0:
            return self._hold(DecisionReason.WARMUP, "ATR14 unavailable")

        closes = tuple(candle.close for candle in reference_candles)
        reference_mean = sum(closes, start=Decimal("0")) / Decimal(
            self.REFERENCE_LOOKBACK_BARS
        )
        variance = sum(
            ((close - reference_mean) ** 2 for close in closes),
            start=Decimal("0"),
        ) / Decimal(self.REFERENCE_LOOKBACK_BARS)
        reference_std = variance.sqrt()
        if reference_std == 0:
            return self._hold(
                DecisionReason.NO_ENTRY,
                "Reference population standard deviation is zero",
            )

        lower_band = (
            reference_mean
            - self.LOWER_BAND_STANDARD_DEVIATIONS * reference_std
        )
        current = context.current_candle
        if not (
            current.low < lower_band
            and current.close > lower_band
            and current.close < reference_mean
        ):
            return self._hold(
                DecisionReason.NO_ENTRY,
                "Strict exhaustion-reclaim conditions not satisfied",
            )

        stop_distance = atr * self.ATR_STOP_MULTIPLIER
        risk_budget = (
            context.equity * self.config.risk_per_trade_percent / Decimal("100")
        )
        affordable = context.cash_usdc / (Decimal("1") + context.entry_fee_rate)
        max_quote = min(affordable, self.config.maximum_position_notional_usdc)
        if risk_budget <= 0 or stop_distance <= 0 or max_quote <= 0:
            return self._hold(
                DecisionReason.INSUFFICIENT_CAPITAL,
                "No positive risk budget or affordable Spot notional",
            )

        return StrategyDecision(
            StrategyAction.ENTER_LONG,
            DecisionReason.EXHAUSTION_RECLAIM_ENTRY,
            "Low pierced and close strictly reclaimed the lower population band",
            risk_budget=risk_budget,
            stop_distance=stop_distance,
            reward_risk_ratio=self.REWARD_RISK_RATIO,
            max_quote_amount=max_quote,
            metadata={
                "atr": atr,
                "reference_mean": reference_mean,
                "reference_standard_deviation": reference_std,
                "lower_band": lower_band,
                "reference_lookback_bars": self.REFERENCE_LOOKBACK_BARS,
                "standard_deviation_ddof": self.STANDARD_DEVIATION_DDOF,
            },
        )

    @staticmethod
    def _hold(reason_code: DecisionReason, reason: str) -> StrategyDecision:
        return StrategyDecision(StrategyAction.HOLD, reason_code, reason)
