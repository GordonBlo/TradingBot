"""Frozen V4-H0 price-breakout strategy."""

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


class V4BreakoutStrategy(BaseStrategy):
    """Enter on a strict close breakout of the previous 20 closed highs."""

    NAME = "V4BreakoutStrategy"
    VERSION = "4.0"
    BREAKOUT_LOOKBACK_BARS = 20

    def __init__(self, config: TrendMomentumConfig) -> None:
        self.config = config

    def evaluate(self, context: StrategyContext) -> StrategyDecision:
        if context.has_position:
            if context.bars_in_position >= self.config.maximum_bars_in_position:
                return StrategyDecision(
                    StrategyAction.EXIT_LONG,
                    DecisionReason.TIME_EXIT,
                    f"Maximum holding period reached ({context.bars_in_position} bars)",
                )
            return self._hold(DecisionReason.NO_ENTRY, "Long position remains valid")

        if (
            context.bars_since_exit is not None
            and context.bars_since_exit <= self.config.cooldown_bars
        ):
            return self._hold(
                DecisionReason.COOLDOWN,
                f"Cooldown active ({context.bars_since_exit}/{self.config.cooldown_bars} bars)",
            )

        required_history = self.BREAKOUT_LOOKBACK_BARS + 1
        if len(context.recent_history) < required_history:
            return self._hold(
                DecisionReason.WARMUP,
                "Previous 20 closed candles unavailable",
            )

        atr = context.indicators.atr
        if atr is None or atr <= 0:
            return self._hold(DecisionReason.WARMUP, "ATR14 unavailable")

        reference_candles = context.recent_history[-required_history:-1]
        reference_high = max(candle.high for candle in reference_candles)
        close = context.current_candle.close
        if close <= reference_high:
            return self._hold(
                DecisionReason.NO_ENTRY,
                "Close did not exceed the previous-20 high",
            )

        stop_distance = atr * self.config.atr_stop_multiplier
        risk_budget = (
            context.equity * self.config.risk_per_trade_percent / Decimal("100")
        )
        affordable = context.cash_usdc / (Decimal("1") + context.entry_fee_rate)
        max_quote = min(
            affordable,
            self.config.maximum_position_notional_usdc,
        )
        if risk_budget <= 0 or stop_distance <= 0 or max_quote <= 0:
            return self._hold(
                DecisionReason.INSUFFICIENT_CAPITAL,
                "No positive risk budget or affordable Spot notional",
            )

        return StrategyDecision(
            StrategyAction.ENTER_LONG,
            DecisionReason.PRICE_BREAKOUT_ENTRY,
            "Close broke strictly above the previous 20 closed-candle highs",
            risk_budget=risk_budget,
            stop_distance=stop_distance,
            reward_risk_ratio=self.config.reward_risk_ratio,
            max_quote_amount=max_quote,
            metadata={
                "atr": atr,
                "breakout_reference_high": reference_high,
                "breakout_lookback_bars": self.BREAKOUT_LOOKBACK_BARS,
            },
        )

    @staticmethod
    def _hold(reason_code: DecisionReason, reason: str) -> StrategyDecision:
        return StrategyDecision(StrategyAction.HOLD, reason_code, reason)
