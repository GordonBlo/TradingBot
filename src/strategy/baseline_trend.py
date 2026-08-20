"""Fixed-parameter trend/momentum baseline for workflow validation."""

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


class TrendMomentumBaselineStrategy(BaseStrategy):
    """One deliberately simple, unoptimized long-only research baseline."""

    NAME = "TrendMomentumBaselineStrategy"
    VERSION = "3.0"

    def __init__(self, config: TrendMomentumConfig) -> None:
        self.config = config

    def evaluate(self, context: StrategyContext) -> StrategyDecision:
        current = context.indicators
        previous = context.previous_indicators

        if not current.is_ready:
            return self._hold(DecisionReason.WARMUP, "Required indicators unavailable")

        assert current.ema_fast is not None
        assert current.ema_slow is not None
        assert current.rsi is not None
        assert current.atr is not None
        assert current.volume_ratio is not None

        if context.has_position:
            if current.ema_fast < current.ema_slow:
                return StrategyDecision(
                    StrategyAction.EXIT_LONG,
                    DecisionReason.TREND_EXIT,
                    (
                        f"Trend invalidated: EMA{self.config.fast_ema_period} "
                        f"{current.ema_fast} < EMA{self.config.slow_ema_period} "
                        f"{current.ema_slow}"
                    ),
                )
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

        if (
            previous is None
            or previous.ema_fast is None
            or previous.ema_slow is None
        ):
            return self._hold(DecisionReason.WARMUP, "Previous EMA values unavailable")

        crossover = (
            previous.ema_fast <= previous.ema_slow
            and current.ema_fast > current.ema_slow
        )
        eligible = (
            crossover
            and current.close > current.ema_slow
            and self.config.rsi_min <= current.rsi <= self.config.rsi_max
            and current.volume_ratio >= self.config.minimum_volume_ratio
        )
        if not eligible:
            return self._hold(DecisionReason.NO_ENTRY, "Entry conditions not satisfied")

        stop_distance = current.atr * self.config.atr_stop_multiplier
        risk_budget = (
            context.equity * self.config.risk_per_trade_percent / Decimal("100")
        )
        affordable = context.cash_usdc / (
            Decimal("1") + context.entry_fee_rate
        )
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
            DecisionReason.EMA_CROSSOVER_ENTRY,
            (
                f"EMA{self.config.fast_ema_period} crossed above "
                f"EMA{self.config.slow_ema_period}; RSI={current.rsi}; "
                f"VolumeRatio={current.volume_ratio}"
            ),
            risk_budget=risk_budget,
            stop_distance=stop_distance,
            reward_risk_ratio=self.config.reward_risk_ratio,
            max_quote_amount=max_quote,
            metadata={
                "ema_fast": current.ema_fast,
                "ema_slow": current.ema_slow,
                "rsi": current.rsi,
                "atr": current.atr,
                "volume_ratio": current.volume_ratio,
            },
        )

    @staticmethod
    def _hold(reason_code: DecisionReason, reason: str) -> StrategyDecision:
        return StrategyDecision(StrategyAction.HOLD, reason_code, reason)

