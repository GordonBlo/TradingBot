"""Scheduled V8-H0 executor for funding-filtered frozen V6 candidates."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping

from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext
from src.strategy.models import (
    DecisionReason,
    StrategyAction,
    StrategyDecision,
    TrendMomentumConfig,
)
from src.strategy.v6_mtf_continuation import V6MTFContinuationStrategy


class V8FundingScheduleStrategy(BaseStrategy):
    """Execute an immutable subset schedule without reevaluating V6 entries."""

    NAME = "V8FundingScheduleStrategy"
    VERSION = "8.0"
    V6_H0_RUN_ID = "e1eef7bdd0c37ad4"
    DERIVATIVES_DATASET_ID = "eec764735d270f9d"
    REWARD_RISK_RATIO = V6MTFContinuationStrategy.REWARD_RISK_RATIO
    MIN_STOP_DISTANCE_FRACTION = (
        V6MTFContinuationStrategy.MIN_STOP_DISTANCE_FRACTION
    )
    MAXIMUM_HOLD_BARS = V6MTFContinuationStrategy.MAXIMUM_HOLD_BARS
    COOLDOWN_BARS = V6MTFContinuationStrategy.COOLDOWN_BARS

    def __init__(
        self,
        config: TrendMomentumConfig,
        *,
        retained_atr_by_signal_close: Mapping[datetime, Decimal],
    ) -> None:
        schedule: dict[datetime, Decimal] = {}
        for timestamp, atr_distance in retained_atr_by_signal_close.items():
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("V8-H0 schedule timestamps must be timezone-aware.")
            normalized = timestamp.astimezone(timezone.utc)
            atr = Decimal(str(atr_distance))
            if atr <= 0:
                raise ValueError("V8-H0 frozen V6 ATR distance must be positive.")
            if normalized in schedule:
                raise ValueError("V8-H0 frozen schedule contains a duplicate timestamp.")
            schedule[normalized] = atr
        self.config = config
        self._schedule: Mapping[datetime, Decimal] = MappingProxyType(schedule)
        self._emitted: set[datetime] = set()
        self._conflicts: set[datetime] = set()

    @property
    def required_history_bars(self) -> int:
        return 0

    @property
    def emitted_candidate_timestamps(self) -> frozenset[datetime]:
        return frozenset(self._emitted)

    @property
    def scheduled_candidate_conflict_timestamps(self) -> frozenset[datetime]:
        return frozenset(self._conflicts)

    @staticmethod
    def _hold(reason_code: DecisionReason, reason: str) -> StrategyDecision:
        return StrategyDecision(StrategyAction.HOLD, reason_code, reason)

    def evaluate(self, context: StrategyContext) -> StrategyDecision:
        signal_close = context.timestamp.astimezone(timezone.utc)
        is_scheduled = signal_close in self._schedule

        if context.has_position:
            if is_scheduled:
                self._conflicts.add(signal_close)
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
            if is_scheduled:
                self._conflicts.add(signal_close)
            return self._hold(
                DecisionReason.COOLDOWN,
                f"Cooldown active ({context.bars_since_exit}/{self.COOLDOWN_BARS} bars)",
            )

        if not is_scheduled:
            return self._hold(
                DecisionReason.NO_ENTRY,
                "Timestamp is absent from the funding-retained frozen V6 schedule",
            )

        atr_distance = self._schedule[signal_close]
        risk_budget = (
            context.equity * self.config.risk_per_trade_percent / Decimal("100")
        )
        affordable = context.cash_usdc / (Decimal("1") + context.entry_fee_rate)
        max_quote = min(affordable, self.config.maximum_position_notional_usdc)
        if risk_budget <= 0 or max_quote <= 0:
            self._conflicts.add(signal_close)
            return self._hold(
                DecisionReason.INSUFFICIENT_CAPITAL,
                "No positive frozen-V6 risk budget or affordable Spot notional",
            )

        self._emitted.add(signal_close)
        return StrategyDecision(
            StrategyAction.ENTER_LONG,
            DecisionReason.MTF_CONTINUATION_ENTRY,
            "Frozen V6 candidate retained by non-positive causal funding",
            risk_budget=risk_budget,
            stop_distance=atr_distance,
            reward_risk_ratio=self.REWARD_RISK_RATIO,
            max_quote_amount=max_quote,
            minimum_stop_distance_fraction=self.MIN_STOP_DISTANCE_FRACTION,
            metadata={
                "frozen_v6_atr_14_4h": atr_distance,
                "source_schedule": "FROZEN_V6_H0",
            },
        )
