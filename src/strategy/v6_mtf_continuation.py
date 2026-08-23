"""Frozen V6-H0 cost-aware multi-timeframe continuation strategy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping

from src.analysis.indicators import IndicatorEngine, atr, ema
from src.models.candle import Candle
from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext
from src.strategy.models import (
    DecisionReason,
    StrategyAction,
    StrategyDecision,
    TrendMomentumConfig,
)


SOURCE_INTERVAL = "15m"
HIGHER_TIMEFRAME_INTERVAL = "4h"
SOURCE_BARS_PER_4H_CANDLE = 16
SOURCE_BAR_DURATION = timedelta(minutes=15)


@dataclass(frozen=True, slots=True)
class HigherTimeframeState:
    latest_candle: Candle
    ema_20: Decimal
    ema_50: Decimal
    atr_14: Decimal

    @property
    def long_regime_active(self) -> bool:
        return (
            self.ema_20 > self.ema_50
            and self.latest_candle.close > self.ema_20
        )


@dataclass(frozen=True, slots=True)
class V6PreparedContext:
    """Immutable causal 4h state lookup for one 15m evaluation stream."""

    completed_4h_candles: tuple[Candle, ...]
    states_by_timestamp: Mapping[datetime, HigherTimeframeState | None]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "states_by_timestamp",
            MappingProxyType(dict(self.states_by_timestamp)),
        )

    def state_at(self, timestamp: datetime) -> HigherTimeframeState | None:
        try:
            return self.states_by_timestamp[timestamp]
        except KeyError as exc:
            raise ValueError(
                "V6 prepared context does not contain the signal candle."
            ) from exc


def _utc_timestamp(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("4h aggregation timestamps must be timezone-aware.")
    return timestamp.astimezone(timezone.utc)


def _is_utc_4h_boundary(timestamp: datetime) -> bool:
    utc_timestamp = _utc_timestamp(timestamp)
    return (
        utc_timestamp.hour % 4 == 0
        and utc_timestamp.minute == 0
        and utc_timestamp.second == 0
        and utc_timestamp.microsecond == 0
    )


def aggregate_completed_4h_candles(
    candles: Sequence[Candle],
    *,
    as_of: datetime,
) -> tuple[Candle, ...]:
    """Aggregate only fully known, UTC-aligned 16-bar 4h candles."""

    as_of_utc = _utc_timestamp(as_of)
    source = tuple(
        candle
        for candle in candles
        if _utc_timestamp(candle.timestamp) <= as_of_utc
    )
    output: list[Candle] = []

    for index, first in enumerate(source):
        if first.interval != SOURCE_INTERVAL or not _is_utc_4h_boundary(
            first.timestamp
        ):
            continue
        group = source[index : index + SOURCE_BARS_PER_4H_CANDLE]
        if len(group) != SOURCE_BARS_PER_4H_CANDLE:
            continue
        expected_timestamps = tuple(
            first.timestamp + SOURCE_BAR_DURATION * offset
            for offset in range(SOURCE_BARS_PER_4H_CANDLE)
        )
        if any(
            candle.timestamp != expected_timestamp
            or candle.interval != SOURCE_INTERVAL
            or candle.symbol != first.symbol
            or not candle.is_closed
            for candle, expected_timestamp in zip(
                group, expected_timestamps, strict=True
            )
        ):
            continue
        output.append(
            Candle(
                timestamp=first.timestamp,
                symbol=first.symbol,
                interval=HIGHER_TIMEFRAME_INTERVAL,
                open=first.open,
                high=max(candle.high for candle in group),
                low=min(candle.low for candle in group),
                close=group[-1].close,
                volume=sum(
                    (candle.volume for candle in group),
                    start=Decimal("0"),
                ),
                is_closed=True,
            )
        )
    return tuple(output)


def higher_timeframe_state(
    candles: Sequence[Candle],
    *,
    as_of: datetime,
) -> HigherTimeframeState | None:
    """Return the latest causal V6 4h state, if all indicators are ready."""

    completed = aggregate_completed_4h_candles(candles, as_of=as_of)
    ema_20 = ema(
        tuple(candle.close for candle in completed),
        V6MTFContinuationStrategy.HIGHER_TIMEFRAME_FAST_EMA_PERIOD,
    )
    ema_50 = ema(
        tuple(candle.close for candle in completed),
        V6MTFContinuationStrategy.HIGHER_TIMEFRAME_SLOW_EMA_PERIOD,
    )
    atr_14 = atr(completed, V6MTFContinuationStrategy.ATR_PERIOD_4H)
    if ema_20 is None or ema_50 is None or atr_14 is None:
        return None
    return HigherTimeframeState(
        latest_candle=completed[-1],
        ema_20=ema_20,
        ema_50=ema_50,
        atr_14=atr_14,
    )


def prepare_v6_context(candles: Sequence[Candle]) -> V6PreparedContext:
    """Build exact causal completed-4h indicator snapshots in O(n)."""

    source = tuple(candles)
    if not source:
        return V6PreparedContext((), {})
    completed = aggregate_completed_4h_candles(
        source,
        as_of=source[-1].timestamp,
    )
    indicators = IndicatorEngine().calculate_research_series(
        completed,
        fast_ema_period=V6MTFContinuationStrategy.HIGHER_TIMEFRAME_FAST_EMA_PERIOD,
        slow_ema_period=V6MTFContinuationStrategy.HIGHER_TIMEFRAME_SLOW_EMA_PERIOD,
        atr_period=V6MTFContinuationStrategy.ATR_PERIOD_4H,
    )
    ready_states: dict[datetime, HigherTimeframeState | None] = {}
    for candle, snapshot in zip(completed, indicators, strict=True):
        completion_timestamp = candle.timestamp + (
            SOURCE_BAR_DURATION * (SOURCE_BARS_PER_4H_CANDLE - 1)
        )
        ready_states[completion_timestamp] = (
            HigherTimeframeState(
                latest_candle=candle,
                ema_20=snapshot.ema_fast,
                ema_50=snapshot.ema_slow,
                atr_14=snapshot.atr,
            )
            if snapshot.ema_fast is not None
            and snapshot.ema_slow is not None
            and snapshot.atr is not None
            else None
        )

    by_timestamp: dict[datetime, HigherTimeframeState | None] = {}
    latest: HigherTimeframeState | None = None
    for candle in source:
        if candle.timestamp in ready_states:
            latest = ready_states[candle.timestamp]
        by_timestamp[candle.timestamp] = latest
    return V6PreparedContext(completed, by_timestamp)


class V6MTFContinuationStrategy(BaseStrategy):
    """Buy a 15m EMA reclaim only inside a completed-4h uptrend."""

    NAME = "V6MTFContinuationStrategy"
    VERSION = "6.0"
    EXECUTION_INTERVAL = SOURCE_INTERVAL
    HIGHER_TIMEFRAME_CONTEXT_INTERVAL = HIGHER_TIMEFRAME_INTERVAL
    SOURCE_BARS_PER_4H_CANDLE = SOURCE_BARS_PER_4H_CANDLE
    HIGHER_TIMEFRAME_FAST_EMA_PERIOD = 20
    HIGHER_TIMEFRAME_SLOW_EMA_PERIOD = 50
    PULLBACK_EMA_PERIOD = 20
    ATR_PERIOD_4H = 14
    BASE_FEE_BPS_PER_SIDE = Decimal("10")
    BASE_SLIPPAGE_BPS_PER_SIDE = Decimal("2")
    ROUND_TRIP_FRICTION_BPS = Decimal("24")
    COST_DISTANCE_MULTIPLE = 4
    MIN_STOP_DISTANCE_BPS = Decimal("96")
    MIN_STOP_DISTANCE_FRACTION = Decimal("0.0096")
    REWARD_RISK_RATIO = Decimal("2")
    MAXIMUM_HOLD_BARS = 96
    COOLDOWN_BARS = 4
    REQUIRED_HISTORY_BARS = (
        HIGHER_TIMEFRAME_SLOW_EMA_PERIOD * SOURCE_BARS_PER_4H_CANDLE
        + SOURCE_BARS_PER_4H_CANDLE
    )

    def __init__(
        self,
        config: TrendMomentumConfig,
        *,
        prepared_context: V6PreparedContext | None = None,
    ) -> None:
        frozen = {
            "fast_ema_period": self.PULLBACK_EMA_PERIOD,
            "atr_period": self.ATR_PERIOD_4H,
            "reward_risk_ratio": self.REWARD_RISK_RATIO,
            "maximum_bars_in_position": self.MAXIMUM_HOLD_BARS,
            "cooldown_bars": self.COOLDOWN_BARS,
        }
        for name, expected in frozen.items():
            if getattr(config, name) != expected:
                raise ValueError(f"V6-H0 {name} is frozen at {expected}.")
        self.config = config
        self._prepared_context = prepared_context

    @property
    def required_history_bars(self) -> int:
        return self.REQUIRED_HISTORY_BARS

    @property
    def requires_full_history(self) -> bool:
        return True

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

        history = tuple(context.recent_history)
        if len(history) < self.PULLBACK_EMA_PERIOD + 1 or not all(
            candle.is_closed
            for candle in history[-(self.PULLBACK_EMA_PERIOD + 1) :]
        ):
            return self._hold(
                DecisionReason.WARMUP,
                "Closed 15m EMA20 history unavailable",
            )

        state = (
            self._prepared_context.state_at(context.current_candle.timestamp)
            if self._prepared_context is not None
            else higher_timeframe_state(
                history,
                as_of=context.current_candle.timestamp,
            )
        )
        if state is None:
            return self._hold(
                DecisionReason.WARMUP,
                "Completed 4h EMA50 or ATR14 history unavailable",
            )
        if not state.long_regime_active:
            return self._hold(
                DecisionReason.NO_ENTRY,
                "Completed 4h long regime is inactive",
            )

        previous = history[-2]
        previous_indicators = context.previous_indicators
        previous_ema = (
            previous_indicators.ema_fast
            if previous_indicators is not None
            and previous_indicators.timestamp == previous.timestamp
            else None
        )
        current_ema = context.indicators.ema_fast
        if previous_ema is None or current_ema is None:
            return self._hold(DecisionReason.WARMUP, "15m EMA20 unavailable")

        current = context.current_candle
        if not (
            previous.close <= previous_ema
            and current.close > current_ema
            and current.close > previous.high
        ):
            return self._hold(
                DecisionReason.NO_ENTRY,
                "15m pullback reclaim continuation conditions not satisfied",
            )

        risk_budget = (
            context.equity * self.config.risk_per_trade_percent / Decimal("100")
        )
        affordable = context.cash_usdc / (Decimal("1") + context.entry_fee_rate)
        max_quote = min(affordable, self.config.maximum_position_notional_usdc)
        if risk_budget <= 0 or state.atr_14 <= 0 or max_quote <= 0:
            return self._hold(
                DecisionReason.INSUFFICIENT_CAPITAL,
                "No positive risk budget, 4h ATR, or affordable Spot notional",
            )

        signal_cost_floor = current.close * self.MIN_STOP_DISTANCE_FRACTION
        return StrategyDecision(
            StrategyAction.ENTER_LONG,
            DecisionReason.MTF_CONTINUATION_ENTRY,
            "Completed-4h uptrend and strict 15m pullback continuation confirmed",
            risk_budget=risk_budget,
            stop_distance=state.atr_14,
            reward_risk_ratio=self.REWARD_RISK_RATIO,
            max_quote_amount=max_quote,
            minimum_stop_distance_fraction=self.MIN_STOP_DISTANCE_FRACTION,
            metadata={
                "atr_4h": state.atr_14,
                "higher_timeframe_ema_20": state.ema_20,
                "higher_timeframe_ema_50": state.ema_50,
                "higher_timeframe_close": state.latest_candle.close,
                "higher_timeframe_candle_timestamp": (
                    state.latest_candle.timestamp.isoformat()
                ),
                "previous_ema_20_15m": previous_ema,
                "current_ema_20_15m": current_ema,
                "signal_cost_floor_distance": signal_cost_floor,
                "minimum_stop_distance_fraction": (
                    self.MIN_STOP_DISTANCE_FRACTION
                ),
                "cost_distance_multiple": self.COST_DISTANCE_MULTIPLE,
                "minimum_stop_distance_bps": self.MIN_STOP_DISTANCE_BPS,
            },
        )

    @staticmethod
    def _hold(reason_code: DecisionReason, reason: str) -> StrategyDecision:
        return StrategyDecision(StrategyAction.HOLD, reason_code, reason)
