"""Causal strategy-to-backtest adapter and signal journal."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal

from src.analysis.indicators import IndicatorEngine
from src.backtest.engine import (
    BacktestEngine,
    PositionExitProvider,
    StopUpdateProvider,
)
from src.backtest.execution import basis_points_rate
from src.backtest.models import (
    BacktestConfig,
    BacktestContext,
    BacktestResult,
    ExitReason,
    OrderAction,
    OrderIntent,
)
from src.historical.dataset import HistoricalDataset
from src.market.intervals import next_open_time
from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext
from src.strategy.models import DecisionReason, StrategyAction, TrendMomentumConfig


@dataclass(frozen=True, slots=True)
class SignalRecord:
    timestamp: datetime
    action: StrategyAction
    close: Decimal
    ema_fast: Decimal | None
    ema_slow: Decimal | None
    rsi: Decimal | None
    atr: Decimal | None
    volume_ratio: Decimal | None
    position_state: str
    reason_code: DecisionReason
    reason: str


class StrategyDecisionAdapter:
    """Translate typed strategy decisions into V2 simulated intents only."""

    def __init__(
        self,
        *,
        dataset: HistoricalDataset,
        strategy: BaseStrategy,
        strategy_config: TrendMomentumConfig,
        backtest_config: BacktestConfig,
        minimum_action_index: int = 0,
    ) -> None:
        if not 0 <= minimum_action_index < len(dataset.candles):
            raise ValueError("Minimum strategy action index is outside the dataset.")
        self._dataset = dataset
        self._strategy = strategy
        self._minimum_action_index = minimum_action_index
        self._fee_rate = basis_points_rate(backtest_config.fee_bps)
        self._requires_full_history = strategy.requires_full_history
        if strategy.required_history_bars < 0:
            raise ValueError("Strategy history requirement cannot be negative.")
        self._history_limit = max(
            strategy_config.fast_ema_period,
            strategy_config.slow_ema_period,
            strategy_config.rsi_period + 1,
            strategy_config.atr_period,
            strategy_config.volume_sma_period,
            strategy.required_history_bars,
        ) + 1
        self._indicators = IndicatorEngine().calculate_research_series(
            dataset.candles,
            fast_ema_period=strategy_config.fast_ema_period,
            slow_ema_period=strategy_config.slow_ema_period,
            rsi_period=strategy_config.rsi_period,
            atr_period=strategy_config.atr_period,
            volume_sma_period=strategy_config.volume_sma_period,
        )
        self._last_completed_count = 0
        self._last_exit_index: int | None = None
        self.signals: list[SignalRecord] = []

    def __call__(self, context: BacktestContext) -> OrderIntent | None:
        if context.account.completed_trade_count > self._last_completed_count:
            self._last_exit_index = context.index
            self._last_completed_count = context.account.completed_trade_count
        bars_since_exit = (
            context.index - self._last_exit_index
            if self._last_exit_index is not None
            else None
        )
        current = self._indicators[context.index]
        previous = self._indicators[context.index - 1] if context.index > 0 else None
        history_start = (
            0
            if self._requires_full_history
            else max(0, len(context.history) - self._history_limit)
        )
        recent_history = context.history[history_start:]
        strategy_context = StrategyContext(
            timestamp=next_open_time(
                context.candle.timestamp, self._dataset.interval
            ),
            current_candle=context.candle,
            recent_history=recent_history,
            indicators=current,
            previous_indicators=previous,
            has_position=context.account.has_position,
            bars_in_position=context.account.bars_in_position,
            equity=context.account.equity,
            cash_usdc=context.account.cash_usdc,
            completed_trade_count=context.account.completed_trade_count,
            bars_since_exit=bars_since_exit,
            entry_fee_rate=self._fee_rate,
        )
        decision = self._strategy.evaluate(strategy_context)
        if context.index < self._minimum_action_index:
            return None
        if decision.action is StrategyAction.HOLD:
            return None

        self.signals.append(
            SignalRecord(
                timestamp=strategy_context.timestamp,
                action=decision.action,
                close=current.close,
                ema_fast=current.ema_fast,
                ema_slow=current.ema_slow,
                rsi=current.rsi,
                atr=current.atr,
                volume_ratio=current.volume_ratio,
                position_state=("LONG" if context.account.has_position else "FLAT"),
                reason_code=decision.reason_code,
                reason=decision.reason,
            )
        )
        if decision.action is StrategyAction.ENTER_LONG:
            return OrderIntent(
                action=OrderAction.BUY,
                risk_budget=decision.risk_budget,
                stop_distance=decision.stop_distance,
                reward_risk_ratio=decision.reward_risk_ratio,
                max_quote_amount=decision.max_quote_amount,
                minimum_stop_distance_fraction=(
                    decision.minimum_stop_distance_fraction
                ),
                reason=decision.reason,
            )
        exit_reason = (
            ExitReason.TIME_EXIT
            if decision.reason_code is DecisionReason.TIME_EXIT
            else ExitReason.TREND_EXIT
        )
        return OrderIntent(
            action=OrderAction.SELL,
            reason=decision.reason,
            exit_reason=exit_reason,
        )


@dataclass(frozen=True, slots=True)
class StrategyPeriodEvaluation:
    backtest: BacktestResult
    signals: tuple[SignalRecord, ...]


def evaluate_strategy_period(
    dataset: HistoricalDataset,
    *,
    strategy: BaseStrategy,
    strategy_config: TrendMomentumConfig,
    backtest_config: BacktestConfig,
    evaluation_start_index: int = 0,
    stop_updates : StopUpdateProvider | None = None,
    position_exits: PositionExitProvider | None = None,
) -> StrategyPeriodEvaluation:
    adapter = StrategyDecisionAdapter(
        dataset=dataset,
        strategy=strategy,
        strategy_config=strategy_config,
        backtest_config=backtest_config,
        minimum_action_index=evaluation_start_index,
    )
    result = BacktestEngine(backtest_config).run(
        dataset,
        adapter,
        stop_updates=stop_updates,
        position_exits=position_exits
        )
    if evaluation_start_index:
        evaluation_dataset = HistoricalDataset(
            symbol=dataset.symbol,
            interval=dataset.interval,
            source=dataset.source,
            candles=dataset.candles[evaluation_start_index:],
        )
        evaluation_count = len(evaluation_dataset.candles)
        exposure = (
            result.metrics.market_exposure_percent
            * Decimal(len(dataset.candles))
            / Decimal(evaluation_count)
        )
        result = replace(
            result,
            dataset=evaluation_dataset,
            equity_curve=result.equity_curve[evaluation_start_index:],
            metrics=replace(result.metrics, market_exposure_percent=exposure),
        )
    return StrategyPeriodEvaluation(result, tuple(adapter.signals))
