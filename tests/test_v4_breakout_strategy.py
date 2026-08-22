from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.analysis.indicators import atr
from src.backtest.models import BacktestConfig, ExitReason
from src.historical.dataset import HistoricalDataset
from src.models.candle import Candle
from src.models.market_data_source import MarketDataSource
from src.models.research_indicator_snapshot import ResearchIndicatorSnapshot
from src.research.evaluation import evaluate_strategy_period
from src.strategy.context import StrategyContext
from src.strategy.models import (
    DecisionReason,
    StrategyAction,
    StrategyDecision,
    TrendMomentumConfig,
)
from src.strategy.v4_breakout import V4BreakoutStrategy


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def candle(
    index: int,
    *,
    close: str = "99",
    open: str = "99",
    high: str = "100",
    low: str = "98",
    volume: str = "10",
) -> Candle:
    return Candle(
        timestamp=BASE + timedelta(minutes=15 * index),
        symbol="BTCUSDC",
        interval="15m",
        open=Decimal(open),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal(volume),
        is_closed=True,
    )


def context(
    history: tuple[Candle, ...],
    *,
    atr_value: str | None = "5",
    bars_since_exit: int | None = None,
    has_position: bool = False,
    bars_in_position: int = 0,
    old_indicators_ready: bool = True,
) -> StrategyContext:
    current = history[-1]
    indicators = ResearchIndicatorSnapshot(
        timestamp=current.timestamp,
        symbol=current.symbol,
        interval=current.interval,
        close=current.close,
        volume=current.volume,
        ema_fast=Decimal("105") if old_indicators_ready else None,
        ema_slow=Decimal("100") if old_indicators_ready else None,
        rsi=Decimal("58") if old_indicators_ready else None,
        atr=Decimal(atr_value) if atr_value is not None else None,
        volume_sma=Decimal("10") if old_indicators_ready else None,
        volume_ratio=Decimal("1") if old_indicators_ready else None,
    )
    return StrategyContext(
        timestamp=current.timestamp + timedelta(minutes=15),
        current_candle=current,
        recent_history=history,
        indicators=indicators,
        previous_indicators=None,
        has_position=has_position,
        bars_in_position=bars_in_position,
        equity=Decimal("1000"),
        cash_usdc=Decimal("1000"),
        completed_trade_count=1 if bars_since_exit is not None else 0,
        bars_since_exit=bars_since_exit,
        entry_fee_rate=Decimal("0"),
    )


def reference_history(
    *, current_close: str, current_high: str = "105"
) -> tuple[Candle, ...]:
    previous = tuple(candle(index) for index in range(20))
    return previous + (
        candle(20, close=current_close, high=current_high),
    )


def evaluate(
    history: tuple[Candle, ...], **kwargs: object
) -> StrategyDecision:
    return V4BreakoutStrategy(TrendMomentumConfig()).evaluate(
        context(history, **kwargs)
    )


def test_insufficient_history_has_no_buy() -> None:
    history = tuple(candle(index) for index in range(19)) + (
        candle(19, close="101", high="105"),
    )

    decision = evaluate(history)

    assert decision.action is StrategyAction.HOLD
    assert decision.reason_code is DecisionReason.WARMUP


def test_current_candle_is_excluded_from_breakout_reference() -> None:
    decision = evaluate(reference_history(current_close="101", current_high="150"))

    assert decision.action is StrategyAction.ENTER_LONG
    assert decision.metadata["breakout_reference_high"] == Decimal("100")


def test_equal_close_is_not_a_breakout() -> None:
    decision = evaluate(reference_history(current_close="100"))

    assert decision.action is StrategyAction.HOLD


def test_strictly_higher_close_is_a_buy_signal() -> None:
    decision = evaluate(reference_history(current_close="101"))

    assert decision.action is StrategyAction.ENTER_LONG
    assert decision.reason_code is DecisionReason.PRICE_BREAKOUT_ENTRY


def test_breakout_uses_exactly_the_previous_twenty_highs() -> None:
    old_outlier = candle(0, high="150")
    previous_twenty = tuple(candle(index) for index in range(1, 21))
    current = candle(21, close="101", high="105")

    decision = evaluate((old_outlier, *previous_twenty, current))

    assert decision.action is StrategyAction.ENTER_LONG
    assert decision.metadata["breakout_reference_high"] == Decimal("100")


def test_atr14_stop_distance_and_two_r_target_are_frozen() -> None:
    decision = evaluate(reference_history(current_close="101"), atr_value="5")

    assert decision.stop_distance == Decimal("10.0")
    assert decision.reward_risk_ratio == Decimal("2.0")


def test_entry_has_no_ema_rsi_or_volume_dependency() -> None:
    decision = evaluate(
        reference_history(current_close="101"),
        old_indicators_ready=False,
    )

    assert decision.action is StrategyAction.ENTER_LONG


@pytest.mark.parametrize("bars_since_exit", (0, 1, 2, 3, 4))
def test_four_bar_cooldown_is_preserved(bars_since_exit: int) -> None:
    decision = evaluate(
        reference_history(current_close="101"),
        bars_since_exit=bars_since_exit,
    )

    assert decision.reason_code is DecisionReason.COOLDOWN


def test_maximum_hold_is_a_time_exit_without_old_filter_logic() -> None:
    decision = evaluate(
        reference_history(current_close="99"),
        has_position=True,
        bars_in_position=96,
        old_indicators_ready=False,
        atr_value=None,
    )

    assert (decision.action, decision.reason_code) == (
        StrategyAction.EXIT_LONG,
        DecisionReason.TIME_EXIT,
    )


def execution_dataset(*, entry_low: Decimal, entry_high: Decimal) -> HistoricalDataset:
    candles = tuple(
        candle(index, close="100", open="100", high="101", low="99")
        for index in range(20)
    )
    signal = candle(20, close="102", open="100", high="103", low="99")
    entry = candle(
        21,
        close="110",
        open="110",
        high=str(entry_high),
        low=str(entry_low),
    )
    return HistoricalDataset(
        symbol="BTCUSDC",
        interval="15m",
        source=MarketDataSource.BINANCE_PUBLIC,
        candles=(*candles, signal, entry),
    )


def run_v4(data: HistoricalDataset):
    strategy_config = TrendMomentumConfig()
    return evaluate_strategy_period(
        data,
        strategy=V4BreakoutStrategy(strategy_config),
        strategy_config=strategy_config,
        backtest_config=BacktestConfig(
            fee_bps=Decimal("0"),
            slippage_bps=Decimal("0"),
        ),
    ).backtest


def test_adapter_executes_at_next_open_with_fill_relative_two_r_target() -> None:
    preliminary = execution_dataset(
        entry_low=Decimal("108"), entry_high=Decimal("130")
    )
    signal_atr = atr(preliminary.candles[:21], 14)
    assert signal_atr is not None
    target = Decimal("110") + Decimal("4") * signal_atr
    data = replace(
        preliminary,
        candles=(
            *preliminary.candles[:-1],
            candle(
                21,
                close="110",
                open="110",
                high=str(target),
                low="108",
            ),
        ),
    )

    trade = run_v4(data).trades[0]

    assert trade.entry_price == Decimal("110")
    assert trade.exit_price == target
    assert trade.exit_reason is ExitReason.TAKE_PROFIT


def test_v4_bracket_keeps_existing_stop_first_policy() -> None:
    data = execution_dataset(entry_low=Decimal("100"), entry_high=Decimal("130"))
    signal_atr = atr(data.candles[:21], 14)
    assert signal_atr is not None

    trade = run_v4(data).trades[0]

    assert trade.entry_price == Decimal("110")
    assert trade.exit_price == Decimal("110") - Decimal("2") * signal_atr
    assert trade.exit_reason is ExitReason.STOP_LOSS
