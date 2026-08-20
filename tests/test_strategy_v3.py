from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.analysis.indicators import IndicatorEngine, atr, ema, rsi, sma
from src.backtest.engine import BacktestEngine
from src.backtest.models import BacktestConfig, ExitReason, OrderAction, OrderIntent
from src.historical.dataset import HistoricalDataset
from src.models.candle import Candle
from src.models.market_data_source import MarketDataSource
from src.models.research_indicator_snapshot import ResearchIndicatorSnapshot
from src.strategy.baseline_trend import TrendMomentumBaselineStrategy
from src.strategy.context import StrategyContext
from src.strategy.models import (
    DecisionReason,
    StrategyAction,
    StrategyDecision,
    TrendMomentumConfig,
)


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def candle(
    index: int,
    close: str = "100",
    *,
    open: str | None = None,
    high: str | None = None,
    low: str | None = None,
    volume: str = "10",
) -> Candle:
    close_value = Decimal(close)
    open_value = Decimal(open) if open is not None else close_value
    return Candle(
        timestamp=BASE + timedelta(minutes=15 * index),
        symbol="BTCUSDC",
        interval="15m",
        open=open_value,
        high=Decimal(high) if high is not None else max(open_value, close_value) + 1,
        low=Decimal(low) if low is not None else min(open_value, close_value) - 1,
        close=close_value,
        volume=Decimal(volume),
        is_closed=True,
    )


def snapshot(
    index: int,
    *,
    close: str = "110",
    fast: str | None = "105",
    slow: str | None = "100",
    rsi_value: str | None = "58",
    atr_value: str | None = "5",
    volume_ratio: str | None = "1.1",
) -> ResearchIndicatorSnapshot:
    return ResearchIndicatorSnapshot(
        timestamp=BASE + timedelta(minutes=15 * index),
        symbol="BTCUSDC",
        interval="15m",
        close=Decimal(close),
        volume=Decimal("10"),
        ema_fast=Decimal(fast) if fast is not None else None,
        ema_slow=Decimal(slow) if slow is not None else None,
        rsi=Decimal(rsi_value) if rsi_value is not None else None,
        atr=Decimal(atr_value) if atr_value is not None else None,
        volume_sma=Decimal("10") if volume_ratio is not None else None,
        volume_ratio=Decimal(volume_ratio) if volume_ratio is not None else None,
    )


def context(
    *,
    current: ResearchIndicatorSnapshot | None = None,
    previous: ResearchIndicatorSnapshot | None = None,
    has_position: bool = False,
    bars_in_position: int = 0,
    bars_since_exit: int | None = None,
    equity: str = "1000",
    cash: str = "1000",
) -> StrategyContext:
    current = current or snapshot(1)
    previous = previous or snapshot(0, fast="99", slow="100")
    current_candle = candle(1, str(current.close))
    return StrategyContext(
        timestamp=current_candle.timestamp + timedelta(minutes=15),
        current_candle=current_candle,
        recent_history=(candle(0), current_candle),
        indicators=current,
        previous_indicators=previous,
        has_position=has_position,
        bars_in_position=bars_in_position,
        equity=Decimal(equity),
        cash_usdc=Decimal(cash),
        completed_trade_count=1 if bars_since_exit is not None else 0,
        bars_since_exit=bars_since_exit,
        entry_fee_rate=Decimal("0.001"),
    )


def test_strategy_contract_accepts_three_actions_and_rejects_invalid_action() -> None:
    enter = StrategyDecision(
        StrategyAction.ENTER_LONG,
        DecisionReason.EMA_CROSSOVER_ENTRY,
        "entry",
        risk_budget=Decimal("5"),
        stop_distance=Decimal("10"),
        reward_risk_ratio=Decimal("2"),
        max_quote_amount=Decimal("50"),
    )
    exit_decision = StrategyDecision(
        StrategyAction.EXIT_LONG, DecisionReason.TREND_EXIT, "exit"
    )
    hold = StrategyDecision(StrategyAction.HOLD, DecisionReason.NO_ENTRY, "hold")

    assert (enter.action, exit_decision.action, hold.action) == (
        StrategyAction.ENTER_LONG,
        StrategyAction.EXIT_LONG,
        StrategyAction.HOLD,
    )
    with pytest.raises(ValueError):
        StrategyDecision("SHORT", DecisionReason.NO_ENTRY, "invalid")  # type: ignore[arg-type]


def test_context_rejects_future_history() -> None:
    current = candle(1)
    with pytest.raises(ValueError, match="future"):
        StrategyContext(
            timestamp=current.timestamp + timedelta(minutes=15),
            current_candle=current,
            recent_history=(current, candle(2)),
            indicators=snapshot(1),
            previous_indicators=snapshot(0),
            has_position=False,
            bars_in_position=0,
            equity=Decimal("1000"),
            cash_usdc=Decimal("1000"),
            completed_trade_count=0,
            bars_since_exit=None,
        )


def test_ema_crossover_entry_has_fixed_atr_risk_and_cash_cap() -> None:
    decision = TrendMomentumBaselineStrategy(TrendMomentumConfig()).evaluate(context())

    assert decision.action is StrategyAction.ENTER_LONG
    assert decision.reason_code is DecisionReason.EMA_CROSSOVER_ENTRY
    assert decision.risk_budget == Decimal("5")
    assert decision.stop_distance == Decimal("10.0")
    assert decision.reward_risk_ratio == Decimal("2.0")
    assert decision.max_quote_amount == Decimal("50")


def test_persistent_bullish_ema_state_does_not_reenter() -> None:
    decision = TrendMomentumBaselineStrategy(TrendMomentumConfig()).evaluate(
        context(previous=snapshot(0, fast="101", slow="100"))
    )
    assert decision.action is StrategyAction.HOLD


@pytest.mark.parametrize(
    "current",
    (
        snapshot(1, close="100"),
        snapshot(1, rsi_value="51.9"),
        snapshot(1, rsi_value="68.1"),
        snapshot(1, volume_ratio="0.79"),
    ),
)
def test_trend_rsi_and_volume_filters_prevent_entry(
    current: ResearchIndicatorSnapshot,
) -> None:
    decision = TrendMomentumBaselineStrategy(TrendMomentumConfig()).evaluate(
        context(current=current)
    )
    assert decision.action is StrategyAction.HOLD


def test_missing_required_indicator_holds_without_zero_substitution() -> None:
    decision = TrendMomentumBaselineStrategy(TrendMomentumConfig()).evaluate(
        context(current=snapshot(1, atr_value=None))
    )
    assert decision.action is StrategyAction.HOLD
    assert decision.reason_code is DecisionReason.WARMUP


@pytest.mark.parametrize("bars_since_exit", (0, 1, 2, 3, 4))
def test_four_bar_cooldown_blocks_entry(bars_since_exit: int) -> None:
    decision = TrendMomentumBaselineStrategy(TrendMomentumConfig()).evaluate(
        context(bars_since_exit=bars_since_exit)
    )
    assert decision.action is StrategyAction.HOLD
    assert decision.reason_code is DecisionReason.COOLDOWN


def test_entry_eligibility_returns_after_cooldown() -> None:
    decision = TrendMomentumBaselineStrategy(TrendMomentumConfig()).evaluate(
        context(bars_since_exit=5)
    )
    assert decision.action is StrategyAction.ENTER_LONG


def test_trend_and_time_exits_have_structured_reasons() -> None:
    strategy = TrendMomentumBaselineStrategy(TrendMomentumConfig())
    trend = strategy.evaluate(
        context(
            current=snapshot(1, fast="99", slow="100"),
            has_position=True,
            bars_in_position=3,
            cash="900",
        )
    )
    timed = strategy.evaluate(
        context(has_position=True, bars_in_position=96, cash="900")
    )

    assert (trend.action, trend.reason_code) == (
        StrategyAction.EXIT_LONG,
        DecisionReason.TREND_EXIT,
    )
    assert (timed.action, timed.reason_code) == (
        StrategyAction.EXIT_LONG,
        DecisionReason.TIME_EXIT,
    )


def _dataset(*candles: Candle) -> HistoricalDataset:
    return HistoricalDataset(
        "BTCUSDC", "15m", MarketDataSource.BINANCE_PUBLIC, tuple(candles)
    )


def test_fill_time_atr_stop_and_two_r_target_use_actual_next_open() -> None:
    data = _dataset(
        candle(0, "100", high="101", low="99"),
        candle(1, "120", open="110", high="130", low="101"),
    )
    intent = OrderIntent(
        OrderAction.BUY,
        risk_budget=Decimal("100"),
        stop_distance=Decimal("10"),
        reward_risk_ratio=Decimal("2"),
        max_quote_amount=Decimal("2000"),
        reason="risk entry",
    )
    result = BacktestEngine(
        BacktestConfig(
            initial_capital_usdc=Decimal("2000"),
            fee_bps=Decimal("0"),
            slippage_bps=Decimal("0"),
        )
    ).run(data, lambda ctx: intent if ctx.index == 0 else None)

    trade = result.trades[0]
    assert trade.entry_price == Decimal("110")
    assert trade.exit_price == Decimal("130")
    assert trade.exit_reason is ExitReason.TAKE_PROFIT
    assert trade.quantity == Decimal("10")


def test_fee_aware_risk_sizing_never_creates_negative_cash() -> None:
    data = _dataset(candle(0), candle(1), candle(2))
    intent = OrderIntent(
        OrderAction.BUY,
        risk_budget=Decimal("1000"),
        stop_distance=Decimal("1"),
        reward_risk_ratio=Decimal("2"),
        max_quote_amount=Decimal("1000"),
        reason="cash capped",
    )
    result = BacktestEngine(BacktestConfig()).run(
        data, lambda ctx: intent if ctx.index == 0 else None
    )

    trade = result.trades[0]
    assert trade.entry_notional + trade.entry_fee <= Decimal("1000")
    assert all(point.cash >= 0 for point in result.equity_curve)


def test_incremental_research_indicators_match_existing_math_at_every_prefix() -> None:
    candles = tuple(
        candle(
            index,
            str(Decimal("100") + Decimal((index * 7) % 13) - Decimal("6")),
            volume=str(Decimal("10") + index % 5),
        )
        for index in range(65)
    )
    series = IndicatorEngine().calculate_research_series(candles)

    for index, current in enumerate(series):
        prefix = candles[: index + 1]
        closes = [item.close for item in prefix]
        volumes = [item.volume for item in prefix]
        assert current.ema_fast == ema(closes, 20)
        assert current.ema_slow == ema(closes, 50)
        assert current.rsi == rsi(closes, 14)
        assert current.atr == atr(prefix, 14)
        assert current.volume_sma == sma(volumes, 20)
