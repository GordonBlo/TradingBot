from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

import src.strategy.v6_mtf_continuation as v6
from src.analysis.indicators import IndicatorEngine, ema
from src.backtest.models import BacktestConfig
from src.historical.dataset import HistoricalDataset
from src.models.candle import Candle
from src.models.market_data_source import MarketDataSource
from src.research.evaluation import evaluate_strategy_period
from src.strategy.context import StrategyContext
from src.strategy.models import TrendMomentumConfig


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def candle(
    index: int,
    *,
    close: Decimal,
    open: Decimal | None = None,
    high: Decimal | None = None,
    low: Decimal | None = None,
) -> Candle:
    open = close if open is None else open
    return Candle(
        timestamp=BASE + timedelta(minutes=15 * index),
        symbol="BTCUSDC",
        interval="15m",
        open=open,
        high=max(open, close) if high is None else high,
        low=min(open, close) if low is None else low,
        close=close,
        volume=Decimal(index % 17 + 1),
        is_closed=True,
    )


def long_history(groups: int = 60, forming_bars: int = 8) -> tuple[Candle, ...]:
    completed = tuple(
        candle(
            group * 16 + offset,
            close=Decimal("100") + Decimal(group) + Decimal(offset) / 100,
            high=Decimal("101") + Decimal(group) + Decimal(offset) / 100,
            low=Decimal("99") + Decimal(group) + Decimal(offset) / 100,
        )
        for group in range(groups)
        for offset in range(16)
    )
    forming = tuple(
        candle(
            len(completed) + offset,
            close=Decimal("500") + offset,
            high=Decimal("900") + offset,
            low=Decimal("1"),
        )
        for offset in range(forming_bars)
    )
    return completed + forming


def strategy_context(
    candles: tuple[Candle, ...],
    indicators,
    index: int,
) -> StrategyContext:
    current = candles[index]
    return StrategyContext(
        timestamp=current.timestamp + timedelta(minutes=15),
        current_candle=current,
        recent_history=candles[: index + 1],
        indicators=indicators[index],
        previous_indicators=indicators[index - 1] if index else None,
        has_position=False,
        bars_in_position=0,
        equity=Decimal("1000"),
        cash_usdc=Decimal("1000"),
        completed_trade_count=0,
        bars_since_exit=None,
        entry_fee_rate=Decimal("0.001"),
    )


def entry_dataset() -> tuple[HistoricalDataset, int]:
    completed = tuple(
        candle(
            group * 16 + offset,
            close=Decimal("100") + group,
            high=Decimal("101") + group,
            low=Decimal("99") + group,
        )
        for group in range(50)
        for offset in range(16)
    )
    previous = candle(800, close=Decimal("140"), high=Decimal("141"))
    signal = candle(801, close=Decimal("160"), high=Decimal("160"))
    entry_and_exit = candle(
        802,
        open=Decimal("200"),
        close=Decimal("200"),
        high=Decimal("250"),
        low=Decimal("199"),
    )
    dataset = HistoricalDataset(
        symbol="BTCUSDC",
        interval="15m",
        source=MarketDataSource.BINANCE_PUBLIC,
        candles=completed + (previous, signal, entry_and_exit),
    )
    return dataset, 801


def test_prepared_4h_states_exactly_match_full_history_reference() -> None:
    candles = long_history()
    prepared = v6.prepare_v6_context(candles)
    indicators = IndicatorEngine().calculate_research_series(
        candles,
        fast_ema_period=20,
        slow_ema_period=50,
        atr_period=14,
    )
    selected_indices = (
        0,
        14,
        15,
        16,
        17,
        783,
        798,
        799,
        800,
        814,
        815,
        816,
        959,
        960,
        967,
    )

    for index in selected_indices:
        current = candles[index]
        reference = v6.higher_timeframe_state(
            candles[: index + 1], as_of=current.timestamp
        )
        optimized = prepared.state_at(current.timestamp)
        assert optimized == reference
        assert (
            optimized.long_regime_active if optimized is not None else None
        ) == (reference.long_regime_active if reference is not None else None)
        assert indicators[index].ema_fast == ema(
            tuple(item.close for item in candles[: index + 1]), 20
        )


def test_reference_and_prepared_decisions_are_exactly_equal() -> None:
    candles = long_history()
    prepared = v6.prepare_v6_context(candles)
    indicators = IndicatorEngine().calculate_research_series(candles)
    reference = v6.V6MTFContinuationStrategy(TrendMomentumConfig())
    optimized = v6.V6MTFContinuationStrategy(
        TrendMomentumConfig(), prepared_context=prepared
    )

    for index in (0, 15, 16, 783, 799, 800, 815, 959, 960, 967):
        context = strategy_context(candles, indicators, index)
        assert optimized.evaluate(context) == reference.evaluate(context)


def test_reference_and_prepared_execution_are_identical() -> None:
    dataset, warmup = entry_dataset()
    config = TrendMomentumConfig()
    backtest = BacktestConfig(
        fee_bps=Decimal("10"),
        slippage_bps=Decimal("2"),
        warmup_candles=warmup,
    )
    reference = evaluate_strategy_period(
        dataset,
        strategy=v6.V6MTFContinuationStrategy(config),
        strategy_config=config,
        backtest_config=backtest,
    )
    optimized = evaluate_strategy_period(
        dataset,
        strategy=v6.V6MTFContinuationStrategy(
            config, prepared_context=v6.prepare_v6_context(dataset.candles)
        ),
        strategy_config=config,
        backtest_config=backtest,
    )

    assert optimized.signals == reference.signals
    assert optimized.backtest.trades == reference.backtest.trades
    assert optimized.backtest.equity_curve == reference.backtest.equity_curve
    assert optimized.backtest.metrics == reference.backtest.metrics


def test_aggregation_is_prepared_once_not_per_decision(monkeypatch) -> None:
    dataset, _ = entry_dataset()
    calls = 0
    reference_aggregate = v6.aggregate_completed_4h_candles

    def counting_aggregate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return reference_aggregate(*args, **kwargs)

    monkeypatch.setattr(v6, "aggregate_completed_4h_candles", counting_aggregate)
    prepared = v6.prepare_v6_context(dataset.candles)
    assert calls == 1

    evaluate_strategy_period(
        dataset,
        strategy=v6.V6MTFContinuationStrategy(
            TrendMomentumConfig(), prepared_context=prepared
        ),
        strategy_config=TrendMomentumConfig(),
        backtest_config=BacktestConfig(),
    )

    assert calls == 1
    with pytest.raises(TypeError):
        prepared.states_by_timestamp[dataset.candles[0].timestamp] = None
