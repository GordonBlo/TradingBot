from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

import src.strategy.v6_mtf_continuation as v6
import src.cli.run_v6_h0_replay as replay
from src.analysis.indicators import IndicatorEngine, ema
from src.backtest.models import BacktestConfig
from src.historical.dataset import HistoricalDataset
from src.models.candle import Candle
from src.models.market_data_source import MarketDataSource
from src.research.evaluation import evaluate_strategy_period
from src.research.multiregime.models import (
    PartitionKind,
    PartitionStatus,
    ResearchWindow,
)
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
        assert prepared.indicators_15m_by_timestamp[current.timestamp] == (
            indicators[index]
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

    class RecordingPreparedV6(v6.V6MTFContinuationStrategy):
        def __init__(self) -> None:
            super().__init__(
                TrendMomentumConfig(), prepared_context=prepared
            )
            self.history_lengths: list[int] = []

        def evaluate(self, context):
            self.history_lengths.append(len(context.recent_history))
            return super().evaluate(context)

    strategy = RecordingPreparedV6()
    evaluate_strategy_period(
        dataset,
        strategy=strategy,
        strategy_config=TrendMomentumConfig(),
        backtest_config=BacktestConfig(),
        prepared_indicators=prepared.indicators_15m_by_timestamp,
    )

    assert calls == 1
    assert strategy.requires_full_history is False
    assert max(strategy.history_lengths) <= 51
    with pytest.raises(TypeError):
        prepared.states_by_timestamp[dataset.candles[0].timestamp] = None


def overlapping_research_windows():
    base_dataset, _ = entry_dataset()
    trailing = tuple(
        candle(
            index,
            close=Decimal("200"),
            high=Decimal("201"),
            low=Decimal("199"),
        )
        for index in range(803, 821)
    )
    dataset = HistoricalDataset(
        symbol="BTCUSDC",
        interval="15m",
        source=MarketDataSource.BINANCE_PUBLIC,
        candles=base_dataset.candles + trailing,
    )
    region_end = BASE + timedelta(minutes=15 * 821)
    region = SimpleNamespace(
        metadata=SimpleNamespace(
            kind=PartitionKind.CONSUMED_RESEARCH,
            start=BASE,
            end=region_end,
        ),
        dataset=dataset,
    )

    def window(window_id: str, start_index: int, end_index: int):
        start = BASE + timedelta(minutes=15 * start_index)
        end = BASE + timedelta(minutes=15 * end_index)
        evaluation = HistoricalDataset(
            symbol=dataset.symbol,
            interval=dataset.interval,
            source=dataset.source,
            candles=dataset.candles[start_index:end_index],
        )
        return ResearchWindow(
            window_id=window_id,
            start=start,
            end=end,
            duration_days=Decimal(end_index - start_index) / Decimal("96"),
            partition_kind=PartitionKind.CONSUMED_RESEARCH,
            data_status=PartitionStatus.CONSUMED_RESEARCH_DATA,
            partial_window=False,
            duration_eligible=True,
            replay_dataset=evaluation,
            evaluation_start_index=0,
        )

    raw = (window("W001", 0, 803), window("W002", 803, 821))
    expanded = replay.expand_complete_region_history(raw, (region,))
    return (region,), expanded


def test_region_cached_overlapping_windows_are_exactly_equivalent() -> None:
    regions, windows = overlapping_research_windows()
    region_contexts = replay.prepare_v6_region_contexts(
        windows=windows, regions=regions
    )
    reference_records = []
    optimized_records = []

    for window in windows:
        per_window = v6.prepare_v6_context(window.replay_dataset.candles)
        shared = replay.prepared_context_for_window(
            window=window,
            regions=regions,
            region_contexts=region_contexts,
        )
        for source_candle in (
            window.replay_dataset.candles[0],
            window.replay_dataset.candles[window.evaluation_start_index],
            window.replay_dataset.candles[-1],
        ):
            assert shared.state_at(source_candle.timestamp) == per_window.state_at(
                source_candle.timestamp
            )
            assert shared.indicators_15m_by_timestamp[source_candle.timestamp] == (
                per_window.indicators_15m_by_timestamp[source_candle.timestamp]
            )

        reference = replay._evaluate_window(
            window=window,
            strategy_config=TrendMomentumConfig(),
            backtest_config=BacktestConfig(),
            prepared_context=per_window,
        )
        optimized = replay._evaluate_window(
            window=window,
            strategy_config=TrendMomentumConfig(),
            backtest_config=BacktestConfig(),
            prepared_context=shared,
        )
        assert optimized[0].signals == reference[0].signals
        assert optimized[0].backtest.trades == reference[0].backtest.trades
        assert optimized[0].backtest.equity_curve == reference[0].backtest.equity_curve
        assert optimized[1:] == reference[1:]
        reference_records.append(reference[1])
        optimized_records.append(optimized[1])

    assert replay.summarize_combined_records(tuple(optimized_records)) == (
        replay.summarize_combined_records(tuple(reference_records))
    )


def test_overlapping_windows_prepare_once_per_region(monkeypatch) -> None:
    regions, windows = overlapping_research_windows()
    calls: list[int] = []
    original = replay.prepare_v6_context

    def counting_prepare(candles):
        calls.append(len(candles))
        return original(candles)

    monkeypatch.setattr(replay, "prepare_v6_context", counting_prepare)
    contexts = replay.prepare_v6_region_contexts(
        windows=windows, regions=regions
    )
    for window in windows:
        replay.prepared_context_for_window(
            window=window,
            regions=regions,
            region_contexts=contexts,
        )

    before, after = replay.preparation_work_totals(
        windows=windows, regions=regions
    )
    assert calls == [len(regions[0].dataset.candles)]
    assert before == sum(len(window.replay_dataset.candles) for window in windows)
    assert after == len(regions[0].dataset.candles)
    assert after < before
