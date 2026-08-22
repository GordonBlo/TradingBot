from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.analysis.indicators import ema
from src.backtest.engine import BacktestEngine
from src.backtest.models import (
    BacktestConfig,
    ExitReason,
    OrderAction,
    OrderIntent,
)
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
from src.strategy.v6_mtf_continuation import (
    V6MTFContinuationStrategy,
    aggregate_completed_4h_candles,
    higher_timeframe_state,
)


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def candle(
    index: int,
    *,
    close: Decimal,
    open: Decimal | None = None,
    high: Decimal | None = None,
    low: Decimal | None = None,
    volume: Decimal = Decimal("1"),
    closed: bool = True,
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
        volume=volume,
        is_closed=closed,
    )


def source_from_4h_closes(
    closes: list[Decimal], *, candle_range: Decimal = Decimal("1")
) -> tuple[Candle, ...]:
    return tuple(
        candle(
            group_index * 16 + source_index,
            close=close,
            high=close + candle_range,
            low=close - candle_range,
            volume=Decimal(source_index + 1),
        )
        for group_index, close in enumerate(closes)
        for source_index in range(16)
    )


def context(
    history: tuple[Candle, ...],
    *,
    bars_since_exit: int | None = None,
    has_position: bool = False,
    bars_in_position: int = 0,
) -> StrategyContext:
    current = history[-1]
    closes = tuple(candle.close for candle in history)
    indicators = ResearchIndicatorSnapshot(
        timestamp=current.timestamp,
        symbol=current.symbol,
        interval=current.interval,
        close=current.close,
        volume=current.volume,
        ema_fast=ema(closes, 20),
        ema_slow=None,
        rsi=None,
        atr=None,
        volume_sma=None,
        volume_ratio=None,
    )
    previous = history[-2] if len(history) > 1 else None
    previous_indicators = (
        ResearchIndicatorSnapshot(
            timestamp=previous.timestamp,
            symbol=previous.symbol,
            interval=previous.interval,
            close=previous.close,
            volume=previous.volume,
            ema_fast=ema(closes[:-1], 20),
            ema_slow=None,
            rsi=None,
            atr=None,
            volume_sma=None,
            volume_ratio=None,
        )
        if previous is not None
        else None
    )
    return StrategyContext(
        timestamp=current.timestamp + timedelta(minutes=15),
        current_candle=current,
        recent_history=history,
        indicators=indicators,
        previous_indicators=previous_indicators,
        has_position=has_position,
        bars_in_position=bars_in_position,
        equity=Decimal("1000"),
        cash_usdc=Decimal("1000"),
        completed_trade_count=1 if bars_since_exit is not None else 0,
        bars_since_exit=bars_since_exit,
        entry_fee_rate=Decimal("0"),
    )


def strategy() -> V6MTFContinuationStrategy:
    return V6MTFContinuationStrategy(TrendMomentumConfig())


def evaluate(history: tuple[Candle, ...], **kwargs: object) -> StrategyDecision:
    return strategy().evaluate(context(history, **kwargs))


def entry_history(
    *,
    candle_range: Decimal = Decimal("1"),
    previous_close: Decimal = Decimal("140"),
    current_close: Decimal = Decimal("160"),
    previous_high: Decimal = Decimal("141"),
    regime_closes: list[Decimal] | None = None,
) -> tuple[Candle, ...]:
    completed = source_from_4h_closes(
        regime_closes or [Decimal("100") + index for index in range(50)],
        candle_range=candle_range,
    )
    previous_index = len(completed)
    previous = candle(
        previous_index,
        close=previous_close,
        high=previous_high,
        low=previous_close - Decimal("1"),
    )
    current = candle(
        previous_index + 1,
        close=current_close,
        high=current_close,
        low=current_close - Decimal("1"),
    )
    return (*completed, previous, current)


def test_exactly_sixteen_aligned_15m_candles_form_correct_4h_ohlcv() -> None:
    raw = tuple(
        candle(
            index,
            open=Decimal("100") + index,
            high=Decimal("110") + index,
            low=Decimal("90") - index,
            close=Decimal("101") + index,
            volume=Decimal(index + 1),
        )
        for index in range(16)
    )

    aggregated = aggregate_completed_4h_candles(raw, as_of=raw[-1].timestamp)

    assert len(aggregated) == 1
    result = aggregated[0]
    assert result.timestamp == BASE
    assert result.interval == "4h"
    assert result.open == Decimal("100")
    assert result.high == Decimal("125")
    assert result.low == Decimal("75")
    assert result.close == Decimal("116")
    assert result.volume == Decimal("136")


def test_4h_utc_alignment_and_incomplete_candle_exclusion() -> None:
    unaligned = tuple(candle(index + 4, close=Decimal("100")) for index in range(16))
    incomplete = tuple(candle(index, close=Decimal("100")) for index in range(15))

    assert aggregate_completed_4h_candles(unaligned, as_of=unaligned[-1].timestamp) == ()
    assert aggregate_completed_4h_candles(incomplete, as_of=incomplete[-1].timestamp) == ()


def test_forming_and_future_15m_candles_do_not_affect_current_4h_state() -> None:
    completed = source_from_4h_closes(
        [Decimal("100") + index for index in range(50)]
    )
    forming_and_future = (
        *completed,
        *(candle(800 + index, close=Decimal("1"), high=Decimal("1000"), low=Decimal("1")) for index in range(21)),
    )

    reference = higher_timeframe_state(completed, as_of=completed[-1].timestamp)
    observed = higher_timeframe_state(
        forming_and_future,
        as_of=forming_and_future[804].timestamp,
    )

    assert reference is not None
    assert observed == reference
    assert observed.latest_candle.timestamp == completed[-1].timestamp - timedelta(hours=3, minutes=45)


@pytest.mark.parametrize(
    ("closes", "active"),
    (
        ([Decimal("100")] * 30 + [Decimal("200")] * 19 + [Decimal("210")], True),
        ([Decimal("100")] * 50, False),
        ([Decimal("200")] * 30 + [Decimal("100")] * 20, False),
        ([Decimal("100")] * 30 + [Decimal("200")] * 19 + [Decimal("150")], False),
    ),
)
def test_completed_4h_regime_uses_strict_ema_and_close_conditions(
    closes: list[Decimal], active: bool
) -> None:
    candles = source_from_4h_closes(closes)
    state = higher_timeframe_state(candles, as_of=candles[-1].timestamp)

    assert state is not None
    assert state.long_regime_active is active


def test_completed_4h_close_equal_to_ema20_is_inactive() -> None:
    closes = [Decimal("100")] * 30 + [Decimal("200")] * 19
    final_close = ema(closes, 20)
    assert final_close is not None
    candles = source_from_4h_closes([*closes, final_close])
    state = higher_timeframe_state(candles, as_of=candles[-1].timestamp)

    assert state is not None
    assert state.ema_20 > state.ema_50
    assert state.latest_candle.close == state.ema_20
    assert state.long_regime_active is False


def test_insufficient_completed_4h_history_has_no_buy() -> None:
    history = entry_history(regime_closes=[Decimal("100") + index for index in range(49)])

    decision = evaluate(history)

    assert (decision.action, decision.reason_code) == (
        StrategyAction.HOLD,
        DecisionReason.WARMUP,
    )


def test_previous_close_equal_to_its_ema_is_accepted() -> None:
    completed = source_from_4h_closes(
        [Decimal("100") + index for index in range(50)]
    )
    previous_ema = ema(tuple(candle.close for candle in completed), 20)
    assert previous_ema is not None

    decision = evaluate(
        entry_history(
            previous_close=previous_ema,
            previous_high=previous_ema + Decimal("1"),
        )
    )

    assert decision.action is StrategyAction.ENTER_LONG


def test_previous_close_above_its_ema_is_rejected() -> None:
    decision = evaluate(
        entry_history(
            previous_close=Decimal("180"),
            previous_high=Decimal("181"),
            current_close=Decimal("200"),
        )
    )

    assert decision.action is StrategyAction.HOLD


def test_current_close_must_be_strictly_above_its_ema() -> None:
    completed = source_from_4h_closes(
        [Decimal("100") + index for index in range(50)]
    )
    previous = candle(800, close=Decimal("140"), high=Decimal("141"), low=Decimal("139"))
    previous_ema = ema(
        tuple(candle.close for candle in (*completed, previous)), 20
    )
    assert previous_ema is not None
    current = candle(801, close=previous_ema, high=previous_ema, low=previous_ema - Decimal("1"))

    decision = evaluate((*completed, previous, current))

    assert decision.action is StrategyAction.HOLD


def test_current_close_must_be_strictly_above_previous_high() -> None:
    decision = evaluate(entry_history(previous_high=Decimal("160")))

    assert decision.action is StrategyAction.HOLD


def test_valid_continuation_buys_without_old_indicator_filters() -> None:
    decision = evaluate(entry_history())

    assert decision.action is StrategyAction.ENTER_LONG
    assert decision.reason_code is DecisionReason.MTF_CONTINUATION_ENTRY
    history = entry_history()
    assert decision.metadata["previous_ema_20_15m"] == ema(
        tuple(candle.close for candle in history[:-1]), 20
    )
    assert decision.metadata["current_ema_20_15m"] == ema(
        tuple(candle.close for candle in history), 20
    )


def test_inactive_completed_4h_regime_has_no_buy() -> None:
    decision = evaluate(
        entry_history(regime_closes=[Decimal("100")] * 50)
    )

    assert decision.action is StrategyAction.HOLD


def _run_trade(*, candle_range: Decimal, stop_first: bool = False):
    signal_history = entry_history(candle_range=candle_range)
    decision = evaluate(signal_history)
    assert decision.action is StrategyAction.ENTER_LONG
    fill = Decimal("200")
    stop_distance = max(
        decision.stop_distance,
        fill * V6MTFContinuationStrategy.MIN_STOP_DISTANCE_FRACTION,
    )
    target = fill + Decimal("2") * stop_distance
    stop = fill - stop_distance
    entry = candle(
        len(signal_history),
        open=fill,
        close=fill,
        high=target if not stop_first else target,
        low=(stop if stop_first else fill - Decimal("0.1")),
    )
    dataset = HistoricalDataset(
        symbol="BTCUSDC",
        interval="15m",
        source=MarketDataSource.BINANCE_PUBLIC,
        candles=(*signal_history, entry),
    )
    result = evaluate_strategy_period(
        dataset,
        strategy=strategy(),
        strategy_config=TrendMomentumConfig(),
        backtest_config=BacktestConfig(
            fee_bps=Decimal("0"),
            slippage_bps=Decimal("0"),
            warmup_candles=len(signal_history) - 1,
        ),
    ).backtest
    return decision, result.trades[0], fill, stop_distance, target, stop


def test_atr14_4h_determines_stop_when_larger_than_cost_floor() -> None:
    decision, trade, fill, stop_distance, target, _ = _run_trade(
        candle_range=Decimal("3")
    )

    assert decision.stop_distance > fill * Decimal("0.0096")
    assert stop_distance == decision.stop_distance
    assert trade.entry_price == fill
    assert trade.exit_price == target
    assert trade.exit_reason is ExitReason.TAKE_PROFIT


def test_96_bps_cost_floor_uses_actual_entry_fill_and_two_r_target() -> None:
    decision, trade, fill, stop_distance, target, _ = _run_trade(
        candle_range=Decimal("0.1")
    )

    assert decision.stop_distance < fill * Decimal("0.0096")
    assert stop_distance == fill * Decimal("0.0096")
    assert trade.entry_price == fill
    assert trade.exit_price == target
    assert target == fill + Decimal("2") * stop_distance
    assert target != Decimal("160") + Decimal("2") * stop_distance


def test_equal_atr_and_cost_floor_uses_the_same_frozen_distance() -> None:
    fill = Decimal("200")
    floor = fill * V6MTFContinuationStrategy.MIN_STOP_DISTANCE_FRACTION
    intent = OrderIntent(
        action=OrderAction.BUY,
        risk_budget=Decimal("1"),
        stop_distance=floor,
        reward_risk_ratio=Decimal("2"),
        max_quote_amount=Decimal("50"),
        minimum_stop_distance_fraction=(
            V6MTFContinuationStrategy.MIN_STOP_DISTANCE_FRACTION
        ),
        reason="V6 equality regression",
    )
    signal = candle(0, close=Decimal("100"))
    target = fill + Decimal("2") * floor
    entry = candle(
        1,
        open=fill,
        close=fill,
        high=target,
        low=fill - Decimal("0.1"),
    )
    dataset = HistoricalDataset(
        symbol="BTCUSDC",
        interval="15m",
        source=MarketDataSource.BINANCE_PUBLIC,
        candles=(signal, entry),
    )

    result = BacktestEngine(
        BacktestConfig(fee_bps=Decimal("0"), slippage_bps=Decimal("0"))
    ).run(dataset, lambda backtest_context: intent if backtest_context.index == 0 else None)

    assert result.trades[0].entry_price == fill
    assert result.trades[0].exit_price == target


def test_stop_first_remains_the_existing_ambiguous_bar_policy() -> None:
    _, trade, _, _, _, stop = _run_trade(
        candle_range=Decimal("0.1"), stop_first=True
    )

    assert trade.exit_price == stop
    assert trade.exit_reason is ExitReason.STOP_LOSS


def test_latest_completed_4h_atr_is_used_and_forming_4h_atr_is_excluded() -> None:
    history = entry_history(candle_range=Decimal("0.1"))
    completed = history[:800]
    expected = higher_timeframe_state(completed, as_of=completed[-1].timestamp)
    decision = evaluate(history)

    assert expected is not None
    assert decision.metadata["atr_4h"] == expected.atr_14
    assert decision.metadata["higher_timeframe_candle_timestamp"] == (
        expected.latest_candle.timestamp.isoformat()
    )


def test_buy_slippage_sets_actual_fill_floor_without_recalculating_signal_atr() -> None:
    signal_history = entry_history(candle_range=Decimal("0.1"))
    decision = evaluate(signal_history)
    assert decision.stop_distance is not None
    market_open = Decimal("200")
    buy_rate = Decimal("0.0002")
    actual_fill = market_open * (Decimal("1") + buy_rate)
    expected_distance = max(
        decision.stop_distance,
        actual_fill * Decimal("0.0096"),
    )
    expected_stop = actual_fill - expected_distance
    entry = candle(
        len(signal_history),
        open=market_open,
        close=market_open,
        high=market_open,
        low=Decimal("1"),
    )
    dataset = HistoricalDataset(
        symbol="BTCUSDC",
        interval="15m",
        source=MarketDataSource.BINANCE_PUBLIC,
        candles=(*signal_history, entry),
    )

    trade = evaluate_strategy_period(
        dataset,
        strategy=strategy(),
        strategy_config=TrendMomentumConfig(),
        backtest_config=BacktestConfig(
            fee_bps=Decimal("10"),
            slippage_bps=Decimal("2"),
            warmup_candles=len(signal_history) - 1,
        ),
    ).backtest.trades[0]

    assert expected_distance == actual_fill * Decimal("0.0096")
    assert expected_distance != decision.metadata["signal_cost_floor_distance"]
    assert trade.entry_price == actual_fill
    assert trade.exit_price == expected_stop * (Decimal("1") - buy_rate)


def test_v6_adapter_preserves_full_causal_history() -> None:
    history = entry_history(
        regime_closes=[Decimal("100") + index for index in range(60)],
        previous_close=Decimal("150"),
        previous_high=Decimal("151"),
        current_close=Decimal("180"),
    )
    entry = candle(len(history), close=Decimal("180"))
    dataset = HistoricalDataset(
        symbol="BTCUSDC",
        interval="15m",
        source=MarketDataSource.BINANCE_PUBLIC,
        candles=(*history, entry),
    )

    class RecordingV6(V6MTFContinuationStrategy):
        def __init__(self) -> None:
            super().__init__(TrendMomentumConfig())
            self.signal_history_length: int | None = None

        def evaluate(self, strategy_context: StrategyContext) -> StrategyDecision:
            if strategy_context.current_candle.timestamp == history[-1].timestamp:
                self.signal_history_length = len(strategy_context.recent_history)
            return super().evaluate(strategy_context)

    recording = RecordingV6()
    evaluate_strategy_period(
        dataset,
        strategy=recording,
        strategy_config=TrendMomentumConfig(),
        backtest_config=BacktestConfig(
            warmup_candles=len(history) - 1,
        ),
    )

    assert len(history) > recording.REQUIRED_HISTORY_BARS
    assert recording.signal_history_length == len(history)


def test_cost_floor_constants_have_no_bps_percent_conversion_error() -> None:
    cls = V6MTFContinuationStrategy

    assert cls.ROUND_TRIP_FRICTION_BPS == Decimal("2") * (
        cls.BASE_FEE_BPS_PER_SIDE + cls.BASE_SLIPPAGE_BPS_PER_SIDE
    )
    assert cls.ROUND_TRIP_FRICTION_BPS == Decimal("24")
    assert cls.COST_DISTANCE_MULTIPLE == 4
    assert cls.MIN_STOP_DISTANCE_BPS == Decimal("96")
    assert cls.MIN_STOP_DISTANCE_BPS == (
        cls.ROUND_TRIP_FRICTION_BPS * cls.COST_DISTANCE_MULTIPLE
    )
    assert cls.MIN_STOP_DISTANCE_FRACTION == Decimal("96") / Decimal("10000")


def test_pre_v6_risk_intent_without_floor_keeps_original_bracket_semantics() -> None:
    intent = OrderIntent(
        action=OrderAction.BUY,
        risk_budget=Decimal("1"),
        stop_distance=Decimal("5"),
        reward_risk_ratio=Decimal("2"),
        max_quote_amount=Decimal("50"),
        reason="pre-V6 regression",
    )
    signal = candle(0, close=Decimal("100"))
    entry = candle(
        1,
        open=Decimal("110"),
        close=Decimal("110"),
        high=Decimal("120"),
        low=Decimal("106"),
    )
    dataset = HistoricalDataset(
        symbol="BTCUSDC",
        interval="15m",
        source=MarketDataSource.BINANCE_PUBLIC,
        candles=(signal, entry),
    )

    trade = BacktestEngine(
        BacktestConfig(fee_bps=Decimal("0"), slippage_bps=Decimal("0"))
    ).run(dataset, lambda backtest_context: intent if backtest_context.index == 0 else None).trades[0]

    assert intent.minimum_stop_distance_fraction is None
    assert trade.entry_price == Decimal("110")
    assert trade.exit_price == Decimal("120")
    assert trade.exit_reason is ExitReason.TAKE_PROFIT


@pytest.mark.parametrize("bars_since_exit", (0, 1, 2, 3, 4))
def test_four_bar_cooldown_is_preserved(bars_since_exit: int) -> None:
    decision = evaluate(entry_history(), bars_since_exit=bars_since_exit)

    assert decision.reason_code is DecisionReason.COOLDOWN


def test_maximum_hold_is_a_time_exit() -> None:
    decision = evaluate(
        entry_history(),
        has_position=True,
        bars_in_position=96,
    )

    assert (decision.action, decision.reason_code) == (
        StrategyAction.EXIT_LONG,
        DecisionReason.TIME_EXIT,
    )
