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
from src.strategy.v5_mean_reversion import V5MeanReversionStrategy


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def candle(
    index: int,
    *,
    close: str = "90",
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
        high=Decimal(high) if high is not None else max(open_value, close_value),
        low=Decimal(low) if low is not None else min(open_value, close_value),
        close=close_value,
        volume=Decimal(volume),
        is_closed=True,
    )


def reference_candles(*, first_highs: bool = False) -> tuple[Candle, ...]:
    closes = ("110",) * 10 + ("90",) * 10
    return tuple(
        candle(
            index,
            close=close,
            high="1000" if first_highs else None,
            low="1" if first_highs else None,
        )
        for index, close in enumerate(closes)
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


def history(
    *,
    current_close: str = "90",
    current_low: str = "79.5",
    current_high: str = "100",
    first_highs: bool = False,
) -> tuple[Candle, ...]:
    return reference_candles(first_highs=first_highs) + (
        candle(
            20,
            close=current_close,
            high=current_high,
            low=current_low,
        ),
    )


def evaluate(
    candles: tuple[Candle, ...], **kwargs: object
) -> StrategyDecision:
    return V5MeanReversionStrategy(TrendMomentumConfig()).evaluate(
        context(candles, **kwargs)
    )


def test_fewer_than_twenty_reference_closes_has_no_buy() -> None:
    candles = tuple(candle(index) for index in range(19)) + (
        candle(19, close="90", high="100", low="79.5"),
    )

    decision = evaluate(candles)

    assert decision.action is StrategyAction.HOLD
    assert decision.reason_code is DecisionReason.WARMUP


def test_exactly_twenty_previous_closes_use_population_distribution() -> None:
    decision = evaluate(history())

    assert decision.action is StrategyAction.ENTER_LONG
    assert decision.metadata["reference_mean"] == Decimal("100")
    assert decision.metadata["reference_standard_deviation"] == Decimal("10")
    assert decision.metadata["lower_band"] == Decimal("80")
    assert decision.metadata["standard_deviation_ddof"] == 0


def test_current_candle_is_excluded_from_reference_distribution() -> None:
    decision = evaluate(history(current_low="79.5"))

    assert decision.action is StrategyAction.ENTER_LONG
    assert decision.metadata["reference_mean"] == Decimal("100")
    assert decision.metadata["lower_band"] == Decimal("80")


def test_reference_uses_exactly_previous_twenty_closes() -> None:
    older_outlier = candle(0, close="1000")
    previous_twenty = tuple(
        candle(index + 1, close=close)
        for index, close in enumerate(("110",) * 10 + ("90",) * 10)
    )
    current = candle(21, close="90", high="100", low="79.5")

    decision = evaluate((older_outlier, *previous_twenty, current))

    assert decision.action is StrategyAction.ENTER_LONG
    assert decision.metadata["reference_mean"] == Decimal("100")


def test_reference_uses_closes_not_highs_or_lows() -> None:
    decision = evaluate(history(first_highs=True))

    assert decision.action is StrategyAction.ENTER_LONG
    assert decision.metadata["reference_mean"] == Decimal("100")
    assert decision.metadata["lower_band"] == Decimal("80")


def test_zero_standard_deviation_has_no_buy() -> None:
    flat = tuple(candle(index, close="100") for index in range(20))
    decision = evaluate(
        flat + (candle(20, close="95", high="100", low="90"),)
    )

    assert decision.action is StrategyAction.HOLD
    assert decision.reason_code is DecisionReason.NO_ENTRY


@pytest.mark.parametrize(
    ("current_low", "current_close"),
    (
        ("80", "90"),
        ("79", "80"),
        ("79", "100"),
        ("79", "79.5"),
        ("79", "101"),
    ),
)
def test_strict_band_and_mean_boundaries_do_not_buy(
    current_low: str, current_close: str
) -> None:
    decision = evaluate(
        history(
            current_low=current_low,
            current_close=current_close,
            current_high="101",
        )
    )

    assert decision.action is StrategyAction.HOLD


def test_strict_exhaustion_reclaim_buys_without_old_filters() -> None:
    decision = evaluate(history(), old_indicators_ready=False)

    assert decision.action is StrategyAction.ENTER_LONG
    assert decision.reason_code is DecisionReason.EXHAUSTION_RECLAIM_ENTRY


def test_atr_stop_distance_and_two_r_target_are_frozen() -> None:
    decision = evaluate(history(), atr_value="5")

    assert decision.stop_distance == Decimal("10")
    assert decision.reward_risk_ratio == Decimal("2")


@pytest.mark.parametrize("bars_since_exit", (0, 1, 2, 3, 4))
def test_four_bar_cooldown_is_preserved(bars_since_exit: int) -> None:
    decision = evaluate(history(), bars_since_exit=bars_since_exit)

    assert decision.reason_code is DecisionReason.COOLDOWN


def test_maximum_hold_is_a_time_exit() -> None:
    decision = evaluate(
        history(),
        has_position=True,
        bars_in_position=96,
        atr_value=None,
        old_indicators_ready=False,
    )

    assert (decision.action, decision.reason_code) == (
        StrategyAction.EXIT_LONG,
        DecisionReason.TIME_EXIT,
    )


def execution_dataset(*, entry_low: Decimal, entry_high: Decimal) -> HistoricalDataset:
    previous = reference_candles()
    signal = candle(20, close="90", high="100", low="79.5")
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
        candles=(*previous, signal, entry),
    )


def run_v5(data: HistoricalDataset):
    strategy_config = TrendMomentumConfig()
    return evaluate_strategy_period(
        data,
        strategy=V5MeanReversionStrategy(strategy_config),
        strategy_config=strategy_config,
        backtest_config=BacktestConfig(
            fee_bps=Decimal("0"),
            slippage_bps=Decimal("0"),
        ),
    ).backtest


def test_adapter_uses_next_open_and_fill_relative_two_r_target() -> None:
    preliminary = execution_dataset(
        entry_low=Decimal("109"), entry_high=Decimal("200")
    )
    signal_atr = atr(preliminary.candles[:21], 14)
    assert signal_atr is not None
    target = Decimal("110") + Decimal("4") * signal_atr
    data = replace(
        preliminary,
        candles=(
            *preliminary.candles[:-1],
            candle(21, close="110", open="110", high=str(target), low="109"),
        ),
    )

    trade = run_v5(data).trades[0]

    assert trade.entry_price == Decimal("110")
    assert trade.exit_price == target
    assert trade.exit_reason is ExitReason.TAKE_PROFIT


def test_v5_bracket_keeps_existing_stop_first_policy() -> None:
    data = execution_dataset(entry_low=Decimal("1"), entry_high=Decimal("200"))
    signal_atr = atr(data.candles[:21], 14)
    assert signal_atr is not None

    trade = run_v5(data).trades[0]

    assert trade.entry_price == Decimal("110")
    assert trade.exit_price == Decimal("110") - Decimal("2") * signal_atr
    assert trade.exit_reason is ExitReason.STOP_LOSS
