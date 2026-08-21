from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from src.backtest.engine import BacktestEngine
from src.backtest.models import (
    BacktestConfig,
    BacktestContext,
    ExitReason,
    OrderAction,
    OrderIntent,
)
from src.historical.dataset import HistoricalDataset
from src.hypotheses.h9_early_failure import H9EarlyFailureExit
from src.models.candle import Candle
from src.models.market_data_source import MarketDataSource


BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


def bar(
    index: int,
    *,
    open: str,
    high: str,
    low: str,
    close: str,
) -> Candle:
    return Candle(
        timestamp=BASE + timedelta(minutes=15 * index),
        symbol="BTCUSDC",
        interval="15m",
        open=Decimal(open),
        high=Decimal(high),
        low=Decimal(low),
        close=Decimal(close),
        volume=Decimal("10"),
        is_closed=True,
    )


def dataset(*candles: Candle) -> HistoricalDataset:
    return HistoricalDataset(
        symbol="BTCUSDC",
        interval="15m",
        source=MarketDataSource.BINANCE_PUBLIC,
        candles=tuple(candles),
    )


def entry_only(
    context: BacktestContext,
) -> OrderIntent | None:
    if context.index == 0:
        return OrderIntent(
            action=OrderAction.BUY,
            quote_amount=Decimal("100"),
            stop_loss=Decimal("95"),
            take_profit=Decimal("110"),
            reason="H9 test entry",
        )

    return None


def engine() -> BacktestEngine:
    return BacktestEngine(
        BacktestConfig(
            fee_bps=Decimal("0"),
            slippage_bps=Decimal("0"),
        )
    )


def test_h9_exits_on_next_bar_open_after_trigger() -> None:
    data = dataset(
        bar(0, open="100", high="101", low="99", close="100"),
        # Held bar 1: -0.5R level = 97.5 is touched.
        bar(1, open="100", high="102", low="97.5", close="99"),
        # H9 must exit here, at open.
        bar(2, open="99", high="103", low="98", close="102"),
    )

    h9 = H9EarlyFailureExit()

    result = engine().run(
        data,
        entry_only,
        position_exits=h9,
    )

    assert h9.trigger_count == 1
    assert len(result.trades) == 1

    trade = result.trades[0]

    assert trade.exit_reason is ExitReason.EARLY_FAILURE_EXIT
    assert trade.exit_time == data.candles[2].timestamp
    assert trade.exit_price == Decimal("99")


def test_original_stop_wins_on_trigger_candle() -> None:
    data = dataset(
        bar(0, open="100", high="101", low="99", close="100"),
        # Hits both -0.5R and the original -1R stop.
        bar(1, open="100", high="101", low="94", close="96"),
        bar(2, open="96", high="100", low="95", close="98"),
    )

    h9 = H9EarlyFailureExit()

    result = engine().run(
        data,
        entry_only,
        position_exits=h9,
    )

    assert h9.trigger_count == 0

    trade = result.trades[0]

    assert trade.exit_reason is ExitReason.STOP_LOSS
    assert trade.exit_price == Decimal("95")


def test_h9_can_trigger_on_fourth_held_bar() -> None:
    data = dataset(
        bar(0, open="100", high="101", low="99", close="100"),
        bar(1, open="100", high="102", low="98", close="100"),
        bar(2, open="100", high="102", low="98", close="100"),
        bar(3, open="100", high="102", low="98", close="100"),
        # Fourth held bar reaches -0.5R.
        bar(4, open="100", high="102", low="97.5", close="99"),
        bar(5, open="99", high="101", low="98", close="100"),
    )

    h9 = H9EarlyFailureExit()

    result = engine().run(
        data,
        entry_only,
        position_exits=h9,
    )

    assert h9.trigger_count == 1
    assert result.trades[0].exit_reason is ExitReason.EARLY_FAILURE_EXIT
    assert result.trades[0].exit_time == data.candles[5].timestamp


def test_h9_does_not_trigger_on_fifth_held_bar() -> None:
    data = dataset(
        bar(0, open="100", high="101", low="99", close="100"),
        bar(1, open="100", high="102", low="98", close="100"),
        bar(2, open="100", high="102", low="98", close="100"),
        bar(3, open="100", high="102", low="98", close="100"),
        bar(4, open="100", high="102", low="98", close="100"),
        # Fifth held bar reaches -0.5R: too late.
        bar(5, open="100", high="102", low="97.5", close="99"),
        bar(6, open="99", high="101", low="98", close="100"),
    )

    h9 = H9EarlyFailureExit()

    result = engine().run(
        data,
        entry_only,
        position_exits=h9,
    )

    assert h9.trigger_count == 0
    assert result.trades[0].exit_reason is ExitReason.END_OF_BACKTEST


def test_existing_strategy_exit_keeps_precedence() -> None:
    def decision(
        context: BacktestContext,
    ) -> OrderIntent | None:
        if context.index == 0:
            return OrderIntent(
                action=OrderAction.BUY,
                quote_amount=Decimal("100"),
                stop_loss=Decimal("95"),
                take_profit=Decimal("110"),
            )

        if context.index == 1:
            return OrderIntent(
                action=OrderAction.SELL,
                exit_reason=ExitReason.TREND_EXIT,
            )

        return None

    data = dataset(
        bar(0, open="100", high="101", low="99", close="100"),
        # Also reaches H9 threshold.
        bar(1, open="100", high="101", low="97.5", close="99"),
        bar(2, open="99", high="101", low="98", close="100"),
    )

    h9 = H9EarlyFailureExit()

    result = engine().run(
        data,
        decision,
        position_exits=h9,
    )

    assert h9.trigger_count == 0
    assert result.trades[0].exit_reason is ExitReason.TREND_EXIT