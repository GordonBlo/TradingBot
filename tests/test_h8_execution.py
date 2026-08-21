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
from src.hypotheses.h8_protection import H8BreakEvenProtection
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


def entry(context: BacktestContext) -> OrderIntent | None:
    if context.index == 0:
        return OrderIntent(
            OrderAction.BUY,
            Decimal("100"),
            stop_loss=Decimal("95"),
            take_profit=Decimal("110"),
            reason="H8 test entry",
        )

    return None


def engine() -> BacktestEngine:
    return BacktestEngine(
        BacktestConfig(
            fee_bps=Decimal("0"),
            slippage_bps=Decimal("0"),
        )
    )


def test_h8_activates_only_on_bar_after_closed_trigger() -> None:
    data = dataset(
        bar(0, open="100", high="101", low="99", close="100"),
        # Entry at 100. High reaches +1R=105.
        # Low also goes below entry, but H8 is NOT active yet.
        bar(1, open="100", high="105", low="99", close="104"),
        # H8 becomes active here.
        bar(2, open="100", high="104", low="99", close="100"),
    )

    protection = H8BreakEvenProtection()

    result = engine().run(
        data,
        entry,
        stop_updates=protection,
    )

    assert protection.activation_count == 1
    assert len(result.trades) == 1

    trade = result.trades[0]

    assert trade.exit_reason is ExitReason.BREAK_EVEN_STOP
    assert trade.exit_time == data.candles[2].timestamp
    assert trade.exit_price == Decimal("100")


def test_h8_gap_below_break_even_uses_worse_open() -> None:
    data = dataset(
        bar(0, open="100", high="101", low="99", close="100"),
        bar(1, open="100", high="105", low="99", close="104"),
        bar(2, open="98", high="100", low="97", close="99"),
    )

    result = engine().run(
        data,
        entry,
        stop_updates=H8BreakEvenProtection(),
    )

    trade = result.trades[0]

    assert trade.exit_reason is ExitReason.BREAK_EVEN_STOP
    assert trade.exit_price == Decimal("98")


def test_h8_preserves_stop_first_when_be_and_target_touch() -> None:
    data = dataset(
        bar(0, open="100", high="101", low="99", close="100"),
        bar(1, open="100", high="105", low="99", close="104"),
        # Both BE=100 and TP=110 touched.
        bar(2, open="100", high="111", low="99", close="105"),
    )

    result = engine().run(
        data,
        entry,
        stop_updates=H8BreakEvenProtection(),
    )

    trade = result.trades[0]

    assert trade.exit_reason is ExitReason.BREAK_EVEN_STOP
    assert trade.exit_price == Decimal("100")


def test_h8_does_not_activate_without_positive_one_r() -> None:
    data = dataset(
        bar(0, open="100", high="101", low="99", close="100"),
        bar(1, open="100", high="104.99", low="99", close="104"),
        bar(2, open="100", high="104", low="99", close="100"),
    )

    protection = H8BreakEvenProtection()

    result = engine().run(
        data,
        entry,
        stop_updates=protection,
    )

    assert protection.activation_count == 0
    assert result.trades[0].exit_reason is ExitReason.END_OF_BACKTEST


def test_original_stop_wins_on_trigger_candle() -> None:
    data = dataset(
        bar(0, open="100", high="101", low="99", close="100"),
        # Same OHLC candle reaches +1R AND original -1R stop.
        # Existing stop acts before any next-bar H8 activation.
        bar(1, open="100", high="105", low="94", close="100"),
        bar(2, open="100", high="101", low="99", close="100"),
    )

    protection = H8BreakEvenProtection()

    result = engine().run(
        data,
        entry,
        stop_updates=protection,
    )

    assert protection.activation_count == 0

    trade = result.trades[0]

    assert trade.exit_reason is ExitReason.STOP_LOSS
    assert trade.exit_price == Decimal("95")