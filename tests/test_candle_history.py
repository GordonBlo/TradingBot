from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.market.candle_history import CandleHistory
from src.models.candle import Candle

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def candle(
    minute: int,
    *,
    close: str | None = None,
    is_closed: bool = True,
    symbol: str = "BTCUSDC",
) -> Candle:
    close_value = Decimal(close or str(minute + 10))
    return Candle(
        timestamp=BASE_TIME + timedelta(minutes=minute),
        symbol=symbol,
        interval="15m",
        open=close_value,
        high=close_value + Decimal("1"),
        low=close_value - Decimal("1"),
        close=close_value,
        volume=Decimal("5"),
        is_closed=is_closed,
    )


def test_history_initializes_chronologically() -> None:
    history = CandleHistory(
        "BTCUSDC",
        "15m",
        max_size=5,
        candles=[candle(2), candle(0), candle(1)],
    )

    assert [item.timestamp for item in history.candles] == [
        candle(0).timestamp,
        candle(1).timestamp,
        candle(2).timestamp,
    ]
    assert history.latest == candle(2)


def test_history_appends_and_updates_duplicate_timestamp() -> None:
    original = candle(0, close="10")
    updated = candle(0, close="10.5")
    history = CandleHistory("BTCUSDC", "15m", max_size=5, candles=[original])

    assert history.upsert(candle(1)) is True
    assert history.upsert(updated) is True
    assert history.upsert(updated) is False
    assert len(history) == 2
    assert history.candles[0].close == Decimal("10.5")


def test_history_enforces_maximum_size() -> None:
    history = CandleHistory(
        "BTCUSDC",
        "15m",
        max_size=3,
        candles=[candle(index) for index in range(5)],
    )

    assert len(history) == 3
    assert [item.timestamp for item in history.candles] == [
        candle(2).timestamp,
        candle(3).timestamp,
        candle(4).timestamp,
    ]


def test_history_rejects_open_or_mismatched_candles() -> None:
    history = CandleHistory("BTCUSDC", "15m", max_size=3)

    with pytest.raises(ValueError, match="completed"):
        history.upsert(candle(0, is_closed=False))
    with pytest.raises(ValueError, match="symbol and interval"):
        history.upsert(candle(0, symbol="ETHUSDC"))

