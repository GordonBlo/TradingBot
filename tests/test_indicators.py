from datetime import datetime, timedelta, timezone
from decimal import Decimal

from src.analysis.indicators import IndicatorEngine, atr, ema, rsi, sma
from src.analysis.market_analyzer import MarketAnalyzer
from src.market.candle_history import CandleHistory
from src.models.candle import Candle

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)


def market_candle(
    index: int,
    *,
    close: Decimal | None = None,
    high: Decimal | None = None,
    low: Decimal | None = None,
    volume: Decimal | None = None,
    is_closed: bool = True,
) -> Candle:
    close_value = close if close is not None else Decimal(index + 1)
    return Candle(
        timestamp=BASE_TIME + timedelta(minutes=15 * index),
        symbol="BTCUSDC",
        interval="15m",
        open=close_value,
        high=high if high is not None else close_value + Decimal("1"),
        low=low if low is not None else close_value - Decimal("1"),
        close=close_value,
        volume=volume if volume is not None else Decimal(index + 1),
        is_closed=is_closed,
    )


def test_sma_uses_known_trailing_sequence() -> None:
    values = [Decimal(value) for value in (1, 2, 3, 4, 5)]

    assert sma(values, 3) == Decimal("4")
    assert sma(values, 6) is None


def test_ema_is_sma_seeded_and_deterministic() -> None:
    values = [Decimal(value) for value in (1, 2, 3, 4, 5)]

    assert ema(values, 3) == Decimal("4")
    assert ema(values, 6) is None


def test_rsi_handles_rising_falling_flat_and_warmup_markets() -> None:
    rising = [Decimal(value) for value in range(1, 16)]
    falling = list(reversed(rising))
    flat = [Decimal("10")] * 15

    assert rsi(rising, 14) == Decimal("100")
    assert rsi(falling, 14) == Decimal("0")
    assert rsi(flat, 14) == Decimal("50")
    assert rsi(rising[:14], 14) is None


def test_atr_uses_true_range_and_wilder_smoothing() -> None:
    candles = [
        market_candle(0, close=Decimal("9"), high=Decimal("10"), low=Decimal("8")),
        market_candle(1, close=Decimal("11"), high=Decimal("12"), low=Decimal("9")),
        market_candle(2, close=Decimal("12"), high=Decimal("13"), low=Decimal("10")),
        market_candle(3, close=Decimal("14"), high=Decimal("15"), low=Decimal("11")),
    ]

    assert atr(candles[:2], 3) is None
    assert atr(candles[:3], 3) == Decimal("8") / Decimal("3")
    assert atr(candles, 3) == Decimal("28") / Decimal("9")


def test_indicator_snapshot_calculates_volume_average_and_ratio() -> None:
    candles = [market_candle(index) for index in range(20)]

    snapshot = IndicatorEngine().calculate(candles)

    assert snapshot is not None
    assert snapshot.sma_20 == Decimal("10.5")
    assert snapshot.sma_50 is None
    assert snapshot.ema_20 == Decimal("10.5")
    assert snapshot.ema_50 is None
    assert snapshot.rsi_14 == Decimal("100")
    assert snapshot.atr_14 == Decimal("2")
    assert snapshot.volume_sma_20 == Decimal("10.5")
    assert snapshot.volume_ratio == Decimal("20") / Decimal("10.5")


def test_indicator_snapshot_keeps_warmup_values_explicitly_unavailable() -> None:
    snapshot = IndicatorEngine().calculate([market_candle(0)])

    assert snapshot is not None
    assert snapshot.close == Decimal("1")
    assert snapshot.volume == Decimal("1")
    assert snapshot.sma_20 is None
    assert snapshot.sma_50 is None
    assert snapshot.ema_20 is None
    assert snapshot.ema_50 is None
    assert snapshot.rsi_14 is None
    assert snapshot.atr_14 is None
    assert snapshot.volume_sma_20 is None
    assert snapshot.volume_ratio is None


def test_market_analyzer_updates_only_for_changed_closed_candles() -> None:
    first = market_candle(0)
    second_open = market_candle(1, is_closed=False)
    second_closed = market_candle(1)
    history = CandleHistory("BTCUSDC", "15m", max_size=50, candles=[first])
    analyzer = MarketAnalyzer()

    assert analyzer.update_with_closed_candle(history, first) is None
    assert analyzer.update_with_closed_candle(history, second_open) is None
    assert len(history) == 1

    snapshot = analyzer.update_with_closed_candle(history, second_closed)

    assert snapshot is not None
    assert snapshot.timestamp == second_closed.timestamp
    assert len(history) == 2
    assert analyzer.update_with_closed_candle(history, second_closed) is None


def test_market_snapshot_formatter_represents_unavailable_values_as_na() -> None:
    analyzer = MarketAnalyzer()
    history = CandleHistory(
        "BTCUSDC", "15m", max_size=10, candles=[market_candle(0)]
    )
    snapshot = analyzer.analyze(history)

    assert snapshot is not None
    rendered = analyzer.format_snapshot(snapshot)
    assert "BTCUSDC | 15m | CLOSED" in rendered
    assert "SMA20: N/A" in rendered
    assert "Volume Ratio: N/A" in rendered
