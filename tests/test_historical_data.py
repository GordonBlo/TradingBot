from datetime import timezone
from decimal import Decimal

import pytest

from src.exchange.historical_data import HistoricalDataError, HistoricalDataService


def raw_kline(
    open_time: int,
    *,
    close_time: int,
    close: str = "2.0",
    high: str = "3.0",
    low: str = "0.5",
) -> list[object]:
    return [
        open_time,
        "1.0",
        high,
        low,
        close,
        "10.25",
        close_time,
        "0",
        0,
        "0",
        "0",
        "0",
    ]


def test_kline_conversion_orders_and_deduplicates_rows() -> None:
    rows = [
        raw_kline(2_000, close_time=2_999, close="2.2"),
        raw_kline(0, close_time=999, close="1.5"),
        raw_kline(1_000, close_time=1_999, close="2.0"),
        raw_kline(1_000, close_time=1_999, close="2.5"),
    ]

    candles = HistoricalDataService.convert_rows(
        rows,
        symbol="BTCUSDC",
        interval="15m",
        server_time_ms=5_000,
    )

    assert [int(candle.timestamp.timestamp() * 1_000) for candle in candles] == [
        0,
        1_000,
        2_000,
    ]
    assert candles[1].close == Decimal("2.5")
    assert candles[0].timestamp.tzinfo == timezone.utc
    assert all(candle.is_closed for candle in candles)


def test_kline_conversion_filters_open_candle_by_default() -> None:
    rows = [
        raw_kline(0, close_time=999),
        raw_kline(1_000, close_time=1_999),
    ]

    closed = HistoricalDataService.convert_rows(
        rows,
        symbol="BTCUSDC",
        interval="15m",
        server_time_ms=1_500,
    )
    all_candles = HistoricalDataService.convert_rows(
        rows,
        symbol="BTCUSDC",
        interval="15m",
        server_time_ms=1_500,
        closed_only=False,
    )

    assert len(closed) == 1
    assert closed[0].is_closed is True
    assert len(all_candles) == 2
    assert all_candles[1].is_closed is False


@pytest.mark.parametrize(
    "row",
    (
        [0, "1.0"],
        raw_kline(0, close_time=999, high="not-a-number"),
        raw_kline(0, close_time=999, high="0.25", low="0.5"),
    ),
)
def test_kline_conversion_rejects_malformed_rows(row: list[object]) -> None:
    with pytest.raises(HistoricalDataError):
        HistoricalDataService.convert_rows(
            [row],
            symbol="BTCUSDC",
            interval="15m",
            server_time_ms=2_000,
        )


def test_historical_service_fetches_extra_row_and_returns_requested_closed_limit() -> None:
    class FakeClient:
        requested_limit: int | None = None

        def get_server_time(self) -> int:
            return 2_500

        def get_klines(
            self, symbol: str, interval: str, *, limit: int
        ) -> list[list[object]]:
            assert symbol == "BTCUSDC"
            assert interval == "15m"
            self.requested_limit = limit
            return [
                raw_kline(0, close_time=999, close="1.0"),
                raw_kline(1_000, close_time=1_999, close="2.0"),
                raw_kline(2_000, close_time=2_999, close="3.0"),
            ]

    client = FakeClient()
    service = HistoricalDataService(client)

    candles = service.load_recent_candles("BTCUSDC", "15m", limit=2)

    assert client.requested_limit == 3
    assert [candle.close for candle in candles] == [Decimal("1.0"), Decimal("2.0")]

