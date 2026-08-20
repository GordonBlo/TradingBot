"""Historical Binance kline loading and Candle conversion."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Protocol

from src.models.candle import Candle
from src.models.market_data_source import MarketDataSource


class HistoricalDataError(RuntimeError):
    """Raised when historical market data is missing or malformed."""


class HistoricalMarketDataClient(Protocol):
    """Read-only client operations required by the historical service."""

    @property
    def source(self) -> MarketDataSource: ...

    def get_server_time(self) -> int: ...

    def get_klines(
        self, symbol: str, interval: str, *, limit: int
    ) -> list[list[object]]: ...


class HistoricalDataService:
    """Load, validate, order, and de-duplicate recent Binance candles."""

    def __init__(self, client: HistoricalMarketDataClient) -> None:
        self._client = client

    @property
    def source(self) -> MarketDataSource:
        """Expose the provenance of every candle loaded by this service."""

        return self._client.source

    @staticmethod
    def _convert_row(
        row: Sequence[object],
        *,
        row_index: int,
        symbol: str,
        interval: str,
        server_time_ms: int,
    ) -> Candle:
        if len(row) < 7:
            raise HistoricalDataError(
                f"Historical kline row {row_index} has fewer than 7 fields."
            )
        try:
            open_time_ms = int(row[0])
            close_time_ms = int(row[6])
            open_price = Decimal(str(row[1]))
            high_price = Decimal(str(row[2]))
            low_price = Decimal(str(row[3]))
            close_price = Decimal(str(row[4]))
            volume = Decimal(str(row[5]))
        except (TypeError, ValueError, InvalidOperation) as exc:
            raise HistoricalDataError(
                f"Historical kline row {row_index} contains invalid values."
            ) from exc

        if open_time_ms < 0 or close_time_ms < open_time_ms:
            raise HistoricalDataError(
                f"Historical kline row {row_index} contains invalid timestamps."
            )
        try:
            return Candle(
                timestamp=datetime.fromtimestamp(
                    open_time_ms / 1_000, tz=timezone.utc
                ),
                symbol=symbol,
                interval=interval,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                is_closed=close_time_ms < server_time_ms,
            )
        except (OverflowError, OSError, ValueError) as exc:
            raise HistoricalDataError(
                f"Historical kline row {row_index} failed candle validation."
            ) from exc

    @classmethod
    def convert_rows(
        cls,
        rows: Sequence[Sequence[object]],
        *,
        symbol: str,
        interval: str,
        server_time_ms: int,
        closed_only: bool = True,
    ) -> list[Candle]:
        """Convert raw kline rows into chronological, unique Candle objects."""

        if server_time_ms < 0:
            raise HistoricalDataError("Binance server time must not be negative.")

        unique: dict[datetime, Candle] = {}
        for row_index, row in enumerate(rows):
            if not isinstance(row, Sequence) or isinstance(
                row, (str, bytes, bytearray)
            ):
                raise HistoricalDataError(
                    f"Historical kline row {row_index} is not a sequence."
                )
            candle = cls._convert_row(
                row,
                row_index=row_index,
                symbol=symbol,
                interval=interval,
                server_time_ms=server_time_ms,
            )
            if closed_only and not candle.is_closed:
                continue
            unique[candle.timestamp] = candle
        return [unique[timestamp] for timestamp in sorted(unique)]

    def load_recent_candles(
        self,
        symbol: str,
        interval: str,
        *,
        limit: int,
        closed_only: bool = True,
    ) -> list[Candle]:
        """Fetch recent candles, including one extra row for open-candle filtering."""

        if not 1 <= limit <= 1_000:
            raise ValueError("Historical candle limit must be between 1 and 1000.")
        server_time_ms = self._client.get_server_time()
        request_limit = min(limit + 1, 1_000) if closed_only else limit
        rows = self._client.get_klines(symbol, interval, limit=request_limit)
        candles = self.convert_rows(
            rows,
            symbol=symbol,
            interval=interval,
            server_time_ms=server_time_ms,
            closed_only=closed_only,
        )
        candles = candles[-limit:]
        if not candles:
            state = "closed " if closed_only else ""
            raise HistoricalDataError(f"Binance returned no usable {state}candles.")
        return candles
