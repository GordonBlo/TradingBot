"""Async Binance kline stream that publishes domain Candle objects."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from websockets.asyncio.client import connect

from src.models.candle import Candle
from src.models.market_data_source import MarketDataSource
from src.utils.logger import get_logger

CandleHandler = Callable[[Candle], Awaitable[None] | None]


class MarketStream:
    """Subscribe to public Binance candles and publish domain Candle objects."""

    WEBSOCKET_BASE_URL = "wss://data-stream.binance.vision"

    def __init__(
        self,
        symbol: str,
        interval: str,
        *,
        reconnect_delay_seconds: float = 2.0,
    ) -> None:
        self.symbol = symbol.upper()
        self.interval = interval
        self.reconnect_delay_seconds = reconnect_delay_seconds
        self._handlers: list[CandleHandler] = []
        self._logger = get_logger(__name__)

    @property
    def source(self) -> MarketDataSource:
        return MarketDataSource.BINANCE_PUBLIC

    @property
    def stream_url(self) -> str:
        stream_name = f"{self.symbol.lower()}@kline_{self.interval}"
        return f"{self.WEBSOCKET_BASE_URL}/ws/{stream_name}"

    def subscribe(self, handler: CandleHandler) -> Callable[[], None]:
        """Register a candle consumer and return a function that unsubscribes it."""

        self._handlers.append(handler)

        def unsubscribe() -> None:
            if handler in self._handlers:
                self._handlers.remove(handler)

        return unsubscribe

    @staticmethod
    def parse_message(message: str | bytes) -> Candle | None:
        """Translate a Binance kline message into the exchange-neutral model."""

        if isinstance(message, bytes):
            message = message.decode("utf-8")
        payload: dict[str, Any] = json.loads(message)
        if isinstance(payload.get("data"), dict):
            payload = payload["data"]
        kline = payload.get("k")
        if not isinstance(kline, dict):
            return None

        return Candle(
            timestamp=datetime.fromtimestamp(
                int(kline["t"]) / 1_000, tz=timezone.utc
            ),
            symbol=str(kline["s"]),
            interval=str(kline["i"]),
            open=Decimal(str(kline["o"])),
            high=Decimal(str(kline["h"])),
            low=Decimal(str(kline["l"])),
            close=Decimal(str(kline["c"])),
            volume=Decimal(str(kline["v"])),
            is_closed=bool(kline["x"]),
        )

    async def _publish(self, candle: Candle) -> None:
        for handler in tuple(self._handlers):
            try:
                result = handler(candle)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:
                self._logger.error(
                    "Candle consumer failed (%s); stream will continue",
                    type(exc).__name__,
                )

    def _log_candle(self, candle: Candle) -> None:
        log_method = self._logger.info if candle.is_closed else self._logger.debug
        state = "closed" if candle.is_closed else "partial update"
        log_method(
            "%s %s candle %s | timestamp: %s | O: %s H: %s L: %s C: %s "
            "V: %s | candle_closed: %s",
            candle.symbol,
            candle.interval,
            state,
            candle.timestamp.isoformat(),
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
            candle.is_closed,
        )

    async def _wait_for_reconnect(self, stop_event: asyncio.Event) -> None:
        try:
            await asyncio.wait_for(
                stop_event.wait(), timeout=self.reconnect_delay_seconds
            )
        except TimeoutError:
            pass

    async def run(self, stop_event: asyncio.Event | None = None) -> None:
        """Consume candles until stopped, reconnecting after transient failures."""

        stop_event = stop_event or asyncio.Event()
        while not stop_event.is_set():
            try:
                async with connect(
                    self.stream_url,
                    open_timeout=10,
                    close_timeout=5,
                    ping_interval=20,
                    ping_timeout=20,
                    max_queue=32,
                ) as websocket:
                    self._logger.info(
                        "Connected to Binance PUBLIC WebSocket for %s %s candles",
                        self.symbol,
                        self.interval,
                    )
                    while not stop_event.is_set():
                        try:
                            message = await asyncio.wait_for(
                                websocket.recv(), timeout=1.0
                            )
                        except TimeoutError:
                            continue

                        try:
                            candle = self.parse_message(message)
                        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                            self._logger.warning(
                                "Ignored malformed Binance market message (%s)",
                                type(exc).__name__,
                            )
                            continue
                        if candle is None:
                            continue
                        self._log_candle(candle)
                        await self._publish(candle)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if stop_event.is_set():
                    break
                self._logger.warning(
                    "Binance WebSocket disconnected (%s); reconnecting in %.1f seconds",
                    type(exc).__name__,
                    self.reconnect_delay_seconds,
                )
                await self._wait_for_reconnect(stop_event)
