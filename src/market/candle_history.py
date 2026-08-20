"""Bounded chronological storage for completed market candles."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from src.models.candle import Candle


class CandleHistory:
    """Maintain unique, completed candles for one symbol and interval."""

    def __init__(
        self,
        symbol: str,
        interval: str,
        *,
        max_size: int,
        candles: Iterable[Candle] = (),
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size must be at least 1.")
        self.symbol = symbol.upper()
        self.interval = interval
        self.max_size = max_size
        self._by_timestamp: dict[datetime, Candle] = {}
        for candle in candles:
            self._validate(candle)
            self._by_timestamp[candle.timestamp] = candle
        self._trim()

    def _validate(self, candle: Candle) -> None:
        if not candle.is_closed:
            raise ValueError("CandleHistory accepts completed candles only.")
        if candle.symbol != self.symbol or candle.interval != self.interval:
            raise ValueError("Candle does not match this history's symbol and interval.")

    def _trim(self) -> None:
        while len(self._by_timestamp) > self.max_size:
            del self._by_timestamp[min(self._by_timestamp)]

    def upsert(self, candle: Candle) -> bool:
        """Insert or replace a candle and report whether history changed."""

        self._validate(candle)
        previous = self._by_timestamp.get(candle.timestamp)
        if previous == candle:
            return False
        self._by_timestamp[candle.timestamp] = candle
        self._trim()
        return True

    @property
    def candles(self) -> tuple[Candle, ...]:
        """Return an immutable chronological view of the retained candles."""

        return tuple(
            self._by_timestamp[timestamp]
            for timestamp in sorted(self._by_timestamp)
        )

    @property
    def latest(self) -> Candle | None:
        """Return the latest completed candle, if any."""

        candles = self.candles
        return candles[-1] if candles else None

    def __len__(self) -> int:
        return len(self._by_timestamp)

