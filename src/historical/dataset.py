"""Immutable historical dataset and date-slicing model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.market.intervals import require_utc
from src.models.candle import Candle
from src.models.market_data_source import MarketDataSource


@dataclass(frozen=True, slots=True)
class HistoricalDataset:
    """Chronological closed candles from one explicit market source."""

    symbol: str
    interval: str
    source: MarketDataSource
    candles: tuple[Candle, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "interval", self.interval.strip())
        object.__setattr__(self, "candles", tuple(self.candles))
        if not self.symbol or not self.interval:
            raise ValueError("Historical dataset symbol and interval are required.")

    @property
    def first_timestamp(self) -> datetime | None:
        return self.candles[0].timestamp if self.candles else None

    @property
    def last_timestamp(self) -> datetime | None:
        return self.candles[-1].timestamp if self.candles else None

    def slice(self, start: datetime, end: datetime) -> HistoricalDataset:
        """Return an inclusive-start/exclusive-end chronological date slice."""

        start = require_utc(start, name="start")
        end = require_utc(end, name="end")
        if start >= end:
            raise ValueError("Historical dataset start must be before end.")
        selected = tuple(
            candle for candle in self.candles if start <= candle.timestamp < end
        )
        return HistoricalDataset(
            symbol=self.symbol,
            interval=self.interval,
            source=self.source,
            candles=selected,
        )
