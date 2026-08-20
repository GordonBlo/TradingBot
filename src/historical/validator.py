"""Strict validation for deterministic historical candle datasets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Sequence

from src.market.intervals import is_interval_aligned, next_open_time
from src.models.candle import Candle


class DatasetValidationError(ValueError):
    """Base error for historical dataset integrity failures."""


class HistoricalDataGapError(DatasetValidationError):
    """Raised when an expected candle-open timestamp is missing."""


class InvalidCandleError(DatasetValidationError):
    """Raised when a candle violates closed Spot OHLCV invariants."""


@dataclass(frozen=True, slots=True)
class DatasetValidationResult:
    candle_count: int
    first_timestamp: datetime
    last_timestamp: datetime
    gaps: tuple[tuple[datetime, datetime], ...]


class DatasetValidator:
    """Reject corrupt, mixed, open, misaligned, or discontinuous data."""

    def validate(
        self,
        candles: Sequence[Candle],
        *,
        symbol: str | None = None,
        interval: str | None = None,
        allow_gaps: bool = False,
    ) -> DatasetValidationResult:
        if not candles:
            raise DatasetValidationError("Historical dataset contains no candles.")

        expected_symbol = (symbol or candles[0].symbol).strip().upper()
        expected_interval = (interval or candles[0].interval).strip()
        previous: Candle | None = None
        seen: set[datetime] = set()
        gaps: list[tuple[datetime, datetime]] = []

        for index, candle in enumerate(candles):
            self._validate_candle(
                candle,
                index=index,
                symbol=expected_symbol,
                interval=expected_interval,
            )
            if candle.timestamp in seen:
                raise DatasetValidationError(
                    f"Duplicate candle timestamp at {candle.timestamp.isoformat()}."
                )
            seen.add(candle.timestamp)

            if previous is not None:
                if candle.timestamp < previous.timestamp:
                    raise DatasetValidationError(
                        "Historical candles are not in chronological order at "
                        f"{candle.timestamp.isoformat()}."
                    )
                expected_timestamp = next_open_time(
                    previous.timestamp, expected_interval
                )
                if candle.timestamp != expected_timestamp:
                    gaps.append((expected_timestamp, candle.timestamp))
                    if not allow_gaps:
                        raise HistoricalDataGapError(
                            "Historical data gap: expected candle at "
                            f"{expected_timestamp.isoformat()}, next candle is "
                            f"{candle.timestamp.isoformat()}."
                        )
            previous = candle

        return DatasetValidationResult(
            candle_count=len(candles),
            first_timestamp=candles[0].timestamp,
            last_timestamp=candles[-1].timestamp,
            gaps=tuple(gaps),
        )

    @staticmethod
    def _validate_candle(
        candle: Candle, *, index: int, symbol: str, interval: str
    ) -> None:
        if candle.timestamp.tzinfo is None or candle.timestamp.utcoffset() != timedelta(0):
            raise InvalidCandleError(f"Candle {index} timestamp must use UTC.")
        if not is_interval_aligned(candle.timestamp, interval):
            raise InvalidCandleError(
                f"Candle {index} timestamp is not aligned to {interval}."
            )
        if candle.symbol != symbol or candle.interval != interval:
            raise InvalidCandleError(
                f"Candle {index} does not match dataset symbol/interval."
            )
        if not candle.is_closed:
            raise InvalidCandleError(f"Candle {index} is not closed.")

        prices = (candle.open, candle.high, candle.low, candle.close)
        if any(price <= Decimal("0") for price in prices):
            raise InvalidCandleError(f"Candle {index} contains a non-positive price.")
        if candle.volume < Decimal("0"):
            raise InvalidCandleError(f"Candle {index} contains negative volume.")
        if candle.high < candle.low:
            raise InvalidCandleError(f"Candle {index} high is below low.")
        if candle.high < candle.open:
            raise InvalidCandleError(f"Candle {index} high is below open.")
        if candle.high < candle.close:
            raise InvalidCandleError(f"Candle {index} high is below close.")
        if candle.low > candle.open:
            raise InvalidCandleError(f"Candle {index} low is above open.")
        if candle.low > candle.close:
            raise InvalidCandleError(f"Candle {index} low is above close.")
