"""Rate-limit-aware chronological Binance public kline pagination."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, TypeVar

from src.exchange.historical_data import HistoricalDataError, HistoricalDataService
from src.exchange.public_market_client import PublicMarketDataError
from src.market.intervals import next_open_time, require_utc, to_unix_milliseconds
from src.models.candle import Candle
from src.models.market_data_source import MarketDataSource
from src.utils.logger import get_logger


class RangeMarketDataClient(Protocol):
    @property
    def source(self) -> MarketDataSource: ...

    def get_server_time(self) -> int: ...

    def get_klines(
        self,
        symbol: str,
        interval: str,
        *,
        limit: int,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> list[list[object]]: ...


@dataclass(frozen=True, slots=True)
class DownloadProgress:
    request_count: int
    candle_count: int
    last_timestamp: datetime | None


@dataclass(frozen=True, slots=True)
class HistoricalDownloadResult:
    candles: tuple[Candle, ...]
    request_count: int


ProgressCallback = Callable[[DownloadProgress], None]
T = TypeVar("T")


class HistoricalRangeDownloader:
    """Download an inclusive-start/exclusive-end closed-candle range."""

    def __init__(
        self,
        client: RangeMarketDataClient,
        *,
        page_limit: int = 1_000,
        max_retries: int = 3,
        base_backoff_seconds: float = 1.0,
        sleeper: Callable[[float], None] = time.sleep,
        logger: logging.Logger | None = None,
    ) -> None:
        if not 1 <= page_limit <= 1_000:
            raise ValueError("Historical page limit must be between 1 and 1000.")
        if max_retries < 0 or base_backoff_seconds < 0:
            raise ValueError("Retry count and backoff must not be negative.")
        if client.source is not MarketDataSource.BINANCE_PUBLIC:
            raise ValueError("Historical downloads require Binance public Spot data.")
        self._client = client
        self.page_limit = page_limit
        self.max_retries = max_retries
        self.base_backoff_seconds = base_backoff_seconds
        self._sleep = sleeper
        self._logger = logger or get_logger(__name__)

    @property
    def source(self) -> MarketDataSource:
        return self._client.source

    def _with_retries(self, operation_name: str, operation: Callable[[], T]) -> T:
        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except PublicMarketDataError as exc:
                if attempt >= self.max_retries:
                    raise HistoricalDataError(
                        f"{operation_name} failed after "
                        f"{self.max_retries + 1} attempts."
                    ) from exc
                delay = (
                    exc.retry_after_seconds
                    if exc.status_code == 429 and exc.retry_after_seconds is not None
                    else self.base_backoff_seconds * (2**attempt)
                )
                self._logger.warning(
                    "%s failed%s; retrying in %.2f seconds "
                    "(%d/%d)",
                    operation_name,
                    " with HTTP 429" if exc.status_code == 429 else "",
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                self._sleep(delay)
        raise AssertionError("Unreachable retry state.")

    def _request_page(
        self,
        symbol: str,
        interval: str,
        *,
        start_time_ms: int,
        end_time_ms: int,
    ) -> list[list[object]]:
        return self._with_retries(
            "Historical page request",
            lambda: self._client.get_klines(
                symbol,
                interval,
                limit=self.page_limit,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
            ),
        )

    def download(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        *,
        progress: ProgressCallback | None = None,
    ) -> HistoricalDownloadResult:
        start = require_utc(start, name="start")
        end = require_utc(end, name="end")
        if start >= end:
            raise ValueError("Historical download start must be before end.")

        symbol = symbol.strip().upper()
        server_time_ms = self._with_retries(
            "Public Binance server-time request", self._client.get_server_time
        )
        cursor = start
        end_time_ms = to_unix_milliseconds(end) - 1
        request_count = 0
        candles_by_timestamp: dict[datetime, Candle] = {}

        while cursor < end:
            cursor_ms = to_unix_milliseconds(cursor)
            rows = self._request_page(
                symbol,
                interval,
                start_time_ms=cursor_ms,
                end_time_ms=end_time_ms,
            )
            request_count += 1
            if not rows:
                break

            unique_rows: dict[int, list[object]] = {}
            for row in rows:
                try:
                    open_time_ms = int(row[0])
                    normalized_row = list(row)
                except (IndexError, TypeError, ValueError) as exc:
                    raise HistoricalDataError(
                        "Historical page contains a malformed row."
                    ) from exc
                previous_row = unique_rows.get(open_time_ms)
                if previous_row is not None and previous_row != normalized_row:
                    raise HistoricalDataError(
                        "Binance returned conflicting rows for open timestamp "
                        f"{open_time_ms}."
                    )
                unique_rows[open_time_ms] = normalized_row

            page = HistoricalDataService.convert_rows(
                list(unique_rows.values()),
                symbol=symbol,
                interval=interval,
                server_time_ms=server_time_ms,
                closed_only=True,
            )
            page = [candle for candle in page if start <= candle.timestamp < end]
            for candle in page:
                previous = candles_by_timestamp.get(candle.timestamp)
                if previous is not None and previous != candle:
                    raise HistoricalDataError(
                        "Binance returned conflicting candles for timestamp "
                        f"{candle.timestamp.isoformat()}."
                    )
                candles_by_timestamp[candle.timestamp] = candle

            try:
                last_open_ms = max(int(row[0]) for row in rows)
                last_open = datetime.fromtimestamp(last_open_ms / 1_000, tz=start.tzinfo)
            except (IndexError, TypeError, ValueError, OSError) as exc:
                raise HistoricalDataError(
                    "Historical page did not contain a valid open timestamp."
                ) from exc
            next_cursor = next_open_time(last_open, interval)
            if next_cursor <= cursor:
                raise HistoricalDataError(
                    "Historical pagination made no forward progress."
                )
            cursor = next_cursor

            if progress is not None:
                progress(
                    DownloadProgress(
                        request_count=request_count,
                        candle_count=len(candles_by_timestamp),
                        last_timestamp=max(candles_by_timestamp, default=None),
                    )
                )
            if len(rows) < self.page_limit:
                break

        ordered = tuple(
            candles_by_timestamp[timestamp]
            for timestamp in sorted(candles_by_timestamp)
        )
        return HistoricalDownloadResult(ordered, request_count)
