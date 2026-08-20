"""Incremental historical-cache update orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from src.exchange.historical_data import HistoricalDataError
from src.historical.dataset import HistoricalDataset
from src.historical.downloader import HistoricalRangeDownloader, ProgressCallback
from src.historical.storage import HistoricalDatasetStore
from src.historical.validator import DatasetValidator
from src.market.intervals import next_open_time, require_utc


@dataclass(frozen=True, slots=True)
class HistoricalUpdateResult:
    dataset: HistoricalDataset
    requested: HistoricalDataset
    downloaded_candle_count: int
    request_count: int


class HistoricalDatasetManager:
    """Load, extend, merge, validate, and persist one continuous cache."""

    def __init__(
        self,
        downloader: HistoricalRangeDownloader,
        store: HistoricalDatasetStore,
        *,
        validator: DatasetValidator | None = None,
        allow_source_gaps: bool = False,
    ) -> None:
        self._downloader = downloader
        self._store = store
        self._validator = validator or DatasetValidator()
        self._allow_source_gaps = allow_source_gaps

    def update(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        *,
        progress: ProgressCallback | None = None,
    ) -> HistoricalUpdateResult:
        start = require_utc(start, name="start")
        end = require_utc(end, name="end")
        if start >= end:
            raise ValueError("Historical update start must be before end.")
        symbol = symbol.strip().upper()
        cached = self._store.load(symbol, interval)

        missing_ranges: list[tuple[datetime, datetime]] = []
        if cached is None:
            missing_ranges.append((start, end))
            existing = ()
        else:
            existing = cached.candles
            first = cached.candles[0].timestamp
            coverage_end = next_open_time(cached.candles[-1].timestamp, interval)
            if start < first:
                missing_ranges.append((start, first))
            if end > coverage_end:
                missing_ranges.append((coverage_end, end))

        downloaded = []
        request_count = 0
        for range_start, range_end in missing_ranges:
            if range_start >= range_end:
                continue
            result = self._downloader.download(
                symbol,
                interval,
                range_start,
                range_end,
                progress=progress,
            )
            downloaded.extend(result.candles)
            request_count += result.request_count

        merged_by_timestamp = {candle.timestamp: candle for candle in existing}
        for candle in downloaded:
            previous = merged_by_timestamp.get(candle.timestamp)
            if previous is not None and previous != candle:
                raise HistoricalDataError(
                    "Downloaded candle conflicts with cached timestamp "
                    f"{candle.timestamp.isoformat()}."
                )
            merged_by_timestamp[candle.timestamp] = candle
        merged = tuple(
            merged_by_timestamp[timestamp] for timestamp in sorted(merged_by_timestamp)
        )
        if not merged:
            raise HistoricalDataError("No closed candles exist in the requested range.")

        dataset = HistoricalDataset(
            symbol=symbol,
            interval=interval,
            source=self._downloader.source,
            candles=merged,
        )
        self._validator.validate(
            merged,
            symbol=symbol,
            interval=interval,
            allow_gaps=self._allow_source_gaps,
        )
        if cached is None or downloaded:
            self._store.save(dataset)

        requested = dataset.slice(start, end)
        if not requested.candles:
            raise HistoricalDataError("No cached candles exist in the requested range.")
        self._validator.validate(
            requested.candles,
            symbol=symbol,
            interval=interval,
            allow_gaps=self._allow_source_gaps,
        )
        return HistoricalUpdateResult(
            dataset=dataset,
            requested=requested,
            downloaded_candle_count=len(downloaded),
            request_count=request_count,
        )
