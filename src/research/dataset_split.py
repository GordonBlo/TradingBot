"""Strict chronological development/validation/out-of-sample splitting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.historical.dataset import HistoricalDataset
from src.market.intervals import next_open_time, require_utc


class ResearchPeriod(str, Enum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    OUT_OF_SAMPLE = "out_of_sample"


@dataclass(frozen=True, slots=True)
class ResearchPeriodRange:
    period: ResearchPeriod
    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.period, ResearchPeriod):
            object.__setattr__(self, "period", ResearchPeriod(self.period))
        object.__setattr__(self, "start", require_utc(self.start, name="start"))
        object.__setattr__(self, "end", require_utc(self.end, name="end"))
        if self.start >= self.end:
            raise ValueError("Research period start must be before end.")


@dataclass(frozen=True, slots=True)
class ResearchSplit:
    development: HistoricalDataset
    validation: HistoricalDataset
    out_of_sample: HistoricalDataset

    def items(self) -> tuple[tuple[ResearchPeriod, HistoricalDataset], ...]:
        return (
            (ResearchPeriod.DEVELOPMENT, self.development),
            (ResearchPeriod.VALIDATION, self.validation),
            (ResearchPeriod.OUT_OF_SAMPLE, self.out_of_sample),
        )


def split_dataset(
    dataset: HistoricalDataset,
    development: ResearchPeriodRange,
    validation: ResearchPeriodRange,
    out_of_sample: ResearchPeriodRange,
) -> ResearchSplit:
    """Split with inclusive starts and exclusive, contiguous ends: ``[start,end)``."""

    expected = (
        ResearchPeriod.DEVELOPMENT,
        ResearchPeriod.VALIDATION,
        ResearchPeriod.OUT_OF_SAMPLE,
    )
    ranges = (development, validation, out_of_sample)
    if tuple(item.period for item in ranges) != expected:
        raise ValueError("Research ranges must be development, validation, then OOS.")
    if development.end != validation.start or validation.end != out_of_sample.start:
        raise ValueError("Research periods must be contiguous and non-overlapping.")

    selected = tuple(dataset.slice(item.start, item.end) for item in ranges)
    for item, period_dataset in zip(ranges, selected, strict=True):
        if not period_dataset.candles:
            raise ValueError(f"{item.period.value} research period has no candles.")
        actual_start = period_dataset.candles[0].timestamp
        actual_end = next_open_time(
            period_dataset.candles[-1].timestamp, period_dataset.interval
        )
        if actual_start != item.start or actual_end != item.end:
            raise ValueError(
                f"Cached data does not fully cover {item.period.value} "
                f"[{item.start.isoformat()}, {item.end.isoformat()})."
            )

    combined = tuple(candle for part in selected for candle in part.candles)
    if len({candle.timestamp for candle in combined}) != len(combined):
        raise ValueError("Research split contains overlapping candles.")
    return ResearchSplit(*selected)

