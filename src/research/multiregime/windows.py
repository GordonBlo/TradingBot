"""Fixed, gap-safe chronological window construction with local warm-up only."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from src.historical.dataset import HistoricalDataset
from src.research.multiregime.models import (
    MultiRegimeConfig,
    PartitionKind,
    PartitionStatus,
    ResearchRegion,
    ResearchWindow,
)


def construct_windows(
    regions: tuple[ResearchRegion, ...],
    config: MultiRegimeConfig,
) -> tuple[ResearchWindow, ...]:
    """Build non-overlapping windows independently inside each safe region."""

    if any(region.metadata.kind is PartitionKind.LOCKED_BLIND_HOLDOUT for region in regions):
        raise ValueError("Locked blind-holdout data cannot be supplied to window construction.")
    ordered = sorted(regions, key=lambda item: item.metadata.start)
    for previous, current in zip(ordered, ordered[1:]):
        if previous.metadata.end > current.metadata.start:
            raise ValueError("Research regions overlap.")

    windows: list[ResearchWindow] = []
    window_number = 1
    target = timedelta(days=config.window_days)
    minimum = Decimal(config.minimum_window_days)
    for region in ordered:
        metadata = region.metadata
        if not region.dataset.candles:
            raise ValueError(f"{metadata.kind.value} has no candles.")
        if any(
            not (metadata.start <= candle.timestamp < metadata.end)
            for candle in region.dataset.candles
        ):
            raise ValueError(
                f"{metadata.kind.value} contains candles outside its safe partition."
            )
        cursor = metadata.start
        while cursor < metadata.end:
            end = min(cursor + target, metadata.end)
            evaluation = region.dataset.slice(cursor, end)
            if not evaluation.candles:
                raise ValueError(
                    f"No candles cover research window [{cursor.isoformat()}, {end.isoformat()})."
                )
            first_index = region.dataset.candles.index(evaluation.candles[0])
            warmup_start = max(0, first_index - config.warmup_candles)
            replay_candles = region.dataset.candles[warmup_start : first_index] + evaluation.candles
            replay = HistoricalDataset(
                symbol=region.dataset.symbol,
                interval=region.dataset.interval,
                source=region.dataset.source,
                candles=replay_candles,
            )
            evaluation_start_index = first_index - warmup_start
            duration = Decimal(str((end - cursor).total_seconds())) / Decimal("86400")
            partial = end - cursor < target
            windows.append(
                ResearchWindow(
                    window_id=f"W{window_number:03d}",
                    start=cursor,
                    end=end,
                    duration_days=duration,
                    partition_kind=metadata.kind,
                    data_status=PartitionStatus.CONSUMED_RESEARCH_DATA,
                    partial_window=partial,
                    duration_eligible=duration >= minimum,
                    replay_dataset=replay,
                    evaluation_start_index=evaluation_start_index,
                )
            )
            cursor = end
            window_number += 1
    _validate_windows(tuple(windows), ordered)
    return tuple(windows)


def _validate_windows(
    windows: tuple[ResearchWindow, ...], regions: list[ResearchRegion]
) -> None:
    for window in windows:
        owner = next(
            (
                region
                for region in regions
                if region.metadata.kind is window.partition_kind
                and region.metadata.start <= window.start
                and window.end <= region.metadata.end
            ),
            None,
        )
        if owner is None:
            raise ValueError("Research window crosses a partition gap or boundary.")
        evaluation = window.evaluation_dataset
        if any(not (window.start <= candle.timestamp < window.end) for candle in evaluation.candles):
            raise ValueError("Evaluation candles escape their declared research window.")
        warmup = window.replay_dataset.candles[: window.evaluation_start_index]
        if any(candle.timestamp >= window.start for candle in warmup):
            raise ValueError("Warm-up candles overlap counted evaluation candles.")
