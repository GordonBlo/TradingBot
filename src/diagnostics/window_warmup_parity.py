"""Diagnostic-only finite-window versus continuous indicator boundary parity."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.analysis.indicators import IndicatorEngine
from src.research.multiregime.models import ResearchRegion, ResearchWindow
from src.strategy.models import TrendMomentumConfig


@dataclass(frozen=True, slots=True)
class IndicatorBoundaryParity:
    window_id: str
    field: str
    local_value: Decimal | None
    continuous_value: Decimal | None
    absolute_difference: Decimal | None
    relative_difference_percent: Decimal | None


def compare_indicator_boundary(
    window: ResearchWindow,
    region: ResearchRegion,
    strategy_config: TrendMomentumConfig,
) -> tuple[IndicatorBoundaryParity, ...]:
    """Compare state at one boundary; never feeds results back into replay."""

    boundary = window.evaluation_dataset.candles[0].timestamp
    continuous_index = next(
        index
        for index, candle in enumerate(region.dataset.candles)
        if candle.timestamp == boundary
    )
    arguments = {
        "fast_ema_period": strategy_config.fast_ema_period,
        "slow_ema_period": strategy_config.slow_ema_period,
        "rsi_period": strategy_config.rsi_period,
        "atr_period": strategy_config.atr_period,
        "volume_sma_period": strategy_config.volume_sma_period,
    }
    engine = IndicatorEngine()
    local = engine.calculate_research_series(
        window.replay_dataset.candles, **arguments
    )[window.evaluation_start_index]
    continuous = engine.calculate_research_series(
        region.dataset.candles[: continuous_index + 1], **arguments
    )[-1]
    rows = []
    for field in (
        "ema_fast",
        "ema_slow",
        "rsi",
        "atr",
        "volume_sma",
        "volume_ratio",
    ):
        local_value = getattr(local, field)
        continuous_value = getattr(continuous, field)
        difference = (
            abs(local_value - continuous_value)
            if local_value is not None and continuous_value is not None
            else None
        )
        relative = (
            difference / abs(continuous_value) * Decimal("100")
            if difference is not None and continuous_value not in (None, 0)
            else None
        )
        rows.append(
            IndicatorBoundaryParity(
                window_id=window.window_id,
                field=field,
                local_value=local_value,
                continuous_value=continuous_value,
                absolute_difference=difference,
                relative_difference_percent=relative,
            )
        )
    return tuple(rows)
