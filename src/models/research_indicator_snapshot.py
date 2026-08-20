"""Typed causal indicator values used by strategy research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class ResearchIndicatorSnapshot:
    """Indicators calculated using candles no later than ``timestamp``."""

    timestamp: datetime
    symbol: str
    interval: str
    close: Decimal
    volume: Decimal
    ema_fast: Decimal | None
    ema_slow: Decimal | None
    rsi: Decimal | None
    atr: Decimal | None
    volume_sma: Decimal | None
    volume_ratio: Decimal | None

    @property
    def is_ready(self) -> bool:
        return all(
            value is not None
            for value in (
                self.ema_fast,
                self.ema_slow,
                self.rsi,
                self.atr,
                self.volume_ratio,
            )
        )
