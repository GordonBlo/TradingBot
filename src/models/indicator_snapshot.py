"""Typed technical-indicator result for the latest completed candle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class IndicatorSnapshot:
    """Latest market measurements; unavailable warm-up values remain None."""

    timestamp: datetime
    symbol: str
    interval: str
    close: Decimal
    sma_20: Decimal | None
    sma_50: Decimal | None
    ema_20: Decimal | None
    ema_50: Decimal | None
    rsi_14: Decimal | None
    atr_14: Decimal | None
    volume: Decimal
    volume_sma_20: Decimal | None
    volume_ratio: Decimal | None

