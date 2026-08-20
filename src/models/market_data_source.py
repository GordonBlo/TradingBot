"""Typed provenance for analytical market data."""

from __future__ import annotations

from enum import Enum


class MarketDataSource(str, Enum):
    """Supported source of price, candle, volume, and exchange metadata."""

    BINANCE_PUBLIC = "binance_public"

    @property
    def display_name(self) -> str:
        """Return an operator-friendly, unambiguous source name."""

        return "BINANCE PUBLIC SPOT"
