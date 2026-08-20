"""Past-only input exposed to a research strategy."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from src.models.candle import Candle
from src.models.research_indicator_snapshot import ResearchIndicatorSnapshot


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """Causal strategy state at the close of ``current_candle``."""

    timestamp: datetime
    current_candle: Candle
    recent_history: Sequence[Candle]
    indicators: ResearchIndicatorSnapshot
    previous_indicators: ResearchIndicatorSnapshot | None
    has_position: bool
    bars_in_position: int
    equity: Decimal
    cash_usdc: Decimal
    completed_trade_count: int
    bars_since_exit: int | None
    entry_fee_rate: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("Strategy timestamp must be timezone-aware.")
        if not self.current_candle.is_closed:
            raise ValueError("Strategy current candle must be closed.")
        if any(
            candle.timestamp > self.current_candle.timestamp
            for candle in self.recent_history
        ):
            raise ValueError("Strategy history cannot contain future candles.")
        if not self.recent_history or self.recent_history[-1] != self.current_candle:
            raise ValueError("Strategy history must end at the current candle.")
        if self.indicators.timestamp != self.current_candle.timestamp:
            raise ValueError("Current indicators must match the current candle.")
        if (
            self.previous_indicators is not None
            and self.previous_indicators.timestamp >= self.indicators.timestamp
        ):
            raise ValueError("Previous indicators must precede current indicators.")
        for name in ("equity", "cash_usdc", "entry_fee_rate"):
            value = Decimal(str(getattr(self, name)))
            if not value.is_finite() or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, value)
        if self.bars_in_position < 0 or self.completed_trade_count < 0:
            raise ValueError("Strategy account counts must not be negative.")
        if self.bars_since_exit is not None and self.bars_since_exit < 0:
            raise ValueError("bars_since_exit must not be negative.")
