"""Market snapshot orchestration and readable presentation."""

from __future__ import annotations

from decimal import Decimal

from src.analysis.indicators import IndicatorEngine
from src.market.candle_history import CandleHistory
from src.models.candle import Candle
from src.models.indicator_snapshot import IndicatorSnapshot


class MarketAnalyzer:
    """Combine completed candle history with typed technical measurements."""

    def __init__(self, indicator_engine: IndicatorEngine | None = None) -> None:
        self._indicator_engine = indicator_engine or IndicatorEngine()

    def analyze(self, history: CandleHistory) -> IndicatorSnapshot | None:
        """Calculate a snapshot for the latest candle in history."""

        return self._indicator_engine.calculate(history.candles)

    def update_with_closed_candle(
        self, history: CandleHistory, candle: Candle
    ) -> IndicatorSnapshot | None:
        """Update history and recalculate only for a new or changed closed candle."""

        if not candle.is_closed:
            return None
        if not history.upsert(candle):
            return None
        return self.analyze(history)

    @staticmethod
    def _number(value: Decimal | None, suffix: str = "") -> str:
        if value is None:
            return "N/A"
        return f"{value:,.2f}{suffix}"

    def format_snapshot(self, snapshot: IndicatorSnapshot) -> str:
        """Render a compact, human-readable market measurement snapshot."""

        return "\n".join(
            (
                f"{snapshot.symbol} | {snapshot.interval} | CLOSED | "
                f"{snapshot.timestamp.isoformat()}",
                f"Price: {self._number(snapshot.close)}",
                f"SMA20: {self._number(snapshot.sma_20)} | "
                f"SMA50: {self._number(snapshot.sma_50)}",
                f"EMA20: {self._number(snapshot.ema_20)} | "
                f"EMA50: {self._number(snapshot.ema_50)}",
                f"RSI14: {self._number(snapshot.rsi_14)} | "
                f"ATR14: {self._number(snapshot.atr_14)}",
                f"Volume: {self._number(snapshot.volume)} | "
                f"Volume SMA20: {self._number(snapshot.volume_sma_20)} | "
                f"Volume Ratio: {self._number(snapshot.volume_ratio, 'x')}",
            )
        )
