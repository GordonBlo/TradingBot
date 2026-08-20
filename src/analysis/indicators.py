"""Precision-preserving technical indicator calculations."""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

from src.models.candle import Candle
from src.models.indicator_snapshot import IndicatorSnapshot
from src.models.research_indicator_snapshot import ResearchIndicatorSnapshot


def _validate_period(period: int) -> None:
    if period < 1:
        raise ValueError("Indicator period must be at least 1.")


def sma(values: Sequence[Decimal], period: int) -> Decimal | None:
    """Return the latest simple moving average for an arbitrary period."""

    _validate_period(period)
    if len(values) < period:
        return None
    return sum(values[-period:], start=Decimal("0")) / Decimal(period)


def ema(values: Sequence[Decimal], period: int) -> Decimal | None:
    """Return an SMA-seeded exponential moving average for an arbitrary period."""

    _validate_period(period)
    if len(values) < period:
        return None
    current = sum(values[:period], start=Decimal("0")) / Decimal(period)
    multiplier = Decimal("2") / Decimal(period + 1)
    for value in values[period:]:
        current = (value - current) * multiplier + current
    return current


def rsi(closes: Sequence[Decimal], period: int) -> Decimal | None:
    """Return Wilder's RSI, using 50 for a completely flat market."""

    _validate_period(period)
    if len(closes) < period + 1:
        return None

    changes = [current - previous for previous, current in zip(closes, closes[1:])]
    initial = changes[:period]
    average_gain = sum(
        (max(change, Decimal("0")) for change in initial),
        start=Decimal("0"),
    ) / Decimal(period)
    average_loss = sum(
        (max(-change, Decimal("0")) for change in initial),
        start=Decimal("0"),
    ) / Decimal(period)

    for change in changes[period:]:
        gain = max(change, Decimal("0"))
        loss = max(-change, Decimal("0"))
        average_gain = (
            average_gain * Decimal(period - 1) + gain
        ) / Decimal(period)
        average_loss = (
            average_loss * Decimal(period - 1) + loss
        ) / Decimal(period)

    if average_gain == 0 and average_loss == 0:
        return Decimal("50")
    if average_loss == 0:
        return Decimal("100")
    if average_gain == 0:
        return Decimal("0")
    relative_strength = average_gain / average_loss
    return Decimal("100") - Decimal("100") / (Decimal("1") + relative_strength)


def atr(candles: Sequence[Candle], period: int) -> Decimal | None:
    """Return Wilder's Average True Range for an arbitrary period."""

    _validate_period(period)
    if len(candles) < period:
        return None

    true_ranges: list[Decimal] = []
    previous_close: Decimal | None = None
    for candle in candles:
        if previous_close is None:
            true_range = candle.high - candle.low
        else:
            true_range = max(
                candle.high - candle.low,
                abs(candle.high - previous_close),
                abs(candle.low - previous_close),
            )
        true_ranges.append(true_range)
        previous_close = candle.close

    current = sum(true_ranges[:period], start=Decimal("0")) / Decimal(period)
    for true_range in true_ranges[period:]:
        current = (
            current * Decimal(period - 1) + true_range
        ) / Decimal(period)
    return current


class IndicatorEngine:
    """Calculate V1 measurements for the latest completed candle."""

    def calculate(self, candles: Sequence[Candle]) -> IndicatorSnapshot | None:
        """Build a typed snapshot from completed candles only."""

        completed_by_timestamp = {
            candle.timestamp: candle for candle in candles if candle.is_closed
        }
        completed = [
            completed_by_timestamp[timestamp]
            for timestamp in sorted(completed_by_timestamp)
        ]
        if not completed:
            return None

        latest = completed[-1]
        if any(
            candle.symbol != latest.symbol or candle.interval != latest.interval
            for candle in completed
        ):
            raise ValueError("Indicator input must contain one symbol and interval.")

        closes = [candle.close for candle in completed]
        volumes = [candle.volume for candle in completed]
        volume_average = sma(volumes, 20)
        volume_ratio = (
            latest.volume / volume_average
            if volume_average is not None and volume_average != 0
            else None
        )
        return IndicatorSnapshot(
            timestamp=latest.timestamp,
            symbol=latest.symbol,
            interval=latest.interval,
            close=latest.close,
            sma_20=sma(closes, 20),
            sma_50=sma(closes, 50),
            ema_20=ema(closes, 20),
            ema_50=ema(closes, 50),
            rsi_14=rsi(closes, 14),
            atr_14=atr(completed, 14),
            volume=latest.volume,
            volume_sma_20=volume_average,
            volume_ratio=volume_ratio,
        )

    def calculate_research_series(
        self,
        candles: Sequence[Candle],
        *,
        fast_ema_period: int = 20,
        slow_ema_period: int = 50,
        rsi_period: int = 14,
        atr_period: int = 14,
        volume_sma_period: int = 20,
    ) -> tuple[ResearchIndicatorSnapshot, ...]:
        """Calculate an O(n), past-only snapshot for every closed candle."""

        for period in (
            fast_ema_period,
            slow_ema_period,
            rsi_period,
            atr_period,
            volume_sma_period,
        ):
            _validate_period(period)
        if not candles:
            return ()
        first = candles[0]
        if any(
            not candle.is_closed
            or candle.symbol != first.symbol
            or candle.interval != first.interval
            for candle in candles
        ):
            raise ValueError(
                "Research indicators require closed candles for one symbol/interval."
            )

        fast_state = _EmaState(fast_ema_period)
        slow_state = _EmaState(slow_ema_period)
        rsi_state = _RsiState(rsi_period)
        atr_state = _AtrState(atr_period)
        volume_state = _SmaState(volume_sma_period)
        snapshots: list[ResearchIndicatorSnapshot] = []
        previous_close: Decimal | None = None

        for candle in candles:
            fast_value = fast_state.update(candle.close)
            slow_value = slow_state.update(candle.close)
            rsi_value = rsi_state.update(candle.close)
            atr_value = atr_state.update(candle, previous_close)
            volume_average = volume_state.update(candle.volume)
            volume_ratio = (
                candle.volume / volume_average
                if volume_average is not None and volume_average != 0
                else None
            )
            snapshots.append(
                ResearchIndicatorSnapshot(
                    timestamp=candle.timestamp,
                    symbol=candle.symbol,
                    interval=candle.interval,
                    close=candle.close,
                    volume=candle.volume,
                    ema_fast=fast_value,
                    ema_slow=slow_value,
                    rsi=rsi_value,
                    atr=atr_value,
                    volume_sma=volume_average,
                    volume_ratio=volume_ratio,
                )
            )
            previous_close = candle.close
        return tuple(snapshots)


class _SmaState:
    def __init__(self, period: int) -> None:
        self.period = period
        self.values: list[Decimal] = []
        self.total = Decimal("0")

    def update(self, value: Decimal) -> Decimal | None:
        self.values.append(value)
        self.total += value
        if len(self.values) > self.period:
            self.total -= self.values[-self.period - 1]
        if len(self.values) < self.period:
            return None
        return self.total / Decimal(self.period)


class _EmaState:
    def __init__(self, period: int) -> None:
        self.period = period
        self.seed_total = Decimal("0")
        self.count = 0
        self.current: Decimal | None = None
        self.multiplier = Decimal("2") / Decimal(period + 1)

    def update(self, value: Decimal) -> Decimal | None:
        self.count += 1
        if self.current is None:
            self.seed_total += value
            if self.count < self.period:
                return None
            self.current = self.seed_total / Decimal(self.period)
            return self.current
        self.current = (value - self.current) * self.multiplier + self.current
        return self.current


class _RsiState:
    def __init__(self, period: int) -> None:
        self.period = period
        self.previous: Decimal | None = None
        self.change_count = 0
        self.gain_total = Decimal("0")
        self.loss_total = Decimal("0")
        self.average_gain: Decimal | None = None
        self.average_loss: Decimal | None = None

    def update(self, close: Decimal) -> Decimal | None:
        if self.previous is None:
            self.previous = close
            return None
        change = close - self.previous
        self.previous = close
        gain = max(change, Decimal("0"))
        loss = max(-change, Decimal("0"))
        self.change_count += 1
        if self.average_gain is None or self.average_loss is None:
            self.gain_total += gain
            self.loss_total += loss
            if self.change_count < self.period:
                return None
            self.average_gain = self.gain_total / Decimal(self.period)
            self.average_loss = self.loss_total / Decimal(self.period)
        else:
            self.average_gain = (
                self.average_gain * Decimal(self.period - 1) + gain
            ) / Decimal(self.period)
            self.average_loss = (
                self.average_loss * Decimal(self.period - 1) + loss
            ) / Decimal(self.period)
        return _rsi_from_averages(self.average_gain, self.average_loss)


def _rsi_from_averages(gain: Decimal, loss: Decimal) -> Decimal:
    if gain == 0 and loss == 0:
        return Decimal("50")
    if loss == 0:
        return Decimal("100")
    if gain == 0:
        return Decimal("0")
    relative_strength = gain / loss
    return Decimal("100") - Decimal("100") / (
        Decimal("1") + relative_strength
    )


class _AtrState:
    def __init__(self, period: int) -> None:
        self.period = period
        self.count = 0
        self.seed_total = Decimal("0")
        self.current: Decimal | None = None

    def update(
        self, candle: Candle, previous_close: Decimal | None
    ) -> Decimal | None:
        true_range = candle.high - candle.low
        if previous_close is not None:
            true_range = max(
                true_range,
                abs(candle.high - previous_close),
                abs(candle.low - previous_close),
            )
        self.count += 1
        if self.current is None:
            self.seed_total += true_range
            if self.count < self.period:
                return None
            self.current = self.seed_total / Decimal(self.period)
            return self.current
        self.current = (
            self.current * Decimal(self.period - 1) + true_range
        ) / Decimal(self.period)
        return self.current
