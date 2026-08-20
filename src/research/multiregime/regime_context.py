"""Causal, diagnostic-only market context for V3.2.1 windows."""

from __future__ import annotations

from decimal import Decimal

from src.analysis.indicators import IndicatorEngine
from src.backtest.metrics import maximum_drawdown_percent
from src.diagnostics.distributions import decimal_mean, median
from src.hypotheses.causal import CausalRollingPercentile
from src.market.intervals import next_open_time
from src.research.multiregime.models import (
    CandleRegimeContext,
    DiagnosticVolatilityRegime,
    LongTrendRegime,
    MarketWindowContext,
    MultiRegimeConfig,
    ResearchWindow,
)


def build_regime_context(
    window: ResearchWindow,
    config: MultiRegimeConfig,
) -> tuple[MarketWindowContext, tuple[CandleRegimeContext, ...]]:
    """Calculate contexts from current/prior closes without exposing them to strategies."""

    candles = window.replay_dataset.candles
    series = IndicatorEngine().calculate_research_series(
        candles,
        fast_ema_period=50,
        slow_ema_period=200,
        rsi_period=14,
        atr_period=14,
        volume_sma_period=20,
    )
    low_rolling = CausalRollingPercentile(
        lookback=config.volatility_lookback,
        percentile_rank=config.volatility_low_percentile,
    )
    high_rolling = CausalRollingPercentile(
        lookback=config.volatility_lookback,
        percentile_rank=config.volatility_high_percentile,
    )
    contexts: list[CandleRegimeContext] = []
    atr_values: list[Decimal] = []
    volume_ratios: list[Decimal] = []
    for index, (candle, current) in enumerate(zip(candles, series, strict=True)):
        atr_percent = (
            current.atr / current.close * Decimal("100")
            if current.atr is not None and current.close > 0
            else None
        )
        low_threshold = low_rolling.threshold()
        high_threshold = high_rolling.threshold()
        low_rolling.observe(atr_percent)
        high_rolling.observe(atr_percent)
        if current.ema_fast is None or current.ema_slow is None:
            long_trend = LongTrendRegime.UNAVAILABLE
        elif current.close > current.ema_slow and current.ema_fast > current.ema_slow:
            long_trend = LongTrendRegime.LONG_TREND_UP
        elif current.close < current.ema_slow and current.ema_fast < current.ema_slow:
            long_trend = LongTrendRegime.LONG_TREND_DOWN
        else:
            long_trend = LongTrendRegime.LONG_TREND_MIXED
        if atr_percent is None or low_threshold is None or high_threshold is None:
            volatility = DiagnosticVolatilityRegime.UNAVAILABLE
        elif atr_percent <= low_threshold:
            volatility = DiagnosticVolatilityRegime.LOW_VOLATILITY
        elif atr_percent >= high_threshold:
            volatility = DiagnosticVolatilityRegime.HIGH_VOLATILITY
        else:
            volatility = DiagnosticVolatilityRegime.MEDIUM_VOLATILITY
        if index >= window.evaluation_start_index:
            contexts.append(
                CandleRegimeContext(
                    signal_timestamp=next_open_time(candle.timestamp, candle.interval),
                    long_trend=long_trend,
                    volatility=volatility,
                    atr_percent=atr_percent,
                )
            )
            if atr_percent is not None:
                atr_values.append(atr_percent)
            if current.volume_ratio is not None:
                volume_ratios.append(current.volume_ratio)

    evaluation = window.evaluation_dataset.candles
    closes = tuple(candle.close for candle in evaluation)
    returns = tuple(
        (current / previous - Decimal("1")) * Decimal("100")
        for previous, current in zip(closes, closes[1:])
        if previous > 0
    )
    realized = _population_standard_deviation(returns)
    market = MarketWindowContext(
        btc_return_percent=(closes[-1] / closes[0] - Decimal("1")) * Decimal("100"),
        btc_maximum_drawdown_percent=maximum_drawdown_percent(closes),
        realized_candle_volatility_percent=realized,
        average_atr_percent=(decimal_mean(atr_values) if atr_values else None),
        median_atr_percent=(median(atr_values) if atr_values else None),
        average_volume_ratio=(decimal_mean(volume_ratios) if volume_ratios else None),
    )
    return market, tuple(contexts)


def _population_standard_deviation(values: tuple[Decimal, ...]) -> Decimal:
    if not values:
        return Decimal("0")
    average = decimal_mean(values)
    variance = decimal_mean((value - average) ** 2 for value in values)
    return variance.sqrt()
