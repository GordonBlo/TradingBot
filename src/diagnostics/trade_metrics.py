"""Build exact per-trade diagnostics from V2 trades, V3 signals, and OHLC."""

from __future__ import annotations

from decimal import Decimal

from src.analysis.indicators import IndicatorEngine
from src.diagnostics.cost_analysis import decompose_trade_costs
from src.diagnostics.excursions import calculate_excursions
from src.diagnostics.models import (
    DiagnosticPeriodInput,
    MarketRegime,
    TradeDiagnostic,
    VolatilityRegime,
)
from src.research.evaluation import SignalRecord
from src.strategy.models import StrategyAction, TrendMomentumConfig


def _percentile(values: list[Decimal], fraction: Decimal) -> Decimal:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * Decimal(len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - Decimal(lower)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _interval_minutes(interval: str) -> int:
    if interval.endswith("m") and interval != "1M":
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    if interval.endswith("d"):
        return int(interval[:-1]) * 24 * 60
    if interval == "1w":
        return 7 * 24 * 60
    raise ValueError(f"Holding minutes are unavailable for interval {interval}.")


def _market_regime(signal: SignalRecord | None) -> MarketRegime:
    if (
        signal is None
        or signal.ema_fast is None
        or signal.ema_slow is None
    ):
        return MarketRegime.UNAVAILABLE
    if signal.close > signal.ema_slow and signal.ema_fast > signal.ema_slow:
        return MarketRegime.TREND_UP
    if signal.close < signal.ema_slow and signal.ema_fast < signal.ema_slow:
        return MarketRegime.TREND_DOWN
    return MarketRegime.RANGE_OR_MIXED


def _volatility_regimes(
    period: DiagnosticPeriodInput,
    strategy_config: TrendMomentumConfig,
) -> dict[object, VolatilityRegime]:
    series = IndicatorEngine().calculate_research_series(
        period.dataset.candles,
        fast_ema_period=strategy_config.fast_ema_period,
        slow_ema_period=strategy_config.slow_ema_period,
        rsi_period=strategy_config.rsi_period,
        atr_period=strategy_config.atr_period,
        volume_sma_period=strategy_config.volume_sma_period,
    )
    timestamp_index = {
        candle.timestamp: index for index, candle in enumerate(period.dataset.candles)
    }
    entry_indices = {
        trade.entry_signal_time: timestamp_index[trade.entry_time] - 1
        for trade in period.trades
        if trade.entry_time in timestamp_index and timestamp_index[trade.entry_time] > 0
    }
    regimes: dict[object, VolatilityRegime] = {}
    for timestamp, signal_index in entry_indices.items():
        values = [
            item.atr / item.close * Decimal("100")
            for item in series[max(0, signal_index - 99) : signal_index + 1]
            if item.atr is not None and item.close > 0
        ]
        current = series[signal_index]
        if current.atr is None or len(values) < 3:
            regimes[timestamp] = VolatilityRegime.UNAVAILABLE
            continue
        current_percent = current.atr / current.close * Decimal("100")
        low = _percentile(values, Decimal("0.3333333333333333333333333333"))
        high = _percentile(values, Decimal("0.6666666666666666666666666667"))
        regimes[timestamp] = (
            VolatilityRegime.LOW
            if current_percent <= low
            else VolatilityRegime.HIGH
            if current_percent >= high
            else VolatilityRegime.MEDIUM
        )
    return regimes


def build_trade_diagnostics(
    period: DiagnosticPeriodInput,
    *,
    strategy_config: TrendMomentumConfig,
    slippage_bps: Decimal,
) -> tuple[TradeDiagnostic, ...]:
    timestamp_index = {
        candle.timestamp: index for index, candle in enumerate(period.dataset.candles)
    }
    entry_signals = {
        signal.timestamp: signal
        for signal in period.signals
        if signal.action is StrategyAction.ENTER_LONG
    }
    volatility = _volatility_regimes(period, strategy_config)
    diagnostics: list[TradeDiagnostic] = []
    interval_minutes = _interval_minutes(period.dataset.interval)

    for trade in period.trades:
        signal = entry_signals.get(trade.entry_signal_time)
        stop_distance = (
            signal.atr * strategy_config.atr_stop_multiplier
            if signal is not None and signal.atr is not None
            else None
        )
        initial_stop = (
            trade.entry_price - stop_distance if stop_distance is not None else None
        )
        initial_risk_amount = (
            stop_distance * trade.quantity if stop_distance is not None else None
        )
        costs = decompose_trade_costs(trade, slippage_bps)
        excursions = calculate_excursions(
            trade,
            period.dataset,
            timestamp_index,
            initial_risk_per_unit=stop_distance,
        )
        entry_atr_percent = (
            signal.atr / signal.close * Decimal("100")
            if signal is not None and signal.atr is not None and signal.close > 0
            else None
        )
        ema_spread = (
            (signal.ema_fast - signal.ema_slow)
            / signal.ema_slow
            * Decimal("100")
            if signal is not None
            and signal.ema_fast is not None
            and signal.ema_slow is not None
            and signal.ema_slow != 0
            else None
        )
        close_spread = (
            (signal.close - signal.ema_slow)
            / signal.ema_slow
            * Decimal("100")
            if signal is not None
            and signal.ema_slow is not None
            and signal.ema_slow != 0
            else None
        )
        valid_risk = initial_risk_amount is not None and initial_risk_amount > 0
        opportunity_value = excursions.mfe * trade.quantity
        adverse_value = excursions.mae * trade.quantity
        diagnostics.append(
            TradeDiagnostic(
                trade_id=trade.trade_id,
                entry_signal_time=trade.entry_signal_time,
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                reference_entry_price=costs.reference_entry_price,
                reference_exit_price=costs.reference_exit_price,
                quantity=trade.quantity,
                exit_reason=trade.exit_reason,
                frictionless_pnl=costs.frictionless_pnl,
                gross_pnl=trade.gross_pnl,
                slippage_drag=costs.slippage_drag,
                fee_cost=trade.total_fee,
                net_pnl=trade.net_pnl,
                return_percent=trade.return_percent,
                initial_stop_price=initial_stop,
                initial_risk_per_unit=stop_distance,
                initial_risk_amount=initial_risk_amount,
                r_multiple=(trade.net_pnl / initial_risk_amount if valid_risk else None),
                bars_held=trade.bars_held,
                holding_minutes=trade.bars_held * interval_minutes,
                highest_observed_price=excursions.highest_observed_price,
                lowest_observed_price=excursions.lowest_observed_price,
                mfe=excursions.mfe,
                mae=excursions.mae,
                mfe_percent=excursions.mfe_percent,
                mae_percent=excursions.mae_percent,
                mfe_r=excursions.mfe_r,
                mae_r=excursions.mae_r,
                bars_to_mfe=excursions.bars_to_mfe,
                bars_to_mae=excursions.bars_to_mae,
                first_four_bar_mae_r=excursions.first_four_bar_mae_r,
                capture_ratio=(
                    trade.net_pnl / opportunity_value
                    if trade.net_pnl > 0 and opportunity_value > 0
                    else None
                ),
                loss_efficiency=(
                    abs(trade.net_pnl) / adverse_value
                    if trade.net_pnl < 0 and adverse_value > 0
                    else None
                ),
                entry_close=signal.close if signal is not None else None,
                entry_rsi=signal.rsi if signal is not None else None,
                entry_atr=signal.atr if signal is not None else None,
                entry_atr_percent=entry_atr_percent,
                entry_volume_ratio=(
                    signal.volume_ratio if signal is not None else None
                ),
                entry_ema_fast=signal.ema_fast if signal is not None else None,
                entry_ema_slow=signal.ema_slow if signal is not None else None,
                entry_ema_spread_percent=ema_spread,
                close_vs_slow_ema_percent=close_spread,
                market_regime=_market_regime(signal),
                volatility_regime=volatility.get(
                    trade.entry_signal_time, VolatilityRegime.UNAVAILABLE
                ),
                entry_utc_hour=trade.entry_signal_time.hour,
                entry_day_of_week=trade.entry_signal_time.strftime("%A"),
                diagnostic_status=(
                    "COMPLETE" if signal is not None and valid_risk else "R_UNAVAILABLE"
                ),
            )
        )
    return tuple(diagnostics)
