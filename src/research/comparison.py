"""Transparent period comparison and cross-period stability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from src.backtest.models import BacktestResult
from src.market.intervals import next_open_time
from src.research.dataset_split import ResearchPeriod


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    period: ResearchPeriod
    start: datetime
    end: datetime
    candles: int
    trades: int
    net_pnl: Decimal
    return_percent: Decimal
    win_rate_percent: Decimal
    profit_factor: Decimal | None
    expectancy: Decimal
    average_win: Decimal
    average_loss: Decimal
    maximum_drawdown_percent: Decimal
    total_fees: Decimal
    market_exposure_percent: Decimal
    benchmark_return_percent: Decimal
    benchmark_maximum_drawdown_percent: Decimal
    low_sample_size: bool


@dataclass(frozen=True, slots=True)
class StabilityDiagnostics:
    development_profitable: bool
    validation_profitable: bool
    out_of_sample_profitable: bool
    development_profit_factor: Decimal | None
    validation_profit_factor: Decimal | None
    out_of_sample_profit_factor: Decimal | None
    development_drawdown_percent: Decimal
    validation_drawdown_percent: Decimal
    out_of_sample_drawdown_percent: Decimal
    oos_to_development_expectancy_ratio: Decimal | None


def comparison_row(
    period: ResearchPeriod,
    baseline: BacktestResult,
    benchmark: BacktestResult,
    *,
    minimum_trades_warning: int,
) -> ComparisonRow:
    metrics = baseline.metrics
    dataset = baseline.dataset
    return ComparisonRow(
        period=period,
        start=dataset.candles[0].timestamp,
        end=next_open_time(dataset.candles[-1].timestamp, dataset.interval),
        candles=len(dataset.candles),
        trades=metrics.total_trades,
        net_pnl=metrics.net_profit,
        return_percent=metrics.total_return_percent,
        win_rate_percent=metrics.win_rate_percent,
        profit_factor=metrics.profit_factor,
        expectancy=metrics.expectancy_per_trade,
        average_win=metrics.average_winning_trade,
        average_loss=metrics.average_losing_trade,
        maximum_drawdown_percent=metrics.maximum_drawdown_percent,
        total_fees=metrics.total_fees_paid,
        market_exposure_percent=metrics.market_exposure_percent,
        benchmark_return_percent=benchmark.metrics.total_return_percent,
        benchmark_maximum_drawdown_percent=(
            benchmark.metrics.maximum_drawdown_percent
        ),
        low_sample_size=metrics.total_trades < minimum_trades_warning,
    )


def stability_diagnostics(
    rows: tuple[ComparisonRow, ComparisonRow, ComparisonRow],
) -> StabilityDiagnostics:
    development, validation, oos = rows
    ratio = (
        oos.expectancy / development.expectancy
        if development.expectancy != 0
        else None
    )
    return StabilityDiagnostics(
        development_profitable=development.net_pnl > 0,
        validation_profitable=validation.net_pnl > 0,
        out_of_sample_profitable=oos.net_pnl > 0,
        development_profit_factor=development.profit_factor,
        validation_profit_factor=validation.profit_factor,
        out_of_sample_profit_factor=oos.profit_factor,
        development_drawdown_percent=development.maximum_drawdown_percent,
        validation_drawdown_percent=validation.maximum_drawdown_percent,
        out_of_sample_drawdown_percent=oos.maximum_drawdown_percent,
        oos_to_development_expectancy_ratio=ratio,
    )

