"""Aggregate post-trade favorable/adverse excursion evidence."""

from __future__ import annotations

from decimal import Decimal

from src.diagnostics.distributions import (
    decimal_mean,
    median,
    optional_mean,
    optional_median,
    summarize_group,
)
from src.diagnostics.models import ExcursionDiagnostics, TradeDiagnostic


def _timing(
    trades: tuple[TradeDiagnostic, ...], attribute: str
) -> tuple[Decimal, Decimal]:
    values = tuple(Decimal(getattr(trade, attribute)) for trade in trades)
    return decimal_mean(values), median(values)


def excursion_diagnostics(
    trades: tuple[TradeDiagnostic, ...], minimum: int
) -> ExcursionDiagnostics:
    winners = tuple(trade for trade in trades if trade.net_pnl > 0)
    losers = tuple(trade for trade in trades if trade.net_pnl < 0)
    all_mfe = _timing(trades, "bars_to_mfe")
    winner_mfe = _timing(winners, "bars_to_mfe")
    loser_mfe = _timing(losers, "bars_to_mfe")
    all_mae = _timing(trades, "bars_to_mae")
    winner_mae = _timing(winners, "bars_to_mae")
    loser_mae = _timing(losers, "bars_to_mae")

    mfe_thresholds = tuple(
        summarize_group(
            "mfe_threshold",
            f">={threshold}R",
            (
                trade
                for trade in trades
                if trade.mfe_r is not None and trade.mfe_r >= threshold
            ),
            total_trades=len(trades),
            minimum_trades_warning=minimum,
        )
        for threshold in map(Decimal, ("0.5", "1", "1.5", "2", "3"))
    )
    mae_thresholds = tuple(
        summarize_group(
            "mae_threshold",
            f">={threshold}R",
            (
                trade
                for trade in trades
                if trade.mae_r is not None and trade.mae_r >= threshold
            ),
            total_trades=len(trades),
            minimum_trades_warning=minimum,
        )
        for threshold in map(Decimal, ("0.5", "1"))
    )
    first_four_half = tuple(
        trade
        for trade in losers
        if trade.first_four_bar_mae_r is not None
        and trade.first_four_bar_mae_r >= Decimal("0.5")
    )
    first_four_one = tuple(
        trade
        for trade in losers
        if trade.first_four_bar_mae_r is not None
        and trade.first_four_bar_mae_r >= Decimal("1")
    )
    return ExcursionDiagnostics(
        median_mfe_r=optional_median(trade.mfe_r for trade in trades),
        median_mae_r=optional_median(trade.mae_r for trade in trades),
        average_bars_to_mfe_all=all_mfe[0],
        median_bars_to_mfe_all=all_mfe[1],
        average_bars_to_mfe_winners=winner_mfe[0],
        median_bars_to_mfe_winners=winner_mfe[1],
        average_bars_to_mfe_losers=loser_mfe[0],
        median_bars_to_mfe_losers=loser_mfe[1],
        average_bars_to_mae_all=all_mae[0],
        median_bars_to_mae_all=all_mae[1],
        average_bars_to_mae_winners=winner_mae[0],
        median_bars_to_mae_winners=winner_mae[1],
        average_bars_to_mae_losers=loser_mae[0],
        median_bars_to_mae_losers=loser_mae[1],
        mean_profitable_capture_ratio=optional_mean(
            trade.capture_ratio for trade in winners
        ),
        median_profitable_capture_ratio=optional_median(
            trade.capture_ratio for trade in winners
        ),
        mean_losing_loss_efficiency=optional_mean(
            trade.loss_efficiency for trade in losers
        ),
        losing_mae_half_r_first_four_percent=(
            Decimal(len(first_four_half)) / Decimal(len(losers)) * Decimal("100")
            if losers
            else None
        ),
        losing_mae_one_r_first_four_percent=(
            Decimal(len(first_four_one)) / Decimal(len(losers)) * Decimal("100")
            if losers
            else None
        ),
        mfe_thresholds=mfe_thresholds,
        mae_thresholds=mae_thresholds,
    )

