"""Performance statistics calculated from trades and the equity curve."""

from __future__ import annotations

from decimal import Decimal
from typing import Sequence

from src.backtest.models import EquityPoint, PerformanceMetrics, Trade


def maximum_drawdown_percent(equity_values: Sequence[Decimal]) -> Decimal:
    """Return peak-to-trough drawdown as a positive percentage magnitude."""

    if not equity_values:
        return Decimal("0")
    peak = Decimal(str(equity_values[0]))
    if peak <= 0:
        raise ValueError("Equity values must be positive for drawdown calculation.")
    maximum = Decimal("0")
    for raw_value in equity_values:
        value = Decimal(str(raw_value))
        if value <= 0:
            raise ValueError("Equity values must be positive for drawdown calculation.")
        peak = max(peak, value)
        maximum = max(maximum, (peak - value) / peak * Decimal("100"))
    return maximum


def _maximum_streak(trades: Sequence[Trade], *, winning: bool) -> int:
    maximum = 0
    current = 0
    for trade in trades:
        matches = trade.net_pnl > 0 if winning else trade.net_pnl < 0
        current = current + 1 if matches else 0
        maximum = max(maximum, current)
    return maximum


def calculate_metrics(
    *,
    initial_capital: Decimal,
    trades: Sequence[Trade],
    equity_curve: Sequence[EquityPoint],
    exposed_bars: int,
    total_bars: int,
) -> PerformanceMetrics:
    initial_capital = Decimal(str(initial_capital))
    final_equity = (
        equity_curve[-1].total_equity if equity_curve else initial_capital
    )
    net_profit = final_equity - initial_capital
    wins = [trade for trade in trades if trade.net_pnl > 0]
    losses = [trade for trade in trades if trade.net_pnl < 0]
    breakeven = [trade for trade in trades if trade.net_pnl == 0]
    gross_profit = sum((trade.net_pnl for trade in wins), Decimal("0"))
    gross_loss = sum((trade.net_pnl for trade in losses), Decimal("0"))
    average_win = gross_profit / len(wins) if wins else Decimal("0")
    average_loss = gross_loss / len(losses) if losses else Decimal("0")
    total_trades = len(trades)
    payoff_ratio = (
        average_win / abs(average_loss) if wins and losses else None
    )
    profit_factor = (
        gross_profit / abs(gross_loss) if gross_loss != 0 else None
    )
    expectancy = (
        sum((trade.net_pnl for trade in trades), Decimal("0")) / total_trades
        if total_trades
        else Decimal("0")
    )
    average_return = (
        sum((trade.return_percent for trade in trades), Decimal("0"))
        / total_trades
        if total_trades
        else Decimal("0")
    )
    return PerformanceMetrics(
        initial_capital=initial_capital,
        final_equity=final_equity,
        net_profit=net_profit,
        total_return_percent=net_profit / initial_capital * Decimal("100"),
        total_trades=total_trades,
        winning_trades=len(wins),
        losing_trades=len(losses),
        breakeven_trades=len(breakeven),
        win_rate_percent=(
            Decimal(len(wins)) / Decimal(total_trades) * Decimal("100")
            if total_trades
            else Decimal("0")
        ),
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        average_winning_trade=average_win,
        average_losing_trade=average_loss,
        largest_winning_trade=(
            max((trade.net_pnl for trade in wins), default=Decimal("0"))
        ),
        largest_losing_trade=(
            min((trade.net_pnl for trade in losses), default=Decimal("0"))
        ),
        payoff_ratio=payoff_ratio,
        profit_factor=profit_factor,
        expectancy_per_trade=expectancy,
        average_return_percent_per_trade=average_return,
        maximum_drawdown_percent=maximum_drawdown_percent(
            [point.total_equity for point in equity_curve]
        ),
        maximum_consecutive_wins=_maximum_streak(trades, winning=True),
        maximum_consecutive_losses=_maximum_streak(trades, winning=False),
        total_fees_paid=sum((trade.total_fee for trade in trades), Decimal("0")),
        market_exposure_percent=(
            Decimal(exposed_bars) / Decimal(total_bars) * Decimal("100")
            if total_bars
            else Decimal("0")
        ),
    )
