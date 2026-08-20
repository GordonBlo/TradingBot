"""Exact per-trade counterfactual fee and slippage accounting."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.backtest.execution import basis_points_rate
from src.backtest.models import Trade
from src.diagnostics.distributions import decimal_mean, median
from src.diagnostics.models import CostDiagnostics, TradeDiagnostic


@dataclass(frozen=True, slots=True)
class TradeCostBreakdown:
    reference_entry_price: Decimal
    reference_exit_price: Decimal
    frictionless_pnl: Decimal
    slippage_adjusted_gross_pnl: Decimal
    slippage_drag: Decimal
    fee_drag: Decimal
    net_pnl: Decimal


def decompose_trade_costs(trade: Trade, slippage_bps: Decimal) -> TradeCostBreakdown:
    """Recover reference prices from deterministic adverse V2 fill equations."""

    slippage_rate = basis_points_rate(slippage_bps)
    reference_entry = trade.entry_price / (Decimal("1") + slippage_rate)
    reference_exit = trade.exit_price / (Decimal("1") - slippage_rate)
    frictionless = (reference_exit - reference_entry) * trade.quantity
    calculated_gross = (trade.exit_price - trade.entry_price) * trade.quantity
    if abs(calculated_gross - trade.gross_pnl) > Decimal("1e-24"):
        raise ValueError(f"Recorded accounting is inconsistent for {trade.trade_id}.")
    gross = trade.gross_pnl
    slippage_drag = gross - frictionless
    net = gross - trade.total_fee
    if abs(net - trade.net_pnl) > Decimal("1e-24"):
        raise ValueError(f"Recorded fee accounting is inconsistent for {trade.trade_id}.")
    net = trade.net_pnl
    if slippage_rate > 0 and slippage_drag > 0:
        raise ValueError(f"Adverse slippage drag is positive for {trade.trade_id}.")
    return TradeCostBreakdown(
        reference_entry_price=reference_entry,
        reference_exit_price=reference_exit,
        frictionless_pnl=frictionless,
        slippage_adjusted_gross_pnl=gross,
        slippage_drag=slippage_drag,
        fee_drag=trade.total_fee,
        net_pnl=net,
    )


def aggregate_costs(
    trades: tuple[TradeDiagnostic, ...], initial_capital: Decimal
) -> CostDiagnostics:
    frictionless = sum(
        (trade.frictionless_pnl for trade in trades), start=Decimal("0")
    )
    gross = sum((trade.gross_pnl for trade in trades), start=Decimal("0"))
    slippage = sum((trade.slippage_drag for trade in trades), start=Decimal("0"))
    fees = sum((trade.fee_cost for trade in trades), start=Decimal("0"))
    net = sum((trade.net_pnl for trade in trades), start=Decimal("0"))
    if (
        abs((frictionless + slippage - fees) - net) > Decimal("1e-24")
        or abs((gross - fees) - net) > Decimal("1e-24")
    ):
        raise ValueError("Aggregate cost decomposition is inconsistent.")
    gross_profitable_movement = sum(
        (trade.gross_pnl for trade in trades if trade.gross_pnl > 0),
        start=Decimal("0"),
    )
    return CostDiagnostics(
        frictionless_pnl=frictionless,
        slippage_adjusted_gross_pnl=gross,
        slippage_drag=slippage,
        fee_drag=fees,
        net_pnl=net,
        fees_percent_initial_capital=fees / initial_capital * Decimal("100"),
        average_fee_per_trade=decimal_mean(trade.fee_cost for trade in trades),
        median_fee_per_trade=median(trade.fee_cost for trade in trades),
        fees_percent_gross_profitable_movement=(
            fees / gross_profitable_movement * Decimal("100")
            if gross_profitable_movement > 0
            else None
        ),
        average_slippage_drag_per_trade=decimal_mean(
            trade.slippage_drag for trade in trades
        ),
        slippage_percent_initial_capital=(
            slippage / initial_capital * Decimal("100")
        ),
        slippage_relative_to_frictionless_pnl=(
            slippage / abs(frictionless) * Decimal("100")
            if frictionless != 0
            else None
        ),
    )
