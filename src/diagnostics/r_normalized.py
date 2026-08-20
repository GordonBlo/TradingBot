"""Risk-normalized post-trade performance diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.backtest.models import Trade
from src.diagnostics.cost_analysis import decompose_trade_costs


@dataclass(frozen=True, slots=True)
class RNormalizedTrade:
    trade_id: str

    initial_risk: Decimal

    frictionless_r: Decimal
    gross_after_slippage_r: Decimal

    fee_cost_r: Decimal
    slippage_cost_r: Decimal
    total_friction_r: Decimal

    net_r: Decimal


@dataclass(frozen=True, slots=True)
class RNormalizedSummary:
    trade_count: int

    winning_trades: int
    losing_trades: int
    breakeven_trades: int

    win_rate_percent: Decimal

    frictionless_expectancy_r: Decimal
    gross_after_slippage_expectancy_r: Decimal
    net_expectancy_r: Decimal

    average_fee_r: Decimal
    average_slippage_cost_r: Decimal
    average_total_friction_r: Decimal

    average_winner_r: Decimal
    average_loser_r: Decimal

    payoff_ratio_r: Decimal | None
    profit_factor_r: Decimal | None

    break_even_win_rate_percent: Decimal | None


def _mean(values: list[Decimal]) -> Decimal:
    if not values:
        return Decimal("0")

    return sum(values, start=Decimal("0")) / Decimal(len(values))


def build_r_normalized_trades(
    *,
    trades: tuple[Trade, ...],
    actual_stop_risk_by_trade_id: dict[str, Decimal],
    slippage_bps: Decimal,
) -> tuple[RNormalizedTrade, ...]:
    """Convert completed trades into initial-risk-normalized R metrics."""

    output: list[RNormalizedTrade] = []

    for trade in trades:
        try:
            initial_risk = actual_stop_risk_by_trade_id[trade.trade_id]
        except KeyError as exc:
            raise ValueError(
                f"Missing initial stop-risk for trade {trade.trade_id}."
            ) from exc

        if initial_risk <= 0:
            raise ValueError(
                f"Initial stop-risk must be positive for trade {trade.trade_id}."
            )

        costs = decompose_trade_costs(
            trade,
            slippage_bps,
        )

        frictionless_r = (
            costs.frictionless_pnl / initial_risk
        )

        gross_after_slippage_r = (
            trade.gross_pnl / initial_risk
        )

        fee_cost_r = (
            trade.total_fee / initial_risk
        )

        slippage_cost_r = (
            abs(costs.slippage_drag) / initial_risk
        )

        total_friction_r = (
            fee_cost_r + slippage_cost_r
        )

        net_r = (
            trade.net_pnl / initial_risk
        )

        # Accounting identity:
        # frictionless result - slippage cost - fee cost = net result.
        reconstructed_net_r = (
            frictionless_r
            - slippage_cost_r
            - fee_cost_r
        )

        if abs(reconstructed_net_r - net_r) > Decimal("1e-20"):
            raise ValueError(
                f"R accounting mismatch for trade {trade.trade_id}."
            )

        output.append(
            RNormalizedTrade(
                trade_id=trade.trade_id,
                initial_risk=initial_risk,
                frictionless_r=frictionless_r,
                gross_after_slippage_r=gross_after_slippage_r,
                fee_cost_r=fee_cost_r,
                slippage_cost_r=slippage_cost_r,
                total_friction_r=total_friction_r,
                net_r=net_r,
            )
        )

    return tuple(output)


def summarize_r_normalized_trades(
    records: tuple[RNormalizedTrade, ...],
) -> RNormalizedSummary:
    """Aggregate R-normalized trade performance."""

    if not records:
        return RNormalizedSummary(
            trade_count=0,
            winning_trades=0,
            losing_trades=0,
            breakeven_trades=0,
            win_rate_percent=Decimal("0"),
            frictionless_expectancy_r=Decimal("0"),
            gross_after_slippage_expectancy_r=Decimal("0"),
            net_expectancy_r=Decimal("0"),
            average_fee_r=Decimal("0"),
            average_slippage_cost_r=Decimal("0"),
            average_total_friction_r=Decimal("0"),
            average_winner_r=Decimal("0"),
            average_loser_r=Decimal("0"),
            payoff_ratio_r=None,
            profit_factor_r=None,
            break_even_win_rate_percent=None,
        )

    winners = [
        record.net_r
        for record in records
        if record.net_r > 0
    ]

    losers = [
        record.net_r
        for record in records
        if record.net_r < 0
    ]

    breakeven = [
        record.net_r
        for record in records
        if record.net_r == 0
    ]

    trade_count = len(records)

    average_winner = _mean(winners)
    average_loser = _mean(losers)

    payoff_ratio = (
        average_winner / abs(average_loser)
        if winners and losers
        else None
    )

    gross_winning_r = sum(
        winners,
        start=Decimal("0"),
    )

    gross_losing_r = abs(
        sum(
            losers,
            start=Decimal("0"),
        )
    )

    profit_factor = (
        gross_winning_r / gross_losing_r
        if gross_losing_r > 0
        else None
    )

    break_even_win_rate = None

    if average_winner > 0 and average_loser < 0:
        average_loss_magnitude = abs(average_loser)

        break_even_win_rate = (
            average_loss_magnitude
            /
            (
                average_winner
                + average_loss_magnitude
            )
            * Decimal("100")
        )

    return RNormalizedSummary(
        trade_count=trade_count,
        winning_trades=len(winners),
        losing_trades=len(losers),
        breakeven_trades=len(breakeven),
        win_rate_percent=(
            Decimal(len(winners))
            / Decimal(trade_count)
            * Decimal("100")
        ),
        frictionless_expectancy_r=_mean(
            [
                record.frictionless_r
                for record in records
            ]
        ),
        gross_after_slippage_expectancy_r=_mean(
            [
                record.gross_after_slippage_r
                for record in records
            ]
        ),
        net_expectancy_r=_mean(
            [
                record.net_r
                for record in records
            ]
        ),
        average_fee_r=_mean(
            [
                record.fee_cost_r
                for record in records
            ]
        ),
        average_slippage_cost_r=_mean(
            [
                record.slippage_cost_r
                for record in records
            ]
        ),
        average_total_friction_r=_mean(
            [
                record.total_friction_r
                for record in records
            ]
        ),
        average_winner_r=average_winner,
        average_loser_r=average_loser,
        payoff_ratio_r=payoff_ratio,
        profit_factor_r=profit_factor,
        break_even_win_rate_percent=break_even_win_rate,
    )