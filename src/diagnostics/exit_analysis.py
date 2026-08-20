"""Exit-reason, stop-loss, and take-profit diagnostics."""

from __future__ import annotations

from decimal import Decimal

from src.backtest.models import ExitReason
from src.diagnostics.distributions import (
    decimal_mean,
    evidence_strength,
    median,
    optional_mean,
)
from src.diagnostics.models import (
    ExitReasonDiagnostics,
    StopLossDiagnostics,
    TakeProfitDiagnostics,
    TradeDiagnostic,
)


def exit_reason_diagnostics(
    trades: tuple[TradeDiagnostic, ...], minimum: int
) -> tuple[ExitReasonDiagnostics, ...]:
    rows: list[ExitReasonDiagnostics] = []
    for reason in ExitReason:
        items = tuple(trade for trade in trades if trade.exit_reason is reason)
        if not items:
            continue
        count = len(items)
        winners = sum(trade.net_pnl > 0 for trade in items)
        rows.append(
            ExitReasonDiagnostics(
                exit_reason=reason,
                trade_count=count,
                percentage_of_trades=(
                    Decimal(count) / Decimal(len(trades)) * Decimal("100")
                ),
                win_rate_percent=Decimal(winners) / Decimal(count) * Decimal("100"),
                gross_pnl=sum(
                    (trade.gross_pnl for trade in items), start=Decimal("0")
                ),
                net_pnl=sum(
                    (trade.net_pnl for trade in items), start=Decimal("0")
                ),
                average_pnl=decimal_mean(trade.net_pnl for trade in items),
                average_r=optional_mean(trade.r_multiple for trade in items),
                average_bars_held=decimal_mean(
                    Decimal(trade.bars_held) for trade in items
                ),
                average_mfe_r=optional_mean(trade.mfe_r for trade in items),
                average_mae_r=optional_mean(trade.mae_r for trade in items),
                evidence_strength=evidence_strength(count, minimum),
            )
        )
    return tuple(rows)


def stop_loss_diagnostics(
    trades: tuple[TradeDiagnostic, ...],
) -> StopLossDiagnostics:
    stopped = tuple(
        trade for trade in trades if trade.exit_reason is ExitReason.STOP_LOSS
    )
    return StopLossDiagnostics(
        count=len(stopped),
        percentage_of_trades=(
            Decimal(len(stopped)) / Decimal(len(trades)) * Decimal("100")
            if trades
            else Decimal("0")
        ),
        average_loss_r=optional_mean(trade.r_multiple for trade in stopped),
        average_bars_before_stop=decimal_mean(
            Decimal(trade.bars_held) for trade in stopped
        ),
        median_bars_before_stop=median(
            Decimal(trade.bars_held) for trade in stopped
        ),
        average_mfe_before_stop=decimal_mean(trade.mfe for trade in stopped),
        average_mfe_r_before_stop=optional_mean(trade.mfe_r for trade in stopped),
    )


def take_profit_diagnostics(
    trades: tuple[TradeDiagnostic, ...],
) -> TakeProfitDiagnostics:
    targets = tuple(
        trade for trade in trades if trade.exit_reason is ExitReason.TAKE_PROFIT
    )
    return TakeProfitDiagnostics(
        count=len(targets),
        percentage_of_trades=(
            Decimal(len(targets)) / Decimal(len(trades)) * Decimal("100")
            if trades
            else Decimal("0")
        ),
        average_bars_to_target=decimal_mean(
            Decimal(trade.bars_held) for trade in targets
        ),
        average_mae_before_target=decimal_mean(trade.mae for trade in targets),
        average_mae_r_before_target=optional_mean(trade.mae_r for trade in targets),
    )

