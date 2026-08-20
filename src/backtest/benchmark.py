"""Accounting benchmark only; this is not a trading strategy."""

from __future__ import annotations

from decimal import Decimal

from src.backtest.execution import basis_points_rate
from src.backtest.models import BacktestContext, OrderAction, OrderIntent


class BuyAndHoldBenchmark:
    """Buy once after the first close and let the engine liquidate at the end."""

    def __init__(self, initial_capital: Decimal, fee_bps: Decimal) -> None:
        fee_rate = basis_points_rate(fee_bps)
        self._entry_notional = Decimal(str(initial_capital)) / (
            Decimal("1") + fee_rate
        )

    def __call__(self, context: BacktestContext) -> OrderIntent | None:
        if context.index != 0:
            return None
        return OrderIntent(
            action=OrderAction.BUY,
            quote_amount=self._entry_notional,
            reason="BUY_AND_HOLD_BENCHMARK entry",
        )
