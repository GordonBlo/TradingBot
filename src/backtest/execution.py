"""Pure simulated fills, fees, slippage, and protective-exit resolution."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.backtest.models import AmbiguousBarPolicy, ExitReason
from src.models.candle import Candle


def basis_points_rate(basis_points: Decimal) -> Decimal:
    """Convert basis points to a decimal rate (10 bps == 0.001)."""

    value = Decimal(str(basis_points))
    if not value.is_finite() or value < 0:
        raise ValueError("Basis points must be finite and non-negative.")
    return value / Decimal("10000")


@dataclass(frozen=True, slots=True)
class ProtectiveExit:
    reason: ExitReason
    reference_price: Decimal


class SimulatedExecutionModel:
    """Deterministic long-Spot execution with adverse slippage only."""

    def __init__(
        self,
        *,
        fee_bps: Decimal,
        slippage_bps: Decimal,
        ambiguous_bar_policy: AmbiguousBarPolicy,
    ) -> None:
        self.fee_rate = basis_points_rate(fee_bps)
        self.slippage_rate = basis_points_rate(slippage_bps)
        self.ambiguous_bar_policy = ambiguous_bar_policy

    def buy_fill_price(self, market_price: Decimal) -> Decimal:
        market_price = Decimal(str(market_price))
        if market_price <= 0:
            raise ValueError("Market price must be greater than zero.")
        return market_price * (Decimal("1") + self.slippage_rate)

    def sell_fill_price(self, market_price: Decimal) -> Decimal:
        market_price = Decimal(str(market_price))
        if market_price <= 0:
            raise ValueError("Market price must be greater than zero.")
        return market_price * (Decimal("1") - self.slippage_rate)

    def fee(self, notional: Decimal) -> Decimal:
        notional = Decimal(str(notional))
        if notional < 0:
            raise ValueError("Notional must not be negative.")
        return notional * self.fee_rate

    def maximum_affordable_notional(self, cash: Decimal) -> Decimal:
        """Return the largest entry notional whose fee also fits in cash."""

        cash = Decimal(str(cash))
        if cash < 0:
            raise ValueError("Cash must not be negative.")
        return cash / (Decimal("1") + self.fee_rate)

    def protective_exit(
        self,
        candle: Candle,
        *,
        stop_loss: Decimal | None,
        take_profit: Decimal | None,
    ) -> ProtectiveExit | None:
        stop_touched = stop_loss is not None and candle.low <= stop_loss
        target_touched = take_profit is not None and candle.high >= take_profit
        if not stop_touched and not target_touched:
            return None
        if stop_touched and target_touched:
            if self.ambiguous_bar_policy is not AmbiguousBarPolicy.STOP_FIRST:
                raise ValueError("Unsupported ambiguous-bar policy.")
            return ProtectiveExit(
                ExitReason.STOP_LOSS,
                min(candle.open, stop_loss),
            )
        if stop_touched:
            return ProtectiveExit(
                ExitReason.STOP_LOSS,
                min(candle.open, stop_loss),
            )
        return ProtectiveExit(
            ExitReason.TAKE_PROFIT,
            max(candle.open, take_profit),
        )
