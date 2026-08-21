"""Precision-safe simulated USDC/BTC Spot account."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from src.backtest.models import (
    AccountSnapshot,
    ExitReason,
    OpenPositionSnapshot,
    Trade,
)


class BacktestAccountError(RuntimeError):
    """Base simulated-account error."""


class InsufficientBalanceError(BacktestAccountError):
    """Raised when a hypothetical fill exceeds available USDC."""


class InvalidOrderIntentError(BacktestAccountError):
    """Raised when an intent violates V2 long-only Spot constraints."""


@dataclass(slots=True)
class Position:
    trade_id: str
    entry_signal_time: datetime
    entry_time: datetime
    entry_index: int
    entry_price: Decimal
    quantity: Decimal
    entry_notional: Decimal
    entry_fee: Decimal
    stop_loss: Decimal | None
    take_profit: Decimal | None
    stop_exit_reason: ExitReason


class BacktestAccount:
    """One-position, non-negative BTC, no-borrowing Spot portfolio."""

    def __init__(self, initial_cash: Decimal) -> None:
        self.initial_cash = Decimal(str(initial_cash))
        if not self.initial_cash.is_finite() or self.initial_cash <= 0:
            raise ValueError("Initial simulated cash must be finite and positive.")
        self.cash = self.initial_cash
        self.btc_quantity = Decimal("0")
        self.realized_pnl = Decimal("0")
        self.fees_paid = Decimal("0")
        self.position: Position | None = None
        self._next_trade_id = 1
        self.completed_trade_count = 0

    def open_long(
        self,
        *,
        signal_time: datetime,
        fill_time: datetime,
        fill_index: int,
        fill_price: Decimal,
        quote_notional: Decimal,
        fee: Decimal,
        stop_loss: Decimal | None,
        take_profit: Decimal | None,
    ) -> Position:
        if self.position is not None:
            raise InvalidOrderIntentError("V2 does not allow pyramiding.")
        fill_price = Decimal(str(fill_price))
        quote_notional = Decimal(str(quote_notional))
        fee = Decimal(str(fee))
        if fill_price <= 0 or quote_notional <= 0 or fee < 0:
            raise InvalidOrderIntentError("Entry price/notional/fee is invalid.")
        total_debit = quote_notional + fee
        if total_debit > self.cash:
            raise InsufficientBalanceError(
                f"Entry requires {total_debit} USDC but only {self.cash} is available."
            )
        if stop_loss is not None and stop_loss >= fill_price:
            raise InvalidOrderIntentError("Long stop-loss must be below entry fill.")
        if take_profit is not None and take_profit <= fill_price:
            raise InvalidOrderIntentError("Long take-profit must be above entry fill.")

        quantity = quote_notional / fill_price
        position = Position(
            trade_id=f"T{self._next_trade_id:06d}",
            entry_signal_time=signal_time,
            entry_time=fill_time,
            entry_index=fill_index,
            entry_price=fill_price,
            quantity=quantity,
            entry_notional=quote_notional,
            entry_fee=fee,
            stop_loss=stop_loss,
            take_profit=take_profit,
            stop_exit_reason=ExitReason.STOP_LOSS,
        )
        self._next_trade_id += 1
        self.cash -= total_debit
        self.btc_quantity = quantity
        self.fees_paid += fee
        self.position = position
        return position

    def open_position_snapshot(self) -> OpenPositionSnapshot | None:
        position = self.position

        if position is None:
            return None

        return OpenPositionSnapshot(
            trade_id=position.trade_id,
            entry_price=position.entry_price,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
        )

    def update_long_stop_loss(
        self,
        *,
        stop_loss: Decimal,
        exit_reason: ExitReason,
    ) -> None:
        position = self.position

        if position is None:
            raise InvalidOrderIntentError(
                "There is no long Spot position to protect."
            )

        stop_loss = Decimal(str(stop_loss))

        if stop_loss <= 0:
            raise InvalidOrderIntentError(
                "Protective stop must be positive."
            )

        if not isinstance(exit_reason, ExitReason):
            exit_reason = ExitReason(exit_reason)

        if exit_reason not in {
            ExitReason.STOP_LOSS,
            ExitReason.BREAK_EVEN_STOP,
        }:
            raise InvalidOrderIntentError(
                "Invalid protective stop exit reason."
            )

        if (
            position.stop_loss is not None
            and stop_loss < position.stop_loss
        ):
            raise InvalidOrderIntentError(
                "Long protective stop cannot be loosened."
            )

        if (
            position.take_profit is not None
            and stop_loss >= position.take_profit
        ):
            raise InvalidOrderIntentError(
                "Protective stop must remain below take-profit."
            )

        position.stop_loss = stop_loss
        position.stop_exit_reason = exit_reason

    def close_long(
        self,
        *,
        signal_time: datetime,
        fill_time: datetime,
        fill_price: Decimal,
        fee: Decimal,
        bars_held: int,
        exit_reason: ExitReason,
    ) -> Trade:
        position = self.position
        if position is None:
            raise InvalidOrderIntentError("There is no long Spot position to exit.")
        fill_price = Decimal(str(fill_price))
        fee = Decimal(str(fee))
        if fill_price <= 0 or fee < 0 or bars_held < 1:
            raise InvalidOrderIntentError("Exit price/fee/bars-held is invalid.")

        exit_notional = position.quantity * fill_price
        gross_pnl = exit_notional - position.entry_notional
        net_pnl = gross_pnl - position.entry_fee - fee
        total_fee = position.entry_fee + fee
        trade = Trade(
            trade_id=position.trade_id,
            entry_signal_time=position.entry_signal_time,
            entry_time=position.entry_time,
            entry_price=position.entry_price,
            exit_signal_time=signal_time,
            exit_time=fill_time,
            exit_price=fill_price,
            quantity=position.quantity,
            entry_notional=position.entry_notional,
            exit_notional=exit_notional,
            entry_fee=position.entry_fee,
            exit_fee=fee,
            total_fee=total_fee,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            return_percent=net_pnl / position.entry_notional * Decimal("100"),
            bars_held=bars_held,
            exit_reason=exit_reason,
        )
        self.cash += exit_notional - fee
        self.btc_quantity = Decimal("0")
        self.realized_pnl += net_pnl
        self.fees_paid += fee
        self.position = None
        self.completed_trade_count += 1
        return trade

    def snapshot(
        self, mark_price: Decimal, *, current_index: int | None = None
    ) -> AccountSnapshot:
        mark_price = Decimal(str(mark_price))
        position_value = self.btc_quantity * mark_price
        unrealized = (
            self.btc_quantity * (mark_price - self.position.entry_price)
            if self.position is not None
            else Decimal("0")
        )
        return AccountSnapshot(
            cash_usdc=self.cash,
            btc_quantity=self.btc_quantity,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=unrealized,
            fees_paid=self.fees_paid,
            equity=self.cash + position_value,
            has_position=self.position is not None,
            bars_in_position=(
                max(0, current_index - self.position.entry_index + 1)
                if self.position is not None and current_index is not None
                else 0
            ),
            completed_trade_count=self.completed_trade_count,
        )
