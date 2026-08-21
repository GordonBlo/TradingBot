"""H8 +1R next-bar price break-even protection."""

from __future__ import annotations

from decimal import Decimal

from src.backtest.models import (
    BacktestContext,
    ExitReason,
    OpenPositionSnapshot,
    ProtectiveStopUpdate,
)


class H8BreakEvenProtection:
    """Freeze the stop at entry after a closed candle reaches +1R."""

    TRIGGER_R = Decimal("1")

    def __init__(self) -> None:
        self._initial_stop_by_trade_id: dict[str, Decimal] = {}
        self._activated_trade_ids: set[str] = set()

    @property
    def activated_trade_ids(self) -> frozenset[str]:
        return frozenset(self._activated_trade_ids)

    @property
    def activation_count(self) -> int:
        return len(self._activated_trade_ids)

    def __call__(
        self,
        context: BacktestContext,
        position: OpenPositionSnapshot,
    ) -> ProtectiveStopUpdate | None:
        if position.trade_id in self._activated_trade_ids:
            return None

        if position.stop_loss is None:
            raise ValueError(
                "H8 requires the frozen H5_Q25 initial stop."
            )

        initial_stop = self._initial_stop_by_trade_id.setdefault(
            position.trade_id,
            position.stop_loss,
        )

        initial_risk = position.entry_price - initial_stop

        if initial_risk <= 0:
            raise ValueError(
                "H8 initial long risk must be positive."
            )

        trigger_price = (
            position.entry_price
            + initial_risk * self.TRIGGER_R
        )

        if context.candle.high < trigger_price:
            return None

        self._activated_trade_ids.add(position.trade_id)

        return ProtectiveStopUpdate(
            stop_loss=position.entry_price,
            exit_reason=ExitReason.BREAK_EVEN_STOP,
        )