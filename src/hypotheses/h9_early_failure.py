"""H9 frozen -0.5R / first-four-bars Early Failure Exit."""

from __future__ import annotations

from decimal import Decimal

from src.backtest.models import (
    BacktestContext,
    ExitReason,
    OpenPositionSnapshot,
    OrderAction,
    OrderIntent,
)


class H9EarlyFailureExit:
    ADVERSE_TRIGGER_R = Decimal("0.5")
    OBSERVATION_BARS = 4

    def __init__(self) -> None:
        self._triggered_trade_ids: set[str] = set()

    @property
    def triggered_trade_ids(self) -> frozenset[str]:
        return frozenset(self._triggered_trade_ids)

    @property
    def trigger_count(self) -> int:
        return len(self._triggered_trade_ids)

    def __call__(
        self,
        context: BacktestContext,
        position: OpenPositionSnapshot,
    ) -> OrderIntent | None:
        if position.trade_id in self._triggered_trade_ids:
            return None

        if not context.account.has_position:
            return None

        # Snapshot semantics define the current held candle number.
        held_bar = context.account.bars_in_position

        if not 1 <= held_bar <= self.OBSERVATION_BARS:
            return None

        if position.stop_loss is None:
            raise ValueError(
                "H9 requires the frozen H5_Q25 initial stop."
            )

        initial_risk = (
            position.entry_price
            - position.stop_loss
        )

        if initial_risk <= 0:
            raise ValueError(
                "H9 initial long risk must be positive."
            )

        trigger_price = (
            position.entry_price
            - initial_risk * self.ADVERSE_TRIGGER_R
        )

        if context.candle.low > trigger_price:
            return None

        self._triggered_trade_ids.add(
            position.trade_id
        )

        return OrderIntent(
            action=OrderAction.SELL,
            reason=(
                "H9 early failure: closed candle reached "
                "-0.5R within first 4 held bars"
            ),
            exit_reason=ExitReason.EARLY_FAILURE_EXIT,
        )