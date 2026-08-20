"""Chronological, offline-only, next-bar-open backtest replay engine."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from itertools import islice
from typing import overload

from src.backtest.account import (
    BacktestAccount,
    InvalidOrderIntentError,
)
from src.backtest.execution import SimulatedExecutionModel
from src.backtest.metrics import calculate_metrics
from src.backtest.models import (
    BacktestConfig,
    BacktestContext,
    BacktestResult,
    EquityPoint,
    ExitReason,
    OrderAction,
    OrderIntent,
    Trade,
)
from src.historical.dataset import HistoricalDataset
from src.historical.validator import DatasetValidator
from src.market.intervals import next_open_time
from src.models.candle import Candle


DecisionProvider = Callable[[BacktestContext], OrderIntent | None]


class _HistoricalView(Sequence[Candle]):
    """Read-only sequence ending at the current candle, never beyond it."""

    __slots__ = ("__candles", "__length")

    def __init__(self, candles: tuple[Candle, ...], length: int) -> None:
        self.__candles = candles
        self.__length = length

    def __len__(self) -> int:
        return self.__length

    @overload
    def __getitem__(self, index: int) -> Candle: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Candle, ...]: ...

    def __getitem__(self, index: int | slice) -> Candle | tuple[Candle, ...]:
        if isinstance(index, slice):
            start, stop, step = index.indices(self.__length)
            return tuple(
                self.__candles[position]
                for position in range(start, stop, step)
            )
        normalized = index + self.__length if index < 0 else index
        if normalized < 0 or normalized >= self.__length:
            raise IndexError("historical view index out of range")
        return self.__candles[normalized]

    def __iter__(self) -> Iterator[Candle]:
        return islice(self.__candles, self.__length)


@dataclass(frozen=True, slots=True)
class _PendingIntent:
    intent: OrderIntent
    signal_time: datetime


class BacktestEngine:
    """Replay validated candles without any network/exchange dependency."""

    ENGINE_VERSION = "2.0"

    def __init__(
        self,
        config: BacktestConfig,
        *,
        validator: DatasetValidator | None = None,
    ) -> None:
        self.config = config
        self._validator = validator or DatasetValidator()
        self._execution = SimulatedExecutionModel(
            fee_bps=config.fee_bps,
            slippage_bps=config.slippage_bps,
            ambiguous_bar_policy=config.ambiguous_bar_policy,
        )

    @staticmethod
    def _validate_state_intent(
        intent: OrderIntent, account: BacktestAccount
    ) -> None:
        if intent.action is OrderAction.BUY and account.position is not None:
            raise InvalidOrderIntentError("Cannot BUY while a Spot position is open.")
        if intent.action is OrderAction.SELL and account.position is None:
            raise InvalidOrderIntentError("Cannot SELL without a Spot position.")

    def _execute_pending(
        self,
        pending: _PendingIntent,
        candle: Candle,
        index: int,
        account: BacktestAccount,
    ) -> Trade | None:
        intent = pending.intent
        if intent.action is OrderAction.BUY:
            fill_price = self._execution.buy_fill_price(candle.open)
            if intent.quote_amount is not None:
                quote_notional = intent.quote_amount
                stop_loss = intent.stop_loss
                take_profit = intent.take_profit
            else:
                assert intent.risk_budget is not None
                assert intent.stop_distance is not None
                assert intent.reward_risk_ratio is not None
                assert intent.max_quote_amount is not None
                risk_notional = (
                    intent.risk_budget / intent.stop_distance * fill_price
                )
                quote_notional = min(
                    risk_notional,
                    intent.max_quote_amount,
                    self._execution.maximum_affordable_notional(account.cash),
                )
                if quote_notional <= 0:
                    raise InvalidOrderIntentError(
                        "Risk-sized BUY has no affordable positive notional."
                    )
                stop_loss = fill_price - intent.stop_distance
                if stop_loss <= 0:
                    raise InvalidOrderIntentError(
                        "Risk-sized BUY stop would not be positive."
                    )
                take_profit = fill_price + (
                    intent.stop_distance * intent.reward_risk_ratio
                )
            fee = self._execution.fee(quote_notional)
            account.open_long(
                signal_time=pending.signal_time,
                fill_time=candle.timestamp,
                fill_index=index,
                fill_price=fill_price,
                quote_notional=quote_notional,
                fee=fee,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            return None

        position = account.position
        if position is None:
            raise InvalidOrderIntentError("Pending SELL has no position to exit.")
        fill_price = self._execution.sell_fill_price(candle.open)
        exit_notional = position.quantity * fill_price
        return account.close_long(
            signal_time=pending.signal_time,
            fill_time=candle.timestamp,
            fill_price=fill_price,
            fee=self._execution.fee(exit_notional),
            bars_held=max(1, index - position.entry_index),
            exit_reason=intent.exit_reason or ExitReason.SIGNAL,
        )

    def run(
        self,
        dataset: HistoricalDataset,
        decisions: DecisionProvider,
    ) -> BacktestResult:
        """Run once, consuming each closed candle in chronological order."""

        self._validator.validate(
            dataset.candles,
            symbol=dataset.symbol,
            interval=dataset.interval,
        )
        candles = dataset.candles
        account = BacktestAccount(self.config.initial_capital_usdc)
        trades: list[Trade] = []
        equity_curve: list[EquityPoint] = []
        pending: _PendingIntent | None = None
        exposed_bars = 0

        for index, candle in enumerate(candles):
            if pending is not None:
                completed = self._execute_pending(pending, candle, index, account)
                if completed is not None:
                    trades.append(completed)
                pending = None

            was_exposed = account.position is not None
            position = account.position
            if position is not None:
                protective = self._execution.protective_exit(
                    candle,
                    stop_loss=position.stop_loss,
                    take_profit=position.take_profit,
                )
                if protective is not None:
                    fill_price = self._execution.sell_fill_price(
                        protective.reference_price
                    )
                    exit_notional = position.quantity * fill_price
                    trades.append(
                        account.close_long(
                            signal_time=candle.timestamp,
                            fill_time=candle.timestamp,
                            fill_price=fill_price,
                            fee=self._execution.fee(exit_notional),
                            bars_held=max(1, index - position.entry_index + 1),
                            exit_reason=protective.reason,
                        )
                    )
            if was_exposed:
                exposed_bars += 1

            snapshot = account.snapshot(candle.close, current_index=index)
            equity_curve.append(
                EquityPoint(
                    timestamp=next_open_time(candle.timestamp, dataset.interval),
                    cash=snapshot.cash_usdc,
                    position_value=snapshot.btc_quantity * candle.close,
                    total_equity=snapshot.equity,
                )
            )

            if index < self.config.warmup_candles:
                continue
            context = BacktestContext(
                index=index,
                candle=candle,
                history=_HistoricalView(candles, index + 1),
                account=snapshot,
            )
            intent = decisions(context)
            if intent is None:
                continue
            if not isinstance(intent, OrderIntent):
                raise InvalidOrderIntentError(
                    "Decision provider must return OrderIntent or None."
                )
            self._validate_state_intent(intent, account)
            if index + 1 < len(candles):
                pending = _PendingIntent(
                    intent,
                    next_open_time(candle.timestamp, dataset.interval),
                )

        if account.position is not None:
            final_candle = candles[-1]
            position = account.position
            final_close_time = next_open_time(
                final_candle.timestamp, dataset.interval
            )
            fill_price = self._execution.sell_fill_price(final_candle.close)
            exit_notional = position.quantity * fill_price
            trades.append(
                account.close_long(
                    signal_time=final_close_time,
                    fill_time=final_close_time,
                    fill_price=fill_price,
                    fee=self._execution.fee(exit_notional),
                    bars_held=max(
                        1, len(candles) - position.entry_index
                    ),
                    exit_reason=ExitReason.END_OF_BACKTEST,
                )
            )
            final_snapshot = account.snapshot(
                final_candle.close, current_index=len(candles) - 1
            )
            equity_curve[-1] = EquityPoint(
                timestamp=final_close_time,
                cash=final_snapshot.cash_usdc,
                position_value=Decimal("0"),
                total_equity=final_snapshot.equity,
            )

        metrics = calculate_metrics(
            initial_capital=self.config.initial_capital_usdc,
            trades=trades,
            equity_curve=equity_curve,
            exposed_bars=exposed_bars,
            total_bars=len(candles),
        )
        return BacktestResult(
            dataset=dataset,
            config=self.config,
            trades=tuple(trades),
            equity_curve=tuple(equity_curve),
            metrics=metrics,
        )
