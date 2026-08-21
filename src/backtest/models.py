"""Typed inputs and outputs for deterministic offline backtests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Sequence

from src.historical.dataset import HistoricalDataset
from src.models.candle import Candle


def _decimal(name: str, value: Decimal) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a valid decimal.") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be finite.")
    return parsed


class OrderAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class ExitReason(str, Enum):
    SIGNAL = "SIGNAL"
    TREND_EXIT = "TREND_EXIT"
    TIME_EXIT = "TIME_EXIT"
    STOP_LOSS = "STOP_LOSS"
    BREAK_EVEN_STOP = "BREAK_EVEN_STOP"
    TAKE_PROFIT = "TAKE_PROFIT"
    END_OF_BACKTEST = "END_OF_BACKTEST"


class AmbiguousBarPolicy(str, Enum):
    STOP_FIRST = "STOP_FIRST"


class ExecutionTiming(str, Enum):
    NEXT_BAR_OPEN = "NEXT_BAR_OPEN"


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    initial_capital_usdc: Decimal = Decimal("1000")
    fee_bps: Decimal = Decimal("10")
    slippage_bps: Decimal = Decimal("2")
    ambiguous_bar_policy: AmbiguousBarPolicy = AmbiguousBarPolicy.STOP_FIRST
    execution_timing: ExecutionTiming = ExecutionTiming.NEXT_BAR_OPEN
    warmup_candles: int = 0

    def __post_init__(self) -> None:
        for name in ("initial_capital_usdc", "fee_bps", "slippage_bps"):
            object.__setattr__(self, name, _decimal(name, getattr(self, name)))
        if self.initial_capital_usdc <= 0:
            raise ValueError("Backtest initial capital must be greater than zero.")
        if not Decimal("0") <= self.fee_bps <= Decimal("10000"):
            raise ValueError("Backtest fee basis points must be in [0, 10000].")
        if not Decimal("0") <= self.slippage_bps < Decimal("10000"):
            raise ValueError("Backtest slippage basis points must be in [0, 10000).")
        if self.warmup_candles < 0:
            raise ValueError("Backtest warm-up candle count must not be negative.")
        if self.execution_timing is not ExecutionTiming.NEXT_BAR_OPEN:
            raise ValueError("V2 supports next-bar-open execution only.")
        if self.ambiguous_bar_policy is not AmbiguousBarPolicy.STOP_FIRST:
            raise ValueError("V2 supports STOP_FIRST ambiguous-bar handling only.")


@dataclass(frozen=True, slots=True)
class OrderIntent:
    action: OrderAction
    quote_amount: Decimal | None = None
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None
    reason: str = "scripted decision"
    risk_budget: Decimal | None = None
    stop_distance: Decimal | None = None
    reward_risk_ratio: Decimal | None = None
    max_quote_amount: Decimal | None = None
    exit_reason: ExitReason | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.action, OrderAction):
            object.__setattr__(self, "action", OrderAction(self.action))
        for name in (
            "quote_amount",
            "stop_loss",
            "take_profit",
            "risk_budget",
            "stop_distance",
            "reward_risk_ratio",
            "max_quote_amount",
        ):
            value = getattr(self, name)
            if value is not None:
                parsed = _decimal(name, value)
                if parsed <= 0:
                    raise ValueError(f"{name} must be greater than zero.")
                object.__setattr__(self, name, parsed)
        if self.exit_reason is not None and not isinstance(
            self.exit_reason, ExitReason
        ):
            object.__setattr__(self, "exit_reason", ExitReason(self.exit_reason))
        risk_fields = (
            self.risk_budget,
            self.stop_distance,
            self.reward_risk_ratio,
            self.max_quote_amount,
        )
        uses_risk_sizing = any(value is not None for value in risk_fields)
        if self.action is OrderAction.BUY:
            if self.quote_amount is None and not all(
                value is not None for value in risk_fields
            ):
                raise ValueError(
                    "BUY intent requires quote_amount or all risk-sizing fields."
                )
            if self.quote_amount is not None and uses_risk_sizing:
                raise ValueError(
                    "BUY intent cannot mix fixed-notional and risk sizing."
                )
            if uses_risk_sizing and any(
                value is not None for value in (self.stop_loss, self.take_profit)
            ):
                raise ValueError(
                    "Risk-sized BUY brackets are derived from the actual fill."
                )
            if self.exit_reason is not None:
                raise ValueError("BUY intent cannot define an exit reason.")
        if self.action is OrderAction.SELL and any(
            value is not None
            for value in (
                self.quote_amount,
                self.stop_loss,
                self.take_profit,
                *risk_fields,
            )
        ):
            raise ValueError("SELL intent exits the full Spot position without brackets.")
        object.__setattr__(self, "reason", self.reason.strip())
        if not self.reason:
            raise ValueError("Order intent reason must not be empty.")


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    cash_usdc: Decimal
    btc_quantity: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    fees_paid: Decimal
    equity: Decimal
    has_position: bool
    bars_in_position: int = 0
    completed_trade_count: int = 0


@dataclass(frozen=True, slots=True)
class OpenPositionSnapshot:
    trade_id: str
    entry_price: Decimal
    stop_loss: Decimal | None
    take_profit: Decimal | None


@dataclass(frozen=True, slots=True)
class ProtectiveStopUpdate:
    stop_loss: Decimal
    exit_reason: ExitReason

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stop_loss",
            _decimal("stop_loss", self.stop_loss),
        )

        if self.stop_loss <= 0:
            raise ValueError("Protective stop must be greater than zero.")

        if not isinstance(self.exit_reason, ExitReason):
            object.__setattr__(
                self,
                "exit_reason",
                ExitReason(self.exit_reason),
            )

        if self.exit_reason not in {
            ExitReason.STOP_LOSS,
            ExitReason.BREAK_EVEN_STOP,
        }:
            raise ValueError(
                "Protective stop update requires a protective exit reason."
            )


@dataclass(frozen=True, slots=True)
class BacktestContext:
    """Decision input containing only the current and already-closed past."""

    index: int
    candle: Candle
    history: Sequence[Candle]
    account: AccountSnapshot


@dataclass(frozen=True, slots=True)
class Trade:
    trade_id: str
    entry_signal_time: datetime
    entry_time: datetime
    entry_price: Decimal
    exit_signal_time: datetime
    exit_time: datetime
    exit_price: Decimal
    quantity: Decimal
    entry_notional: Decimal
    exit_notional: Decimal
    entry_fee: Decimal
    exit_fee: Decimal
    total_fee: Decimal
    gross_pnl: Decimal
    net_pnl: Decimal
    return_percent: Decimal
    bars_held: int
    exit_reason: ExitReason


@dataclass(frozen=True, slots=True)
class EquityPoint:
    timestamp: datetime
    cash: Decimal
    position_value: Decimal
    total_equity: Decimal


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    initial_capital: Decimal
    final_equity: Decimal
    net_profit: Decimal
    total_return_percent: Decimal
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate_percent: Decimal
    gross_profit: Decimal
    gross_loss: Decimal
    average_winning_trade: Decimal
    average_losing_trade: Decimal
    largest_winning_trade: Decimal
    largest_losing_trade: Decimal
    payoff_ratio: Decimal | None
    profit_factor: Decimal | None
    expectancy_per_trade: Decimal
    average_return_percent_per_trade: Decimal
    maximum_drawdown_percent: Decimal
    maximum_consecutive_wins: int
    maximum_consecutive_losses: int
    total_fees_paid: Decimal
    market_exposure_percent: Decimal


@dataclass(frozen=True, slots=True)
class BacktestResult:
    dataset: HistoricalDataset
    config: BacktestConfig
    trades: tuple[Trade, ...]
    equity_curve: tuple[EquityPoint, ...]
    metrics: PerformanceMetrics
