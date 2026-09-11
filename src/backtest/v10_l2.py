"""Deterministic, offline L2 execution primitives; no strategy or dataset loader.

Opt-in infrastructure for future preregistrations, not an amendment to any frozen
experiment. See docs/v10_l2_execution.md for accounting and event conventions.
"""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Context, Decimal, localcontext
from typing import Literal, Sequence

from src.backtest.execution import SimulatedExecutionModel
from src.backtest.models import AmbiguousBarPolicy


ZERO = Decimal(0)


class L2IntegrityError(ValueError):
    """Invalid simulated input; no result may be accepted from this replay."""


def utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise L2IntegrityError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def number(value: Decimal, *, positive: bool = False) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise L2IntegrityError("finite Decimal required")
    if positive and value <= ZERO:
        raise L2IntegrityError("positive Decimal required")
    return value


def duration(value: timedelta) -> None:
    if not isinstance(value, timedelta) or value < timedelta(0):
        raise L2IntegrityError("nonnegative timedelta required")


def canonical(value: object) -> str:
    def encode(item: object) -> str | dict:
        if isinstance(item, Decimal):
            return str(item)
        if isinstance(item, datetime):
            return utc(item).isoformat()
        if isinstance(item, timedelta):
            return {"days": item.days, "seconds": item.seconds, "microseconds": item.microseconds}
        raise TypeError(type(item).__name__)

    return json.dumps(value, default=encode, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Quote:
    source_at: datetime
    available_at: datetime
    bid: Decimal
    ask: Decimal

    def __post_init__(self) -> None:
        if utc(self.source_at) > utc(self.available_at):
            raise L2IntegrityError("quote source is in the future")
        if number(self.bid, positive=True) > number(self.ask, positive=True):
            raise L2IntegrityError("crossed executable quote")


@dataclass(frozen=True)
class Decision:
    at: datetime
    signal_available_at: datetime
    action: Literal["ENTER", "EXIT"]
    quantity: Decimal | None = None

    def __post_init__(self) -> None:
        if utc(self.signal_available_at) > utc(self.at):
            raise L2IntegrityError("decision precedes signal availability")
        if self.action not in ("ENTER", "EXIT"):
            raise L2IntegrityError("only LONG ENTER/EXIT decisions are supported")
        if self.action == "ENTER":
            number(self.quantity, positive=True)
        elif self.quantity is not None:
            raise L2IntegrityError("EXIT closes the entire position; quantity must be absent")


@dataclass(frozen=True)
class ExecutionRules:
    fee_bps: Decimal
    slippage_bps: Decimal
    latency: timedelta
    max_quote_age: timedelta

    def __post_init__(self) -> None:
        for value in (self.fee_bps, self.slippage_bps):
            if not ZERO <= number(value) < Decimal(10000):
                raise L2IntegrityError("cost bps must be in [0, 10000)")
        duration(self.latency)
        duration(self.max_quote_age)


@dataclass(frozen=True)
class Fill:
    decision: Decision
    executed_at: datetime
    quote: Quote
    price: Decimal
    quantity: Decimal
    notional: Decimal
    fee: Decimal


@dataclass(frozen=True)
class Trade:
    entry: Fill
    exit: Fill
    holding_time: timedelta
    mid_price_move: Decimal
    executable_gross_pnl: Decimal
    filled_gross_pnl: Decimal
    spread_cost: Decimal
    slippage_cost: Decimal
    fees: Decimal
    net_pnl: Decimal
    mid_return: Decimal
    executable_gross_return: Decimal
    gross_return: Decimal
    net_return: Decimal


@dataclass(frozen=True)
class Replay:
    session_id: str
    rules: ExecutionRules
    input_sha256: str
    trades: tuple[Trade, ...]
    initial_cash: Decimal
    final_cash: Decimal

    def summary(self) -> dict[str, Decimal | int | None]:
        with localcontext(Context(prec=50)):
            count = len(self.trades)
            gains = sum((max(t.net_pnl, ZERO) for t in self.trades), ZERO)
            losses = -sum((min(t.net_pnl, ZERO) for t in self.trades), ZERO)
            spread = sum((t.spread_cost for t in self.trades), ZERO)
            slippage = sum((t.slippage_cost for t in self.trades), ZERO)
            fees = sum((t.fees for t in self.trades), ZERO)
            notional = sum((t.entry.notional + t.exit.notional for t in self.trades), ZERO)
            return {
                "trade_count": count,
                "gross_expectancy": sum((t.gross_return for t in self.trades), ZERO) / count if count else None,
                "net_expectancy": sum((t.net_return for t in self.trades), ZERO) / count if count else None,
                "profit_factor": gains / losses if losses else None,
                "winning_net_pnl": gains,
                "losing_net_pnl_absolute": losses,
                "win_rate": Decimal(sum(t.net_pnl > ZERO for t in self.trades)) / count if count else None,
                "turnover_notional": notional,
                "turnover_over_initial_cash": notional / self.initial_cash,
                "average_cost_per_trade": (spread + slippage + fees) / count if count else None,
                "total_spread_cost": spread,
                "total_slippage_cost": slippage,
                "total_fees": fees,
                "total_cost": spread + slippage + fees,
            }

    def to_json(self) -> str:
        return canonical({"schema_version": 1, **asdict(self), "summary": self.summary()})

    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()


def replay(
    *, session_id: str, quotes: Sequence[Quote], decisions: Sequence[Decision],
    rules: ExecutionRules, initial_cash: Decimal,
) -> Replay:
    """Replay one caller-isolated session. Unclosed positions fail explicitly.

    Quotes carry forward only within max_quote_age, measured from source time.
    Each decision executes exactly at decision.at + latency, never at a future
    quote. Quotes available at that exact time are usable. No automatic exits.
    """
    if not isinstance(session_id, str) or not session_id.strip():
        raise L2IntegrityError("session identity required")
    number(initial_cash, positive=True)
    quotes, decisions = tuple(quotes), tuple(decisions)
    times = [utc(q.available_at) for q in quotes]
    if any(a >= b for a, b in zip(times, times[1:])):
        raise L2IntegrityError("quote availability must be strictly increasing")
    if any(a.source_at > b.source_at for a, b in zip(quotes, quotes[1:])):
        raise L2IntegrityError("quote source timestamps regressed")
    decision_times = [utc(d.at) for d in decisions]
    if any(a >= b for a, b in zip(decision_times, decision_times[1:])):
        raise L2IntegrityError("decisions must be strictly increasing")
    with localcontext(Context(prec=50)):
        model = SimulatedExecutionModel(
            fee_bps=rules.fee_bps, slippage_bps=rules.slippage_bps,
            ambiguous_bar_policy=AmbiguousBarPolicy.STOP_FIRST,
        )
        cash = initial_cash
        position: Fill | None = None
        last_fill: datetime | None = None
        trades: list[Trade] = []
        for decision in decisions:
            if last_fill is not None and utc(decision.at) <= last_fill:
                raise L2IntegrityError("decision must follow the preceding fill; pending decisions prohibited")
            execution_at = utc(decision.at) + rules.latency
            index = bisect_right(times, execution_at) - 1
            if index < 0:
                raise L2IntegrityError("no available executable quote")
            quote = quotes[index]
            if execution_at - utc(quote.source_at) > rules.max_quote_age:
                raise L2IntegrityError("executable quote is stale")
            if decision.action == "ENTER":
                if position is not None:
                    raise L2IntegrityError("overlapping LONG position")
                quantity = decision.quantity
                price = model.buy_fill_price(quote.ask)
            else:
                if position is None:
                    raise L2IntegrityError("EXIT without a LONG position")
                quantity = position.quantity
                price = model.sell_fill_price(quote.bid)
            assert quantity is not None
            notional = quantity * price
            fee = model.fee(notional)
            fill = Fill(decision, execution_at, quote, price, quantity, notional, fee)
            if decision.action == "ENTER":
                if notional + fee > cash:
                    raise L2IntegrityError("insufficient cash; borrowing prohibited")
                cash -= notional + fee
                position = fill
            else:
                assert position is not None
                entry_mid = (position.quote.bid + position.quote.ask) / 2
                exit_mid = (quote.bid + quote.ask) / 2
                mid_move = quantity * (exit_mid - entry_mid)
                executable = quantity * (quote.bid - position.quote.ask)
                gross = notional - position.notional
                spread = mid_move - executable
                slippage = executable - gross
                fees = position.fee + fee
                net = gross - fees
                denominator = position.notional
                trades.append(Trade(
                    position, fill, execution_at - position.executed_at,
                    mid_move, executable, gross, spread, slippage, fees, net,
                    mid_move / denominator, executable / denominator,
                    gross / denominator, net / denominator,
                ))
                cash += notional - fee
                position = None
            last_fill = execution_at
        if position is not None:
            raise L2IntegrityError("unclosed LONG position; supply an explicit EXIT")
        binding = canonical({
            "session_id": session_id, "quotes": [asdict(q) for q in quotes],
            "decisions": [asdict(d) for d in decisions], "rules": asdict(rules),
            "initial_cash": initial_cash,
        })
        return Replay(session_id, rules, hashlib.sha256(binding.encode()).hexdigest(), tuple(trades), initial_cash, cash)


@dataclass(frozen=True)
class EventRules:
    threshold: Decimal
    rearm_at_or_below: Decimal
    persistence_samples: int
    cooldown: timedelta

    def __post_init__(self) -> None:
        if number(self.rearm_at_or_below) > number(self.threshold):
            raise L2IntegrityError("rearm threshold must not exceed trigger threshold")
        if type(self.persistence_samples) is not int or self.persistence_samples < 1:
            raise L2IntegrityError("persistence must be a positive integer")
        duration(self.cooldown)


@dataclass(frozen=True)
class SignalState:
    at: datetime
    source_at: datetime
    available_at: datetime
    value: Decimal

    def __post_init__(self) -> None:
        if not utc(self.source_at) <= utc(self.available_at) <= utc(self.at):
            raise L2IntegrityError("signal source/availability violates causality")
        number(self.value)


@dataclass(frozen=True)
class SignalEvent:
    crossing_at: datetime
    decision_at: datetime
    signal_available_at: datetime


class EventCompressor:
    """One reserved event at a time, released explicitly at trade exit.

    Initial state is unarmed. Requires an observed rearm value, then a strict
    threshold crossing and N consecutive one-second qualifying observations.
    Missing seconds reset persistence/arming. Busy/cooldown states never queue.
    """

    def __init__(self, rules: EventRules) -> None:
        self.rules = rules
        self._last: datetime | None = None
        self._busy = False
        self._armed = False
        self._count = 0
        self._crossing: datetime | None = None
        self._cooldown_until: datetime | None = None

    def observe(self, state: SignalState) -> SignalEvent | None:
        at = utc(state.at)
        if self._last is not None and at <= self._last:
            raise L2IntegrityError("signal timestamps must be strictly increasing")
        if self._last is not None and at != self._last + timedelta(seconds=1):
            self._armed, self._count = False, 0
        self._last = at
        if self._busy or (self._cooldown_until is not None and at < self._cooldown_until):
            return None
        if state.value <= self.rules.rearm_at_or_below:
            self._armed, self._count = True, 0
        elif state.value <= self.rules.threshold:
            self._count = 0
        elif self._armed:
            if self._count == 0:
                self._crossing = at
            self._count += 1
            if self._count == self.rules.persistence_samples:
                self._busy, self._armed = True, False
                assert self._crossing is not None
                return SignalEvent(self._crossing, at, utc(state.available_at))
        return None

    def release(self, exit_at: datetime) -> None:
        at = utc(exit_at)
        if not self._busy or self._last is None or at < self._last:
            raise L2IntegrityError("release must follow a reserved event and observed states")
        self._busy, self._armed, self._count = False, False, 0
        self._last = at
        self._cooldown_until = at + self.rules.cooldown
