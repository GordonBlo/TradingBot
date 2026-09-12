"""Deterministic depth-aware Spot L2 execution research primitives."""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Context, Decimal, InvalidOperation, localcontext
from typing import Any, Literal, Protocol

from src.backtest.execution import SimulatedExecutionModel
from src.backtest.models import AmbiguousBarPolicy

ZERO = Decimal(0)
FULL_FILL_OR_REJECT = "FULL_FILL_OR_REJECT"
ALLOW_PARTIAL_FILL = "ALLOW_PARTIAL_FILL"
FillMode = Literal["FULL_FILL_OR_REJECT", "ALLOW_PARTIAL_FILL"]


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
    def encode(item: object) -> str | dict[str, int]:
        if isinstance(item, Decimal):
            return str(item)
        if isinstance(item, datetime):
            return utc(item).isoformat()
        if isinstance(item, timedelta):
            return {"days": item.days, "seconds": item.seconds, "microseconds": item.microseconds}
        raise TypeError(type(item).__name__)

    return json.dumps(value, default=encode, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class DepthLevel:
    price: Decimal
    quantity: Decimal

    def __post_init__(self) -> None:
        number(self.price, positive=True)
        number(self.quantity, positive=True)


def _validate_side(levels: tuple[DepthLevel, ...], *, bids: bool) -> None:
    if not isinstance(levels, tuple) or not levels:
        raise L2IntegrityError("both depth sides require immutable nonempty levels")
    seen: dict[Decimal, Decimal] = {}
    for level in levels:
        if not isinstance(level, DepthLevel):
            raise L2IntegrityError("depth side contains an invalid level")
        previous_quantity = seen.get(level.price)
        if previous_quantity is not None:
            if previous_quantity != level.quantity:
                raise L2IntegrityError("duplicate price has conflicting quantities")
            raise L2IntegrityError("duplicate depth price")
        seen[level.price] = level.quantity
    for previous, current in zip(levels, levels[1:]):
        ordered = previous.price > current.price if bids else previous.price < current.price
        if not ordered:
            raise L2IntegrityError("book levels are not strictly price-monotonic")


@dataclass(frozen=True)
class DepthSnapshot:
    symbol: str
    source_at: datetime
    available_at: datetime
    sequence_id: int
    bids: tuple[DepthLevel, ...]
    asks: tuple[DepthLevel, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol or self.symbol != self.symbol.upper():
            raise L2IntegrityError("uppercase depth symbol provenance required")
        if utc(self.source_at) > utc(self.available_at):
            raise L2IntegrityError("depth source is in the future")
        if type(self.sequence_id) is not int or self.sequence_id < 0:
            raise L2IntegrityError("nonnegative depth sequence provenance required")
        _validate_side(self.bids, bids=True)
        _validate_side(self.asks, bids=False)
        if self.bids[0].price >= self.asks[0].price:
            raise L2IntegrityError("crossed or locked depth book")

    @property
    def best_bid(self) -> Decimal:
        return self.bids[0].price

    @property
    def best_ask(self) -> Decimal:
        return self.asks[0].price

    @property
    def midpoint(self) -> Decimal:
        return (self.best_bid + self.best_ask) / Decimal(2)


@dataclass(frozen=True)
class QuantityFilter:
    min_qty: Decimal | None
    max_qty: Decimal | None
    step_size: Decimal | None

    def __post_init__(self) -> None:
        for value in (self.min_qty, self.max_qty, self.step_size):
            if value is not None:
                number(value, positive=True)
        if self.min_qty is not None and self.max_qty is not None and self.min_qty > self.max_qty:
            raise L2IntegrityError("quantity filter minimum exceeds maximum")

    def validate(self, quantity: Decimal, *, label: str) -> None:
        number(quantity, positive=True)
        if self.min_qty is not None and quantity < self.min_qty:
            raise L2IntegrityError(f"{label} quantity is below minimum")
        if self.max_qty is not None and quantity > self.max_qty:
            raise L2IntegrityError(f"{label} quantity is above maximum")
        if self.step_size is not None and quantity % self.step_size != ZERO:
            raise L2IntegrityError(f"{label} quantity violates step size")


@dataclass(frozen=True)
class NotionalFilter:
    filter_type: Literal["MIN_NOTIONAL", "NOTIONAL"]
    min_notional: Decimal | None
    max_notional: Decimal | None
    apply_min_to_market: bool
    apply_max_to_market: bool
    avg_price_mins: int

    def __post_init__(self) -> None:
        if self.filter_type not in ("MIN_NOTIONAL", "NOTIONAL"):
            raise L2IntegrityError("unsupported notional filter")
        if self.filter_type == "MIN_NOTIONAL" and (
            self.min_notional is None or self.max_notional is not None or self.apply_max_to_market
        ):
            raise L2IntegrityError("invalid MIN_NOTIONAL semantics")
        for value in (self.min_notional, self.max_notional):
            if value is not None:
                number(value, positive=True)
        if self.min_notional is not None and self.max_notional is not None and self.min_notional > self.max_notional:
            raise L2IntegrityError("notional filter minimum exceeds maximum")
        if type(self.apply_min_to_market) is not bool or type(self.apply_max_to_market) is not bool:
            raise L2IntegrityError("market-notional applicability must be boolean")
        if type(self.avg_price_mins) is not int or self.avg_price_mins < 0:
            raise L2IntegrityError("avgPriceMins must be a nonnegative integer")

    def validate_market(self, notional: Decimal) -> None:
        number(notional, positive=True)
        if self.apply_min_to_market and self.min_notional is not None and notional < self.min_notional:
            raise L2IntegrityError(f"market notional is below {self.filter_type} minimum")
        if self.apply_max_to_market and self.max_notional is not None and notional > self.max_notional:
            raise L2IntegrityError("market notional is above NOTIONAL maximum")


@dataclass(frozen=True)
class ExchangeConstraints:
    symbol: str
    lot_size: QuantityFilter
    market_lot_size: QuantityFilter | None
    notional_filters: tuple[NotionalFilter, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol or self.symbol != self.symbol.upper():
            raise L2IntegrityError("uppercase exchange symbol required")
        if not isinstance(self.lot_size, QuantityFilter):
            raise L2IntegrityError("LOT_SIZE constraint required")
        if self.market_lot_size is not None and not isinstance(self.market_lot_size, QuantityFilter):
            raise L2IntegrityError("invalid MARKET_LOT_SIZE constraint")
        if not isinstance(self.notional_filters, tuple):
            raise L2IntegrityError("notional constraints must be immutable")
        if any(not isinstance(item, NotionalFilter) for item in self.notional_filters):
            raise L2IntegrityError("invalid notional constraint")
        kinds = [item.filter_type for item in self.notional_filters]
        if len(kinds) != len(set(kinds)):
            raise L2IntegrityError("duplicate notional constraint")

    def validate_market_quantity(self, quantity: Decimal, *, label: str) -> None:
        self.lot_size.validate(quantity, label=label)
        if self.market_lot_size is not None:
            self.market_lot_size.validate(quantity, label=label)

    def validate_market_notional(self, notional: Decimal) -> None:
        for rule in self.notional_filters:
            rule.validate_market(notional)


def _exchange_decimal(value: object, *, field: str, zero_disables: bool = False) -> Decimal | None:
    if isinstance(value, (bool, float)) or not isinstance(value, (str, int, Decimal)):
        raise L2IntegrityError(f"exchangeInfo {field} must be an exact decimal value")
    try:
        result = value if isinstance(value, Decimal) else Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise L2IntegrityError(f"exchangeInfo {field} is invalid") from exc
    if not result.is_finite() or result < ZERO:
        raise L2IntegrityError(f"exchangeInfo {field} is invalid")
    if result == ZERO and zero_disables:
        return None
    if result <= ZERO:
        raise L2IntegrityError(f"exchangeInfo {field} must be positive")
    return result


def _exchange_bool(item: Mapping[str, object], field: str) -> bool:
    value = item.get(field)
    if type(value) is not bool:
        raise L2IntegrityError(f"exchangeInfo {field} must be boolean")
    return value


def _exchange_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise L2IntegrityError(f"exchangeInfo {field} must be a nonnegative integer")
    return value


def _quantity_filter(item: Mapping[str, object]) -> QuantityFilter:
    return QuantityFilter(
        _exchange_decimal(item.get("minQty"), field="minQty", zero_disables=True),
        _exchange_decimal(item.get("maxQty"), field="maxQty", zero_disables=True),
        _exchange_decimal(item.get("stepSize"), field="stepSize", zero_disables=True),
    )


def exchange_constraints_from_exchange_info(
    payload: Mapping[str, object], *, symbol: str = "BTCUSDC"
) -> ExchangeConstraints:
    """Translate public Binance Spot exchangeInfo without using PRICE_FILTER."""
    symbols = payload.get("symbols") if isinstance(payload, Mapping) else None
    if not isinstance(symbols, Sequence) or isinstance(symbols, (str, bytes)):
        raise L2IntegrityError("exchangeInfo symbols are missing")
    matches = [item for item in symbols if isinstance(item, Mapping) and item.get("symbol") == symbol]
    if len(matches) != 1:
        raise L2IntegrityError("exchangeInfo must contain exactly one requested symbol")
    raw_filters = matches[0].get("filters")
    if not isinstance(raw_filters, Sequence) or isinstance(raw_filters, (str, bytes)):
        raise L2IntegrityError("exchangeInfo symbol filters are missing")
    filters: dict[str, Mapping[str, object]] = {}
    for item in raw_filters:
        if not isinstance(item, Mapping) or not isinstance(item.get("filterType"), str):
            raise L2IntegrityError("exchangeInfo contains a malformed filter")
        filter_type = item["filterType"]
        if filter_type in {"LOT_SIZE", "MARKET_LOT_SIZE", "MIN_NOTIONAL", "NOTIONAL"}:
            if filter_type in filters:
                raise L2IntegrityError(f"duplicate exchangeInfo {filter_type} filter")
            filters[filter_type] = item
    if "LOT_SIZE" not in filters:
        raise L2IntegrityError("exchangeInfo LOT_SIZE filter is missing")
    notionals: list[NotionalFilter] = []
    if item := filters.get("MIN_NOTIONAL"):
        notionals.append(NotionalFilter(
            "MIN_NOTIONAL",
            _exchange_decimal(item.get("minNotional"), field="minNotional"),
            None,
            _exchange_bool(item, "applyToMarket"),
            False,
            _exchange_int(item.get("avgPriceMins"), field="avgPriceMins"),
        ))
    if item := filters.get("NOTIONAL"):
        notionals.append(NotionalFilter(
            "NOTIONAL",
            _exchange_decimal(item.get("minNotional"), field="minNotional", zero_disables=True),
            _exchange_decimal(item.get("maxNotional"), field="maxNotional", zero_disables=True),
            _exchange_bool(item, "applyMinToMarket"),
            _exchange_bool(item, "applyMaxToMarket"),
            _exchange_int(item.get("avgPriceMins"), field="avgPriceMins"),
        ))
    return ExchangeConstraints(
        symbol,
        _quantity_filter(filters["LOT_SIZE"]),
        _quantity_filter(filters["MARKET_LOT_SIZE"]) if "MARKET_LOT_SIZE" in filters else None,
        tuple(notionals),
    )


class PublicExchangeInfoClient(Protocol):
    def get_exchange_info(self, symbol: str | None = None) -> dict[str, Any]: ...


def load_public_exchange_constraints(
    client: PublicExchangeInfoClient, *, symbol: str = "BTCUSDC"
) -> ExchangeConstraints:
    """Load only unauthenticated exchangeInfo through the existing public client."""
    return exchange_constraints_from_exchange_info(client.get_exchange_info(symbol), symbol=symbol)


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
            raise L2IntegrityError("EXIT closes available position quantity; quantity must be absent")


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
    snapshot: DepthSnapshot
    requested_quantity: Decimal
    filled_quantity: Decimal
    fill_ratio: Decimal
    levels_consumed: int
    best_quote: Decimal
    book_vwap: Decimal
    vwap: Decimal
    worst_book_price: Decimal
    worst_fill_price: Decimal
    quote_notional: Decimal
    depth_slippage_per_unit: Decimal
    depth_slippage_cost: Decimal
    spread_cost: Decimal
    additional_slippage_cost: Decimal
    total_execution_slippage_vs_midpoint: Decimal
    fee: Decimal

    @property
    def price(self) -> Decimal:
        return self.vwap

    @property
    def quantity(self) -> Decimal:
        return self.filled_quantity

    @property
    def notional(self) -> Decimal:
        return self.quote_notional


@dataclass(frozen=True)
class Trade:
    entry: Fill
    exit_fills: tuple[Fill, ...]
    exit_timestamp: datetime
    holding_time: timedelta
    mid_price_move: Decimal
    top_of_book_gross_pnl: Decimal
    executable_gross_pnl: Decimal
    filled_gross_pnl: Decimal
    spread_cost: Decimal
    depth_slippage_cost: Decimal
    additional_slippage_cost: Decimal
    fees: Decimal
    total_cost: Decimal
    net_pnl: Decimal
    mid_return: Decimal
    executable_gross_return: Decimal
    gross_return: Decimal
    net_return: Decimal

    @property
    def exit(self) -> Fill:
        return self.exit_fills[-1]


@dataclass(frozen=True)
class Replay:
    session_id: str
    rules: ExecutionRules
    constraints: ExchangeConstraints
    fill_mode: FillMode
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
            depth = sum((t.depth_slippage_cost for t in self.trades), ZERO)
            additional = sum((t.additional_slippage_cost for t in self.trades), ZERO)
            fees = sum((t.fees for t in self.trades), ZERO)
            notional = sum((
                t.entry.notional + sum((fill.notional for fill in t.exit_fills), ZERO)
                for t in self.trades
            ), ZERO)
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
                "average_cost_per_trade": (spread + depth + additional + fees) / count if count else None,
                "total_spread_cost": spread,
                "total_depth_slippage_cost": depth,
                "total_additional_slippage_cost": additional,
                "total_fees": fees,
                "total_cost": spread + depth + additional + fees,
            }

    def to_json(self) -> str:
        return canonical({"schema_version": 2, **asdict(self), "summary": self.summary()})

    def sha256(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()


@dataclass
class _Position:
    entry: Fill
    remaining_quantity: Decimal
    exit_fills: list[Fill]


def _make_fill(
    *, decision: Decision, execution_at: datetime, snapshot: DepthSnapshot,
    requested_quantity: Decimal, buy: bool, fill_mode: FillMode,
    model: SimulatedExecutionModel,
) -> Fill:
    levels = snapshot.asks if buy else snapshot.bids
    remaining = requested_quantity
    consumed: list[tuple[DepthLevel, Decimal]] = []
    for level in levels:
        take = min(remaining, level.quantity)
        if take > ZERO:
            consumed.append((level, take))
            remaining -= take
        if remaining == ZERO:
            break
    if remaining > ZERO and fill_mode == FULL_FILL_OR_REJECT:
        raise L2IntegrityError("insufficient causal depth for full fill")
    filled = requested_quantity - remaining
    if filled <= ZERO:
        raise L2IntegrityError("causal depth supplied no executable quantity")
    book_notional = sum((level.price * take for level, take in consumed), ZERO)
    book_vwap = book_notional / filled
    best = levels[0].price
    worst_book = consumed[-1][0].price
    if buy:
        vwap = model.buy_fill_price(book_vwap)
        worst_fill = model.buy_fill_price(worst_book)
        depth_per_unit = book_vwap - best
        spread_cost = filled * (best - snapshot.midpoint)
        additional = filled * (vwap - book_vwap)
        total_slippage = filled * (vwap - snapshot.midpoint)
    else:
        vwap = model.sell_fill_price(book_vwap)
        worst_fill = model.sell_fill_price(worst_book)
        depth_per_unit = best - book_vwap
        spread_cost = filled * (snapshot.midpoint - best)
        additional = filled * (book_vwap - vwap)
        total_slippage = filled * (snapshot.midpoint - vwap)
    depth_cost = filled * depth_per_unit
    if spread_cost + depth_cost + additional != total_slippage:
        raise L2IntegrityError("non-additive fill cost attribution")
    quote_notional = filled * vwap
    return Fill(
        decision, execution_at, snapshot, requested_quantity, filled,
        filled / requested_quantity, len(consumed), best, book_vwap, vwap,
        worst_book, worst_fill, quote_notional, depth_per_unit, depth_cost,
        spread_cost, additional, total_slippage, model.fee(quote_notional),
    )


def _complete_trade(position: _Position) -> Trade:
    entry = position.entry
    exits = tuple(position.exit_fills)
    quantity = entry.filled_quantity
    if sum((fill.filled_quantity for fill in exits), ZERO) != quantity:
        raise L2IntegrityError("exit fills do not reconcile to entry quantity")
    entry_mid = entry.snapshot.midpoint
    mid_move = sum((fill.filled_quantity * (fill.snapshot.midpoint - entry_mid) for fill in exits), ZERO)
    top_gross = sum((fill.filled_quantity * fill.best_quote for fill in exits), ZERO) - quantity * entry.best_quote
    executable = sum((fill.filled_quantity * fill.book_vwap for fill in exits), ZERO) - quantity * entry.book_vwap
    gross = sum((fill.notional for fill in exits), ZERO) - entry.notional
    spread = entry.spread_cost + sum((fill.spread_cost for fill in exits), ZERO)
    depth = entry.depth_slippage_cost + sum((fill.depth_slippage_cost for fill in exits), ZERO)
    additional = entry.additional_slippage_cost + sum((fill.additional_slippage_cost for fill in exits), ZERO)
    fees = entry.fee + sum((fill.fee for fill in exits), ZERO)
    net = gross - fees
    if mid_move - spread != top_gross or top_gross - depth != executable or executable - additional != gross:
        raise L2IntegrityError("non-additive trade execution attribution")
    if mid_move - spread - depth - additional - fees != net:
        raise L2IntegrityError("non-additive trade net attribution")
    denominator = entry.notional
    return Trade(
        entry, exits, exits[-1].executed_at, exits[-1].executed_at - entry.executed_at,
        mid_move, top_gross, executable, gross, spread, depth, additional, fees,
        spread + depth + additional + fees, net, mid_move / denominator,
        executable / denominator, gross / denominator, net / denominator,
    )


def replay(
    *, session_id: str, snapshots: Sequence[DepthSnapshot], decisions: Sequence[Decision],
    rules: ExecutionRules, constraints: ExchangeConstraints, fill_mode: FillMode,
    initial_cash: Decimal,
) -> Replay:
    """Replay one isolated session using the latest causally available depth."""
    if not isinstance(session_id, str) or not session_id.strip():
        raise L2IntegrityError("session identity required")
    if fill_mode not in (FULL_FILL_OR_REJECT, ALLOW_PARTIAL_FILL):
        raise L2IntegrityError("caller must choose an explicit supported fill mode")
    number(initial_cash, positive=True)
    snapshots, decisions = tuple(snapshots), tuple(decisions)
    if any(snapshot.symbol != constraints.symbol for snapshot in snapshots):
        raise L2IntegrityError("depth snapshot symbol does not match exchange constraints")
    times = [utc(snapshot.available_at) for snapshot in snapshots]
    if any(a >= b for a, b in zip(times, times[1:])):
        raise L2IntegrityError("depth availability must be strictly increasing")
    if any(a.source_at > b.source_at for a, b in zip(snapshots, snapshots[1:])):
        raise L2IntegrityError("depth source timestamps regressed")
    if any(a.sequence_id > b.sequence_id for a, b in zip(snapshots, snapshots[1:])):
        raise L2IntegrityError("depth sequence provenance regressed")
    decision_times = [utc(decision.at) for decision in decisions]
    if any(a >= b for a, b in zip(decision_times, decision_times[1:])):
        raise L2IntegrityError("decisions must be strictly increasing")
    with localcontext(Context(prec=50)):
        model = SimulatedExecutionModel(
            fee_bps=rules.fee_bps, slippage_bps=rules.slippage_bps,
            ambiguous_bar_policy=AmbiguousBarPolicy.STOP_FIRST,
        )
        cash = initial_cash
        position: _Position | None = None
        last_fill: datetime | None = None
        trades: list[Trade] = []
        for decision in decisions:
            if last_fill is not None and utc(decision.at) <= last_fill:
                raise L2IntegrityError("decision must follow the preceding fill; pending decisions prohibited")
            execution_at = utc(decision.at) + rules.latency
            index = bisect_right(times, execution_at) - 1
            if index < 0:
                raise L2IntegrityError("no causally available depth snapshot")
            snapshot = snapshots[index]
            if execution_at - utc(snapshot.source_at) > rules.max_quote_age:
                raise L2IntegrityError("executable depth is stale")
            if decision.action == "ENTER":
                if position is not None:
                    raise L2IntegrityError("overlapping LONG position")
                assert decision.quantity is not None
                requested = decision.quantity
                constraints.validate_market_quantity(requested, label="requested")
                fill = _make_fill(
                    decision=decision, execution_at=execution_at, snapshot=snapshot,
                    requested_quantity=requested, buy=True, fill_mode=fill_mode, model=model,
                )
                constraints.validate_market_quantity(fill.filled_quantity, label="filled")
                constraints.validate_market_notional(fill.notional)
                if fill.notional + fill.fee > cash:
                    raise L2IntegrityError("insufficient cash; borrowing prohibited")
                cash -= fill.notional + fill.fee
                position = _Position(fill, fill.filled_quantity, [])
            else:
                if position is None:
                    raise L2IntegrityError("EXIT without a LONG position")
                requested = position.remaining_quantity
                constraints.validate_market_quantity(requested, label="requested")
                fill = _make_fill(
                    decision=decision, execution_at=execution_at, snapshot=snapshot,
                    requested_quantity=requested, buy=False, fill_mode=fill_mode, model=model,
                )
                constraints.validate_market_quantity(fill.filled_quantity, label="filled")
                constraints.validate_market_notional(fill.notional)
                residual = requested - fill.filled_quantity
                if residual > ZERO:
                    constraints.validate_market_quantity(residual, label="residual position")
                position.exit_fills.append(fill)
                position.remaining_quantity = residual
                cash += fill.notional - fill.fee
                if residual == ZERO:
                    trades.append(_complete_trade(position))
                    position = None
            last_fill = execution_at
        if position is not None:
            raise L2IntegrityError("unclosed or partially closed LONG position")
        binding = canonical({
            "session_id": session_id,
            "snapshots": [asdict(snapshot) for snapshot in snapshots],
            "decisions": [asdict(decision) for decision in decisions],
            "rules": asdict(rules),
            "constraints": asdict(constraints),
            "fill_mode": fill_mode,
            "initial_cash": initial_cash,
        })
        return Replay(
            session_id, rules, constraints, fill_mode,
            hashlib.sha256(binding.encode()).hexdigest(), tuple(trades), initial_cash, cash,
        )


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
    """Compress causal one-second states using externally frozen rules."""

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
