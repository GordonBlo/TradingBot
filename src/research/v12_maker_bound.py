"""Conditional maker quantity bounds, not an exchange/order/strategy simulator.

Only normalized, integrity-checked public observations may be supplied. No
network, data loader, account, fees, PnL, strategy, or predictive model exists
here. See docs/v12_conditional_maker_bound.md for the conditional theorem.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Context, Decimal, localcontext
from enum import Enum

from src.backtest.v10_l2 import L2IntegrityError, number, utc


D = Decimal
ARITHMETIC = Context(prec=80)
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _integer(value: int, label: str) -> None:
    if type(value) is not int or value < 0:
        raise L2IntegrityError(f"nonnegative integer {label} required")


def _quantity(value: Decimal) -> None:
    number(value, positive=True)
    # Bounded input domain ensures exact addition/subtraction in our context.
    if len(value.as_tuple().digits) > 36 or abs(value.as_tuple().exponent) > 18:
        raise L2IntegrityError("Decimal outside exact arithmetic domain")


def _epoch_us(value: datetime) -> int:
    delta = utc(value) - EPOCH
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


class State(str, Enum):
    DECIDED = "DECIDED"
    SUBMISSION_PENDING = "SUBMISSION_PENDING"
    CONDITIONALLY_WORKING = "CONDITIONALLY_WORKING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED_ASSUMED = "REJECTED_ASSUMED"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True)
class Timing:
    """Exogenous scenario inputs; offset = exchange clock minus local UTC."""
    submission_latency_us: int
    cancellation_latency_us: int
    exchange_offset_min_us: int
    exchange_offset_max_us: int

    def __post_init__(self) -> None:
        _integer(self.submission_latency_us, "submission latency")
        _integer(self.cancellation_latency_us, "cancellation latency")
        if any(type(x) is not int for x in (
            self.exchange_offset_min_us, self.exchange_offset_max_us,
        )) or self.exchange_offset_min_us > self.exchange_offset_max_us:
            raise L2IntegrityError("ordered integer exchange clock bounds required")


@dataclass(frozen=True)
class Assumptions:
    accepted_and_resting_until_cancel: bool
    ordinary_price_priority: bool
    fixed_external_order_instructions: bool
    no_self_trade_prevention: bool
    filters_and_continuous_status_valid: bool
    arrival_best_ask: Decimal | None

    def __post_init__(self) -> None:
        for item in (
            self.accepted_and_resting_until_cancel, self.ordinary_price_priority,
            self.fixed_external_order_instructions, self.no_self_trade_prevention,
            self.filters_and_continuous_status_valid,
        ):
            if type(item) is not bool:
                raise L2IntegrityError("explicit Boolean assumptions required")
        if self.arrival_best_ask is not None:
            _quantity(self.arrival_best_ask)

    def supported(self) -> bool:
        return all((
            self.accepted_and_resting_until_cancel, self.ordinary_price_priority,
            self.fixed_external_order_instructions, self.no_self_trade_prevention,
            self.filters_and_continuous_status_valid,
        )) and self.arrival_best_ask is not None


@dataclass(frozen=True)
class Order:
    order_id: str
    session_id: str
    limit: Decimal
    quantity: Decimal
    decision_utc: datetime
    decision_monotonic_ns: int

    def __post_init__(self) -> None:
        if not self.order_id or not self.session_id:
            raise L2IntegrityError("order and session identity required")
        _quantity(self.limit)
        _quantity(self.quantity)
        utc(self.decision_utc)
        _integer(self.decision_monotonic_ns, "decision monotonic time")


@dataclass(frozen=True)
class Receipt:
    session_id: str
    connection_id: str
    stream: str
    ordinal: int
    utc: datetime
    monotonic_ns: int
    payload_sha256: str

    def __post_init__(self) -> None:
        if not self.session_id or not self.connection_id or self.stream not in (
            "TRADE", "AGGTRADE", "DEPTH", "BOOK_TICKER", "CONTROL",
        ):
            raise L2IntegrityError("public receipt identity required")
        _integer(self.ordinal, "ingestion ordinal")
        _integer(self.monotonic_ns, "receipt monotonic time")
        utc(self.utc)
        if not isinstance(self.payload_sha256, str) or len(self.payload_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.payload_sha256
        ):
            raise L2IntegrityError("lowercase SHA-256 receipt binding required")


@dataclass(frozen=True)
class PublicTrade:
    trade_id: int
    price: Decimal
    quantity: Decimal
    buyer_is_maker: bool
    trade_time_us: int
    event_time_us: int
    timestamp_resolution_us: int
    receipt: Receipt
    symbol: str = "BTCUSDC"

    def __post_init__(self) -> None:
        for value, label in ((self.trade_id, "trade ID"), (self.trade_time_us, "T"),
                             (self.event_time_us, "E")):
            _integer(value, label)
        _quantity(self.price)
        _quantity(self.quantity)
        if self.timestamp_resolution_us not in (1, 1000) or type(
            self.timestamp_resolution_us
        ) is not int:
            raise L2IntegrityError("explicit MICROSECOND or MILLISECOND precision required")
        if self.event_time_us < self.trade_time_us:
            raise L2IntegrityError("trade event time precedes trade time")
        if type(self.buyer_is_maker) is not bool or self.symbol != "BTCUSDC":
            raise L2IntegrityError("BTCUSDC buyer-is-maker semantics required")
        if self.receipt.stream != "TRADE":
            raise L2IntegrityError("only individual @trade observations can credit volume")


@dataclass(frozen=True)
class Credit:
    trade_id: int
    quantity: Decimal
    price: Decimal
    observed_at: datetime
    ingestion_ordinal: int


class ConditionalMakerBound:
    """One hypothetical order, one tape; credit is a conditional lower bound.

    Initial high-water IDs must come from valid pre-decision stream coverage.
    Feed EVERY envelope in ingestion order, including auxiliary observations.
    The upstream adapter verifies hashes, reconstructions and clock provenance.
    """

    def __init__(self, order: Order, timing: Timing, assumptions: Assumptions, *,
                 initial_trade_id: int, initial_depth_id: int,
                 initial_ordinal: int, coverage_established: bool):
        for value, label in ((initial_trade_id, "initial trade ID"),
                             (initial_depth_id, "initial depth ID"),
                             (initial_ordinal, "initial ordinal")):
            _integer(value, label)
        if type(coverage_established) is not bool:
            raise L2IntegrityError("explicit coverage status required")
        self.order, self.timing, self.assumptions = order, timing, assumptions
        self.state = State.DECIDED
        self.reason: str | None = None
        self.credited = D(0)
        self._credits: list[Credit] = []
        self._trade_id, self._depth_id, self._ordinal = (
            initial_trade_id, initial_depth_id, initial_ordinal,
        )
        self._seen: dict[int, tuple] = {}
        self._connections: dict[str, str] = {}
        self._last_trade_time: int | None = None
        self._last_receipt_utc = utc(order.decision_utc)
        self._last_receipt_ns = order.decision_monotonic_ns
        self._now_utc, self._now_ns = self._last_receipt_utc, self._last_receipt_ns
        self._working_ns = order.decision_monotonic_ns + timing.submission_latency_us * 1000
        self._working_upper = (_epoch_us(order.decision_utc)
                               + timing.submission_latency_us + timing.exchange_offset_max_us)
        self._cancel_ns: int | None = None
        self._cancel_lower: int | None = None
        if not coverage_established:
            self._invalidate("initial stream/book/clock coverage unknown")

    @property
    def credits(self) -> tuple[Credit, ...]:
        return tuple(self._credits)

    def _invalidate(self, reason: str) -> None:
        if self.state != State.INDETERMINATE:
            self.reason = reason
        self.state = State.INDETERMINATE

    def advance(self, *, as_of: datetime, monotonic_ns: int) -> None:
        """Local causal clock, never a claim about exchange acknowledgement."""
        now = utc(as_of)
        _integer(monotonic_ns, "as-of monotonic time")
        if now < self._now_utc or monotonic_ns < self._now_ns:
            self._invalidate("local replay clock regressed")
            raise L2IntegrityError(self.reason)
        self._now_utc, self._now_ns = now, monotonic_ns
        if self.state == State.SUBMISSION_PENDING and monotonic_ns >= self._working_ns:
            ask = self.assumptions.arrival_best_ask
            if ask is not None and self.order.limit >= ask:
                if self.assumptions.accepted_and_resting_until_cancel:
                    self._invalidate("acceptance contradicts assumed crossing arrival")
                else:
                    self.state = State.REJECTED_ASSUMED
            elif self.assumptions.supported():
                self.state = State.CONDITIONALLY_WORKING
            else:
                self._invalidate("acceptance/matching assumptions unsupported")
        if self.state == State.CANCEL_PENDING and monotonic_ns >= self._cancel_ns:
            self.state = State.CANCELLED

    def submit(self) -> None:
        if self.state != State.DECIDED:
            raise L2IntegrityError("submission requires DECIDED")
        if self._now_ns != self.order.decision_monotonic_ns or self._now_utc != utc(
            self.order.decision_utc
        ):
            raise L2IntegrityError("submission must be anchored at the declared decision")
        self.state = State.SUBMISSION_PENDING

    def request_cancel(self, *, as_of: datetime, monotonic_ns: int) -> None:
        self.advance(as_of=as_of, monotonic_ns=monotonic_ns)
        if self.state not in (State.CONDITIONALLY_WORKING, State.PARTIALLY_FILLED):
            raise L2IntegrityError("cancel requires a conditionally working order")
        self._cancel_ns = monotonic_ns + self.timing.cancellation_latency_us * 1000
        self._cancel_lower = (_epoch_us(as_of) + self.timing.cancellation_latency_us
                              + self.timing.exchange_offset_min_us)
        self.state = State.CANCEL_PENDING

    def _receive(self, receipt: Receipt, *, as_of: datetime, monotonic_ns: int) -> bool:
        if utc(receipt.utc) > utc(as_of) or receipt.monotonic_ns > monotonic_ns:
            self._invalidate("future receipt")
            raise L2IntegrityError(self.reason)
        self.advance(as_of=as_of, monotonic_ns=monotonic_ns)
        if self.state == State.INDETERMINATE:
            return False
        if receipt.session_id != self.order.session_id:
            self._invalidate("session changed")
        elif receipt.ordinal != self._ordinal + 1:
            self._invalidate("ingestion ordinal gap/regression")
        elif utc(receipt.utc) < self._last_receipt_utc or (
            receipt.monotonic_ns < self._last_receipt_ns
        ):
            self._invalidate("receipt clock regressed")
        elif receipt.stream in self._connections and (
            self._connections[receipt.stream] != receipt.connection_id
        ):
            self._invalidate("connection changed without continuous coverage")
        if self.state == State.INDETERMINATE:
            return False
        self._ordinal = receipt.ordinal
        self._last_receipt_utc, self._last_receipt_ns = utc(receipt.utc), receipt.monotonic_ns
        self._connections[receipt.stream] = receipt.connection_id
        return True

    def observe_auxiliary(self, receipt: Receipt, *, as_of: datetime,
                          monotonic_ns: int) -> None:
        """Healthy aggTrade/bookTicker/control envelopes: NEVER fill evidence."""
        if receipt.stream not in ("AGGTRADE", "BOOK_TICKER", "CONTROL"):
            raise L2IntegrityError("auxiliary receipt cannot hide individual trades/depth")
        self._receive(receipt, as_of=as_of, monotonic_ns=monotonic_ns)

    def observe_depth(self, first_id: int, final_id: int, receipt: Receipt, *,
                      as_of: datetime, monotonic_ns: int) -> None:
        _integer(first_id, "depth U")
        _integer(final_id, "depth u")
        if first_id > final_id or receipt.stream != "DEPTH":
            raise L2IntegrityError("invalid depth update interval")
        if not self._receive(receipt, as_of=as_of, monotonic_ns=monotonic_ns):
            return
        if final_id <= self._depth_id:
            return
        if first_id > self._depth_id + 1:
            self._invalidate("dropped depth update")
            return
        self._depth_id = final_id

    def observe_boundary(self, reason: str, receipt: Receipt, *, as_of: datetime,
                         monotonic_ns: int) -> None:
        """Any outage, resync, snapshot, stale/unknown coverage or status change.

        A later valid snapshot cannot restore THIS order's missing history.
        Previous valid credit remains a bound; no additional credit is allowed.
        """
        if not reason:
            raise L2IntegrityError("explicit integrity boundary reason required")
        if self._receive(receipt, as_of=as_of, monotonic_ns=monotonic_ns):
            self._invalidate(reason)

    def observe_trade(self, trade: PublicTrade, *, as_of: datetime,
                      monotonic_ns: int) -> None:
        if not self._receive(trade.receipt, as_of=as_of, monotonic_ns=monotonic_ns):
            return
        fingerprint = (trade.price, trade.quantity, trade.buyer_is_maker,
                       trade.trade_time_us, trade.event_time_us, trade.timestamp_resolution_us)
        if trade.trade_id in self._seen:
            if self._seen[trade.trade_id] != fingerprint:
                self._invalidate("conflicting duplicate trade")
            return
        if trade.trade_id != self._trade_id + 1:
            self._invalidate("dropped/unordered individual trade")
            return
        if self._last_trade_time is not None and trade.trade_time_us < self._last_trade_time:
            self._invalidate("exchange trade time regressed")
            return
        latest_possible_receipt_exchange = (_epoch_us(trade.receipt.utc)
                                           + self.timing.exchange_offset_max_us)
        if trade.event_time_us > latest_possible_receipt_exchange:
            self._invalidate("exchange timestamp incompatible with receipt clock bounds")
            return
        self._seen[trade.trade_id] = fingerprint
        self._trade_id, self._last_trade_time = trade.trade_id, trade.trade_time_us
        if self.state not in (State.CONDITIONALLY_WORKING, State.PARTIALLY_FILLED,
                              State.CANCEL_PENDING, State.CANCELLED):
            return
        latest_trade_time = trade.trade_time_us + trade.timestamp_resolution_us - 1
        if trade.trade_time_us <= self._working_upper:
            return  # Arrival ties and uncertain timestamp buckets give zero credit.
        if self._cancel_lower is not None and latest_trade_time >= self._cancel_lower:
            return  # Cancellation ties/races with uncertain ordering give zero credit.
        if not trade.buyer_is_maker or trade.price >= self.order.limit:
            return
        with localcontext(ARITHMETIC):
            quantity = min(self.order.quantity - self.credited, trade.quantity)
            self.credited += quantity
        self._credits.append(Credit(trade.trade_id, quantity, self.order.limit,
                                    utc(trade.receipt.utc), trade.receipt.ordinal))
        if self.credited == self.order.quantity:
            self.state = State.FILLED
        elif self.state == State.CONDITIONALLY_WORKING:
            self.state = State.PARTIALLY_FILLED
        # CANCEL_PENDING / CANCELLED retain cancellation state and partial quantity.
