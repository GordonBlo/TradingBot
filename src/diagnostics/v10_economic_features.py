"""Frozen V10 causal sampling. No file discovery, authorization or live entry point."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from typing import Any

from src.microstructure.v10 import _TimelineMachine
from src.orderbook.features import event_flow_features, state_features
from src.research.v10_collection_preregistration import timestamp, utc
from src.research.v10_collection_readiness import require
from src.research.v10_economic_discovery_preregistration import (
    FLOW_FEATURES,
    INTERACTIONS,
    L2_FEATURES,
    WINDOWS,
)

FEATURES = (*L2_FEATURES, *FLOW_FEATURES, *INTERACTIONS)
HORIZONS = (5, 30, 60, 300)
ZERO = Decimal(0)
SECOND = timedelta(seconds=1)
LATENCY = timedelta(milliseconds=100)
DEPTH_FLOW = (
    "bid_depth_added",
    "bid_depth_removed",
    "ask_depth_added",
    "ask_depth_removed",
    "update_count",
)


@dataclass(frozen=True)
class Quote:
    query_at: datetime
    available_at: datetime
    source_at: datetime
    record_index: int
    bid: Decimal
    ask: Decimal

    @property
    def mid(self) -> Decimal:
        return (self.bid + self.ask) / 2

    def provenance(self) -> dict:
        return {
            "query_at": timestamp(self.query_at),
            "available_at": timestamp(self.available_at),
            "source_at": timestamp(self.source_at),
            "record_index": self.record_index,
            "bid": str(self.bid),
            "ask": str(self.ask),
        }


@dataclass(frozen=True)
class Sample:
    at: datetime
    features: tuple[Decimal | None, ...]
    targets: dict[int, Decimal]
    economics: dict[int, dict]
    provenance: dict


@dataclass(frozen=True)
class SessionSamples:
    session_id: str
    started_at: datetime
    ended_at: datetime
    rows: tuple[Sample, ...]
    exclusions: dict[str, int]


def quote_economics(entry: Quote, exit_quote: Quote) -> dict:
    """Exact unit quote-price response, not an order, position or sized fill."""
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        for quote in (entry, exit_quote):
            require(
                all(
                    isinstance(x, Decimal) and x.is_finite() and x > 0
                    for x in (quote.bid, quote.ask)
                )
                and quote.bid < quote.ask,
                "invalid quote",
            )
            require(
                quote.available_at <= quote.query_at
                and quote.source_at <= quote.query_at,
                "future economic quote",
            )
        require(entry.query_at < exit_quote.query_at, "reversed economic interval")
        a, b, me, mx = entry.ask, exit_quote.bid, entry.mid, exit_quote.mid
        result: dict[str, Any] = {
            "unadjusted_quote_return_bps": Decimal(10000) * (b / a - 1)
        }
        for name, fee_bps, slip_bps in (("base", 10, 2), ("stress", 20, 4)):
            f, s = Decimal(fee_bps) / 10000, Decimal(slip_bps) / 10000
            paid, received = a * (1 + s), b * (1 - s)
            result[name] = {
                "mid_move": 10000 * (mx - me) / paid,
                "executable_gross": 10000 * (b - a) / paid,
                "spread": 10000 * ((a - me) + (mx - b)) / paid,
                "adverse": 10000 * ((paid - a) + (b - received)) / paid,
                "fees": 10000 * f * (paid + received) / paid,
                "net": 10000 * (received - paid - f * (paid + received)) / paid,
            }
        return result


class SamplingMachine(_TimelineMachine):
    """Observe the frozen acquisition state machine, including buffered diff bridging."""

    def __init__(self, max_levels: int):
        super().__init__(max_levels_per_side=max_levels)
        self.source_at: datetime | None = None
        self.synchronized_at: datetime | None = None
        self.first_trade_at: datetime | None = None
        self.flows: deque = deque()
        self.trades: dict[int, deque] = {w: deque() for w in WINDOWS}
        self.trade_sums = {w: [ZERO] * 5 for w in WINDOWS}

    def _apply(self, event, *, received, record_index):
        before = self.depth_counters.reconstructed_updates
        flow = event_flow_features(self.book, event)
        valid = super()._apply(event, received=received, record_index=record_index)
        if valid and self.depth_counters.reconstructed_updates > before:
            self.source_at = event.event_time
            if self.synchronized_at is None:
                self.synchronized_at = received
            self.flows.append(
                (received, event.event_time, tuple(flow[name] for name in DEPTH_FLOW))
            )
        return valid

    def accept(self, kind: str, record: dict) -> None:
        if kind == "DEPTH":
            if record["record_type"] in ("RESYNC_BOUNDARY", "REST_SNAPSHOT"):
                self.source_at = None
                self.synchronized_at = None
                self.flows.clear()
                if record["record_type"] == "RESYNC_BOUNDARY":
                    self.first_trade_at = None
                    for w in WINDOWS:
                        self.trades[w].clear()
                        self.trade_sums[w] = [ZERO] * 5
            self.depth(record)
        else:
            observed = self.aggtrade(record)
            if observed is not None:
                if self.first_trade_at is None:
                    self.first_trade_at = observed.received_at
                trade = observed.trade
                buy = not trade.buyer_is_maker
                values = (
                    Decimal(int(buy)),
                    Decimal(int(not buy)),
                    trade.quantity if buy else ZERO,
                    ZERO if buy else trade.quantity,
                    trade.price * trade.quantity * (1 if buy else -1),
                )
                for w in WINDOWS:
                    self.trades[w].append(
                        (observed.received_at, observed.exchange_event_at, values)
                    )
                    self.trade_sums[w] = [
                        a + b for a, b in zip(self.trade_sums[w], values)
                    ]
        depth = self.depth_counters
        trades = self.aggtrades.counters
        require(
            not any(
                (
                    depth.sequence_gaps,
                    depth.invalid_events,
                    depth.crossed_invalid_book_states,
                    trades.invalid_events,
                    trades.id_regressions,
                    trades.conflicting_duplicates,
                    trades.aggregate_id_gap_events,
                    trades.underlying_trade_id_gap_events,
                )
            ),
            "raw reconstruction integrity failure",
        )

    def expire(self, at: datetime) -> None:
        while self.flows and self.flows[0][0] <= at - SECOND:
            self.flows.popleft()
        for w in WINDOWS:
            while self.trades[w] and self.trades[w][0][0] <= at - timedelta(seconds=w):
                _, _, values = self.trades[w].popleft()
                self.trade_sums[w] = [a - b for a, b in zip(self.trade_sums[w], values)]

    def quote(self, at: datetime) -> Quote | None:
        available, source = self.state_available_at, self.source_at
        if (
            available is None
            or source is None
            or self.book.last_update_id is None
            or self.state_record_index is None
        ):
            return None
        if not (
            timedelta(0) <= at - available <= SECOND
            and timedelta(0) <= at - source <= SECOND
        ):
            return None
        bid, ask = self.book.best_bid, self.book.best_ask
        if bid is None or ask is None or bid[0] >= ask[0]:
            raise ValueError("invalid sampled book")
        return Quote(at, available, source, self.state_record_index, bid[0], ask[0])

    def features(self, at: datetime) -> tuple[Decimal | None, ...]:
        self.expire(at)
        require(
            all(source <= at for _, source, _ in self.flows), "future depth flow source"
        )
        require(
            all(source <= at for _, source, _ in self.trades[30]),
            "future trade flow source",
        )
        values = state_features(self.book)
        sums = [sum((item[2][i] for item in self.flows), ZERO) for i in range(5)]
        values.update({f"{name}_1s": sums[i] for i, name in enumerate(DEPTH_FLOW[:4])})
        values["update_intensity_1s"] = sums[4]
        for w in WINDOWS:
            buy_n, sell_n, buy_q, sell_q, signed_notional = self.trade_sums[w]
            signed = buy_q - sell_q
            imbalance = signed / (buy_q + sell_q) if buy_q + sell_q else ZERO
            ordered = (signed, buy_n, sell_n, buy_q, sell_q, signed_notional, imbalance)
            names = [name for name in FLOW_FEATURES if name.endswith(f"_{w}s")]
            values.update(zip(names, ordered))
            for name in ("depth_imbalance_20", "microprice_minus_mid_bps"):
                value = values[name]
                values[f"{name}_x_flow_imbalance_{w}s"] = (
                    value * imbalance if value is not None else None
                )
        return tuple(values[name] for name in FEATURES)


def sample_session(
    records: Iterable[tuple[str, dict]],
    *,
    session_id: str,
    started_at: datetime,
    ended_at: datetime,
    max_levels: int,
) -> SessionSamples:
    """Consume a canonical merged iterator once. Keep only grid quotes/features, not raw books."""
    start, end = utc(timestamp(started_at)), utc(timestamp(ended_at))
    require(start < end, "invalid session interval")
    end = min(end, start + timedelta(seconds=10800))
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        machine = SamplingMachine(max_levels)
        iterator = iter(records)
        pending = next(iterator, None)
        previous_key = None
        indices = {"DEPTH": 0, "AGGTRADE": 0}
        quotes: dict[datetime, Quote | None] = {}
        features: dict[datetime, tuple] = {}
        rejected: Counter = Counter()
        at = start.replace(microsecond=0)
        if at < start:
            at += SECOND
        while at <= end:
            for query in (at, at + LATENCY):
                if query > end:
                    continue
                while (
                    pending is not None and utc(pending[1]["received_at_utc"]) <= query
                ):
                    kind, record = pending
                    require(
                        kind in indices and record["session_id"] == session_id,
                        "source identity mismatch",
                    )
                    received = utc(record["received_at_utc"])
                    key = (
                        received,
                        0 if kind == "DEPTH" else 1,
                        record["record_index"],
                    )
                    require(
                        start <= received
                        and (previous_key is None or key > previous_key),
                        "noncanonical causal merge",
                    )
                    require(record["record_index"] == indices[kind], "source index gap")
                    indices[kind] += 1
                    previous_key = key
                    machine.accept(kind, record)
                    machine.expire(received)
                    pending = next(iterator, None)
                quotes[query] = machine.quote(query)
                if query == at:
                    first_trade, synchronized = (
                        machine.first_trade_at,
                        machine.synchronized_at,
                    )
                    if first_trade is None or synchronized is None:
                        rejected["unobserved_stream_warmup"] += 1
                    elif at - timedelta(seconds=30) < max(
                        start, first_trade, synchronized
                    ):
                        rejected["warmup"] += 1
                    elif quotes[query] is None:
                        rejected["missing_or_stale_decision_quote"] += 1
                    else:
                        features[at] = machine.features(at)
            at += SECOND
        rows = []
        for at, vector in features.items():
            endpoints = [at, at + LATENCY]
            for horizon in HORIZONS:
                endpoints.extend(
                    (
                        at + timedelta(seconds=horizon),
                        at + timedelta(seconds=horizon) + LATENCY,
                    )
                )
            if any(point > end for point in endpoints):
                rejected["terminal_target_exclusion"] += 1
                continue
            if any(quotes.get(point) is None for point in endpoints):
                rejected["missing_or_stale_common_endpoint"] += 1
                continue
            selected: dict[datetime, Quote] = {}
            for point in endpoints:
                quote = quotes[point]
                if quote is None:
                    raise ValueError("missing common endpoint")
                selected[point] = quote
            targets, economics = {}, {}
            for horizon in HORIZONS:
                exit_at = at + timedelta(seconds=horizon)
                targets[horizon] = 10000 * (
                    selected[exit_at].mid / selected[at].mid - 1
                )
                economics[horizon] = quote_economics(
                    selected[at + LATENCY], selected[exit_at + LATENCY]
                )
            rows.append(
                Sample(
                    at,
                    vector,
                    targets,
                    economics,
                    {timestamp(q.query_at): q.provenance() for q in selected.values()},
                )
            )
        require(len(rows) >= 1202, "fewer than 1202 common rows in session")
        return SessionSamples(
            session_id, start, end, tuple(rows), dict(sorted(rejected.items()))
        )
