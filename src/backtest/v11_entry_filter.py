"""Research-only V11 preparation mechanics. No strategy, data loader or orders.

Consumes an already-fixed parent schedule. It cannot generate or resize trades.
Latency-aware fills are an explicit proposed V11 amendment, NOT a V6 replay.
Full depth or an integrity failure: insufficient liquidity never disappears from
the paired sample. The old V10 primitives supply accounting, not its executor.
"""
from __future__ import annotations

import hashlib
import math
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Context, Decimal, localcontext
from itertools import pairwise

from src.backtest.execution import SimulatedExecutionModel
from src.backtest.models import AmbiguousBarPolicy
from src.backtest.v10_l2 import (
    FULL_FILL_OR_REJECT,
    Decision,
    DepthSnapshot,
    ExchangeConstraints,
    Fill,
    L2IntegrityError,
    Trade,
    _complete_trade,
    _make_fill,
    _Position,
    canonical,
    number,
    utc,
)
from src.strategy.v6_mtf_continuation import V6MTFContinuationStrategy as V6

D = Decimal
BAR = timedelta(minutes=15)
LATENCY = timedelta(milliseconds=100)
MAX_OBSERVATION_DELAY = timedelta(seconds=1)
BASE_COSTS = (D(10), D(2))
STRESS_COSTS = (D(20), D(4))


def bar_open(at: datetime) -> datetime:
    at = utc(at)
    return at.replace(minute=(at.minute // 15) * 15, second=0, microsecond=0)


@dataclass(frozen=True)
class ParentCandidate:
    candidate_id: str
    candle_open: datetime
    candle_close: datetime
    decision_received_at: datetime
    finalized: bool
    atr14_4h: Decimal
    atr_available_at: datetime
    quantity: Decimal

    def __post_init__(self) -> None:
        if not self.candidate_id or self.finalized is not True:
            raise L2IntegrityError("finalized parent candidate identity required")
        if utc(self.candle_open) != bar_open(self.candle_open) or (
            utc(self.candle_close) - utc(self.candle_open) != BAR
        ):
            raise L2IntegrityError("parent requires an aligned completed 15m candle")
        if not utc(self.candle_close) <= utc(self.decision_received_at) <= (
            utc(self.candle_close) + MAX_OBSERVATION_DELAY
        ):
            raise L2IntegrityError("parent decision precedes close or is late")
        if utc(self.atr_available_at) > utc(self.decision_received_at):
            raise L2IntegrityError("future parent ATR")
        number(self.atr14_4h, positive=True)
        number(self.quantity, positive=True)


@dataclass(frozen=True)
class FilterDecision:
    candidate_id: str
    bucket_close: datetime
    source_available_at: datetime
    prediction: float | None

    def accepted(self, parent: ParentCandidate) -> bool:
        if self.candidate_id != parent.candidate_id:
            raise L2IntegrityError("filter candidate identity mismatch")
        if utc(self.bucket_close) != utc(parent.candle_close) or (
            utc(self.source_available_at) > utc(self.bucket_close)
        ):
            raise L2IntegrityError("future or misaligned L2 score")
        if self.prediction is None:
            return False
        if isinstance(self.prediction, bool) or not isinstance(self.prediction, (int, float)):
            raise L2IntegrityError("numeric score required")
        if not math.isfinite(self.prediction):
            raise L2IntegrityError("nonfinite score")
        return self.prediction > 0


@dataclass(frozen=True)
class ReferenceTrade:
    parent: ParentCandidate
    trade: Trade
    stress_trade: Trade
    initial_distance: Decimal
    stop: Decimal
    target: Decimal
    trigger_at: datetime
    exit_reason: str
    cooldown_until: datetime


@dataclass(frozen=True)
class PairedRow:
    reference: ReferenceTrade
    decision: FilterDecision
    accepted: bool
    parent_net_r: Decimal
    filtered_net_r: Decimal
    opportunity_improvement_r: Decimal


@dataclass(frozen=True)
class PairedResult:
    rows: tuple[PairedRow, ...]
    parent_count: int
    accepted_count: int
    parent_expectancy_r: Decimal
    accepted_expectancy_r: Decimal | None
    opportunity_improvement_r: Decimal


class _Books:
    def __init__(self, snapshots: Sequence[DepthSnapshot], constraints: ExchangeConstraints):
        self.items = tuple(snapshots)
        self.times = tuple(utc(s.available_at) for s in self.items)
        if not self.items or constraints.symbol != "BTCUSDC":
            raise L2IntegrityError("BTCUSDC Spot depth required")
        if any(s.symbol != "BTCUSDC" for s in self.items):
            raise L2IntegrityError("wrong depth symbol")
        for a, b in zip(self.items, self.items[1:]):
            if (a.available_at >= b.available_at or a.source_at > b.source_at
                    or a.sequence_id >= b.sequence_id):
                raise L2IntegrityError("unordered or ambiguous depth provenance")
        self.constraints = constraints

    def at_or_after(self, at: datetime) -> tuple[int, DepthSnapshot]:
        index = bisect_left(self.times, utc(at))
        if index == len(self.items) or self.times[index] - utc(at) > MAX_OBSERVATION_DELAY:
            raise L2IntegrityError("missing executable observation within latency budget")
        snapshot = self.items[index]
        self.fresh(snapshot)
        return index, snapshot

    @staticmethod
    def fresh(snapshot: DepthSnapshot) -> None:
        if utc(snapshot.available_at) - utc(snapshot.source_at) > MAX_OBSERVATION_DELAY:
            raise L2IntegrityError("stale depth")

    def fill(self, decision: Decision, snapshot: DepthSnapshot, quantity: Decimal,
             costs: tuple[Decimal, Decimal]) -> Fill:
        self.constraints.validate_market_quantity(quantity, label="fixed parent")
        model = SimulatedExecutionModel(fee_bps=costs[0], slippage_bps=costs[1],
                                       ambiguous_bar_policy=AmbiguousBarPolicy.STOP_FIRST)
        fill = _make_fill(
            decision=decision, execution_at=snapshot.available_at, snapshot=snapshot,
            requested_quantity=quantity, buy=decision.action == "ENTER",
            fill_mode=FULL_FILL_OR_REJECT, model=model,
        )
        self.constraints.validate_market_notional(fill.notional)
        return fill

    @staticmethod
    def executable_bid(snapshot: DepthSnapshot, quantity: Decimal) -> Decimal:
        """Full-quantity liquidation VWAP before additional slippage/fees."""
        remaining, notional = quantity, D(0)
        for level in snapshot.bids:
            take = min(remaining, level.quantity)
            notional += take * level.price
            remaining -= take
            if remaining == 0:
                return notional / quantity
        raise L2IntegrityError("insufficient bid depth for protective observation")


def build_reference(
    candidates: Sequence[ParentCandidate], snapshots: Sequence[DepthSnapshot], *,
    constraints: ExchangeConstraints, initial_cash: Decimal,
) -> tuple[ReferenceTrade, ...]:
    """Execute ALL supplied parents before filtering; no divergent account state.

    The caller binds a V6-derived schedule; this function validates chronology,
    inventory, cooldown and cash, and never silently skips conflicting parents.
    Sequence-verified, gap-free reconstructed books are an upstream prerequisite;
    update IDs need not be consecutive because a diff update can span IDs.
    """
    candidates = tuple(candidates)
    if not candidates or len({c.candidate_id for c in candidates}) != len(candidates):
        raise L2IntegrityError("nonempty unique parent schedule required")
    if any(a.candle_close >= b.candle_close for a, b in pairwise(candidates)):
        raise L2IntegrityError("parent schedule is not chronological")
    number(initial_cash, positive=True)
    books = _Books(snapshots, constraints)
    with localcontext(Context(prec=50)):
        cash, results = initial_cash, []
        for parent in candidates:
            if results and parent.candle_close < results[-1].cooldown_until:
                raise L2IntegrityError("parent overlaps reference position/cooldown; no replacements")
            entry_decision = Decision(parent.decision_received_at, parent.decision_received_at,
                                      "ENTER", parent.quantity)
            entry_index, entry_book = books.at_or_after(parent.decision_received_at + LATENCY)
            if entry_book.available_at >= parent.candle_close + BAR:
                raise L2IntegrityError("entry escaped the next parent bar")
            entry = books.fill(entry_decision, entry_book, parent.quantity, BASE_COSTS)
            if entry.notional + entry.fee > cash:
                raise L2IntegrityError("reference cash insufficient; no leverage")
            distance = max(parent.atr14_4h, entry.price * V6.MIN_STOP_DISTANCE_FRACTION)
            stop, target = entry.price - distance, entry.price + V6.REWARD_RISK_RATIO * distance
            if stop <= 0:
                raise L2IntegrityError("invalid parent stop geometry")
            deadline = parent.candle_close + V6.MAXIMUM_HOLD_BARS * BAR
            trigger_at, reason = deadline, "MAX_HOLD"
            previous_at = entry_book.available_at
            for snapshot_index in range(entry_index, len(books.items)):
                snapshot = books.items[snapshot_index]
                # A gap while exposed cannot silently hide an earlier trigger.
                if min(snapshot.available_at, deadline) - previous_at > MAX_OBSERVATION_DELAY:
                    raise L2IntegrityError("unobserved protective path")
                if snapshot.available_at > deadline:
                    break
                books.fresh(snapshot)
                if snapshot is not entry_book and snapshot.source_at < entry.executed_at:
                    # A delayed pre-entry update is not a post-entry price path.
                    previous_at = snapshot.available_at
                    continue
                bid = books.executable_bid(snapshot, parent.quantity)
                if bid <= stop or bid >= target:
                    trigger_at = snapshot.available_at
                    reason = "STOP" if bid <= stop else "TARGET"
                    break
                previous_at = snapshot.available_at
            else:
                if previous_at < deadline:
                    raise L2IntegrityError("unclosed parent path")
            exit_decision = Decision(trigger_at, trigger_at, "EXIT")
            _, exit_book = books.at_or_after(trigger_at + LATENCY)
            exit_fill = books.fill(exit_decision, exit_book, parent.quantity, BASE_COSTS)
            trade = _complete_trade(_Position(entry, D(0), [exit_fill]))
            # Stress reprices exactly these observations and this quantity. No
            # new stops, signals, account schedule or capital-dependent sizing.
            stress_entry = books.fill(entry_decision, entry_book, parent.quantity, STRESS_COSTS)
            stress_exit = books.fill(exit_decision, exit_book, parent.quantity, STRESS_COSTS)
            stress = _complete_trade(_Position(stress_entry, D(0), [stress_exit]))
            cash += trade.net_pnl
            # V6 adapter observes an exit at its candle close (index zero),
            # blocks bars_since_exit <= 4, and permits the fifth later close.
            cooldown_until = bar_open(exit_fill.executed_at) + (V6.COOLDOWN_BARS + 2) * BAR
            results.append(ReferenceTrade(parent, trade, stress, distance, stop, target,
                                          trigger_at, reason, cooldown_until))
        return tuple(results)


def reference_sha256(reference: Sequence[ReferenceTrade]) -> str:
    return hashlib.sha256(canonical([asdict(row) for row in reference]).encode()).hexdigest()


def pair_filter(reference: Sequence[ReferenceTrade], decisions: Sequence[FilterDecision], *,
                expected_reference_sha256: str) -> PairedResult:
    """Subset-only ledger. No execution, strategy callback, cash or replacements."""
    reference, decisions = tuple(reference), tuple(decisions)
    if not reference or reference_sha256(reference) != expected_reference_sha256:
        raise L2IntegrityError("reference schedule hash mismatch")
    parent_ids = [r.parent.candidate_id for r in reference]
    if len(set(parent_ids)) != len(parent_ids):
        raise L2IntegrityError("duplicate reference candidate")
    if [d.candidate_id for d in decisions] != parent_ids:
        raise L2IntegrityError("filter must account for every parent exactly once in order")
    with localcontext(Context(prec=50)):
        rows = []
        for item, decision in zip(reference, decisions):
            accepted = decision.accepted(item.parent)
            parent_r = item.trade.net_pnl / (item.initial_distance * item.parent.quantity)
            filtered_r = parent_r if accepted else D(0)
            rows.append(PairedRow(item, decision, accepted, parent_r, filtered_r,
                                  filtered_r - parent_r))
        n, k = len(rows), sum(row.accepted for row in rows)
        return PairedResult(
            tuple(rows), n, k, sum((r.parent_net_r for r in rows), D(0)) / n,
            sum((r.filtered_net_r for r in rows), D(0)) / k if k else None,
            sum((r.opportunity_improvement_r for r in rows), D(0)) / n,
        )
