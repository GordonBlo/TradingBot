"""Offline causal V6 parent materialization and frozen V9 subset ledger.

No collection, fit, alternative strategy or orders. Call only on synthetic
inputs until a separate prospective protocol and execution are authorized.
"""
from __future__ import annotations

import hashlib
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import ROUND_FLOOR, Context, Decimal, localcontext
from fractions import Fraction
from itertools import pairwise
from math import lcm
from pathlib import Path

from src.backtest.v10_l2 import (
    DepthSnapshot,
    ExchangeConstraints,
    L2IntegrityError,
    canonical,
    utc,
)
from src.backtest.v11_entry_filter import (
    BAR,
    LATENCY,
    MAX_OBSERVATION_DELAY,
    FilterDecision,
    PairedResult,
    ParentCandidate,
    ReferenceTrade,
    _Books,
    bar_open,
    build_reference,
    pair_filter,
    reference_sha256,
)
from src.models.candle import Candle
from src.research.v11_score import FROZEN_FEATURES, load_bundle, score, sha256
from src.strategy.context import StrategyContext
from src.strategy.models import StrategyAction, TrendMomentumConfig
from src.strategy.v6_mtf_continuation import V6MTFContinuationStrategy as V6
from src.strategy.v6_mtf_continuation import prepare_v6_context

D = Decimal
INITIAL_CASH = D(1000)
BUNDLE = Path('research/v11_preparation/v9_static_score/score_bundle.json')
BUNDLE_SHA = 'c2fb47fb459a97024ebedc12af0354d6c28792e2dce87c1207eba2ba5e4664e2'
V6_BINDINGS = {
    'research/v6_mtf_continuation/e1eef7bdd0c37ad4/manifest.json': '2dcc805e71f7b95cf6e39b6d628f301a6123bdcbe73b24401aa80c8f15ffb6de',
    'src/strategy/v6_mtf_continuation.py': '2bc13ee68bb55c7dfaa2098fbe1460082004705e44978429e0fb607ba45f2653',
    'src/strategy/models.py': '367564e45b109423c9dccd78fa664f28ac62c2aefbff179035b426e591f2bef6',
    'src/analysis/indicators.py': 'f8398e679d6dca797073e772aa581423eef369c5e9192356dfb75c24c6133d22',
}


@dataclass(frozen=True)
class ClosedBar:
    candle: Candle
    received_at: datetime

    def __post_init__(self):
        c = self.candle
        if (not c.is_closed or c.symbol != 'BTCUSDC' or c.interval != '15m'
                or c.timestamp != bar_open(c.timestamp) or min(c.open, c.low, c.close) <= 0):
            raise L2IntegrityError('closed positive aligned BTCUSDC 15m candle required')
        close = utc(c.timestamp) + BAR
        if not close <= utc(self.received_at) <= close + MAX_OBSERVATION_DELAY:
            raise L2IntegrityError('candle receipt before close or too late')


@dataclass(frozen=True)
class ScoreBucket:
    close: datetime
    available_at: datetime  # maximum source availability of all nonmissing features
    values: tuple[float | None, ...]
    feature_order: tuple[str, ...] = tuple(FROZEN_FEATURES)

    def __post_init__(self):
        if (utc(self.available_at) > utc(self.close) or self.close.microsecond
                or len(self.values) != len(FROZEN_FEATURES)
                or not isinstance(self.values, tuple)
                or self.feature_order != tuple(FROZEN_FEATURES)):
            raise L2IntegrityError('future, misordered or malformed one-second L2 bucket')


@dataclass(frozen=True)
class CandidateAudit:
    candidate_id: str
    signal_close: datetime
    parent_cash_at_signal: Decimal
    risk_budget: Decimal
    max_quote: Decimal
    atr_available_at: datetime
    quantity: Decimal
    reason: str


@dataclass(frozen=True)
class MaterializedLedger:
    binding: dict
    input_sha256: str
    reference: tuple[ReferenceTrade, ...]
    candidates: tuple[CandidateAudit, ...]
    paired: PairedResult | None
    risk: dict

    def write_once(self, path: Path):
        body = canonical(asdict(self)).encode()
        with path.open('xb') as stream:
            stream.write(body)
        return hashlib.sha256(body).hexdigest()


def engineering_binding(root: Path) -> dict:
    for name, expected in V6_BINDINGS.items():
        if sha256(root/name) != expected:
            raise L2IntegrityError('frozen V6 generator/config source binding mismatch')
    config = TrendMomentumConfig()
    V6(config)  # Existing V6 constructor independently enforces its frozen fields.
    return {'v6_id': 'e1eef7bdd0c37ad4', 'v6_sources': V6_BINDINGS,
            'config': asdict(config), 'initial_cash': INITIAL_CASH,
            'score_bundle_sha256': BUNDLE_SHA, 'rule': 'prediction > 0',
            'sizing': 'V6 risk/notional intent; largest feasible exchange lot at execution book',
            'generator_decimal_precision': 28, 'execution_decimal_precision': 50,
            'adapter_sha256': sha256(root/'src/backtest/v11_entry_filter.py'),
            'materializer_sha256': sha256(root/'src/research/v11_materializer.py')}


def sized_quantity(book: DepthSnapshot, constraints: ExchangeConstraints, *,
                   risk_budget: Decimal, atr: Decimal, max_quote: Decimal) -> Decimal:
    """Precommitted risk intent resolves at arrival, never from later prices.

    At one depth level this is the old V6 engine sizing, rounded DOWN to the
    exchange lot. For multiple levels solve monotone risk/notional limits.
    Insufficient displayed depth fails; it does not create a partial fill.
    """
    steps = [rule.step_size for rule in (constraints.lot_size, constraints.market_lot_size)
             if rule is not None and rule.step_size is not None]
    if not steps:
        raise L2IntegrityError('explicit exchange quantity step required')
    fractions = [Fraction(step) for step in steps]
    denominator = lcm(*(f.denominator for f in fractions))
    numerator = lcm(*(f.numerator*(denominator//f.denominator) for f in fractions))
    with localcontext(Context(prec=50)):
        step = D(numerator)/D(denominator)
        price = book.best_ask*D('1.0002')
        upper = min(risk_budget/max(atr, price*V6.MIN_STOP_DISTANCE_FRACTION), max_quote/price)
        for rule in (constraints.lot_size, constraints.market_lot_size):
            if rule is not None and rule.max_qty is not None:
                upper = min(upper, rule.max_qty)
        lots = int((upper/step).to_integral_value(rounding=ROUND_FLOOR))
        if sum((level.quantity for level in book.asks), D(0)) < step*lots:
            raise L2IntegrityError('insufficient displayed depth; no liquidity downsizing')

        def feasible(count):
            q = step*count
            left, total = q, D(0)
            for level in book.asks:
                take = min(left, level.quantity)
                total += take*level.price
                left -= take
                if left == 0:
                    break
            fill = total/q*D('1.0002')
            return q*fill <= max_quote and q*max(atr, fill*V6.MIN_STOP_DISTANCE_FRACTION) <= risk_budget

        lo, hi = 0, lots
        while lo < hi:
            mid = (lo+hi+1)//2
            if feasible(mid):
                lo = mid
            else:
                hi = mid-1
        q = lo*step
        constraints.validate_market_quantity(q, label='V6 resolved')
        return q


def _input_hash(bars, books, buckets):
    digest = hashlib.sha256()
    for kind, rows in (('bars', bars), ('books', books), ('scores', buckets)):
        digest.update(kind.encode())
        for row in rows:
            digest.update(canonical(asdict(row)).encode()+b'\n')
    return digest.hexdigest()


def _risk_report(reference, decisions, snapshots):
    times = [s.available_at for s in snapshots]
    output = {}
    with localcontext(Context(prec=50)):
        for label, filtered in (('parent', False), ('filtered', True)):
            cash, peak, max_dd, exposure, turnover = INITIAL_CASH, INITIAL_CASH, D(0), 0., D(0)
            for item, decision in zip(reference, decisions, strict=True):
                if filtered and not decision.accepted(item.parent):
                    continue
                trade, q = item.trade, item.parent.quantity
                if trade.entry.notional+trade.entry.fee > cash:
                    raise L2IntegrityError('paired arm lacks cash for fixed quantity; no borrowing or resizing')
                cash -= trade.entry.notional + trade.entry.fee
                start = bisect_left(times, trade.entry.executed_at)
                end = bisect_left(times, trade.exit.executed_at)
                for index in range(start, end):
                    liquidation = _Books.executable_bid(snapshots[index], q)*q*D('.9998')*D('.999')
                    mark = cash+liquidation
                    peak = max(peak, mark)
                    max_dd = max(max_dd, peak-mark)
                cash += trade.exit.notional-trade.exit.fee
                peak = max(peak, cash)
                max_dd = max(max_dd, peak-cash)
                exposure += trade.holding_time.total_seconds()
                turnover += trade.entry.notional+trade.exit.notional
            output[label] = {'final_cash': cash, 'max_liquidation_drawdown_usdc': max_dd,
                             'position_seconds': exposure, 'turnover_usdc': turnover}
    return output


def materialize(bars: Sequence[ClosedBar], snapshots: Sequence[DepthSnapshot],
                buckets: Sequence[ScoreBucket], *, constraints: ExchangeConstraints,
                cutoff: datetime, evaluation_start: datetime, evaluation_end: datetime,
                root: Path = Path('.')) -> MaterializedLedger:
    """One chronological parent pass. Scores cannot enter the parent state."""
    binding = engineering_binding(root)
    bundle = load_bundle(root/BUNDLE, expected_sha256=BUNDLE_SHA)
    bars, snapshots, buckets = tuple(bars), tuple(snapshots), tuple(buckets)
    cutoff, evaluation_start, evaluation_end = map(utc, (cutoff, evaluation_start, evaluation_end))
    if (not bars or not cutoff < bars[0].candle.timestamp
            or evaluation_start < bars[0].candle.timestamp+timedelta(days=9)
            or evaluation_end <= evaluation_start
            or bars[-1].candle.timestamp+BAR < evaluation_end):
        raise L2IntegrityError('strict post-cutoff data, nine-day warmup and complete endpoint required')
    for a, b in pairwise(bars):
        if b.candle.timestamp-a.candle.timestamp != BAR or b.received_at <= a.received_at:
            raise L2IntegrityError('nonconsecutive or unordered closed candles')
    if any(s.source_at <= cutoff for s in snapshots) or any(b.available_at <= cutoff for b in buckets):
        raise L2IntegrityError('pre-cutoff L2 input')
    if any(a.close >= b.close for a, b in pairwise(buckets)):
        raise L2IntegrityError('duplicate or unordered L2 buckets')
    books = _Books(snapshots, constraints)
    times = books.times
    by_close = {b.close: b for b in buckets}
    candles = tuple(b.candle for b in bars)
    with localcontext(Context(prec=50)):
        with localcontext(Context(prec=28)):
            prepared = prepare_v6_context(candles)
        strategy = V6(TrendMomentumConfig(), prepared_context=prepared)
        cash, references, audits, decisions = INITIAL_CASH, [], [], []
        pending, last_exit = None, None
        for index, bar in enumerate(bars):
            close = bar.candle.timestamp+BAR
            if pending is not None and pending.trade.exit.executed_at <= bar.received_at:
                cash += pending.trade.net_pnl
                last_exit = bar_open(pending.trade.exit.executed_at)
                pending = None
            if close < evaluation_start or close >= evaluation_end or pending is not None:
                continue
            since_exit = int((bar.candle.timestamp-last_exit)/BAR) if last_exit is not None else None
            context = StrategyContext(
                timestamp=bar.received_at, current_candle=bar.candle,
                recent_history=candles[max(0, index-20):index+1],
                indicators=prepared.indicators_15m_by_timestamp[bar.candle.timestamp],
                previous_indicators=prepared.indicators_15m_by_timestamp[candles[index-1].timestamp] if index else None,
                has_position=False, bars_in_position=0, equity=cash, cash_usdc=cash,
                completed_trade_count=len(references), bars_since_exit=since_exit, entry_fee_rate=D('.001'),
            )
            with localcontext(Context(prec=28)):
                intent = strategy.evaluate(context)
            if intent.action != StrategyAction.ENTER_LONG:
                continue
            state = prepared.state_at(bar.candle.timestamp)
            atr_close = state.latest_candle.timestamp + timedelta(hours=4)
            atr_index = int((atr_close-BAR-candles[0].timestamp)/BAR)
            atr_available = bars[atr_index].received_at
            if atr_close > close or atr_available > bar.received_at:
                raise L2IntegrityError('future completed-4h state')
            _, entry_book = books.at_or_after(bar.received_at+LATENCY)
            q = sized_quantity(entry_book, constraints, risk_budget=intent.risk_budget,
                               atr=intent.stop_distance, max_quote=intent.max_quote_amount)
            # Identity depends only on the causal signal and parent intent, not
            # future prices, final dataset hash, or the filter outcome.
            identity = hashlib.sha256(canonical({
                'v6': binding['v6_sources'], 'bar': asdict(bar),
                'risk_budget': intent.risk_budget, 'max_quote': intent.max_quote_amount,
                'atr': intent.stop_distance,
            }).encode()).hexdigest()
            parent = ParentCandidate(identity, bar.candle.timestamp, close, bar.received_at,
                                     True, intent.stop_distance, atr_available, q)
            lo = bisect_left(times, bar.received_at)
            hi = bisect_right(times, close + 96*BAR + LATENCY + MAX_OBSERVATION_DELAY)
            ref, = build_reference([parent], snapshots[lo:hi], constraints=constraints, initial_cash=cash)
            if references and close < references[-1].cooldown_until:
                raise L2IntegrityError('materializer/adapter cooldown mismatch')
            bucket = by_close.get(close)
            value = score(bundle['model'], bucket.values) if bucket is not None else None
            decision = FilterDecision(identity, close, bucket.available_at if bucket else close, value)
            decision.accepted(parent)  # Validate provenance before recording.
            references.append(ref)
            decisions.append(decision)
            audits.append(CandidateAudit(identity, close, cash, intent.risk_budget,
                                          intent.max_quote_amount, atr_available, q,
                                          'POSITIVE_SCORE' if value is not None and value > 0 else
                                          'NONPOSITIVE_SCORE' if value is not None else 'MISSING_BUCKET'))
            pending = ref
        references = tuple(references)
        paired = pair_filter(references, decisions, expected_reference_sha256=reference_sha256(references)) if references else None
        binding.update({'cutoff': cutoff, 'evaluation_start': evaluation_start, 'evaluation_end': evaluation_end,
                        'exchange_constraints': asdict(constraints), 'status': 'ENGINEERING_NOT_PREREGISTRATION'})
        return MaterializedLedger(binding, _input_hash(bars, snapshots, buckets), references,
                                  tuple(audits), paired, _risk_report(references, decisions, snapshots))
