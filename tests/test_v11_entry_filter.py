from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext

import pytest

from src.backtest.v10_l2 import (
    DepthLevel,
    DepthSnapshot,
    ExchangeConstraints,
    QuantityFilter,
)
from src.backtest.v11_entry_filter import (
    BAR,
    LATENCY,
    FilterDecision,
    L2IntegrityError,
    ParentCandidate,
    build_reference,
    pair_filter,
    reference_sha256,
)

D = Decimal
B = datetime(2026, 9, 24, tzinfo=UTC)
CONSTRAINTS = ExchangeConstraints('BTCUSDC', QuantityFilter(D('.1'), D(100), D('.1')), None, ())


def at(ms):
    return B + timedelta(milliseconds=ms)


def book(ms, bid='100', ask='101', *, qty='10', source=None, seq=None, levels=None):
    return DepthSnapshot('BTCUSDC', at(ms if source is None else source), at(ms),
                         ms if seq is None else seq,
                         tuple(DepthLevel(D(p), D(q)) for p, q in levels) if levels else
                         (DepthLevel(D(bid), D(qty)),), (DepthLevel(D(ask), D(qty)),))


def parent(identity='a', shift=0):
    return ParentCandidate(identity, at(shift)-BAR, at(shift), at(shift+10), True,
                           D(2), at(shift), D(1))


def reference(candidates=None, snapshots=None, cash='1000'):
    return build_reference(candidates or [parent()], snapshots or [
        book(0, bid='90', ask='91'),  # Pre-entry stop-like price is irrelevant.
        book(109), book(110), book(200, bid='98', ask='99'),
        book(299, bid='97', ask='98'), book(300, bid='95', ask='96'),
    ], constraints=CONSTRAINTS, initial_cash=D(cash))


def decision(p, value):
    return FilterDecision(p.candidate_id, p.candle_close, p.candle_close, value)


def test_latency_ask_entry_bid_exit_and_adverse_gap():
    item, = reference()
    assert item.trade.entry.executed_at == at(110)
    assert item.trade.entry.price == D('101.0202')
    assert item.trigger_at == at(200)
    assert item.exit_reason == 'STOP'
    assert item.trade.exit.executed_at == at(300)
    assert item.trade.exit.price == D('94.9810')
    assert item.stop == D('99.0202')
    assert item.target == D('105.0202')
    assert item.cooldown_until == B + 6 * BAR
    assert item.trade.exit.price < item.stop


def test_costs_are_charged_once_and_stress_preserves_path():
    item, = reference()
    t, s = item.trade, item.stress_trade
    assert t.net_pnl == t.exit.notional - t.entry.notional - t.entry.fee - t.exit.fee
    assert t.net_pnl == t.mid_price_move - t.spread_cost - t.depth_slippage_cost - t.additional_slippage_cost - t.fees
    assert t.entry.fee == t.entry.notional * D('.001')
    assert s.entry.fee == s.entry.notional * D('.002')
    assert s.entry.price == D(101) * D('1.0004')
    assert s.exit.price == D(95) * D('.9996')
    assert s.entry.snapshot is t.entry.snapshot and s.exit.snapshot is t.exit.snapshot
    assert s.entry.quantity == t.entry.quantity == s.exit.quantity


def test_ordered_target_then_stop_is_not_rewritten_stop_first():
    item, = reference(snapshots=[book(110), book(200, bid='106', ask='107'),
                                book(300, bid='98', ask='99')])
    assert item.exit_reason == 'TARGET'
    assert item.trade.exit.price < item.stop  # Target trigger does not guarantee target fill.


@pytest.mark.parametrize('field,value', [
    ('finalized', False), ('decision_received_at', at(-1)),
    ('decision_received_at', at(1001)), ('atr_available_at', at(11)),
    ('quantity', D(0)), ('candle_open', at(-899999)),
])
def test_invalid_parent_provenance(field, value):
    with pytest.raises(L2IntegrityError):
        replace(parent(), **{field: value})


@pytest.mark.parametrize('snapshots', [
    [book(109)],  # No quote at/after arrival.
    [book(1111)],  # First available quote is beyond deadline.
    [book(110, source=-1000)],
    [book(110), book(1500)],  # Missing protective path.
    [book(110), book(110)],
    [book(110), book(200, source=100)],
    [book(110), book(200, seq=100)],
    [book(110, qty='.5')],  # Never downsize to manufacture a fill.
    [book(110), book(200, bid='98', ask='99', qty='.5')],
    [book(110), book(200, bid='98', ask='99'), book(300, qty='.5')],
])
def test_missing_stale_unordered_or_insufficient_depth_fails(snapshots):
    with pytest.raises(L2IntegrityError):
        reference(snapshots=snapshots)


def test_exchange_time_in_future_is_rejected():
    with pytest.raises(L2IntegrityError, match='future'):
        book(110, source=111)


def test_delayed_pre_entry_update_does_not_trigger_exit():
    item, = reference(snapshots=[book(110, source=90),
                                book(120, bid='98', ask='99', source=100),
                                book(200, bid='106', ask='107'), book(300, bid='106', ask='107')])
    assert item.exit_reason == 'TARGET'
    assert item.trigger_at == at(200)


def test_depth_vwap_uses_entire_quantity_for_protective_trigger():
    item, = reference(snapshots=[book(110),
        book(200, bid='100', ask='101', levels=[('100', '.5'), ('96', '.5')]),
        book(300, bid='99', ask='100', levels=[('99', '.5'), ('95', '.5')])])
    assert item.trigger_at == at(200)
    assert item.trade.exit.book_vwap == D(97)
    assert item.trade.exit.levels_consumed == 2
    assert item.trade.exit.quantity == D(1)


def test_no_leverage_or_quantity_rounding():
    with pytest.raises(L2IntegrityError, match='cash'):
        reference(cash='100')
    with pytest.raises(L2IntegrityError, match='step'):
        reference(candidates=[replace(parent(), quantity=D('1.01'))])


def test_fixed_schedule_rejections_cannot_free_position_or_cooldown():
    with pytest.raises(L2IntegrityError, match='no replacements'):
        reference(candidates=[parent(), parent('replacement', 900000)])
    # Four later signal bars still blocked; fifth later close is permitted.
    with pytest.raises(L2IntegrityError, match='cooldown'):
        reference(candidates=[parent(), parent('too_early', 4500000)])
    shift = 5400000
    refs = reference(candidates=[parent(), parent('b', shift)], snapshots=[
        book(110), book(200, bid='98', ask='99'), book(300, bid='98', ask='99'),
        book(shift+110), book(shift+200, bid='106', ask='107'),
        book(shift+300, bid='106', ask='107'),
    ])
    digest = reference_sha256(refs)
    paired = pair_filter(refs, [decision(refs[0].parent, -1), decision(refs[1].parent, 1)],
                         expected_reference_sha256=digest)
    assert paired.parent_count == 2 and paired.accepted_count == 1
    assert paired.rows[1].reference is refs[1]
    assert paired.rows[0].filtered_net_r == 0
    assert paired.rows[1].filtered_net_r == paired.rows[1].parent_net_r
    with localcontext() as ctx:
        ctx.prec = 50
        assert paired.opportunity_improvement_r == -paired.rows[0].parent_net_r / 2
    assert reference_sha256(refs) == digest
    for bad in ([decision(refs[1].parent, 1)], [decision(refs[0].parent, 1)]*2,
                [decision(refs[1].parent, 1), decision(refs[0].parent, 1)]):
        with pytest.raises(L2IntegrityError, match='every parent'):
            pair_filter(refs, bad, expected_reference_sha256=digest)
    with pytest.raises(L2IntegrityError, match='hash'):
        pair_filter(refs, [], expected_reference_sha256='bad')


@pytest.mark.parametrize('prediction,expected', [(1, True), (0, False), (-1, False), (None, False)])
def test_single_score_rule(prediction, expected):
    refs = reference()
    result = pair_filter(refs, [decision(parent(), prediction)],
                         expected_reference_sha256=reference_sha256(refs))
    assert result.rows[0].accepted is expected
    assert result.accepted_count == int(expected)


def test_future_filter_and_nonfinite_score_fail():
    for item in (replace(decision(parent(), 1), source_available_at=at(1)),
                 replace(decision(parent(), 1), bucket_close=at(-1000)),
                 decision(parent(), float('nan'))):
        with pytest.raises(L2IntegrityError):
            item.accepted(parent())


def test_hold_deadline_96_bars_and_latency():
    # One observation per second is the explicit upper bound, not a new data rate.
    snapshots = [book(110)] + [book(ms) for ms in range(1000, 86400001, 1000)]
    snapshots += [book(86400100)]
    item, = reference(snapshots=snapshots)
    assert item.exit_reason == 'MAX_HOLD'
    assert item.trigger_at == B + 96 * BAR
    assert item.trade.exit.executed_at == item.trigger_at + LATENCY


def test_decimal_context_does_not_change_reference():
    with localcontext() as ctx:
        ctx.prec = 9
        low = reference()
    with localcontext() as ctx:
        ctx.prec = 35
        high = reference()
    assert low == high
