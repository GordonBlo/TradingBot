"""Fabricated tapes and independent price/time matching witnesses only."""
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
from hashlib import sha256
from itertools import product

import pytest

from src.research.v12_maker_bound import (
    Assumptions, ConditionalMakerBound, L2IntegrityError, Order, PublicTrade,
    Receipt, State, Timing,
)


D = Decimal
B = datetime(2026, 10, 4, tzinfo=UTC)
BASE_US = int(B.timestamp()) * 1_000_000
ASSUMPTIONS = Assumptions(True, True, True, True, True, D(105))
TIMING = Timing(100, 100, 0, 0)  # Fabricated exogenous times, not selected latency.


def at(us):
    return B + timedelta(microseconds=us)


class Tape:
    def __init__(self, *, assumptions=ASSUMPTIONS, timing=TIMING, coverage=True):
        self.ordinal = 0
        self.trade_id = 0
        self.engine = ConditionalMakerBound(
            Order('hypothetical', 'synthetic', D(100), D(2), B, 0), timing,
            assumptions, initial_trade_id=0, initial_depth_id=10,
            initial_ordinal=0, coverage_established=coverage,
        )
        if coverage:
            self.engine.submit()

    def receipt(self, us, stream='TRADE', connection=None):
        self.ordinal += 1
        return Receipt('synthetic', connection or stream, stream, self.ordinal,
                       at(us), us * 1000, sha256(str(self.ordinal).encode()).hexdigest())

    def trade(self, us=200, *, price='99', qty='1', maker=True, received=None,
              trade_id=None, resolution=1, connection=None):
        received = us + 10 if received is None else received
        self.trade_id = self.trade_id + 1 if trade_id is None else trade_id
        trade = PublicTrade(self.trade_id, D(price), D(qty), maker,
                            BASE_US + us, BASE_US + us + 1, resolution,
                            self.receipt(received, connection=connection))
        self.engine.observe_trade(trade, as_of=at(received), monotonic_ns=received * 1000)
        return trade

    def boundary(self, reason, us=200):
        self.engine.observe_boundary(reason, self.receipt(us, 'CONTROL'),
                                     as_of=at(us), monotonic_ns=us * 1000)

    def depth(self, first, final, us=190):
        self.engine.observe_depth(first, final, self.receipt(us, 'DEPTH'),
                                  as_of=at(us), monotonic_ns=us * 1000)


def oracle(initial, frames, *, insert_own):
    """Separate exact matching oracle with explicit hidden orders, not estimator.

    Orders are [price, remaining, arrival rank, visible cap]. Cancellation
    instructions name real external order identities. A frame can contain an
    execution plus replenishment before the next public depth observation.
    """
    book = {name: [D(price), D(qty), rank, D(cap)]
            for rank, (name, price, qty, cap) in enumerate(initial)}
    next_rank = len(book)
    if insert_own:
        book['OUR'] = [D(100), D(2), next_rank, D(2)]
        next_rank += 1
    fills, trades, depth = D(0), [], []

    def visible():
        return sum((min(qty, cap) for price, qty, _, cap in book.values()
                    if price == 100), D(0))

    depth.append(visible())
    for frame in frames:
        prints = []
        for action in frame:
            if action[0] == 'add':
                _, name, price, qty = action
                book[name] = [D(price), D(qty), next_rank, D(qty)]
                next_rank += 1
            elif action[0] == 'cancel':
                _, name, qty = action
                if name in book:
                    book[name][1] -= min(book[name][1], D(qty))
            else:
                assert action[0] == 'sell'
                remaining = D(action[1])
                for name, item in sorted(book.items(), key=lambda p: (-p[1][0], p[1][2])):
                    price, qty, _, _ = item
                    take = min(qty, remaining)
                    if take:
                        prints.append((price, take))
                        item[1] -= take
                        remaining -= take
                        if name == 'OUR':
                            fills += take
                    if not remaining:
                        break
        trades.append(tuple(prints))
        depth.append(visible())
    return fills, tuple(trades), tuple(depth)


def replay_prints(prints):
    tape = Tape()
    us = 200
    for frame in prints:
        for price, quantity in frame:
            tape.trade(us, price=str(price), qty=str(quantity))
            us += 20
    return tape.engine


BASE_BOOK = [('a', '100', '2', '2'), ('lower', '99', '50', '50')]


@pytest.mark.parametrize('case,frames,expected_set', [
    ('same-price entirely ahead', [(('sell', '1'),)], {D(0)}),
    ('trade-through while working', [(('sell', '4'),)], {D(2)}),
    ('partial trade-through', [(('sell', '3'),)], {D(1)}),
    ('multiple price levels', [(('sell', '2'),), (('sell', '1'),)], {D(1)}),
    ('cancellation ahead', [(('cancel', 'a', '2'),), (('sell', '1'),)], {D(1)}),
    ('addition behind', [(('add', 'new', '100', '2'),), (('sell', '4'),)], {D(2)}),
])
def test_reference_world_fill_sets(case, frames, expected_set):
    actual, _, _ = oracle(BASE_BOOK, frames, insert_own=True)
    _, historical, _ = oracle(BASE_BOOK, frames, insert_own=False)
    assert {actual} == expected_set, case
    engine = replay_prints(historical)
    assert engine.credited <= min(expected_set)
    assert all(c.price == D(100) for c in engine.credits)


def test_cancellation_ahead_or_behind_same_observations_different_actual_fills():
    histories, actual_fills = [], set()
    for cancelled in ('a', 'behind'):
        frames = [(('add', 'behind', '100', '2'),),
                  (('cancel', cancelled, '2'),), (('sell', '2'),)]
        _, prints, depth = oracle(BASE_BOOK, frames, insert_own=False)
        histories.append((prints, depth))
        actual_fills.add(oracle(BASE_BOOK, frames, insert_own=True)[0])
    assert histories[0] == histories[1]
    assert actual_fills == {D(0), D(2)}
    engine = replay_prints(histories[0][0])
    assert engine.credited == min(actual_fills) == 0


def test_hidden_ahead_or_replenishment_same_public_observations():
    hidden = [('a', '100', '4', '2'), ('lower', '99', '50', '50')]
    hidden_frames = [(('sell', '2'),), (('sell', '2'),)]
    replenished_frames = [(('sell', '2'), ('add', 'new', '100', '2')), (('sell', '2'),)]
    hidden_public = oracle(hidden, hidden_frames, insert_own=False)[1:]
    replenished_public = oracle(BASE_BOOK, replenished_frames, insert_own=False)[1:]
    assert hidden_public == replenished_public
    compatible = {oracle(hidden, hidden_frames, insert_own=True)[0],
                  oracle(BASE_BOOK, replenished_frames, insert_own=True)[0]}
    assert compatible == {D(0), D(2)}
    assert replay_prints(hidden_public[0]).credited == min(compatible) == 0


def test_exact_exhaustive_counterfactual_bound_4096_worlds():
    # All choices fixed before running; no historical outcomes or parameters.
    count = 0
    for ahead, better, added, sell1, sell2, cancel_ahead in product(range(4), repeat=6):
        initial = [('ahead', '100', str(ahead), str(ahead)),
                   ('better', '101', str(better), str(better)),
                   ('lower', '99', '50', '50')]
        frames = [(('sell', str(sell1)),),
                  (('add', 'new', '100', str(added)), ('cancel', 'ahead', str(cancel_ahead))),
                  (('sell', str(sell2)),)]
        _, prints, _ = oracle(initial, frames, insert_own=False)
        actual, _, _ = oracle(initial, frames, insert_own=True)
        engine = replay_prints(prints)
        assert engine.credited <= actual, (initial, frames)
        count += 1
    assert count == 4096


def test_touch_and_price_moving_away_are_not_fill_evidence():
    tape = Tape()
    tape.trade(price='100', qty='20')
    tape.trade(230, price='102', qty='20', maker=False)
    tape.depth(11, 12, 245)
    assert tape.engine.credited == 0
    assert tape.engine.state == State.CONDITIONALLY_WORKING


@pytest.mark.parametrize('us,received', [(90, 95), (90, 210), (100, 210)])
def test_trade_before_or_tied_with_working_assumption(us, received):
    tape = Tape()
    tape.trade(us, received=received, qty='2')
    # Compatible fills include zero: before insertion or ambiguous simultaneous arrival.
    assert tape.engine.credited == 0


def test_partial_credit_cap_and_each_trade_consumed_once():
    tape = Tape()
    first = tape.trade(qty='.75')
    assert tape.engine.state == State.PARTIALLY_FILLED
    duplicate = replace(first, receipt=tape.receipt(225))
    tape.engine.observe_trade(duplicate, as_of=at(225), monotonic_ns=225000)
    assert tape.engine.credited == D('.75')
    tape.trade(240, qty='10')
    assert tape.engine.state == State.FILLED
    assert tape.engine.credited == D(2)
    assert [c.quantity for c in tape.engine.credits] == [D('.75'), D('1.25')]


def test_conflicting_duplicate_latches_uncertainty_preserving_previous_bound():
    tape = Tape()
    first = tape.trade(qty='.5')
    conflict = replace(first, quantity=D(1), receipt=tape.receipt(225))
    tape.engine.observe_trade(conflict, as_of=at(225), monotonic_ns=225000)
    assert tape.engine.state == State.INDETERMINATE
    assert tape.engine.credited == D('.5')  # Compatible future totals [.5, 2].


@pytest.mark.parametrize('depth_first', [True, False])
def test_depth_trade_receipt_order_never_changes_credit(depth_first):
    tape = Tape()
    if depth_first:
        tape.depth(11, 12, 205)
        tape.trade(200, received=210)
    else:
        tape.trade(200, received=210)
        tape.depth(11, 12, 215)
    assert tape.engine.credited == 1  # Same conditional compatible minimum.


def test_exchange_time_regression_is_not_reordered_into_favorable_history():
    tape = Tape()
    tape.trade(200, price='100')
    tape.trade(150, received=230, qty='2')
    assert tape.engine.state == State.INDETERMINATE
    assert tape.engine.credited == 0  # Missing chronology admits zero.


@pytest.mark.parametrize('problem', [
    'depth_gap', 'trade_gap', 'reconnect', 'resync', 'outage',
    'stale observations', 'truncated book coverage', 'clock uncertainty',
    'symbol status unknown', 'payload checksum failure',
])
def test_integrity_problem_no_credit_and_no_snapshot_repair(problem):
    tape = Tape()
    if problem == 'depth_gap':
        tape.depth(12, 12)
    elif problem == 'trade_gap':
        tape.trade(trade_id=2)
    elif problem == 'reconnect':
        tape.trade(price='100')
        tape.trade(230, connection='new connection')
    else:
        tape.boundary(problem)
    assert tape.engine.state == State.INDETERMINATE
    assert tape.engine.credited == 0
    tape.boundary('later valid REST snapshot', 250)
    tape.trade(300, qty='2')
    assert tape.engine.state == State.INDETERMINATE
    assert tape.engine.credited == 0


def test_lifecycle_and_cancel_request_racing_with_fill():
    tape = Tape()
    assert tape.engine.state == State.SUBMISSION_PENDING
    tape.engine.advance(as_of=at(100), monotonic_ns=100000)
    assert tape.engine.state == State.CONDITIONALLY_WORKING
    tape.engine.request_cancel(as_of=at(200), monotonic_ns=200000)
    assert tape.engine.state == State.CANCEL_PENDING
    tape.trade(250, qty='1')  # Guaranteed before assumed cancellation at 300.
    assert tape.engine.state == State.CANCEL_PENDING
    assert tape.engine.credited == 1
    tape.trade(300, qty='1', received=310)  # Boundary tie: no further credit.
    assert tape.engine.state == State.CANCELLED
    assert tape.engine.credited == 1
    tape.trade(330, qty='10')
    assert tape.engine.credited == 1


def test_full_fill_during_cancel_pending_and_late_pre_cancel_evidence():
    tape = Tape()
    tape.engine.advance(as_of=at(150), monotonic_ns=150000)
    tape.engine.request_cancel(as_of=at(200), monotonic_ns=200000)
    tape.trade(250, qty='2', received=350)  # Late receipt; exchange T certainly before cancel.
    assert tape.engine.state == State.FILLED
    assert tape.engine.credited == 2


def test_timestamp_buckets_and_clock_bounds_intersecting_cancel_give_zero():
    tape = Tape(timing=Timing(100, 100, -10, 10))
    tape.engine.advance(as_of=at(150), monotonic_ns=150000)
    tape.engine.request_cancel(as_of=at(500), monotonic_ns=500000)
    tape.trade(200, received=1300, resolution=1000)  # [200,1199] overlaps cancel [590,610].
    assert tape.engine.credited == 0


def test_uncertain_arrival_and_same_timestamp_messages_give_zero():
    tape = Tape(timing=Timing(100, 100, -10, 10))
    tape.trade(105, received=200)
    assert tape.engine.credited == 0
    tape.depth(11, 11, 200)
    tape.trade(130, received=200)
    assert tape.engine.credited == 1  # Ordinal provides local order, not venue order.


def test_crossing_at_assumed_arrival_rejected_not_converted_to_taker():
    tape = Tape(assumptions=replace(ASSUMPTIONS, arrival_best_ask=D(100),
                                    accepted_and_resting_until_cancel=False))
    tape.trade(qty='2')
    assert tape.engine.state == State.REJECTED_ASSUMED
    assert tape.engine.credited == 0


def test_contradictory_acceptance_assumptions_do_not_define_an_empty_fill_set():
    tape = Tape(assumptions=replace(ASSUMPTIONS, arrival_best_ask=D(100)))
    tape.trade(qty='2')
    assert tape.engine.state == State.INDETERMINATE
    assert tape.engine.credited == 0


def test_submission_cannot_be_backdated_after_clock_advance():
    engine = ConditionalMakerBound(
        Order('x', 'synthetic', D(100), D(2), B, 0), TIMING, ASSUMPTIONS,
        initial_trade_id=0, initial_depth_id=10, initial_ordinal=0,
        coverage_established=True,
    )
    assert engine.state == State.DECIDED
    engine.advance(as_of=at(200), monotonic_ns=200000)
    with pytest.raises(L2IntegrityError, match='anchored'):
        engine.submit()


@pytest.mark.parametrize('field', [
    'accepted_and_resting_until_cancel', 'ordinary_price_priority',
    'fixed_external_order_instructions', 'no_self_trade_prevention',
    'filters_and_continuous_status_valid', 'arrival_best_ask',
])
def test_unasserted_assumption_keeps_unconditional_minimum_zero(field):
    value = None if field == 'arrival_best_ask' else False
    tape = Tape(assumptions=replace(ASSUMPTIONS, **{field: value}))
    tape.trade(qty='2')
    assert tape.engine.state == State.INDETERMINATE
    assert tape.engine.credited == 0


def test_missing_initial_coverage_not_converted_to_empty_queue():
    tape = Tape(coverage=False)
    assert tape.engine.state == State.INDETERMINATE
    assert tape.engine.credited == 0


def test_aggregate_and_bookticker_observations_never_double_count_trades():
    tape = Tape()
    for us, stream in [(180, 'AGGTRADE'), (190, 'BOOK_TICKER')]:
        tape.engine.observe_auxiliary(tape.receipt(us, stream), as_of=at(us),
                                      monotonic_ns=us * 1000)
    assert tape.engine.credited == 0
    tape.trade()
    assert tape.engine.credited == 1


def test_future_receipt_rejected_and_latched():
    tape = Tape()
    trade = PublicTrade(1, D(99), D(2), True, BASE_US + 200, BASE_US + 201, 1,
                        tape.receipt(210))
    with pytest.raises(L2IntegrityError, match='future receipt'):
        tape.engine.observe_trade(trade, as_of=at(209), monotonic_ns=209000)
    assert tape.engine.state == State.INDETERMINATE
    assert tape.engine.credited == 0


def test_exchange_source_after_receipt_incompatible_with_clock_mapping():
    tape = Tape()
    tape.trade(300, received=200)
    assert tape.engine.state == State.INDETERMINATE
    assert tape.engine.credited == 0


@pytest.mark.parametrize('problem', ['ordinal', 'monotonic', 'utc', 'session'])
def test_receipt_provenance_regressions_latch(problem):
    tape = Tape()
    tape.trade(price='100')
    receipt = tape.receipt(230)
    changes = {'ordinal': {'ordinal': 4}, 'monotonic': {'monotonic_ns': 190000},
               'utc': {'utc': at(190)}, 'session': {'session_id': 'different'}}[problem]
    trade = PublicTrade(2, D(99), D(2), True, BASE_US + 220, BASE_US + 221, 1,
                        replace(receipt, **changes))
    tape.engine.observe_trade(trade, as_of=at(230), monotonic_ns=230000)
    assert tape.engine.state == State.INDETERMINATE
    assert tape.engine.credited == 0


def test_decimal_reproducibility_independent_of_callers_context():
    results = []
    for precision in (3, 28, 70):
        with localcontext() as context:
            context.prec = precision
            tape = Tape()
            tape.trade(qty='.123456789123456789')
            tape.trade(230, qty='.987654321987654321')
            results.append((tape.engine.credited, tape.engine.credits))
    assert results[0] == results[1] == results[2]
    assert results[0][0] == D('1.111111111111111110')


@pytest.mark.parametrize('quantity', [D('NaN'), D('Infinity'), D(0), D(-1), 1.0, D('1e-19')])
def test_malformed_numeric_input_not_silently_coerced(quantity):
    with pytest.raises(L2IntegrityError):
        Order('x', 'synthetic', D(100), quantity, B, 0)


def test_state_errors_and_timing_validation():
    engine = Tape().engine
    with pytest.raises(L2IntegrityError, match='DECIDED'):
        engine.submit()
    with pytest.raises(L2IntegrityError, match='working'):
        engine.request_cancel(as_of=at(50), monotonic_ns=50000)
    with pytest.raises(L2IntegrityError):
        Timing(-1, 100, 0, 0)
    with pytest.raises(L2IntegrityError):
        Timing(100, 100, 1, 0)


def test_no_strategy_pnl_network_or_private_execution_surface():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / 'src/research/v12_maker_bound.py').read_text()
    for forbidden in ('execute_once(', 'requests.', 'websockets', 'net_pnl',
                      'create_order', 'v11_entry_filter', 'v11_score', 'read_csv'):
        assert forbidden not in source
