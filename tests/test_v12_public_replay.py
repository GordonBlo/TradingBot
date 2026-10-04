"""Fabricated exact-byte public messages; no historical research data/network."""
import asyncio
import base64
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from threading import Event

import pytest

from src.microstructure.v10 import MicrostructureIntegrityError, _canonical
from src.microstructure.v12 import (
    DEFAULT_POLICY,
    PHASE1,
    IngestionJournal,
    Policy,
    PublicCollectorV2,
    PublicRawTransport,
    Reading,
    phase1_binding,
)
from src.research.v12_maker_bound import Assumptions, Order, Timing
from src.research.v12_public_replay import (
    MakerPlan,
    certify_session,
    load_session,
    maker_replay,
    replay,
)

B = datetime(2026, 10, 4, tzinfo=UTC)
US = int(B.timestamp()) * 1_000_000
D = Decimal
ROOT = Path(__file__).resolve().parents[1]


class Clock:
    def __init__(self):
        self.us = 0
        self.step = 1
        self.wall_offset = 0

    def __call__(self):
        result = Reading(B + timedelta(microseconds=self.us + self.wall_offset), self.us * 1000)
        self.us += self.step
        return result


class Tape:
    def __init__(self, path, policy=DEFAULT_POLICY, buffered=False):
        self.clock = Clock()
        self.policy = policy
        self.journal = IngestionJournal(path, 'synthetic', clock=self.clock, fsync=False,
                                        policy=policy, clock_id='synthetic-clock')
        self.events = []
        self.add(10, 'CLOCK', {'serverTime': US + 5})
        self.add(20, 'SYMBOL_RULES', {'symbols': [{
            'symbol': 'BTCUSDC', 'status': 'TRADING', 'isSpotTradingAllowed': True,
            'orderTypes': ['LIMIT', 'LIMIT_MAKER'], 'filters': [
                {'filterType': 'PRICE_FILTER', 'tickSize': '1', 'minPrice': '1', 'maxPrice': '1000000'},
                {'filterType': 'LOT_SIZE', 'stepSize': '0.1', 'minQty': '0.1', 'maxQty': '100'},
                {'filterType': 'MIN_NOTIONAL', 'minNotional': '1'},
            ]}]})
        self.add(30, 'CONNECT_START', {})
        self.add(40, 'CONNECTED', {})
        snapshot = {'lastUpdateId': 10, 'bids': [['99', '5'], ['90', '2']], 'asks': [['105', '5']]}
        depth = {'e': 'depthUpdate', 's': 'BTCUSDC', 'U': 11, 'u': 11, 'E': US + 45, 'b': [], 'a': []}
        if buffered:
            self.add(50, 'DIFF_DEPTH', depth)
            self.add(60, 'REST_SNAPSHOT', snapshot)
        else:
            self.add(50, 'REST_SNAPSHOT', snapshot)
            self.add(60, 'DIFF_DEPTH', depth)
        self.trade(70, 1, price='101')
        self.agg(80, 1, 1, 1, price='101')
        self.add(90, 'BOOK_TICKER', {'s': 'BTCUSDC', 'u': 11, 'b': '99', 'B': '5', 'a': '105', 'A': '5'})
        self.prefix = len(self.events)
        self.path = path
        manifest = {'schema_version': 'V12_ENGINEERING_SESSION_1', 'engineering_only': True,
                    'session_id': 'synthetic', 'clock_id': 'synthetic-clock',
                    'policy': asdict(policy), 'phase1_binding': phase1_binding(ROOT)}
        (path / 'session.manifest.json').write_text(_canonical(manifest), encoding='utf-8')

    def add(self, us, kind, payload, *, connection='ws-0', epoch=0, details=None):
        self.clock.us = us
        streams = {'TRADE': 'btcusdc@trade', 'AGGTRADE': 'btcusdc@aggTrade',
                   'DIFF_DEPTH': 'btcusdc@depth@100ms', 'BOOK_TICKER': 'btcusdc@bookTicker'}
        if kind in streams:
            raw = _canonical({'stream': streams[kind], 'data': payload}).encode()
            e = self.journal.capture(raw, connection_id=connection, epoch=epoch, unit='MICROSECOND')
        else:
            if kind in ('CLOCK', 'SYMBOL_RULES', 'REST_SNAPSHOT') and details is None:
                details = {'timestamp_unit': 'MICROSECOND',
                           'request_started': Reading(B + timedelta(microseconds=us - 10), (us - 10) * 1000).payload(),
                           'response_received': Reading(B + timedelta(microseconds=us - 1), (us - 1) * 1000).payload()}
            e = self.journal.capture(_canonical(payload).encode(), record_type=kind, stream='CONTROL',
                                     connection_id=connection, epoch=epoch,
                                     unit='MICROSECOND' if details else 'NONE', details=details)
        self.events.append(e)
        return e

    def trade(self, us, tid, *, price='99', qty='1', maker=True, time=None, event_time=None):
        time = US + us - 5 if time is None else time
        return self.add(us, 'TRADE', {'e': 'trade', 's': 'BTCUSDC', 't': tid, 'p': price, 'q': qty,
                                     'm': maker, 'T': time, 'E': time if event_time is None else event_time})

    def agg(self, us, aid, first, last, *, price='99', qty='1', maker=True):
        return self.add(us, 'AGGTRADE', {'e': 'aggTrade', 's': 'BTCUSDC', 'a': aid, 'f': first, 'l': last,
                                       'p': price, 'q': qty, 'm': maker, 'T': US + us - 5, 'E': US + us - 5})

    def close(self, us=1000, clean=True):
        self.add(us, 'DISCONNECTED', {'expected': True})
        self.add(us + 10, 'SESSION_CLOSE', {'clean': clean})
        self.journal.close()

    def plan(self, *, limit='100', quantity='2', assumptions=None, cancel=None):
        return MakerPlan(Order('hypothetical', 'synthetic', D(limit), D(quantity),
                               B + timedelta(microseconds=100), 100_000),
                         Timing(50, 50, -20, 20),
                         assumptions or Assumptions(True, True, True, True, True, D(105)),
                         self.prefix, B + timedelta(microseconds=cancel) if cancel else None,
                         cancel * 1000 if cancel else None)

    def twice(self):
        first, second = replay(self.events, self.policy), replay(self.events, self.policy)
        assert first.canonical_result() == second.canonical_result()
        return first

    def bound_twice(self, **kwargs):
        plan = self.plan(**kwargs)
        first = maker_replay(self.events, plan, self.policy)
        assert _canonical(first) == _canonical(maker_replay(self.events, plan, self.policy))
        assert first['unconditional_public_credit'] == '0'
        return first


@pytest.mark.parametrize('buffered', [False, True])
def test_bridge_replay_and_persistence(tmp_path, buffered):
    t = Tape(tmp_path / 'session', buffered=buffered)
    t.trade(200, 2)
    t.agg(210, 2, 2, 2)
    t.close()
    machine = t.twice()
    assert not machine.failures
    assert machine.bridges == 1
    certificate = certify_session(t.path)
    assert certificate['status'] == 'ENGINEERING_REPLAY_CERTIFIED'
    assert not certificate['research_eligible']
    assert t.bound_twice()['certified_conditional_credit'] == '1'
    (t.path / 'closure.summary.json').write_text(_canonical(certificate), encoding='utf-8')
    assert load_session(t.path)[1] == t.events
    with (t.path / 'events.jsonl').open('ab') as f:
        f.write(b' ')
    with pytest.raises(MicrostructureIntegrityError, match='closed session'):
        load_session(t.path)


@pytest.mark.parametrize('price,qty,maker,expected', [
    ('100', '1', True, '0'), ('101', '1', True, '0'), ('99', '1', True, '1'),
    ('99', '0.3', True, '0.3'), ('99', '9', True, '2'), ('99', '1', False, '0'),
])
def test_only_distinct_strict_trade_through_credits(tmp_path, price, qty, maker, expected):
    t = Tape(tmp_path / 'session')
    e = t.trade(200, 2, price=price, qty=qty, maker=maker)
    t.add(205, 'TRADE', e['payload'])  # repeated execution never consumes volume again
    t.agg(210, 2, 2, 2, price=price, qty=qty, maker=maker)
    t.close()
    assert t.twice().duplicates == 1
    bound = t.bound_twice()
    assert bound['certified_conditional_credit'] == expected
    assert len(bound['provisional_ledger']) <= 1


def test_hidden_liquidity_and_depth_removal_give_no_extra_credit(tmp_path):
    t = Tape(tmp_path / 'session')
    t.add(180, 'DIFF_DEPTH', {'e': 'depthUpdate', 's': 'BTCUSDC', 'U': 12, 'u': 12,
                            'E': US + 170, 'b': [['99', '0']], 'a': []})
    t.trade(200, 2, price='100')  # unknown hidden/iceberg or priority remains compatible
    t.agg(210, 2, 2, 2, price='100')
    t.close()
    assert not t.twice().failures
    assert t.bound_twice()['certified_conditional_credit'] == '0'


@pytest.mark.parametrize('case,reason', [
    ('depth_gap', 'DEPTH_GAP'), ('dropped_trade', 'TRADE_ID_GAP'),
    ('trade_regression', 'TRADE_ID_REGRESSION'), ('trade_time_regression', 'EXCHANGE_TRADE_TIME_REGRESSION'),
    ('quantity', 'AGGREGATE_TRADE_RECONCILIATION_FAILURE'), ('side', 'AGGREGATE_TRADE_RECONCILIATION_FAILURE'),
    ('overlap', 'AGGREGATE_RANGE_OVERLAP_OR_GAP'), ('conflicting_duplicate', 'CONFLICTING_DUPLICATE_TRADE'),
    ('exchange_future', 'EXCHANGE_TIME_AFTER_POSSIBLE_RECEIPT'), ('ticker', 'BOOK_TICKER_DEPTH_DISAGREEMENT'),
    ('reconnect', 'DISCONNECTED_INTERVAL'), ('backlog', 'PROCESSING_BACKLOG_LIMIT'),
    ('clock_jump', 'CLOCK_DISCONTINUITY'), ('epoch', 'MARKET_OUTSIDE_CONNECTED_EPOCH'),
])
def test_adversarial_histories_latch_uncertainty(tmp_path, case, reason):
    t = Tape(tmp_path / 'session')
    if case == 'depth_gap':
        t.add(180, 'DIFF_DEPTH', {'e': 'depthUpdate', 's': 'BTCUSDC', 'U': 13, 'u': 13,
                                'E': US + 170, 'b': [], 'a': []})
        t.add(185, 'RESYNC_START', {})
        t.add(190, 'REST_SNAPSHOT', {'lastUpdateId': 13, 'bids': [['99', '5']], 'asks': [['105', '5']]})
    if case == 'reconnect':
        t.add(180, 'DISCONNECTED', {'expected': False})
        t.add(185, 'RESYNC_START', {})
        t.add(190, 'RECONNECT_START', {}, connection='ws-1', epoch=1)
        t.add(195, 'RECONNECTED', {}, connection='ws-1', epoch=1)
    if case == 'backlog':
        t.clock.step = 400_000
    if case == 'clock_jump':
        t.clock.wall_offset = 100_000
    tid = 3 if case == 'dropped_trade' else 0 if case == 'trade_regression' else 2
    time = US + 1 if case == 'trade_time_regression' else None
    event_time = US + 100_000 if case == 'exchange_future' else None
    t.trade(200, tid, time=time, event_time=event_time)
    t.clock.step = 1
    # restore forward time after deliberately slow processing
    next_us = 2_000_210 if case == 'backlog' else 210
    t.agg(next_us, 2, tid, tid, qty='2' if case == 'quantity' else '1', maker=case != 'side')
    if case == 'conflicting_duplicate':
        t.trade(next_us + 10, tid, qty='2')
    if case == 'overlap':
        t.agg(next_us + 10, 3, tid, tid)
    if case == 'ticker':
        t.add(next_us + 10, 'BOOK_TICKER', {'s': 'BTCUSDC', 'u': 11, 'b': '98', 'B': '5', 'a': '105', 'A': '5'})
    if case == 'epoch':
        t.add(next_us + 10, 'BOOK_TICKER', {'s': 'BTCUSDC', 'u': 11, 'b': '99', 'B': '5', 'a': '105', 'A': '5'}, epoch=1)
    t.close(next_us + 100)
    machine = t.twice()
    assert reason in {f['reason'] for f in machine.failures}
    assert certify_session(t.path)['status'] == 'ENGINEERING_CAPTURE_NOT_ELIGIBLE'
    # Clock-regressed histories are rejected by the authoritative engine too.
    if case not in ('backlog',):
        assert t.bound_twice()['certified_conditional_credit'] is None


@pytest.mark.parametrize('trade_first', [True, False])
def test_receipt_order_not_hindsight_exchange_sort(tmp_path, trade_first):
    t = Tape(tmp_path / 'session')
    depth = {'e': 'depthUpdate', 's': 'BTCUSDC', 'U': 12, 'u': 12, 'E': US + 175, 'b': [], 'a': []}
    if trade_first:
        t.trade(200, 2, time=US + 180)
        t.add(205, 'DIFF_DEPTH', depth)
    else:
        t.add(200, 'DIFF_DEPTH', depth)
        t.trade(205, 2, time=US + 180)
    t.agg(210, 2, 2, 2)
    t.close()
    assert not t.twice().failures
    assert [e['ordinal'] for e in t.twice().trace] == list(range(1, len(t.events) + 1))
    assert t.bound_twice()['certified_conditional_credit'] == '1'


@pytest.mark.parametrize('time,expected', [(275, '1'), (280, '0'), (290, '0')])
def test_cancel_race_excludes_uncertain_bucket(tmp_path, time, expected):
    t = Tape(tmp_path / 'session')
    t.trade(310, 2, time=US + time)
    t.agg(320, 2, 2, 2)
    t.close()
    assert t.bound_twice(cancel=250)['certified_conditional_credit'] == expected


@pytest.mark.parametrize('limit,quantity', [('89', '2'), ('100.5', '2'), ('100', '0.01')])
def test_unknown_truncated_coverage_or_invalid_filters(tmp_path, limit, quantity):
    t = Tape(tmp_path / 'session', policy=Policy(snapshot_limit=2, max_levels=2))
    t.trade(200, 2, price='88')
    t.agg(210, 2, 2, 2, price='88')
    t.close()
    assert t.bound_twice(limit=limit, quantity=quantity)['certified_conditional_credit'] is None


def test_ambiguous_acceptance_and_intermediate_ticker(tmp_path):
    t = Tape(tmp_path / 'session')
    t.add(180, 'BOOK_TICKER', {'s': 'BTCUSDC', 'u': 12, 'b': '100', 'B': '1', 'a': '105', 'A': '5'})
    t.trade(200, 2)
    t.agg(210, 2, 2, 2)
    t.close()
    assert t.twice().tickers[-1]['status'] == 'UNVERIFIABLE_INTERMEDIATE'
    assumptions = Assumptions(False, True, True, True, True, D(105))
    assert t.bound_twice(assumptions=assumptions)['certified_conditional_credit'] is None


def test_decision_cannot_use_unprocessed_prefix(tmp_path):
    t = Tape(tmp_path / 'session')
    t.trade(200, 2)
    t.agg(210, 2, 2, 2)
    t.close()
    p = t.plan()
    p = replace(p, order=replace(p.order, decision_monotonic_ns=91_000,
                                 decision_utc=B + timedelta(microseconds=91)))
    with pytest.raises(MicrostructureIntegrityError, match='processed prefix'):
        maker_replay(t.events, p)


@pytest.mark.parametrize('tamper', ['raw', 'unit', 'ordinal', 'stream', 'http'])
def test_envelope_tampering_rejected(tmp_path, tamper):
    t = Tape(tmp_path / 'session')
    t.close()
    events = [dict(e) for e in t.events]
    if tamper == 'raw':
        events[6]['raw_payload_base64'] = base64.b64encode(b'{}').decode()
    elif tamper == 'unit':
        events[6]['exchange_timestamp_unit'] = 'GUESS'
    elif tamper == 'ordinal':
        events[6]['ingestion_ordinal'] = 100
    elif tamper == 'stream':
        events[6]['stream_name'] = 'btcusdc@aggTrade'
    else:
        events[0]['details'] = dict(events[0]['details'], response_received=Reading(B + timedelta(seconds=1), 1_000_000_000).payload())
    if tamper == 'http':
        assert replay(events).failures
    else:
        with pytest.raises(MicrostructureIntegrityError):
            replay(events)


def test_missing_aggregate_members_and_unacknowledged_raw(tmp_path):
    t = Tape(tmp_path / 'session')
    t.agg(200, 2, 2, 3, qty='2')
    t.close()
    assert t.twice().aggregates[2]['missing'] == [2, 3]
    assert certify_session(t.path)['status'] == 'ENGINEERING_CAPTURE_NOT_ELIGIBLE'
    (t.path / 'persistence.acks.jsonl').write_text('', encoding='utf-8')
    with pytest.raises(MicrostructureIntegrityError, match='unacknowledged'):
        load_session(t.path)


def test_allowlist_no_orders_or_credentials():
    transport = PublicRawTransport()
    for path in ('/api/v3/order', '/api/v3/account', '/api/v3/userDataStream'):
        with pytest.raises(ValueError, match='allowlist'):
            transport.get(path)
    with pytest.raises(ValueError, match='allowlist'):
        transport.get('/api/v3/depth', {'signature': 'forbidden'})


def test_phase1_identity_and_explicit_short_duration(tmp_path):
    assert set(phase1_binding(ROOT)) == set(PHASE1)
    for duration in (0, 31):
        with pytest.raises(ValueError, match='bounded'):
            asyncio.run(PublicCollectorV2(ROOT).capture_engineering(tmp_path / 'never', seconds=duration))
    assert not (tmp_path / 'never').exists()


def test_unaggregated_individual_never_certifies(tmp_path):
    t = Tape(tmp_path / 'session')
    t.trade(200, 2)
    t.close()
    assert t.twice().trades.keys() - t.twice().reconciled_trade_ids == {2}
    assert not certify_session(t.path)['checks']['trade_aggregate_reconciliation']
    assert t.bound_twice()['certified_conditional_credit'] is None


def test_freshness_and_cancel_clock_validation(tmp_path):
    t = Tape(tmp_path / 'session')
    t.trade(200, 2)
    t.agg(210, 2, 2, 2)
    t.close()
    assert maker_replay(t.events, t.plan(), Policy(book_max_age_us=10))['certified_conditional_credit'] is None
    with pytest.raises(MicrostructureIntegrityError, match='working start'):
        maker_replay(t.events, t.plan(cancel=120))


def test_exact_json_duplicate_key_and_invalid_decimal(tmp_path):
    t = Tape(tmp_path / 'session')
    t.clock.us = 200
    e = t.journal.capture(b'{"s":"BTCUSDC","s":"BTCUSDC"}', connection_id='ws-0')
    t.events.append(e)
    t.close()
    assert 'INVALID_PAYLOAD' in {f['reason'] for f in t.twice().failures}
    assert t.bound_twice()['certified_conditional_credit'] is None


def test_collector_orchestration_with_public_synthetic_transports(tmp_path):
    """All IO is fabricated; rendezvous fixes snapshot/stream order."""
    from src.microstructure.v12 import ReceiptClock
    source = Tape(tmp_path / 'fixture')
    source.close()
    payloads = {e['record_type']: e['payload'] for e in source.events}
    clock = ReceiptClock()
    ready = Event()

    class Transport:
        def get(self, path, params):
            start = clock()
            kind = {'/api/v3/time': 'CLOCK', '/api/v3/exchangeInfo': 'SYMBOL_RULES', '/api/v3/depth': 'REST_SNAPSHOT'}[path]
            payload = payloads[kind]
            if kind == 'CLOCK':
                payload = {'serverTime': start.payload()['utc_epoch_us']}
            response = clock()
            if kind == 'REST_SNAPSHOT':
                ready.set()
            return _canonical(payload).encode(), {'request_started': start.payload(),
                'response_received': response.payload(), 'timestamp_unit': 'MICROSECOND'}

    class WS:
        def __init__(self):
            self.index = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def recv(self, decode=False):
            assert decode is False
            while not ready.is_set():
                await asyncio.sleep(.001)
            if self.index == 4:
                await asyncio.Future()  # collector timeout/cancellation ends the bounded stream
            kind, stream = [('DIFF_DEPTH', 'btcusdc@depth@100ms'), ('TRADE', 'btcusdc@trade'),
                            ('AGGTRADE', 'btcusdc@aggTrade'), ('BOOK_TICKER', 'btcusdc@bookTicker')][self.index]
            self.index += 1
            p = dict(payloads[kind])
            now = clock().payload()['utc_epoch_us']
            for field in ('E', 'T'):
                if field in p:
                    p[field] = now
            return _canonical({'stream': stream, 'data': p}).encode()

    certificate = asyncio.run(PublicCollectorV2(ROOT, clock=clock, transport=Transport(),
                                               connect_factory=lambda *a, **kw: WS()).capture_engineering(
        tmp_path / 'capture', seconds=.5))
    assert certificate['status'] == 'ENGINEERING_REPLAY_CERTIFIED'
    _, events = load_session(tmp_path / 'capture')
    kinds = {e['record_type'] for e in events}
    assert {'CONNECT_START', 'CONNECTED', 'SNAPSHOT_REQUEST_START', 'SNAPSHOT_RECEIVED',
            'BOOK_SYNC_ESTABLISHED', 'DISCONNECTED', 'SESSION_CLOSE'} <= kinds
    assert certificate['phase1_binding'] == phase1_binding(ROOT)
    assert certificate == certify_session(tmp_path / 'capture')
