"""Phase-3 operational failure injection only; no network or research outcomes."""
import asyncio
import errno
from builtins import BaseExceptionGroup
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, Lock

import pytest

from src.microstructure import v12
from src.microstructure.v10 import _canonical, sha256_file
from src.research.v12_public_replay import certify_session, load_session


def test_journals_close_when_final_session_record_hits_disk_failure(tmp_path, monkeypatch):
    """A failed close record must not bypass deterministic file cleanup."""
    journals = []

    class DiskFullWriter:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def write(self, value):
            raise OSError(errno.ENOSPC, 'fabricated disk-full during SESSION_CLOSE')

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    class FaultJournal(v12.IngestionJournal):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            journals.append(self)

        def control(self, kind, **kwargs):
            if kind == 'SESSION_CLOSE':
                self.raw = DiskFullWriter(self.raw)
            return super().control(kind, **kwargs)

    class OfflineTransport:
        def get(self, path, params):
            raise OSError(errno.ENETDOWN, 'fabricated public transport unavailable')

    class OfflineConnection:
        async def __aenter__(self):
            raise OSError(errno.ENETDOWN, 'fabricated public stream unavailable')

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(v12, 'IngestionJournal', FaultJournal)
    collector = v12.PublicCollectorV2(
        Path(__file__).resolve().parents[1], transport=OfflineTransport(),
        connect_factory=lambda *args, **kwargs: OfflineConnection(),
    )
    try:
        with pytest.raises(OSError, match='fabricated disk-full'):
            asyncio.run(collector.capture_engineering(tmp_path / 'failed_engineering', seconds=.001))
        assert len(journals) == 1
        assert journals[0].raw.closed and journals[0].acks.closed, (
            'SESSION_CLOSE write failure skipped journal.close(); raw and ack handles remain open'
        )
    finally:
        # Isolated test cleanup only. Never repairs or rewrites session bytes.
        for journal in journals:
            journal.close()


class Handle:
    """Local write/close injection; preserves underlying persisted bytes."""
    def __init__(self, wrapped, close_error=None):
        self.wrapped = wrapped
        self.close_error = close_error
        self.write_error = None
        self.close_attempts = 0

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def write(self, text):
        if self.write_error is not None:
            raise self.write_error
        return self.wrapped.write(text)

    def close(self):
        self.close_attempts += 1
        if self.close_error is not None:
            error, self.close_error = self.close_error, None
            raise error
        self.wrapped.close()


class SyntheticClock:
    def __init__(self):
        self.us = 10_000
        self.base = datetime(2026, 10, 5, tzinfo=UTC)
        self.lock = Lock()

    def __call__(self):
        with self.lock:
            reading = v12.Reading(self.base + timedelta(microseconds=self.us), self.us * 1000)
            self.us += 1
            return reading


def leaves(error):
    if isinstance(error, BaseExceptionGroup):
        return [leaf for child in error.exceptions for leaf in leaves(child)]
    return [error]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """Fixed fabricated public schema, with injectable ownership failures."""
    config = {}
    journals, connections = [], []
    started, released, worker_done = Event(), Event(), Event()
    clock = SyntheticClock()
    real_journal = v12.IngestionJournal

    class Journal(real_journal):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.raw = Handle(self.raw, config.get('raw_close'))
            self.acks = Handle(self.acks, config.get('ack_close'))
            journals.append(self)
            self.snapshot_seen = False

        def capture(self, *args, **kwargs):
            event = super().capture(*args, **kwargs)
            if event['record_type'] == 'REST_SNAPSHOT':
                self.snapshot_seen = True
            return event

        def control(self, kind, **kwargs):
            if kind == 'SESSION_CLOSE':
                self.raw.write_error = config.get('write')
                self.acks.write_error = config.get('ack_write')
            return super().control(kind, **kwargs)

    class Transport:
        def get(self, path, params):
            request = clock()
            if path == '/api/v3/time':
                payload = {'serverTime': request.payload()['utc_epoch_us']}
            elif path == '/api/v3/exchangeInfo':
                payload = {'symbols': [{'symbol': 'BTCUSDC', 'status': 'TRADING',
                    'isSpotTradingAllowed': True, 'orderTypes': ['LIMIT_MAKER'], 'filters': [
                        {'filterType': 'PRICE_FILTER', 'tickSize': '1', 'minPrice': '1', 'maxPrice': '1000'},
                        {'filterType': 'LOT_SIZE', 'stepSize': '1', 'minQty': '1', 'maxQty': '100'},
                        {'filterType': 'MIN_NOTIONAL', 'minNotional': '1'}]}]}
            else:
                if config.get('block_worker'):
                    started.set()
                    released.wait(5)  # Failure safeguard only; synchronization uses events.
                    worker_done.set()
                    if config.get('worker_error'):
                        raise config['worker_error']
                payload = {'lastUpdateId': 10, 'bids': [['99', '5']], 'asks': [['105', '5']]}
            return _canonical(payload).encode(), {'timestamp_unit': 'MICROSECOND',
                'request_started': request.payload(), 'response_received': clock().payload()}

    class Connection:
        def __init__(self):
            self.index, self.close_attempts = 0, 0
            connections.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            self.close_attempts += 1
            released.set()
            if config.get('ws_close'):
                raise config['ws_close']
            return False

        async def recv(self, **kwargs):
            if config.get('cancel'):
                if config.get('block_worker'):
                    while not started.is_set():
                        await asyncio.sleep(0)
                raise asyncio.CancelledError('fabricated Ctrl+C cancellation')
            while not journals[-1].snapshot_seen:
                await asyncio.sleep(0)
            if self.index == 4:
                await asyncio.Future()
            now = clock().payload()['utc_epoch_us']
            payloads = [
                ('btcusdc@depth@100ms', {'e': 'depthUpdate', 's': 'BTCUSDC', 'E': now,
                    'U': 11, 'u': 11, 'b': [], 'a': []}),
                ('btcusdc@trade', {'e': 'trade', 's': 'BTCUSDC', 'E': now, 'T': now,
                    't': 1, 'p': '100', 'q': '1', 'm': True}),
                ('btcusdc@aggTrade', {'e': 'aggTrade', 's': 'BTCUSDC', 'E': now, 'T': now,
                    'a': 1, 'f': 1, 'l': 1, 'p': '100', 'q': '1', 'm': True}),
                ('btcusdc@bookTicker', {'s': 'BTCUSDC', 'u': 11, 'b': '99', 'B': '5', 'a': '105', 'A': '5'}),
            ]
            stream, payload = payloads[self.index]
            self.index += 1
            return _canonical({'stream': stream, 'data': payload}).encode()

    class LoopDeadline:
        def __init__(self):
            self.connection_prefix = len(connections)

        def time(self):
            return 100 if len(connections) > self.connection_prefix and connections[-1].index == 4 else 0

    class AsyncFacade:
        """Only the collector's deadline is virtual; scheduler remains real."""
        def get_running_loop(self):
            return LoopDeadline()

        def __getattr__(self, name):
            return getattr(asyncio, name)

    monkeypatch.setattr(v12, 'asyncio', AsyncFacade())
    monkeypatch.setattr(v12, 'IngestionJournal', Journal)
    collector = v12.PublicCollectorV2(Path(__file__).resolve().parents[1], clock=clock,
                                     transport=Transport(), connect_factory=lambda *a, **kw: Connection())
    yield {'config': config, 'journals': journals, 'connections': connections,
           'collector': collector, 'path': tmp_path / 'capture', 'worker_done': worker_done}
    released.set()
    for journal in journals:
        journal.close()


@pytest.mark.parametrize('case', ['write', 'raw_close', 'both', 'ack_close', 'all'])
def test_independent_cleanup_attempts_and_explicit_errors(harness, case):
    config = harness['config']
    write = OSError(errno.ENOSPC, 'original SESSION_CLOSE write failure')
    raw_close = OSError(errno.EIO, 'event journal close failure')
    ack_close = OSError(errno.EIO, 'ack journal close failure')
    if case in ('write', 'both', 'all'):
        config['write'] = write
    if case in ('raw_close', 'both', 'all'):
        config['raw_close'] = raw_close
    if case in ('ack_close', 'all'):
        config['ack_close'] = ack_close
    with pytest.raises(BaseException) as caught:
        asyncio.run(harness['collector'].capture_engineering(harness['path'], seconds=.03))
    errors = leaves(caught.value)
    expected = [e for name, e in (('write', write), ('raw_close', raw_close), ('ack_close', ack_close)) if name in config]
    assert errors == expected  # Original object and deterministic primary-first order.
    journal = harness['journals'][0]
    assert journal.raw.close_attempts == journal.acks.close_attempts == 1
    assert harness['connections'][0].close_attempts == 1
    assert not (harness['path'] / 'closure.summary.json').exists()
    failure = v12.strict_json((harness['path'] / 'closure.failure.json').read_bytes())
    assert failure['status'] == 'FAILED' and failure['eligible'] is False
    hashes = {p.name: sha256_file(p) for p in harness['path'].iterdir()}
    journal.close()
    journal.close()
    assert journal.raw.closed and journal.acks.closed
    assert {p.name: sha256_file(p) for p in harness['path'].iterdir()} == hashes


def test_acknowledgement_failure_is_detectable_and_restart_does_not_repair(harness):
    harness['config']['ack_write'] = OSError(errno.ENOSPC, 'missing final acknowledgement')
    with pytest.raises(OSError, match='missing final acknowledgement'):
        asyncio.run(harness['collector'].capture_engineering(harness['path'], seconds=.03))
    with pytest.raises(ValueError, match='unacknowledged'):
        load_session(harness['path'])
    before = {p.name: sha256_file(p) for p in harness['path'].iterdir()}
    harness['config'].clear()
    with pytest.raises(FileExistsError):
        asyncio.run(harness['collector'].capture_engineering(harness['path'], seconds=.03))
    certificate = asyncio.run(harness['collector'].capture_engineering(harness['path'].with_name('new_session'), seconds=.03))
    assert certificate['status'] == 'ENGINEERING_REPLAY_CERTIFIED'
    assert {p.name: sha256_file(p) for p in harness['path'].iterdir()} == before


def test_normal_graceful_close_and_repeated_cleanup(harness):
    certificate = asyncio.run(harness['collector'].capture_engineering(harness['path'], seconds=.03))
    assert certificate['status'] == 'ENGINEERING_REPLAY_CERTIFIED'
    assert not (harness['path'] / 'closure.failure.json').exists()
    assert certify_session(harness['path']) == certificate
    journal = harness['journals'][0]
    assert journal.raw.closed and journal.acks.closed
    before = {p.name: sha256_file(p) for p in harness['path'].iterdir()}
    journal.close()
    journal.close()
    assert journal.raw.close_attempts == journal.acks.close_attempts == 1
    assert {p.name: sha256_file(p) for p in harness['path'].iterdir()} == before


@pytest.mark.parametrize('worker_fails', [False, True])
def test_cancellation_drains_owned_http_worker_and_closes_resources(harness, worker_fails):
    harness['config'].update(cancel=True, block_worker=True)
    secondary = OSError(errno.EIO, 'late worker cleanup error')
    if worker_fails:
        harness['config']['worker_error'] = secondary
    with pytest.raises(BaseException) as caught:
        asyncio.run(harness['collector'].capture_engineering(harness['path'], seconds=1))
    errors = leaves(caught.value)
    assert isinstance(errors[0], asyncio.CancelledError)
    assert errors[1:] == ([secondary] if worker_fails else [])
    assert harness['worker_done'].is_set()
    assert harness['connections'][0].close_attempts == 1
    journal = harness['journals'][0]
    assert journal.raw.closed and journal.acks.closed
    assert not (harness['path'] / 'closure.summary.json').exists()
    assert not certify_session(harness['path'])['checks']['clean_session_close']


def test_websocket_cleanup_error_remains_explicit(harness):
    error = OSError(errno.EIO, 'WebSocket cleanup failure')
    harness['config']['ws_close'] = error
    with pytest.raises(BaseExceptionGroup) as caught:
        asyncio.run(harness['collector'].capture_engineering(harness['path'], seconds=.03))
    assert leaves(caught.value) == [error]
    assert harness['connections'][0].close_attempts == 1
    assert all(journal.raw.closed and journal.acks.closed for journal in harness['journals'])
    assert not (harness['path'] / 'closure.summary.json').exists()


def test_http_read_and_cleanup_error_both_preserved(monkeypatch):
    primary, secondary = OSError('HTTP read failed'), OSError('HTTP response close failed')

    class Response:
        close_attempted = False

        def __enter__(self):
            return self

        def read(self):
            raise primary

        def __exit__(self, *args):
            self.close_attempted = True
            raise secondary

    response = Response()
    monkeypatch.setattr(v12, 'urlopen', lambda *args, **kwargs: response)
    with pytest.raises(BaseExceptionGroup) as caught:
        v12.PublicRawTransport().get('/api/v3/time')
    assert leaves(caught.value) == [primary, secondary]
    assert response.close_attempted


def test_partial_journal_construction_closes_first_handle(tmp_path, monkeypatch):
    original = Path.open
    opened = []
    primary = OSError(errno.ENOSPC, 'ack journal open failed')

    def open_file(path, *args, **kwargs):
        if path.name == 'persistence.acks.jsonl':
            raise primary
        file = original(path, *args, **kwargs)
        opened.append(file)
        return file

    monkeypatch.setattr(Path, 'open', open_file)
    with pytest.raises(OSError) as caught:
        v12.IngestionJournal(tmp_path / 'partial', 'synthetic')
    assert caught.value is primary
    assert len(opened) == 1 and opened[0].closed


@pytest.mark.parametrize('failure_side', ['raw', 'ack'])
def test_fsync_failure_propagates_and_does_not_create_clean_seal(harness, monkeypatch, failure_side):
    original = v12.os.fsync
    error = OSError(errno.EIO, f'{failure_side} journal fsync failure')
    injected = []

    def fsync(fd):
        if harness['journals'] and not injected:
            journal = harness['journals'][0]
            target = getattr(journal, 'raw' if failure_side == 'raw' else 'acks')
            if fd == target.fileno():
                injected.append(True)
                raise error
        return original(fd)

    monkeypatch.setattr(v12.os, 'fsync', fsync)
    with pytest.raises(BaseException) as caught:
        asyncio.run(harness['collector'].capture_engineering(harness['path'], seconds=.03))
    assert leaves(caught.value)[0] is error
    assert all(journal.raw.closed and journal.acks.closed for journal in harness['journals'])
    assert not (harness['path'] / 'closure.summary.json').exists()
    assert v12.strict_json((harness['path'] / 'closure.failure.json').read_bytes())['status'] == 'FAILED'


def test_second_cancellation_cannot_skip_shutdown_cleanup(harness, monkeypatch):
    harness['config']['cancel'] = True

    async def scenario():
        draining, release = asyncio.Event(), asyncio.Event()

        async def drain(*tasks, **kwargs):
            draining.set()
            await release.wait()
            return await asyncio.gather(*tasks, **kwargs)

        monkeypatch.setattr(v12.asyncio, 'gather', drain, raising=False)
        owner = asyncio.create_task(harness['collector'].capture_engineering(harness['path'], seconds=1))
        await asyncio.wait_for(draining.wait(), 5)
        owner.cancel('second shutdown cancellation')
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(BaseExceptionGroup) as caught:
            await owner
        assert all(isinstance(error, asyncio.CancelledError) for error in leaves(caught.value))
        assert len(leaves(caught.value)) == 2

    asyncio.run(scenario())
    assert all(journal.raw.closed and journal.acks.closed for journal in harness['journals'])
    assert harness['connections'][0].close_attempts == 1
    assert not (harness['path'] / 'closure.summary.json').exists()


def test_failure_artifact_write_failure_is_secondary_and_visible(harness, monkeypatch):
    primary = OSError(errno.ENOSPC, 'SESSION_CLOSE disk-full')
    secondary = OSError(errno.ENOSPC, 'failure-artifact disk-full')
    harness['config']['write'] = primary
    original = Path.open

    def open_file(path, *args, **kwargs):
        if path.name == 'closure.failure.json':
            raise secondary
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', open_file)
    with pytest.raises(BaseExceptionGroup) as caught:
        asyncio.run(harness['collector'].capture_engineering(harness['path'], seconds=.03))
    assert leaves(caught.value) == [primary, secondary]
    assert all(journal.raw.closed and journal.acks.closed for journal in harness['journals'])
    assert not (harness['path'] / 'closure.summary.json').exists()


def test_startup_failure_after_opening_journals_still_cleans_up(harness, monkeypatch):
    primary = OSError(errno.ENOSPC, 'manifest write failed')
    original = Path.write_text

    def write_text(path, *args, **kwargs):
        if path.name == 'session.manifest.json':
            raise primary
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'write_text', write_text)
    with pytest.raises(OSError) as caught:
        asyncio.run(harness['collector'].capture_engineering(harness['path'], seconds=.03))
    assert caught.value is primary
    assert all(journal.raw.closed and journal.acks.closed for journal in harness['journals'])
    assert not harness['connections']
    assert not (harness['path'] / 'closure.summary.json').exists()
