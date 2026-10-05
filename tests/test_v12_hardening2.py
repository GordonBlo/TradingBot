"""Fabricated sustained-ingress/seal faults; never maker or economic evaluation."""
import asyncio
from builtins import ExceptionGroup
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.microstructure import v12_operational as ops
from src.microstructure.v10 import sha256_file
from tests import test_v12_operational_readiness as operational

harness = operational.harness


class FrameQueue:
    def __init__(self):
        self.values = deque()

    def __len__(self):
        return len(self.values)

    def put(self, frame):
        self.values.append(frame)


@pytest.fixture
def sustained(harness, monkeypatch, tmp_path):
    collector = harness['collector']
    original_connect = collector.connect_factory
    original_journal = ops.OperationalJournal
    options = {'operational_policy': ops.OperationalPolicy(budget_bytes_per_second=1_000_000_000),
               'disk_usage': lambda p: SimpleNamespace(free=10 ** 12),
               'identity_registry': tmp_path / 'identities'}

    class Journal(original_journal):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.snapshot_seen = False
            self.raw = operational.Handle(self.raw, harness['config'].get('raw_close'))
            self.acks = operational.Handle(self.acks, harness['config'].get('ack_close'))
            harness['journals'].append(self)

        def capture(self, *args, **kwargs):
            event = super().capture(*args, **kwargs)
            if event['record_type'] == 'REST_SNAPSHOT':
                self.snapshot_seen = True
            return event

    def connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connection.recv_messages = SimpleNamespace(frames=FrameQueue())
        return connection

    monkeypatch.setattr(ops, 'OperationalJournal', Journal)
    collector.connect_factory = connect

    def run(path=None, **extra):
        return asyncio.run(collector.capture_engineering_soak(path or harness['path'], seconds=.03,
                          engineering_authorized=True, **(options | extra)))

    return {'harness': harness, 'run': run, 'options': options, 'path': harness['path']}


def hashes(path):
    return {p.name: sha256_file(p) for p in path.iterdir() if p.is_file()}


def test_clean_engineering_seal_and_read_only_second_classification(sustained):
    result = sustained['run']()
    assert result['certificate']['status'] == 'ENGINEERING_REPLAY_CERTIFIED'
    assert result['seal']['research_status'] == 'ENGINEERING_ONLY'
    assert result['classification']['classification'] == 'INELIGIBLE'
    assert result['classification']['research_eligible'] is False
    before = hashes(sustained['path'])
    assert ops.classify_session(sustained['path']) == result['classification']
    assert ops.classify_session(sustained['path']) == result['classification']
    assert hashes(sustained['path']) == before
    assert result['seal']['guard_high_water']['writer_pending_including_current'] == 1
    assert result['seal']['completion_count'] == sustained['harness']['journals'][0].ordinal


@pytest.mark.parametrize('case', ['availability', 'ws', 'writer', 'replay'])
def test_persisted_guard_rejection(sustained, monkeypatch, case):
    journal_class = ops.OperationalJournal
    original = journal_class.complete
    injected = []

    def complete(journal, event, machine):
        if event['record_type'] == 'TRADE' and not injected:
            injected.append(True)
            if case == 'availability':
                journal.clock.us += 1_100_000  # Coherent fabricated UTC and monotonic delay.
            elif case == 'ws':
                journal.observe_ws_frames(129)
        return original(journal, event, machine)

    original_capture = journal_class.capture

    def capture(journal, *args, **kwargs):
        if case in ('writer', 'replay') and not injected and b'"e":"trade"' in args[0]:
            injected.append(True)
            if case == 'writer':
                journal.writer_backlog += 1
            else:
                journal.pending[-1] = {}  # Fabricated queued work, not source data.
            try:
                return original_capture(journal, *args, **kwargs)
            finally:
                if case == 'writer':
                    journal.writer_backlog -= 1
                else:
                    journal.pending.pop(-1)
        return original_capture(journal, *args, **kwargs)

    monkeypatch.setattr(journal_class, 'capture', capture)
    monkeypatch.setattr(journal_class, 'complete', complete)
    result = sustained['run']()
    assert result['classification']['classification'] == 'INELIGIBLE'
    assert result['seal']['guard_violations']
    assert (sustained['path'] / 'availability.jsonl').exists()
    assert all(j.raw.closed and j.acks.closed and j.availability.closed
               for j in sustained['harness']['journals'])


def test_every_ws_enqueue_is_observed(sustained):
    sustained['run']()
    journal = sustained['harness']['journals'][0]
    queue = FrameQueue()
    journal.watch_queue(SimpleNamespace(recv_messages=SimpleNamespace(frames=queue)))
    for _ in range(129):
        queue.put(object())
    assert journal.high_water['ws_buffered_frames'] == 129
    assert journal.violations[-1]['reason'] == 'WS_FRAME_BACKLOG'


def test_low_disk_preflight_creates_no_session(sustained):
    with pytest.raises(OSError, match='DISK_RESERVE_BREACH'):
        sustained['run'](disk_usage=lambda p: SimpleNamespace(free=1))
    assert not sustained['path'].exists()


def test_disk_breach_during_write_preserves_failed_history(sustained):
    calls = []

    def disk(path):
        calls.append(path)
        return SimpleNamespace(free=10 ** 12 if len(calls) < 8 else 1)

    with pytest.raises((OSError, ExceptionGroup)):
        sustained['run'](disk_usage=disk)
    assert ops.classify_session(sustained['path'])['classification'] == 'FAILED'
    assert not (sustained['path'] / 'seal.json').exists()


@pytest.mark.parametrize('fault', ['temp_write', 'seal_fsync', 'publish', 'collision', 'replay', 'missing_ack', 'close'])
def test_seal_failures_never_produce_eligibility(sustained, monkeypatch, fault):
    if fault in ('temp_write', 'seal_fsync'):
        original = ops.write_exclusive

        def write(path, value):
            if path.name == 'seal.pending.json':
                if fault == 'seal_fsync':
                    # Exact fsync fault through real exclusive writer.
                    original_fsync = ops.os.fsync

                    def fail(fd):
                        raise OSError('fabricated seal fsync failure')

                    monkeypatch.setattr(ops.os, 'fsync', fail)
                    try:
                        return original(path, value)
                    finally:
                        monkeypatch.setattr(ops.os, 'fsync', original_fsync)
                raise OSError('fabricated temporary seal write failure')
            return original(path, value)

        monkeypatch.setattr(ops, 'write_exclusive', write)
    elif fault in ('publish', 'collision'):
        original = ops._atomic_publish

        def publish(source, destination):
            if fault == 'collision':
                destination.write_bytes(b'original conflicting seal')
                return original(source, destination)
            raise OSError('fabricated atomic publication failure')

        monkeypatch.setattr(ops, '_atomic_publish', publish)
    elif fault == 'replay':
        from src.research import v12_public_replay

        monkeypatch.setattr(v12_public_replay, 'certify_session',
                            lambda p: (_ for _ in ()).throw(ValueError('fabricated replay failure')))
    elif fault == 'missing_ack':
        original = ops.OperationalJournal.close

        def close(journal):
            original(journal)
            path = journal.directory / 'persistence.acks.jsonl'
            lines = path.read_bytes().splitlines(keepends=True)
            path.write_bytes(b''.join(lines[:-1]))  # Synthetic corruption only.

        monkeypatch.setattr(ops.OperationalJournal, 'close', close)
    else:
        sustained['harness']['config']['raw_close'] = OSError('fabricated shutdown close failure')
    with pytest.raises((OSError, ValueError, ExceptionGroup)):
        sustained['run']()
    assert ops.classify_session(sustained['path'])['classification'] == 'FAILED'
    if fault == 'collision':
        assert (sustained['path'] / 'seal.json').read_bytes() == b'original conflicting seal'
    else:
        assert not (sustained['path'] / 'seal.json').exists()


def test_duplicate_identity_and_restart_preserve_previous_session(sustained):
    sustained['harness']['config']['raw_close'] = OSError('fabricated FAILED session')
    with pytest.raises((OSError, ExceptionGroup)):
        sustained['run'](session_id='fixed-id')
    before = hashes(sustained['path'])
    sustained['harness']['config'].clear()
    with pytest.raises(FileExistsError):
        sustained['run'](sustained['path'].with_name('same-id-new-path'), session_id='fixed-id')
    with pytest.raises(FileExistsError):
        sustained['run'](session_id='different-id')
    result = sustained['run'](sustained['path'].with_name('new-session'), session_id='new-id')
    assert result['classification']['research_status'] == 'ENGINEERING_ONLY'
    assert hashes(sustained['path']) == before


def test_active_requires_live_owner_not_stale_marker(tmp_path):
    path = tmp_path / 'session'
    path.mkdir()
    ops.ACTIVE_OWNERS.add(path.resolve())
    assert ops.classify_session(path)['classification'] == 'ACTIVE'
    ops.ACTIVE_OWNERS.discard(path.resolve())
    assert ops.classify_session(path)['classification'] == 'FAILED'


@pytest.mark.parametrize('duration,authorized', [(901, True), (0, True), (float('nan'), True), (900, False), (True, True)])
def test_bounds_and_explicit_authorization_fail_before_creation(tmp_path, duration, authorized):
    from src.microstructure.v12 import PublicCollectorV2

    collector = PublicCollectorV2(Path(__file__).resolve().parents[1])
    with pytest.raises(ValueError):
        asyncio.run(collector.capture_engineering_soak(tmp_path / 'not-created', seconds=duration,
                                                      engineering_authorized=authorized))
    assert not (tmp_path / 'not-created').exists()


def test_failure_precedence_and_future_classifier_mechanics():
    assert ops._eligibility_classification(engineering_only=False, failure=True,
                checks_pass=True, guards_pass=True, prospective_binding_valid=True) == 'FAILED'
    assert ops._eligibility_classification(engineering_only=False, failure=False,
                checks_pass=True, guards_pass=True, prospective_binding_valid=True) == 'ELIGIBLE'
    assert ops._eligibility_classification(engineering_only=True, failure=False,
                checks_pass=True, guards_pass=True, prospective_binding_valid=True) == 'INELIGIBLE'


def test_config_and_completion_corruption_reject_sealed_session(sustained):
    sustained['run']()
    path = sustained['path'] / 'availability.jsonl'
    path.write_bytes(path.read_bytes()[:-10])
    assert ops.classify_session(sustained['path'])['classification'] == 'FAILED'


def test_disk_sizing_uses_configured_reserve_plus_full_budget(tmp_path):
    policy = replace(ops.OperationalPolicy(), reserve_bytes=1000, budget_bytes_per_second=100)
    guard = ops.DiskProtection(tmp_path, policy, 900, lambda p: SimpleNamespace(free=90_999))
    with pytest.raises(OSError):
        guard.check(preflight=True)
