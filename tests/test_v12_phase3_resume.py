"""Operational raw-only Phase-3 checks; never apply a hypothetical maker plan."""
import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from src.microstructure.v10 import MicrostructureIntegrityError, sha256_file
from src.research.v12_public_replay import certify_session
from tests import test_v12_operational_readiness as operational
from tests.test_v12_public_replay import US, Tape

harness = operational.harness


def history(path):
    return {p.name: sha256_file(p) for p in path.iterdir() if p.is_file()}


@pytest.mark.parametrize('case', ['abrupt', 'truncated_raw', 'truncated_ack'])
def test_interrupted_history_is_not_certifiable_and_is_never_repaired(tmp_path, case):
    tape = Tape(tmp_path / 'session')
    if case == 'abrupt':
        # Emulates killed-process bytes: acknowledged prefix, no SESSION_CLOSE.
        tape.journal.close()
    else:
        tape.close()
        name = 'events.jsonl' if case == 'truncated_raw' else 'persistence.acks.jsonl'
        path = tape.path / name
        path.write_bytes(path.read_bytes()[:-10])  # Synthetic fault only.
    before = history(tape.path)
    if case == 'abrupt':
        first = certify_session(tape.path)
        assert first == certify_session(tape.path)
        assert first['status'] == 'ENGINEERING_CAPTURE_NOT_ELIGIBLE'
        assert first['checks']['clean_session_close'] is False
    else:
        for _ in range(2):
            with pytest.raises((ValueError, MicrostructureIntegrityError)):
                certify_session(tape.path)
    assert history(tape.path) == before


@pytest.mark.parametrize('case,failed_check', [
    ('disconnect', 'no_integrity_failures_or_resync'),
    ('snapshot_failure', 'no_integrity_failures_or_resync'),
    ('bootstrap_reconnect', 'no_integrity_failures_or_resync'),
    ('depth_gap', 'no_integrity_failures_or_resync'),
    ('trade_gap', 'no_integrity_failures_or_resync'),
    ('unresolved_reconciliation', 'trade_aggregate_reconciliation'),
])
def test_operational_failures_remain_ineligible_after_clean_close(tmp_path, case, failed_check):
    tape = Tape(tmp_path / 'session', buffered=case == 'bootstrap_reconnect')
    if case in ('disconnect', 'bootstrap_reconnect'):
        tape.add(180, 'DISCONNECTED', {'expected': False})
        tape.add(185, 'RESYNC_START', {})
        tape.add(190, 'RECONNECT_START', {}, connection='ws-1', epoch=1)
        tape.add(195, 'RECONNECTED', {}, connection='ws-1', epoch=1)
        tape.add(200, 'REST_SNAPSHOT', {'lastUpdateId': 11,
                 'bids': [['99', '5']], 'asks': [['105', '5']]})
        tape.add(210, 'DIFF_DEPTH', {'e': 'depthUpdate', 's': 'BTCUSDC',
                 'E': US + 205, 'U': 12, 'u': 12, 'b': [], 'a': []},
                 connection='ws-1', epoch=1)
    elif case == 'snapshot_failure':
        tape.add(180, 'PUBLIC_REQUEST_FAILED', {'path': '/api/v3/depth',
                 'reason': 'OSError', 'message': 'fabricated snapshot outage'})
    elif case == 'depth_gap':
        tape.add(180, 'DIFF_DEPTH', {'e': 'depthUpdate', 's': 'BTCUSDC',
                 'E': US + 175, 'U': 13, 'u': 13, 'b': [], 'a': []})
    elif case == 'trade_gap':
        tape.trade(180, 3)
        tape.agg(190, 2, 3, 3)
    else:
        tape.trade(180, 2)  # Trailing unmatched capture edge rejects the whole session.
    tape.close()
    before = history(tape.path)
    first = certify_session(tape.path)
    assert first == certify_session(tape.path)
    assert first['status'] == 'ENGINEERING_CAPTURE_NOT_ELIGIBLE'
    assert first['checks'][failed_check] is False
    assert first['research_eligible'] is first['acquisition_authorized'] is False
    assert history(tape.path) == before


def test_existing_engineering_duration_guard_cannot_be_bypassed(tmp_path):
    from src.microstructure.v12 import PublicCollectorV2

    collector = PublicCollectorV2(Path(__file__).resolve().parents[1])
    with pytest.raises(ValueError, match='<=30 seconds'):
        asyncio.run(collector.capture_engineering(tmp_path / 'not_created', seconds=31))
    assert not (tmp_path / 'not_created').exists()


def test_actual_abrupt_process_exit_preserves_unclosed_history(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = ('import os,sys; from pathlib import Path; '
              'from tests.test_v12_public_replay import Tape; '
              'tape=Tape(Path(sys.argv[1])); os._exit(17)')
    path = tmp_path / 'abrupt_child'
    child = subprocess.run([sys.executable, '-c', script, str(path)], cwd=root,
                           capture_output=True, text=True, check=False)
    assert child.returncode == 17, child.stderr
    before = history(path)
    assert certify_session(path)['checks']['clean_session_close'] is False
    assert history(path) == before


def test_rest_snapshot_failure_is_persisted_and_resources_close(harness, monkeypatch):
    from src.microstructure import v12

    original = harness['collector'].transport.get

    def get(path, params):
        if path == '/api/v3/depth':
            raise OSError('fabricated REST snapshot unavailable')
        return original(path, params)

    class Deadline:
        calls = 0

        def time(self):
            self.calls += 1
            return 100 if self.calls > 15 else 0

    deadline = Deadline()
    monkeypatch.setattr(v12.asyncio, 'get_running_loop', lambda: deadline, raising=False)
    monkeypatch.setattr(harness['collector'].transport, 'get', get)
    certificate = asyncio.run(harness['collector'].capture_engineering(harness['path'], seconds=.03))
    assert certificate['status'] == 'ENGINEERING_CAPTURE_NOT_ELIGIBLE'
    assert 'PUBLIC_REQUEST_FAILED' in {f['reason'] for f in certificate['failures']}
    assert all(j.raw.closed and j.acks.closed for j in harness['journals'])
    assert all(c.close_attempts == 1 for c in harness['connections'])


def test_disconnect_during_snapshot_bootstrap_cannot_be_rehabilitated(harness):
    original = harness['collector'].connect_factory
    first = True

    def connect(*args, **kwargs):
        nonlocal first
        connection = original(*args, **kwargs)
        if first:
            first = False

            async def recv(**kwargs):
                raise OSError('fabricated disconnect during bootstrap')

            connection.recv = recv
        return connection

    harness['collector'].connect_factory = connect
    certificate = asyncio.run(harness['collector'].capture_engineering(harness['path'], seconds=.03))
    assert len(harness['connections']) == 2
    assert certificate['status'] == 'ENGINEERING_CAPTURE_NOT_ELIGIBLE'
    assert 'DISCONNECTED_INTERVAL' in {f['reason'] for f in certificate['failures']}
    assert all(j.raw.closed and j.acks.closed for j in harness['journals'])
    assert all(c.close_attempts == 1 for c in harness['connections'])
