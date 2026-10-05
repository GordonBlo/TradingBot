"""Engineering-only sustained V12 guards, completion journal and immutable seals.

No maker plan, prediction, return, economics or prospective creation interface.
The original raw envelope/replay semantics remain unchanged.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import uuid
from builtins import BaseExceptionGroup
from dataclasses import asdict, dataclass
from pathlib import Path

from src.cli.import_v10_github_artifacts import _atomic_publish
from src.microstructure.v10 import MicrostructureIntegrityError, _canonical, sha256_file
from src.microstructure.v12 import (
    STREAMS,
    IngestionJournal,
    _close_handles,
    _raise_failures,
    digest,
    phase1_binding,
    strict_json,
)

FILES = ('session.manifest.json', 'events.jsonl', 'persistence.acks.jsonl',
         'availability.jsonl', 'closure.summary.json', 'metadata.final.json', 'terminal.json')
ACTIVE_OWNERS = set()  # Only this live process can attest ACTIVE; stale files never suffice.


class OperationalGuardViolation(RuntimeError):
    """Persisted operational rejection, distinct from journal corruption."""


@dataclass(frozen=True)
class OperationalPolicy:
    availability_limit_ns: int = 1_000_000_000
    ws_frame_limit: int = 128
    writer_backlog_limit: int = 1
    replay_backlog_limit: int = 1
    reserve_bytes: int = 10 * 1024 ** 3
    budget_bytes_per_second: int = 279_902  # Ceil(4 * measured 69975.4422 B/s).
    refresh_seconds: int = 30
    maximum_records: int = 100_000
    maximum_reconciliation_references: int = 1_000_000

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise ValueError('positive integer operational limits required')
        if self.refresh_seconds >= 60:
            raise ValueError('refresh must precede the existing 60s freshness boundary')


def write_exclusive(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as file:
        file.write(_canonical(value))
        file.flush()
        os.fsync(file.fileno())


class DiskProtection:
    def __init__(self, path, policy, seconds, disk_usage=shutil.disk_usage):
        self.path, self.policy, self.disk_usage = path, policy, disk_usage
        self.budget = math.ceil(seconds * policy.budget_bytes_per_second)
        self.minimum_free = None
        self.checks = 0

    def check(self, pending_bytes=0, *, preflight=False):
        free = self.disk_usage(self.path).free
        self.minimum_free = free if self.minimum_free is None else min(self.minimum_free, free)
        self.checks += 1
        required = self.policy.reserve_bytes + (self.budget if preflight else pending_bytes)
        if free < required:
            error = OSError(f'DISK_RESERVE_BREACH: free={free}, required={required}')
            error.add_note('V12 journal persistence failure')
            raise error
        return free


class OperationalJournal(IngestionJournal):
    operational_enabled = True

    def __init__(self, *args, operational_policy, disk, **kwargs):
        self.operational_policy, self.disk = operational_policy, disk
        self.pending = {}
        self.writer_backlog = 0
        self.ws_frames = None
        self.queue_observations, self.violations = [], []
        self.high_water = {'ws_buffered_frames': 0, 'writer_pending_including_current': 0,
                           'replay_pending_including_current': 0}
        self.reconciliation_references = 0
        self.total_written_bytes = 0
        self.completed = 0
        self.shutdown_started_ns = None
        super().__init__(*args, **kwargs)
        try:
            self.availability = (self.directory / 'availability.jsonl').open('x', encoding='utf-8', newline='\n')
        except BaseException as exc:  # noqa: BLE001 -- close every owned handle on partial startup
            _close_handles((('raw', self.raw), ('ack', self.acks)), exc)

    def manifest_fields(self, workspace):
        sources = ('src/microstructure/v12_operational.py', 'src/cli/v12_engineering_soak.py',
                   'src/cli/import_v10_github_artifacts.py')
        config = {'operational_policy': asdict(self.operational_policy),
                  'collection_policy': asdict(self.policy), 'session_budget_bytes': self.disk.budget,
                  'engineering_only': True}
        return {'operational_version': 'V12_OPERATIONAL_HARDENING_2',
                'operational_policy': asdict(self.operational_policy),
                'operational_config': config,
                'operational_config_sha256': digest(_canonical(config).encode()),
                'operational_source_sha256': {source: sha256_file(workspace / source) for source in sources},
                'availability_semantics': 'Receipt through raw fsync, ack fsync and live replay; excludes completion-journal own future write',
                'queue_semantics': 'Every enqueue of library-local buffered WS data frames; not messages/kernel/network residence',
                'memory_bounds': {'maximum_records': self.operational_policy.maximum_records,
                                  'maximum_reconciliation_references': self.operational_policy.maximum_reconciliation_references},
                'durability': 'File flush/fsync and same-volume no-replace atomic publication; directory durability/power-loss guarantee UNKNOWN',
                'prospective_creation_authorized': False}

    def persist_manifest(self, manifest):
        self.disk.check(len(_canonical(manifest).encode()))
        write_exclusive(self.directory / 'session.manifest.json', manifest)

    def violation(self, reason, value, limit):
        reading = self.clock().payload()
        self.violations.append({'reason': reason, 'value': value, 'limit': limit,
                                'observed_at': reading, 'after_ingestion_ordinal': self.ordinal})

    def observe_ws_frames(self, count):
        if type(count) is not int or count < 0:
            raise MicrostructureIntegrityError('invalid local WS frame queue observation')
        self.ws_frames = count
        self.high_water['ws_buffered_frames'] = max(self.high_water['ws_buffered_frames'], count)
        self.queue_observations.append({'frames': count, 'observed_at': self.clock().payload()})
        if count > self.operational_policy.ws_frame_limit:
            self.violation('WS_FRAME_BACKLOG', count, self.operational_policy.ws_frame_limit)

    def watch_queue(self, ws):
        # websockets' assembler SimpleQueue is a data-FRAME queue. Hook every
        # enqueue, including bursts above its soft backpressure high-water mark.
        queue = ws.recv_messages.frames
        self.ws_queue = queue
        original = queue.put

        def put(frame):
            original(frame)
            self.observe_ws_frames(len(queue))

        queue.put = put
        self.observe_ws_frames(len(queue))

    def capture(self, raw, **kwargs):
        receipt = self.clock()  # Common ingress boundary precedes disk/guard work.
        # Writes and live replay are synchronous in the single ingress owner;
        # there is no asynchronous writer queue. Counts include the current item.
        initial_violations = len(self.violations)
        if hasattr(self, 'ws_queue'):
            self.observe_ws_frames(len(self.ws_queue))
        self.writer_backlog += 1
        self.high_water['writer_pending_including_current'] = max(
            self.high_water['writer_pending_including_current'], self.writer_backlog)
        replay_pending = len(self.pending) + 1
        self.high_water['replay_pending_including_current'] = max(
            self.high_water['replay_pending_including_current'], replay_pending)
        for reason, value, limit in (
            ('WRITER_BACKLOG', self.writer_backlog, self.operational_policy.writer_backlog_limit),
            ('REPLAY_BACKLOG', replay_pending, self.operational_policy.replay_backlog_limit),
        ):
            if value > limit:
                self.violation(reason, value, limit)
        try:
            estimate = 4 * len(raw) + 32768  # Envelope plus completion evidence headroom.
            self.disk.check(estimate)
            if self.total_written_bytes + estimate > self.disk.budget:
                self.violation('SESSION_BYTE_BUDGET', self.total_written_bytes + estimate, self.disk.budget)
            if self.ordinal >= self.operational_policy.maximum_records:
                self.violation('RECORD_CAPACITY', self.ordinal + 1, self.operational_policy.maximum_records)
            kwargs.setdefault('ingress_queue_depth', self.ws_frames)
            kwargs['_receipt'] = receipt
            event = super().capture(raw, **kwargs)
            self.pending[event['ingestion_ordinal']] = {
                'receipt_monotonic_ns': event['receipt_monotonic_ns'],
                'raw_fsync_completed_ns': self.raw_fsync_completed_ns,
                'ack_fsync_completed_ns': self.ack_fsync_completed_ns,
                'writer_backlog': self.writer_backlog, 'replay_backlog': replay_pending,
                'new_capture_violations': len(self.violations) - initial_violations,
            }
            return event
        finally:
            self.writer_backlog -= 1

    def before_live(self, event, machine):
        if event['record_type'] == 'AGGTRADE':
            p = event['payload']
            if type(p.get('f')) is int and type(p.get('l')) is int:
                self.reconciliation_references += max(0, p['l'] - p['f'] + 1)
        if self.reconciliation_references > self.operational_policy.maximum_reconciliation_references:
            self.violation('RECONCILIATION_CAPACITY', self.reconciliation_references,
                           self.operational_policy.maximum_reconciliation_references)
        if event['record_type'] in STREAMS.values():
            for name, value, age in (('CLOCK_FRESHNESS', machine.clock_mapping, self.policy.clock_max_age_us),
                                     ('METADATA_FRESHNESS', machine.metadata, self.policy.metadata_max_age_us)):
                observed_age = None if value is None else event['receipt_monotonic_ns'] - value['available_ns']
                if observed_age is None or observed_age > age * 1000:
                    self.violation(name, observed_age, age * 1000)

    def complete(self, event, machine):
        ordinal = event['ingestion_ordinal']
        timing = self.pending.pop(ordinal)
        trace_hash = digest(_canonical(machine.trace[-1]).encode())
        live = self.clock()
        delay = live.monotonic_ns - timing['receipt_monotonic_ns']
        if delay > self.operational_policy.availability_limit_ns:
            self.violation('LOCAL_AVAILABILITY_DELAY', delay, self.operational_policy.availability_limit_ns)
        if event['record_type'] == 'DISCONNECTED' and event['payload'].get('expected'):
            self.shutdown_started_ns = event['receipt_monotonic_ns']
        row = {'ingestion_ordinal': ordinal, 'session_id': self.session_id, 'clock_id': self.clock_id,
               **{k: v for k, v in timing.items() if k != 'new_capture_violations'},
               'live_processing_completed_ns': live.monotonic_ns,
               'live_processing_completed_utc': live.payload()['utc'], 'total_availability_delay_ns': delay,
               'live_trace_sha256': trace_hash, 'queue_observations': self.queue_observations,
               'queue_high_water': dict(self.high_water), 'guard_violations': self.violations,
               'disk_free_bytes': self.disk.minimum_free,
               'upstream_queue_residence': 'UNKNOWN'}
        # Completion is persisted AFTER the measured stages. Its own future
        # persistence completion cannot be put inside the same record.
        encoded = _canonical(row) + '\n'
        try:
            self.disk.check(len(encoded.encode()))
            self.availability.write(encoded)
            self.availability.flush()
            os.fsync(self.availability.fileno())
        except BaseException as exc:
            exc.add_note('V12 journal persistence failure')
            raise
        self.queue_observations = []
        self.completed += 1
        self.total_written_bytes = self.raw.tell() + self.acks.tell() + self.availability.tell()
        if self.violations and event['record_type'] != 'SESSION_CLOSE':
            raise OperationalGuardViolation('operational guard latched; see availability journal')

    def close(self):
        failures = []
        for name, handle in (('raw', self.raw), ('ack', self.acks), ('availability', self.availability)):
            try:
                if not handle.closed:
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException as exc:  # noqa: BLE001 -- fsync failure cannot skip independent closes
                exc.add_note(f'V12 final fsync: {name}')
                failures.append(exc)
        try:
            _close_handles((('raw', self.raw), ('ack', self.acks), ('availability', self.availability)))
        except BaseException as exc:  # noqa: BLE001 -- expose every close failure
            failures.append(exc)
        self.closed_at = self.clock()
        _raise_failures('V12 operational journal shutdown failures', failures)


def flatten(error):
    return [leaf for child in error.exceptions for leaf in flatten(child)] if isinstance(error, BaseExceptionGroup) else [error]


def prepare_identity(directory, session_id, registry):
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}', session_id):
        raise ValueError('safe immutable session ID required')
    if directory.exists():
        raise FileExistsError('session directory already exists; never repair/restart in place')
    registry.mkdir(parents=True, exist_ok=True)
    write_exclusive(registry / (session_id + '.json'), {'session_id': session_id,
                    'directory': str(directory.resolve()), 'scope': 'ENGINEERING_ONLY'})


def _verify_completion(directory, events):
    rows = []
    with (directory / 'availability.jsonl').open('rb') as file:
        for line in file:
            if not line.endswith(b'\n'):
                raise MicrostructureIntegrityError('truncated availability record')
            rows.append(strict_json(line))
    if len(rows) != len(events):
        raise MicrostructureIntegrityError('missing per-event availability completion')
    with (directory / 'persistence.acks.jsonl').open('rb') as file:
        acks = [strict_json(line) for line in file]
    for event, ack, row in zip(events, acks, rows):
        receipt, raw, ack_end, live = (row[k] for k in ('receipt_monotonic_ns', 'raw_fsync_completed_ns',
                                      'ack_fsync_completed_ns', 'live_processing_completed_ns'))
        if (row['session_id'], row['clock_id'], row['ingestion_ordinal'], receipt) != (
            event['session_id'], event['monotonic_clock_id'], event['ingestion_ordinal'], event['receipt_monotonic_ns']
        ) or not receipt <= raw <= ack_end <= live or raw != ack['persistence_complete_monotonic_ns'] or (
            row['total_availability_delay_ns'] != live - receipt
        ):
            raise MicrostructureIntegrityError('availability provenance/order mismatch')
    return rows


def _eligibility_classification(*, engineering_only, failure, checks_pass, guards_pass,
                               prospective_binding_valid=False):
    """Pure future classifier mechanics; no prospective session creation API."""
    if failure:
        return 'FAILED'
    if engineering_only or not checks_pass or not guards_pass or not prospective_binding_valid:
        return 'INELIGIBLE'
    return 'ELIGIBLE'


def seal_session(directory, journal, certificate, terminal):
    from src.research.v12_public_replay import load_session

    manifest, events = load_session(directory)
    rows = _verify_completion(directory, events)
    journal.disk.check()
    metadata = [{'ordinal': e['ingestion_ordinal'], 'type': e['record_type'],
                 'raw_payload_sha256': e['raw_payload_sha256'],
                 'canonical_payload_sha256': e['canonical_payload_sha256']} for e in events
                if e['record_type'] in ('CLOCK', 'SYMBOL_RULES')]
    write_exclusive(directory / 'metadata.final.json', {'metadata_records': metadata})
    journal.disk.check()
    write_exclusive(directory / 'terminal.json', terminal)
    classification = _eligibility_classification(engineering_only=True, failure=False,
                     checks_pass=all(certificate['checks'].values()), guards_pass=not journal.violations)
    seal = {'schema': 'V12_IMMUTABLE_ENGINEERING_SESSION_SEAL_1', 'status': 'SEALED',
            'session_id': manifest['session_id'], 'classification': classification,
            'research_status': 'ENGINEERING_ONLY', 'research_eligible': False,
            'acquisition_authorized': False, 'prospective': False,
            'start_utc': events[0]['received_at_utc'], 'end_utc': events[-1]['received_at_utc'],
            'start_monotonic_ns': events[0]['receipt_monotonic_ns'],
            'end_monotonic_ns': events[-1]['receipt_monotonic_ns'], 'monotonic_namespace': manifest['clock_id'],
            'artifact_sha256': {name: sha256_file(directory / name) for name in FILES},
            'source_sha256': manifest['source_sha256'] | manifest['operational_source_sha256'],
            'config_sha256': manifest['operational_config_sha256'],
            'runtime_identity': {name: manifest[name] for name in ('python', 'websockets', 'source_commit', 'utc_clock', 'monotonic_clock')},
            'replay_sha256': certificate['replay_sha256'], 'replay_byte_identical': certificate['checks']['deterministic_replay'],
            'lifecycle_checks': certificate['checks'], 'integrity_failures': certificate['failures'],
            'guard_high_water': journal.high_water, 'guard_violations': journal.violations,
            'failure_reasons': terminal['reasons'], 'completion_count': len(rows),
            'disk_checks': journal.disk.checks, 'minimum_free_bytes': journal.disk.minimum_free,
            'session_budget_bytes': journal.disk.budget, 'reserve_bytes': journal.operational_policy.reserve_bytes,
            'shutdown_seconds': (journal.closed_at.monotonic_ns - journal.shutdown_started_ns) / 1e9
                if journal.shutdown_started_ns is not None else None,
            'durability': manifest['durability']}
    seal['seal_body_sha256'] = digest(_canonical(seal).encode())
    temporary = directory / 'seal.pending.json'
    journal.disk.check(len(_canonical(seal).encode()))
    write_exclusive(temporary, seal)
    expected = sha256_file(temporary)
    _atomic_publish(temporary, directory / 'seal.json')
    if sha256_file(directory / 'seal.json') != expected:
        raise MicrostructureIntegrityError('published seal differs from exclusive temporary bytes')
    return seal


def classify_session(directory):
    """Read-only repeated classification; never writes, seals, repairs or promotes."""
    try:
        if (directory / 'closure.failure.json').exists():
            return {'classification': 'FAILED', 'research_status': 'ENGINEERING_ONLY', 'research_eligible': False}
        if not (directory / 'seal.json').exists():
            active = directory.resolve() in ACTIVE_OWNERS
            # External/read-after-crash readers fail closed without a seal;
            # only the live owner process can expose ACTIVE.
            return {'classification': 'ACTIVE' if active else 'FAILED',
                    'research_status': 'ENGINEERING_ONLY', 'research_eligible': False}
        seal = strict_json((directory / 'seal.json').read_bytes())
        body_hash = seal.pop('seal_body_sha256')
        if digest(_canonical(seal).encode()) != body_hash:
            raise MicrostructureIntegrityError('seal body hash mismatch')
        if seal['status'] != 'SEALED' or seal['classification'] not in ('INELIGIBLE', 'FAILED') or (
            seal['research_status'] != 'ENGINEERING_ONLY' or seal['research_eligible'] or seal['prospective']
        ):
            raise MicrostructureIntegrityError('engineering seal classification/scope mismatch')
        if set(seal['artifact_sha256']) != set(FILES):
            raise MicrostructureIntegrityError('incomplete sealed artifact binding')
        for name, expected in seal['artifact_sha256'].items():
            if sha256_file(directory / name) != expected:
                raise MicrostructureIntegrityError('seal artifact hash mismatch')
        from src.research.v12_public_replay import load_session

        manifest, events = load_session(directory)
        rows = _verify_completion(directory, events)
        if not rows or seal['session_id'] != manifest['session_id'] or seal['monotonic_namespace'] != manifest['clock_id']:
            raise MicrostructureIntegrityError('seal session/namespace mismatch')
        if seal['source_sha256'] != manifest['source_sha256'] | manifest['operational_source_sha256'] or (
            seal['config_sha256'] != manifest['operational_config_sha256']
        ):
            raise MicrostructureIntegrityError('source/config seal binding mismatch')
        if digest(_canonical(manifest['operational_config']).encode()) != seal['config_sha256']:
            raise MicrostructureIntegrityError('config hash mismatch')
        return {'classification': seal['classification'], 'research_status': 'ENGINEERING_ONLY',
                'research_eligible': False, 'seal_verified': True}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {'classification': 'FAILED', 'research_status': 'ENGINEERING_ONLY',
                'research_eligible': False, 'reason': str(exc)}


async def capture_sustained(collector, directory, *, seconds, engineering_authorized=False,
                            operational_policy=None, disk_usage=shutil.disk_usage,
                            session_id=None, identity_registry=None):
    if engineering_authorized is not True or isinstance(seconds, bool) or not 0 < seconds <= 900:
        raise ValueError('explicit engineering authorization and 0 < duration <=900 seconds required')
    policy = operational_policy or OperationalPolicy()
    bindings = phase1_binding(collector.workspace)
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    disk = DiskProtection(directory.parent, policy, seconds, disk_usage)
    disk.check(preflight=True)
    session_id = session_id or uuid.uuid4().hex
    registry = identity_registry or collector.workspace / 'reports/v12_engineering_identities'
    prepare_identity(directory, session_id, registry)
    journal, failures = None, []
    try:
        journal = OperationalJournal(directory, session_id, clock=collector.clock, policy=collector.policy,
                                     operational_policy=policy, disk=disk)
        ACTIVE_OWNERS.add(directory.resolve())
        await collector._capture_engineering(directory, seconds, bindings, journal)
    except BaseException as exc:  # noqa: BLE001 -- preserve operation and cleanup failures
        failures.append(exc)
    if journal is not None:
        try:
            journal.close()
        except BaseException as exc:  # noqa: BLE001 -- seal prohibited after any close/fsync failure
            failures.append(exc)
    ACTIVE_OWNERS.discard(directory.resolve())
    operational_only = bool(failures) and all(isinstance(e, OperationalGuardViolation)
                                            for error in failures for e in flatten(error))
    terminal = {'status': 'INELIGIBLE' if operational_only else 'FAILED' if failures else 'CLOSED',
                'engineering_only': True, 'reasons': [str(e) for error in failures for e in flatten(error)]}
    try:
        if failures and not operational_only:
            if directory.exists():
                write_exclusive(directory / 'closure.failure.json', terminal)
            _raise_failures('V12 sustained capture failures', failures)
        if journal is None:
            _raise_failures('V12 startup failed', failures)
        from src.research.v12_public_replay import certify_session

        certificate = certify_session(directory)
        write_exclusive(directory / 'closure.summary.json', certificate)
        seal = seal_session(directory, journal, certificate, terminal)
        classification = classify_session(directory)
        if not classification.get('seal_verified'):
            raise MicrostructureIntegrityError('final seal verification failed')
        return {'seal': seal, 'certificate': certificate, 'classification': classification}
    except BaseException as exc:
        if directory.exists() and not (directory / 'closure.failure.json').exists():
            try:
                write_exclusive(directory / 'closure.failure.json', {'status': 'FAILED',
                                'engineering_only': True, 'reasons': [str(exc)]})
            except BaseException as secondary:  # noqa: BLE001 -- expose failure-evidence write too
                _raise_failures('V12 seal/failure-evidence errors', [exc, secondary])
        raise
