"""V12 public collector V2. Smoke <=30s; separately authorized engineering <=900s.

Reuses V10 canonical persistence/source hashes and the existing L2 reconstructor
through replay. No credentials, private endpoint, strategy or order interface.
"""
from __future__ import annotations

import asyncio
import base64
import ctypes
import hashlib
import json
import os
import platform
import time
import uuid
from builtins import BaseExceptionGroup
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from src.exchange.public_market_client import PublicMarketDataClient
from src.microstructure.v10 import (
    MicrostructureIntegrityError,
    _canonical,
    _utc_text,
    sha256_file,
    source_commit,
)
from src.research.v12_maker_bound import _epoch_us

SCHEMA = "V12_PUBLIC_EXECUTION_OBSERVATION_1"
VERSION = "V12_PUBLIC_MICROSTRUCTURE_COLLECTOR_2"
STREAMS = {
    "btcusdc@depth@100ms": "DIFF_DEPTH", "btcusdc@trade": "TRADE",
    "btcusdc@aggTrade": "AGGTRADE", "btcusdc@bookTicker": "BOOK_TICKER",
}
WS_URL = ("wss://data-stream.binance.vision/stream?streams="
          + "/".join(STREAMS) + "&timeUnit=MICROSECOND")
PUBLIC_PATHS = {"/api/v3/time", "/api/v3/exchangeInfo", "/api/v3/depth"}
PHASE1 = (
    "docs/v12_conditional_maker_bound.md", "src/research/v12_maker_bound.py",
    "tests/test_v12_maker_bound.py", "reports/v12_maker_bounds/phase1_verification.json",
)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def strict_json(raw: bytes) -> dict:
    def reject(value):
        raise MicrostructureIntegrityError(f"nonfinite JSON constant {value}")
    def pairs(items):
        result = {}
        for name, value in items:
            if name in result:
                raise MicrostructureIntegrityError("duplicate JSON key")
            result[name] = value
        return result
    result = json.loads(raw.decode("utf-8"), parse_constant=reject, object_pairs_hook=pairs)
    if not isinstance(result, dict):
        raise MicrostructureIntegrityError("public JSON object required")
    return result


def phase1_binding(workspace: Path) -> dict:
    proof = strict_json((workspace / PHASE1[-1]).read_bytes())
    if proof["decision"] != "CONDITIONAL_MAKER_BOUND_READY":
        raise MicrostructureIntegrityError("Phase-1 bound not ready")
    for name, expected in proof["artifact_sha256"].items():
        if name not in PHASE1 or sha256_file(workspace / name) != expected:
            raise MicrostructureIntegrityError("Phase-1 artifact binding changed")
    if set(proof["artifact_sha256"]) != set(PHASE1[:-1]):
        raise MicrostructureIntegrityError("incomplete Phase-1 binding")
    return {name: sha256_file(workspace / name) for name in PHASE1}


@dataclass(frozen=True)
class Reading:
    utc: datetime
    monotonic_ns: int
    sampling_span_ns: int = 0

    def payload(self):
        return {"utc": _utc_text(self.utc), "utc_epoch_us": _epoch_us(self.utc),
                "monotonic_ns": self.monotonic_ns, "sampling_span_ns": self.sampling_span_ns}


class ReceiptClock:
    def __init__(self):
        self.precise_windows_utc = None
        if os.name == 'nt':
            self.precise_windows_utc = ctypes.windll.kernel32.GetSystemTimePreciseAsFileTime
            self.precise_windows_utc.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)]
            self.precise_windows_utc.restype = None

    def __call__(self) -> Reading:
        before = time.perf_counter_ns()
        if self.precise_windows_utc is not None:
            filetime = ctypes.c_ulonglong()
            self.precise_windows_utc(ctypes.byref(filetime))
            now = datetime(1970, 1, 1, tzinfo=UTC) + timedelta(
                microseconds=filetime.value // 10 - 11_644_473_600_000_000)
        else:
            now = datetime.now(UTC)
        after = time.perf_counter_ns()
        return Reading(now, after, after - before)


@dataclass(frozen=True)
class Policy:
    snapshot_limit: int = 5000
    max_levels: int = 5000
    max_processing_delay_us: int = 1_000_000
    clock_jump_tolerance_us: int = 10_000
    metadata_max_age_us: int = 60_000_000
    clock_max_age_us: int = 60_000_000
    book_max_age_us: int = 1_000_000
    max_bridge_events: int = 10_000

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError("positive integer engineering policy required")
        if self.snapshot_limit > 5000 or self.max_levels > self.snapshot_limit:
            raise ValueError("unsupported snapshot/retained depth bound")


DEFAULT_POLICY = Policy()


def _raise_failures(message, failures):
    """Original failure first; never replace it with a secondary cleanup error."""
    failures = list({id(exc): exc for exc in failures}.values())
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise BaseExceptionGroup(message, failures) from failures[0]


def _close_handles(handles, primary=None):
    failures = [primary] if primary is not None else []
    for name, handle in handles:
        try:
            if not handle.closed:
                handle.close()
        except BaseException as exc:  # noqa: BLE001 -- every cleanup attempted, failures re-raised
            exc.add_note(f"V12 cleanup: {name}")
            failures.append(exc)
    _raise_failures("V12 journal cleanup failures", failures)


@contextmanager
def _http_resource(manager):
    resource = manager.__enter__()
    primary = None
    try:
        yield resource
    except BaseException as exc:  # noqa: BLE001 -- retain interrupts until resource cleanup
        primary = exc
    failures = [primary] if primary is not None else []
    try:
        manager.__exit__(type(primary) if primary else None, primary,
                         primary.__traceback__ if primary else None)
    except BaseException as exc:  # noqa: BLE001 -- cleanup failure grouped with original cause
        exc.add_note("V12 cleanup: public HTTP response")
        failures.append(exc)
        raise BaseExceptionGroup("V12 HTTP operation/cleanup failures", failures) from failures[0]
    _raise_failures("V12 HTTP operation/cleanup failures", failures)


@asynccontextmanager
async def _websocket_resource(manager):
    resource = await manager.__aenter__()
    primary = None
    try:
        yield resource
    except BaseException as exc:  # noqa: BLE001 -- retain cancellation until resource cleanup
        primary = exc
    failures = [primary] if primary is not None else []
    try:
        await manager.__aexit__(type(primary) if primary else None, primary,
                                primary.__traceback__ if primary else None)
    except BaseException as exc:  # noqa: BLE001 -- cleanup failure grouped with original cause
        exc.add_note("V12 cleanup: public WebSocket")
        failures.append(exc)
        # Do not let the reconnect handler consume a cleanup error as an outage.
        raise BaseExceptionGroup("V12 WebSocket operation/cleanup failures", failures) from failures[0]
    _raise_failures("V12 WebSocket operation failed", failures)


class IngestionJournal:
    """Single event-loop ingress boundary: ordinal and clocks BEFORE decoding.

    One append-only journal removes V10's cross-file merge ambiguity. Exact
    persistence completion is in a separately hashed acknowledgement journal;
    it cannot truthfully be serialized inside a record before that write ends.
    """
    def __init__(self, directory: Path, session_id: str, *, clock=None,
                 policy=DEFAULT_POLICY, fsync=True, clock_id=None):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory, self.session_id = directory, session_id
        self.clock, self.policy, self.fsync = clock or ReceiptClock(), policy, fsync
        self.clock_id = clock_id or uuid.uuid4().hex
        self.ordinal = 0
        self.mapping_id = None
        self.indices = {}
        self.last_reading = None
        self.raw = (directory / "events.jsonl").open("x", encoding="utf-8", newline="\n")
        try:
            self.acks = (directory / "persistence.acks.jsonl").open("x", encoding="utf-8", newline="\n")
        except BaseException as exc:  # noqa: BLE001 -- partial construction must release event file
            _close_handles((("event journal after acknowledgement-open failure", self.raw),), exc)

    def capture(self, raw: bytes, *, record_type=None, stream=None,
                connection_id="control", epoch=0, unit="NONE", details=None,
                ingress_queue_depth=None, _receipt=None):
        receipt = _receipt if _receipt is not None else self.clock()
        self.ordinal += 1
        ordinal = self.ordinal
        decode_start = self.clock()
        error = None
        try:
            decoded = strict_json(raw)
            if record_type is None:
                name = decoded.get("stream")
                record_type = STREAMS.get(name)
                if record_type is None or not isinstance(decoded.get("data"), dict):
                    raise MicrostructureIntegrityError("unexpected combined public stream")
                stream = name
                payload = decoded["data"]
            else:
                payload = decoded
        except (ValueError, UnicodeError, MicrostructureIntegrityError) as exc:
            decoded, payload = None, None
            record_type, stream = "INVALID_PAYLOAD", stream or "CONTROL"
            error = str(exc)
        decoded_at = self.clock()
        dispatch = self.clock()
        key = (connection_id, stream)
        index = self.indices.get(key, 0)
        self.indices[key] = index + 1
        jumps = False
        if self.last_reading is not None:
            wall_delta = _epoch_us(receipt.utc) - _epoch_us(self.last_reading.utc)
            mono_delta = (receipt.monotonic_ns - self.last_reading.monotonic_ns) // 1000
            jumps = wall_delta < 0 or mono_delta < 0 or abs(wall_delta - mono_delta) > (
                self.policy.clock_jump_tolerance_us
            )
        self.last_reading = receipt
        exchange = {name: payload.get(name) if payload else None for name in ("E", "T")}
        scale = {"MICROSECOND": 1, "MILLISECOND": 1000, "NONE": None}.get(unit)
        intervals = {}
        for name, value in exchange.items():
            intervals[name] = ([value * scale, value * scale + scale - 1]
                               if type(value) is int and value >= 0 and scale else None)
        event = {
            "schema_version": SCHEMA, "collector_version": VERSION,
            "record_type": record_type, "venue": "BINANCE_SPOT", "symbol": "BTCUSDC",
            "session_id": self.session_id, "connection_id": connection_id,
            "reconnect_epoch": epoch, "stream_name": stream,
            "stream_record_index": index, "ingestion_ordinal": ordinal,
            "received_at_utc": _utc_text(receipt.utc), "receipt_utc_epoch_us": _epoch_us(receipt.utc),
            "receipt_monotonic_ns": receipt.monotonic_ns, "clock_sampling_span_ns": receipt.sampling_span_ns,
            "monotonic_clock_id": self.clock_id, "host_boot_id": None,
            "clock_mapping_id": self.mapping_id, "clock_discontinuity": jumps,
            "exchange_timestamp_unit": unit, "exchange_timestamps_raw": exchange,
            "exchange_time_intervals_us": intervals,
            "decode_start_monotonic_ns": decode_start.monotonic_ns,
            "decode_end_monotonic_ns": decoded_at.monotonic_ns,
            "dispatch_monotonic_ns": dispatch.monotonic_ns,
            "processing_available_utc": _utc_text(dispatch.utc),
            "processing_delay_ns": dispatch.monotonic_ns - receipt.monotonic_ns,
            "ingress_queue_depth": ingress_queue_depth,
            "backlog_observability": "UNKNOWN" if ingress_queue_depth is None else "OBSERVED",
            "observed_drop_count": 0, "drop_count_semantics": "application drops only; transport loss unknown",
            "raw_payload_base64": base64.b64encode(raw).decode("ascii"),
            "raw_payload_sha256": digest(raw), "decoded_payload": decoded, "payload": payload,
            "canonical_payload_sha256": digest(_canonical(decoded).encode()),
            "persistence_complete_monotonic_ns": None,
            "persistence_completion_reference": f"persistence.acks.jsonl#{ordinal}",
            "details": details or {}, "parse_error": error,
        }
        encoded = _canonical(event) + "\n"
        try:
            self.raw.write(encoded)
            self.raw.flush()
            if self.fsync:
                os.fsync(self.raw.fileno())
            completed = self.clock()
            if getattr(self, 'operational_enabled', False):
                self.raw_fsync_completed_ns = completed.monotonic_ns
            ack = {"ingestion_ordinal": ordinal, "event_sha256": digest(encoded.encode()),
                   "persistence_complete_monotonic_ns": completed.monotonic_ns}
            self.acks.write(_canonical(ack) + "\n")
            self.acks.flush()
            if self.fsync:
                os.fsync(self.acks.fileno())
            if getattr(self, 'operational_enabled', False):
                self.ack_fsync_completed_ns = self.clock().monotonic_ns
        except BaseException as exc:
            exc.add_note('V12 journal persistence failure')
            raise
        return event

    def control(self, kind, *, connection_id="control", epoch=0, **details):
        return self.capture(_canonical(details).encode(), record_type=kind, stream="CONTROL",
                            connection_id=connection_id, epoch=epoch, details=details)

    def close(self):
        _close_handles((("event journal", self.raw), ("acknowledgement journal", self.acks)))


class PublicRawTransport:
    """Exact HTTP bytes, allowlisted market-data-only GET endpoints."""
    def __init__(self, clock=None):
        self.clock = clock or ReceiptClock()

    def get(self, path, params=None):
        params = params or {}
        if path not in PUBLIC_PATHS or set(params) - {"symbol", "limit"}:
            raise ValueError("endpoint/parameters outside public market allowlist")
        if params.get("symbol", "BTCUSDC") != "BTCUSDC":
            raise ValueError("BTCUSDC only")
        url = PublicMarketDataClient.REST_BASE_URL + path
        if params:
            url += "?" + urlencode(params)
        started = self.clock()
        request = Request(url, headers={"X-MBX-TIME-UNIT": "MICROSECOND"}, method="GET")
        with _http_resource(urlopen(request, timeout=4)) as response:
            raw = response.read()
            received = self.clock()
            status = response.status
        return raw, {"url": url, "method": "GET", "status": status,
                     "request_started": started.payload(), "response_received": received.payload(),
                     "timestamp_unit": "MICROSECOND"}


class PublicCollectorV2:
    def __init__(self, workspace: Path, *, clock=None, transport=None,
                 connect_factory=connect, policy=DEFAULT_POLICY):
        self.workspace, self.policy = workspace, policy
        self.clock = clock or ReceiptClock()
        self.transport = transport or PublicRawTransport(self.clock)
        self.connect_factory = connect_factory

    async def capture_engineering_soak(self, directory: Path, *, seconds: float,
                                       engineering_authorized=False, **options):
        from src.microstructure.v12_operational import capture_sustained

        return await capture_sustained(self, directory, seconds=seconds,
                                       engineering_authorized=engineering_authorized, **options)

    async def capture_engineering(self, directory: Path, *, seconds: float):
        if not 0 < seconds <= 30:
            raise ValueError("engineering capture must be explicitly bounded to <=30 seconds")
        bindings = phase1_binding(self.workspace)
        from src.research.v12_public_replay import certify_session

        journal = IngestionJournal(directory, uuid.uuid4().hex, clock=self.clock, policy=self.policy)
        failures = []
        try:
            await self._capture_engineering(directory, seconds, bindings, journal)
        except BaseException as exc:  # noqa: BLE001 -- all failures re-raised after journal cleanup
            failures.append(exc)
        try:
            journal.close()
        except BaseException as exc:  # noqa: BLE001 -- retain secondary cleanup failure
            failures.append(exc)
        if failures:
            def describe(exc):
                return {"type": type(exc).__name__, "message": str(exc),
                        "notes": getattr(exc, '__notes__', []),
                        "causes": [describe(e) for e in exc.exceptions]
                        if isinstance(exc, BaseExceptionGroup) else []}
            try:
                # Separate append-only failure evidence; never rewrite raw/history.
                with (directory / 'closure.failure.json').open('x', encoding='utf-8') as output:
                    output.write(_canonical({"status": "FAILED", "session_id": journal.session_id,
                                             "eligible": False, "failures": [describe(e) for e in failures]}))
                    output.flush()
                    os.fsync(output.fileno())
            except BaseException as exc:  # noqa: BLE001 -- failure to persist failure evidence is explicit
                exc.add_note('V12 cleanup: failure-artifact persistence; no successful closure exists')
                failures.append(exc)
            _raise_failures("V12 operation/shutdown failures", failures)
        certificate = certify_session(directory)
        (directory / "closure.summary.json").write_text(_canonical(certificate), encoding="utf-8")
        return certificate

    async def _capture_engineering(self, directory, seconds, bindings, journal):
        from src.research.v12_public_replay import ReplayMachine
        manifest = {"schema_version": "V12_ENGINEERING_SESSION_1", "collector_version": VERSION,
                    "session_id": journal.session_id, "clock_id": journal.clock_id,
                    "policy": asdict(self.policy), "phase1_binding": bindings,
                    "source_commit": source_commit(self.workspace), "python": platform.python_version(),
                    "websockets": version("websockets"),
                    "monotonic_clock": vars(time.get_clock_info('perf_counter')),
                    "utc_clock": 'GetSystemTimePreciseAsFileTime' if os.name == 'nt' else 'datetime.now(UTC)',
                    "source_sha256": {name: sha256_file(self.workspace / name) for name in (
                        "src/microstructure/v12.py", "src/research/v12_public_replay.py",
                        "src/microstructure/v10.py", "src/orderbook/book.py", "src/orderbook/models.py")},
                    "websocket_url": WS_URL, "engineering_only": True,
                    "research_eligibility": "NOT_AUTHORIZED", "duration_cap_seconds": seconds,
                    "private_data": False, "orders": False, "commission": "UNAVAILABLE_PUBLIC_ONLY"}
        if getattr(journal, 'operational_enabled', False):
            manifest.update(journal.manifest_fields(self.workspace))
        if getattr(journal, 'operational_enabled', False):
            journal.persist_manifest(manifest)
        else:
            (directory / "session.manifest.json").write_text(_canonical(manifest), encoding="utf-8")
        machine = ReplayMachine(self.policy)

        def feed(event):
            if getattr(journal, 'operational_enabled', False):
                journal.before_live(event, machine)
            machine.feed(event)
            if getattr(journal, 'operational_enabled', False):
                journal.complete(event, machine)
            return event

        def control(kind, **kwargs):
            return feed(journal.control(kind, **kwargs))

        async def http(path, kind, params, connection_id, epoch):
            nonlocal synced_before
            request_id = f"http-{journal.ordinal + 1}"
            control("SNAPSHOT_REQUEST_START" if kind == "REST_SNAPSHOT" else "PUBLIC_REQUEST_START",
                    connection_id=connection_id, epoch=epoch, request_id=request_id, path=path)
            try:
                worker = asyncio.create_task(asyncio.to_thread(self.transport.get, path, params))
                http_workers.append(worker)
                # Cancelling the ingest task must not orphan an in-flight HTTP worker.
                raw, details = await asyncio.shield(worker)
                details["request_id"] = request_id
                event = feed(journal.capture(raw, record_type=kind, stream="REST",
                                             connection_id=request_id, epoch=epoch,
                                             unit="MICROSECOND", details=details))
                if kind == "CLOCK":
                    journal.mapping_id = f"clock-{event['ingestion_ordinal']}"
                if kind == "REST_SNAPSHOT":
                    control("SNAPSHOT_RECEIVED", connection_id=connection_id, epoch=epoch,
                            request_id=request_id, snapshot_ordinal=event["ingestion_ordinal"])
                    if machine.synced and not synced_before:
                        control("BOOK_SYNC_ESTABLISHED", connection_id=connection_id, epoch=epoch,
                                depth_id=machine.book.last_update_id)
                        if epoch or machine.resync_seen:
                            control("RESYNC_COMPLETE", connection_id=connection_id, epoch=epoch)
                        synced_before = True
                return event
            except (OSError, ValueError, MicrostructureIntegrityError) as exc:
                if 'V12 journal persistence failure' in getattr(exc, '__notes__', ()):
                    raise
                control("PUBLIC_REQUEST_FAILED", connection_id=connection_id, epoch=epoch,
                        request_id=request_id, reason=type(exc).__name__, message=str(exc), path=path)
                observed_http_failures.add(id(exc))
                return None

        tasks, http_workers, observed_http_failures = [], [], set()
        synced_before = False
        epoch, clean = 0, False
        loop = asyncio.get_running_loop()
        deadline = loop.time() + seconds
        primary = None
        try:
            await http("/api/v3/time", "CLOCK", {}, "control", 0)
            await http("/api/v3/exchangeInfo", "SYMBOL_RULES", {"symbol": "BTCUSDC"}, "control", 0)
            if getattr(journal, 'operational_enabled', False):
                async def refresh_metadata():
                    while True:
                        await asyncio.sleep(journal.operational_policy.refresh_seconds)
                        await http('/api/v3/time', 'CLOCK', {}, 'control', 0)
                        await http('/api/v3/exchangeInfo', 'SYMBOL_RULES', {'symbol': 'BTCUSDC'}, 'control', 0)

                tasks.append(asyncio.create_task(refresh_metadata()))
            while loop.time() < deadline:
                connection_id = f"ws-{epoch}"
                control("CONNECT_START" if epoch == 0 else "RECONNECT_START",
                        connection_id=connection_id, epoch=epoch)
                try:
                    async with _websocket_resource(self.connect_factory(WS_URL, open_timeout=4, close_timeout=1,
                                                    ping_interval=20, ping_timeout=20,
                                                    max_queue=128)) as ws:
                        if getattr(journal, 'operational_enabled', False):
                            journal.watch_queue(ws)
                        control("CONNECTED" if epoch == 0 else "RECONNECTED",
                                connection_id=connection_id, epoch=epoch)
                        snapshot_task = asyncio.create_task(http(
                            "/api/v3/depth", "REST_SNAPSHOT", {"symbol": "BTCUSDC", "limit": self.policy.snapshot_limit},
                            connection_id, epoch))
                        tasks.append(snapshot_task)
                        synced_before = False
                        while loop.time() < deadline:
                            try:
                                raw = await asyncio.wait_for(ws.recv(decode=False),
                                                             min(.25, max(.001, deadline - loop.time())))
                            except TimeoutError:
                                continue
                            failures_before = len(machine.failures)
                            event = feed(journal.capture(raw, connection_id=connection_id,
                                                         epoch=epoch, unit="MICROSECOND"))
                            if machine.synced and not synced_before:
                                control("BOOK_SYNC_ESTABLISHED", connection_id=connection_id, epoch=epoch,
                                        depth_id=machine.book.last_update_id)
                                if epoch or machine.resync_seen:
                                    control("RESYNC_COMPLETE", connection_id=connection_id, epoch=epoch)
                            synced_before = machine.synced
                            if len(machine.failures) > failures_before and event["record_type"] == "DIFF_DEPTH" and not machine.synced:
                                control("GAP_DETECTED", connection_id=connection_id, epoch=epoch,
                                        reason=machine.failures[-1]["reason"])
                                control("RESYNC_START", connection_id=connection_id, epoch=epoch)
                                if snapshot_task.done():
                                    snapshot_task = asyncio.create_task(http(
                                        "/api/v3/depth", "REST_SNAPSHOT", {"symbol": "BTCUSDC", "limit": self.policy.snapshot_limit},
                                        connection_id, epoch))
                                    tasks.append(snapshot_task)
                        control("DISCONNECTED", connection_id=connection_id, epoch=epoch,
                                reason="BOUNDED_ENGINEERING_END", expected=True)
                    clean = True
                    break
                except (OSError, ValueError, WebSocketException, MicrostructureIntegrityError) as exc:
                    if 'V12 journal persistence failure' in getattr(exc, '__notes__', ()):
                        raise
                    control("DISCONNECTED", connection_id=connection_id, epoch=epoch,
                            reason=type(exc).__name__, message=str(exc), expected=False)
                    control("RESYNC_START", connection_id=connection_id, epoch=epoch)
                    epoch += 1
                    await asyncio.sleep(min(.5, max(0, deadline - loop.time())))
        except BaseException as exc:  # noqa: BLE001 -- retain interruption while all resources drain
            primary = exc

        async def shutdown():
            failures = []
            for task in tasks:
                try:
                    if not task.done():
                        task.cancel()
                except BaseException as exc:  # noqa: BLE001 -- continue independent task cleanup
                    exc.add_note('V12 cleanup: background task cancellation')
                    failures.append(exc)
            for name, owned in (('snapshot ingestion tasks', tasks), ('public HTTP workers', http_workers)):
                try:
                    results = await asyncio.gather(*owned, return_exceptions=True)
                    for result in results:
                        if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
                            # A worker error already reported by its HTTP caller is not secondary cleanup failure.
                            if id(result) in observed_http_failures:
                                continue
                            result.add_note(f'V12 cleanup: {name}')
                            failures.append(result)
                except BaseException as exc:  # noqa: BLE001 -- preserve errors and continue cleanup
                    exc.add_note(f'V12 cleanup: draining {name}')
                    failures.append(exc)
            try:
                control("SESSION_CLOSE", clean=clean and primary is None and not failures,
                        engineering_only=True)
            except BaseException as exc:  # noqa: BLE001 -- SESSION_CLOSE must not bypass file cleanup
                exc.add_note('V12 shutdown: SESSION_CLOSE persistence')
                failures.append(exc)
            return failures

        cleanup_task = asyncio.create_task(shutdown())
        interruptions = []
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError as exc:
                # A second cancellation cannot skip the remaining owned cleanup steps.
                interruptions.append(exc)
        failures = ([primary] if primary is not None else []) + interruptions + cleanup_task.result()
        _raise_failures('V12 capture/shutdown failures', failures)
