"""Synchronized public BTCUSDC Spot depth and aggTrade collection."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from heapq import merge
from pathlib import Path
from typing import Any

from websockets.asyncio.client import connect

from src.exchange.public_market_client import PublicMarketDataClient
from src.orderbook.book import ApplyStatus, ReconstructedOrderBook
from src.orderbook.models import DepthDataError, DepthDiffEvent, DepthSnapshot
from src.orderbook.recorder import DepthIntegrityCounters, DepthSynchronizer, V9DepthRecorder
from src.orderflow.models import AggregateTrade


SYMBOL = "BTCUSDC"
COLLECTOR_VERSION = "V10_PUBLIC_L2_AGGTRADES_1"
DEPTH_STREAM = "BTCUSDC_DIFF_DEPTH_100MS"
DEPTH_SNAPSHOT_STREAM = "BTCUSDC_REST_DEPTH_SNAPSHOT"
DEPTH_CONTROL_STREAM = "V10_DEPTH_SYNCHRONIZER_CONTROL"
AGGTRADE_STREAM = "BTCUSDC_AGGTRADE"
DEPTH_URL = "wss://data-stream.binance.vision/ws/btcusdc@depth@100ms"
AGGTRADE_URL = "wss://data-stream.binance.vision/ws/btcusdc@aggTrade"
DEPTH_SCHEMA = "V10_MICROSTRUCTURE_DEPTH_RAW_1"
AGGTRADE_SCHEMA = "V10_MICROSTRUCTURE_AGGTRADE_RAW_1"
MANIFEST_SCHEMA = "V10_MICROSTRUCTURE_SESSION_MANIFEST_1"
SUMMARY_SCHEMA = "V10_MICROSTRUCTURE_CLOSURE_SUMMARY_1"


class MicrostructureIntegrityError(ValueError):
    """Raised when raw collection or causal replay cannot be trusted."""


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise MicrostructureIntegrityError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _utc_text(value: datetime) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def _from_text(value: object) -> datetime:
    if not isinstance(value, str):
        raise MicrostructureIntegrityError("stored timestamp is invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MicrostructureIntegrityError("stored timestamp is invalid") from exc
    return _utc(parsed)


def _from_milliseconds(value: object, *, field: str) -> datetime:
    if type(value) is not int or value < 0:
        raise MicrostructureIntegrityError(f"aggTrade {field} is invalid")
    return datetime(1970, 1, 1, tzinfo=UTC) + timedelta(milliseconds=value)


def _decimal(value: object, *, field: str) -> Decimal:
    if not isinstance(value, (str, Decimal)):
        raise MicrostructureIntegrityError(f"aggTrade {field} must be exact decimal text")
    try:
        result = value if isinstance(value, Decimal) else Decimal(value)
    except InvalidOperation as exc:
        raise MicrostructureIntegrityError(f"aggTrade {field} is invalid") from exc
    if not result.is_finite() or result <= 0:
        raise MicrostructureIntegrityError(f"aggTrade {field} must be finite and positive")
    return result


def _integer(payload: Mapping[str, Any], field: str) -> int:
    value = payload.get(field)
    if type(value) is not int or value < 0:
        raise MicrostructureIntegrityError(f"aggTrade {field} is invalid")
    return value


def _canonical(value: object) -> str:
    def encode(item: object) -> str:
        if isinstance(item, Decimal):
            return str(item)
        if isinstance(item, datetime):
            return _utc_text(item)
        raise TypeError(type(item).__name__)

    return json.dumps(
        value,
        default=encode,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_commit(workspace: str | Path = ".") -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(workspace),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip().lower()
    return commit if len(commit) == 40 and all(character in "0123456789abcdef" for character in commit) else None


class _RawStore:
    def __init__(
        self,
        path: str | Path,
        *,
        session_id: str,
        schema: str,
        fsync_each_record: bool = True,
    ) -> None:
        self.path = Path(path)
        self.session_id = session_id
        self.schema = schema
        self.fsync_each_record = fsync_each_record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8", newline="\n")
        self.record_count = 0
        self._last_received: datetime | None = None

    def _append(
        self,
        *,
        record_type: str,
        stream_identity: str,
        received_at: datetime,
        payload: dict[str, Any],
    ) -> None:
        received = _utc(received_at)
        if self._last_received is not None and received < self._last_received:
            raise MicrostructureIntegrityError("raw receipt timestamps regress")
        if not isinstance(payload, dict):
            raise MicrostructureIntegrityError("raw stream payload must be an object")
        record = {
            "schema_version": self.schema,
            "session_id": self.session_id,
            "record_index": self.record_count,
            "record_type": record_type,
            "stream_identity": stream_identity,
            "exchange_event_timestamp_ms": payload.get("E"),
            "received_at_utc": _utc_text(received),
            "payload": payload,
        }
        try:
            self._stream.write(_canonical(record) + "\n")
            self._stream.flush()
            if self.fsync_each_record:
                os.fsync(self._stream.fileno())
        except OSError as exc:
            raise MicrostructureIntegrityError("raw event could not be persisted") from exc
        self.record_count += 1
        self._last_received = received

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> _RawStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def records(
        path: str | Path,
        *,
        schema: str,
        session_id: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        previous_received: datetime | None = None
        with Path(path).open("r", encoding="utf-8") as stream:
            for expected_index, line in enumerate(stream):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise MicrostructureIntegrityError(
                        f"invalid raw JSON at line {expected_index + 1}"
                    ) from exc
                if (
                    not isinstance(record, dict)
                    or record.get("schema_version") != schema
                    or record.get("record_index") != expected_index
                    or (session_id is not None and record.get("session_id") != session_id)
                    or not isinstance(record.get("payload"), dict)
                ):
                    raise MicrostructureIntegrityError("raw record identity or schema mismatch")
                expected_streams = {
                    DEPTH_SCHEMA: {
                        "DIFF_DEPTH": DEPTH_STREAM,
                        "REST_SNAPSHOT": DEPTH_SNAPSHOT_STREAM,
                        "RESYNC_BOUNDARY": DEPTH_CONTROL_STREAM,
                    },
                    AGGTRADE_SCHEMA: {"AGG_TRADE": AGGTRADE_STREAM},
                }
                expected_stream = expected_streams[schema].get(record.get("record_type"))
                if expected_stream is None or record.get("stream_identity") != expected_stream:
                    raise MicrostructureIntegrityError("raw stream identity mismatch")
                if record.get("exchange_event_timestamp_ms") != record["payload"].get("E"):
                    raise MicrostructureIntegrityError("raw exchange timestamp does not match payload")
                received = _from_text(record.get("received_at_utc"))
                if previous_received is not None and received < previous_received:
                    raise MicrostructureIntegrityError("raw receipt timestamps regress")
                previous_received = received
                yield record


class DepthRawStore(_RawStore):
    def __init__(self, path: str | Path, *, session_id: str, fsync_each_record: bool = True) -> None:
        super().__init__(
            path,
            session_id=session_id,
            schema=DEPTH_SCHEMA,
            fsync_each_record=fsync_each_record,
        )

    def append(
        self, *, record_type: str, received_at: datetime, payload: dict[str, Any]
    ) -> None:
        identities = {
            "DIFF_DEPTH": DEPTH_STREAM,
            "REST_SNAPSHOT": DEPTH_SNAPSHOT_STREAM,
            "RESYNC_BOUNDARY": DEPTH_CONTROL_STREAM,
        }
        if record_type not in identities:
            raise MicrostructureIntegrityError("unknown depth record type")
        self._append(
            record_type=record_type,
            stream_identity=identities[record_type],
            received_at=received_at,
            payload=payload,
        )


class V10DepthSynchronizer(DepthSynchronizer):
    """V9 semantics plus a persisted boundary for causally exact reconnect replay."""

    def __init__(
        self,
        store: DepthRawStore,
        *,
        max_levels_per_side: int = 5_000,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(store, max_levels_per_side=max_levels_per_side)
        self._clock = clock or (lambda: datetime.now(UTC))

    def request_resync(self) -> None:
        self.store.append(
            record_type="RESYNC_BOUNDARY",
            received_at=self._clock(),
            payload={"reason": "BOOK_STATE_INVALIDATED"},
        )
        super().request_resync()


class AggTradeRawStore(_RawStore):
    def __init__(self, path: str | Path, *, session_id: str, fsync_each_record: bool = True) -> None:
        super().__init__(
            path,
            session_id=session_id,
            schema=AGGTRADE_SCHEMA,
            fsync_each_record=fsync_each_record,
        )

    def append(self, *, received_at: datetime, payload: dict[str, Any]) -> None:
        self._append(
            record_type="AGG_TRADE",
            stream_identity=AGGTRADE_STREAM,
            received_at=received_at,
            payload=payload,
        )


@dataclass(frozen=True, slots=True)
class ObservedAggTrade:
    trade: AggregateTrade
    exchange_event_at: datetime
    received_at: datetime


def parse_aggtrade_payload(
    payload: Mapping[str, Any], *, received_at: datetime
) -> ObservedAggTrade:
    """Parse one raw public WebSocket aggTrade without losing maker semantics."""
    if payload.get("e") != "aggTrade" or payload.get("s") != SYMBOL:
        raise MicrostructureIntegrityError("only BTCUSDC Spot aggTrade is supported")
    event_time = _from_milliseconds(payload.get("E"), field="E")
    trade_time = _from_milliseconds(payload.get("T"), field="T")
    if event_time < trade_time:
        raise MicrostructureIntegrityError("aggTrade event time precedes trade time")
    buyer_is_maker = payload.get("m")
    best_price_match = payload.get("M")
    if type(buyer_is_maker) is not bool:
        raise MicrostructureIntegrityError("aggTrade buyer-is-maker flag is invalid")
    if best_price_match is not None and type(best_price_match) is not bool:
        raise MicrostructureIntegrityError("aggTrade best-match flag is invalid")
    trade = AggregateTrade(
        aggregate_trade_id=_integer(payload, "a"),
        price=_decimal(payload.get("p"), field="p"),
        quantity=_decimal(payload.get("q"), field="q"),
        first_trade_id=_integer(payload, "f"),
        last_trade_id=_integer(payload, "l"),
        timestamp=trade_time,
        buyer_is_maker=buyer_is_maker,
        best_price_match=best_price_match,
    )
    return ObservedAggTrade(trade, event_time, _utc(received_at))


@dataclass(slots=True)
class AggTradeIntegrityCounters:
    raw_events: int = 0
    accepted_events: int = 0
    duplicate_events: int = 0
    conflicting_duplicates: int = 0
    id_regressions: int = 0
    timestamp_regressions: int = 0
    aggregate_id_gap_events: int = 0
    missing_aggregate_trade_ids: int = 0
    underlying_trade_id_gap_events: int = 0
    missing_underlying_trade_ids: int = 0
    invalid_events: int = 0
    reconnect_count: int = 0
    first_aggregate_trade_id: int | None = None
    last_aggregate_trade_id: int | None = None
    first_event_timestamp: str | None = None
    last_event_timestamp: str | None = None

    def as_dict(self) -> dict[str, int | str | None]:
        return asdict(self)


class AggTradeIntegrityTracker:
    def __init__(self) -> None:
        self.counters = AggTradeIntegrityCounters()
        self._previous: ObservedAggTrade | None = None
        self._previous_payload: str | None = None

    def process(
        self, payload: Mapping[str, Any], *, received_at: datetime
    ) -> ObservedAggTrade | None:
        self.counters.raw_events += 1
        try:
            observed = parse_aggtrade_payload(payload, received_at=received_at)
        except (MicrostructureIntegrityError, ValueError):
            self.counters.invalid_events += 1
            raise
        current_payload = _canonical(payload)
        previous = self._previous
        if previous is not None:
            current = observed.trade
            prior = previous.trade
            if current.aggregate_trade_id == prior.aggregate_trade_id:
                if current_payload == self._previous_payload:
                    self.counters.duplicate_events += 1
                    return None
                self.counters.conflicting_duplicates += 1
                self.counters.invalid_events += 1
                raise MicrostructureIntegrityError("conflicting duplicate aggregate trade ID")
            if current.aggregate_trade_id < prior.aggregate_trade_id:
                self.counters.id_regressions += 1
                self.counters.invalid_events += 1
                raise MicrostructureIntegrityError("aggregate trade ID regressed")
            if (
                observed.exchange_event_at < previous.exchange_event_at
                or current.timestamp < prior.timestamp
                or observed.received_at < previous.received_at
            ):
                self.counters.timestamp_regressions += 1
                self.counters.invalid_events += 1
                raise MicrostructureIntegrityError("aggregate trade timestamp regressed")
            aggregate_gap = current.aggregate_trade_id - prior.aggregate_trade_id - 1
            if aggregate_gap:
                self.counters.aggregate_id_gap_events += 1
                self.counters.missing_aggregate_trade_ids += aggregate_gap
            if current.first_trade_id <= prior.last_trade_id:
                self.counters.id_regressions += 1
                self.counters.invalid_events += 1
                raise MicrostructureIntegrityError("underlying trade IDs overlap or regress")
            underlying_gap = current.first_trade_id - prior.last_trade_id - 1
            if underlying_gap:
                self.counters.underlying_trade_id_gap_events += 1
                self.counters.missing_underlying_trade_ids += underlying_gap
        else:
            self.counters.first_aggregate_trade_id = observed.trade.aggregate_trade_id
            self.counters.first_event_timestamp = _utc_text(observed.exchange_event_at)
        self._previous = observed
        self._previous_payload = current_payload
        self.counters.accepted_events += 1
        self.counters.last_aggregate_trade_id = observed.trade.aggregate_trade_id
        self.counters.last_event_timestamp = _utc_text(observed.exchange_event_at)
        return observed


class AggTradeSynchronizer:
    def __init__(self, store: AggTradeRawStore) -> None:
        self.store = store
        self.tracker = AggTradeIntegrityTracker()

    @property
    def counters(self) -> AggTradeIntegrityCounters:
        return self.tracker.counters

    def record(
        self, payload: dict[str, Any], *, received_at: datetime
    ) -> ObservedAggTrade | None:
        self.store.append(received_at=received_at, payload=payload)
        return self.tracker.process(payload, received_at=received_at)


class AggTradeRecorder:
    """Bounded public aggTrade recorder with reconnect and duplicate accounting."""

    WEBSOCKET_URL = AGGTRADE_URL

    def __init__(
        self,
        synchronizer: AggTradeSynchronizer,
        *,
        connect_factory: Callable[..., Any] = connect,
        reconnect_delay_seconds: float = 1.0,
    ) -> None:
        if reconnect_delay_seconds < 0:
            raise ValueError("reconnect delay must be nonnegative")
        self.synchronizer = synchronizer
        self.connect_factory = connect_factory
        self.reconnect_delay_seconds = reconnect_delay_seconds

    async def _message(self, websocket: Any, timeout: float) -> dict[str, Any] | None:
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=max(0.01, timeout))
        except TimeoutError:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            payload = json.loads(raw)
        except (TypeError, UnicodeError, json.JSONDecodeError) as exc:
            raise MicrostructureIntegrityError("aggTrade WebSocket message is invalid JSON") from exc
        if not isinstance(payload, dict):
            raise MicrostructureIntegrityError("aggTrade WebSocket payload is not an object")
        return payload

    async def run(self, *, duration_seconds: float) -> AggTradeIntegrityCounters:
        if not 0 < duration_seconds <= 86_400:
            raise ValueError("aggTrade recording duration must be within one day")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + duration_seconds
        while loop.time() < deadline:
            try:
                async with self.connect_factory(
                    self.WEBSOCKET_URL,
                    open_timeout=10,
                    close_timeout=5,
                    ping_interval=20,
                    ping_timeout=20,
                    max_queue=128,
                ) as websocket:
                    while loop.time() < deadline:
                        payload = await self._message(websocket, min(1.0, deadline - loop.time()))
                        if payload is not None:
                            self.synchronizer.record(payload, received_at=datetime.now(UTC))
            except asyncio.CancelledError:
                raise
            except Exception:
                if loop.time() >= deadline:
                    break
                self.synchronizer.counters.reconnect_count += 1
                await asyncio.sleep(min(self.reconnect_delay_seconds, max(0.0, deadline - loop.time())))
        return self.synchronizer.counters


class DualStreamCollector:
    def __init__(self, *, depth: V9DepthRecorder, aggtrades: AggTradeRecorder) -> None:
        self.depth = depth
        self.aggtrades = aggtrades

    async def run(
        self, *, duration_seconds: float
    ) -> tuple[DepthIntegrityCounters, AggTradeIntegrityCounters]:
        depth_result, trade_result = await asyncio.gather(
            self.depth.run(duration_seconds=duration_seconds),
            self.aggtrades.run(duration_seconds=duration_seconds),
        )
        return depth_result, trade_result


@dataclass(frozen=True, slots=True)
class CausalDepthState:
    available_at: datetime
    raw_record_index: int
    last_update_id: int
    bids: tuple[tuple[Decimal, Decimal], ...]
    asks: tuple[tuple[Decimal, Decimal], ...]


@dataclass(frozen=True, slots=True)
class CausalTradeContext:
    observed_trade: ObservedAggTrade
    depth: CausalDepthState | None


@dataclass(frozen=True, slots=True)
class OfflineReplaySummary:
    session_id: str
    depth_counters: dict[str, int | str | None]
    aggtrade_counters: dict[str, int | str | None]
    causal_trade_count: int
    trades_without_depth: int
    final_depth_valid: bool
    final_depth_update_id: int | None
    final_depth_sha256: str | None
    synchronized_timeline_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merged_records(
    depth_path: str | Path,
    aggtrade_path: str | Path,
    *,
    session_id: str,
) -> Iterator[tuple[str, dict[str, Any]]]:
    def tagged(
        records: Iterator[dict[str, Any]], kind: str, priority: int
    ) -> Iterator[tuple[tuple[datetime, int, int], str, dict[str, Any]]]:
        for record in records:
            yield (
                (_from_text(record["received_at_utc"]), priority, record["record_index"]),
                kind,
                record,
            )

    depth = tagged(
        _RawStore.records(depth_path, schema=DEPTH_SCHEMA, session_id=session_id),
        "DEPTH",
        0,
    )
    trades = tagged(
        _RawStore.records(aggtrade_path, schema=AGGTRADE_SCHEMA, session_id=session_id),
        "AGGTRADE",
        1,
    )
    for _, kind, record in merge(depth, trades, key=lambda item: item[0]):
        yield kind, record


class _TimelineMachine:
    def __init__(self, *, max_levels_per_side: int) -> None:
        self.book = ReconstructedOrderBook(max_levels_per_side=max_levels_per_side)
        self.depth_counters = DepthIntegrityCounters()
        self.aggtrades = AggTradeIntegrityTracker()
        self.pending: list[DepthDiffEvent] = []
        self.state_available_at: datetime | None = None
        self.state_record_index: int | None = None

    def depth(self, record: Mapping[str, Any]) -> None:
        received = _from_text(record.get("received_at_utc"))
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise MicrostructureIntegrityError("depth payload is invalid")
        record_type = record.get("record_type")
        if record_type == "RESYNC_BOUNDARY":
            self.depth_counters.resync_count += 1
            self.book.last_update_id = None
            self.pending = []
            self.state_available_at = None
            self.state_record_index = None
            return
        if record_type == "DIFF_DEPTH":
            self.depth_counters.diff_events += 1
            try:
                event = DepthDiffEvent.from_payload(payload)
            except DepthDataError:
                self.depth_counters.invalid_events += 1
                self.book.last_update_id = None
                self.pending = []
                self.state_available_at = None
                return
            timestamp = event.event_time.isoformat()
            if self.depth_counters.first_event_timestamp is None:
                self.depth_counters.first_event_timestamp = timestamp
            self.depth_counters.last_event_timestamp = timestamp
            if self.book.last_update_id is None:
                self.pending.append(event)
                return
            self._apply(event, received=received, record_index=record["record_index"])
            return
        if record_type != "REST_SNAPSHOT":
            raise MicrostructureIntegrityError("unknown raw depth record type")
        try:
            snapshot = DepthSnapshot.from_payload(payload, received_at=received)
        except DepthDataError:
            self.depth_counters.invalid_events += 1
            self.book.last_update_id = None
            self.pending = []
            self.state_available_at = None
            return
        self.book.reset(snapshot)
        self.depth_counters.snapshots += 1
        if not self.book.is_valid:
            self.depth_counters.crossed_invalid_book_states += 1
            self.book.last_update_id = None
            self.pending = []
            self.state_available_at = None
            return
        buffered, self.pending = self.pending, []
        self.state_available_at = received
        self.state_record_index = record["record_index"]
        for event in buffered:
            if not self._apply(event, received=received, record_index=record["record_index"]):
                break

    def _apply(self, event: DepthDiffEvent, *, received: datetime, record_index: int) -> bool:
        status = self.book.apply(event)
        if status is ApplyStatus.STALE:
            self.depth_counters.stale_events += 1
            return True
        if status is ApplyStatus.SEQUENCE_GAP:
            self.depth_counters.sequence_gaps += 1
            self.book.last_update_id = None
            self.pending = [event]
            self.state_available_at = None
            return False
        self.depth_counters.reconstructed_updates += 1
        if status is ApplyStatus.INVALID_BOOK:
            self.depth_counters.crossed_invalid_book_states += 1
            self.book.last_update_id = None
            self.pending = [event]
            self.state_available_at = None
            return False
        self.state_available_at = received
        self.state_record_index = record_index
        return True

    def aggtrade(self, record: Mapping[str, Any]) -> ObservedAggTrade | None:
        payload = record.get("payload")
        if not isinstance(payload, dict):
            raise MicrostructureIntegrityError("aggTrade payload is invalid")
        try:
            return self.aggtrades.process(
                payload,
                received_at=_from_text(record.get("received_at_utc")),
            )
        except (MicrostructureIntegrityError, ValueError):
            return None

    def causal_state(self) -> CausalDepthState | None:
        if (
            self.book.last_update_id is None
            or not self.book.is_valid
            or self.state_available_at is None
            or self.state_record_index is None
        ):
            return None
        bids, asks = self.book.top_levels(self.book.max_levels_per_side)
        return CausalDepthState(
            self.state_available_at,
            self.state_record_index,
            self.book.last_update_id,
            bids,
            asks,
        )


def iter_causal_trade_contexts(
    depth_path: str | Path,
    aggtrade_path: str | Path,
    *,
    session_id: str,
    max_levels_per_side: int = 5_000,
) -> Iterator[CausalTradeContext]:
    """Yield each valid unique aggTrade with L2 available at its receipt time."""
    machine = _TimelineMachine(max_levels_per_side=max_levels_per_side)
    for kind, record in _merged_records(
        depth_path, aggtrade_path, session_id=session_id
    ):
        if kind == "DEPTH":
            machine.depth(record)
            continue
        observed = machine.aggtrade(record)
        if observed is not None:
            state = machine.causal_state()
            if state is not None and state.available_at > observed.received_at:
                raise MicrostructureIntegrityError("future depth leaked into aggTrade context")
            yield CausalTradeContext(observed, state)


def replay_synchronized_session(
    depth_path: str | Path,
    aggtrade_path: str | Path,
    *,
    session_id: str,
    max_levels_per_side: int = 5_000,
) -> OfflineReplaySummary:
    """Deterministically validate both raw logs without retaining full contexts."""
    machine = _TimelineMachine(max_levels_per_side=max_levels_per_side)
    timeline = hashlib.sha256()
    causal_trade_count = 0
    trades_without_depth = 0
    for kind, record in _merged_records(
        depth_path, aggtrade_path, session_id=session_id
    ):
        timeline.update(kind.encode())
        timeline.update(b":")
        timeline.update(_canonical(record).encode())
        timeline.update(b"\n")
        if kind == "DEPTH":
            machine.depth(record)
            continue
        observed = machine.aggtrade(record)
        if observed is not None:
            causal_trade_count += 1
            if machine.causal_state() is None:
                trades_without_depth += 1
    state = machine.causal_state()
    final_depth_hash = hashlib.sha256(_canonical(asdict(state)).encode()).hexdigest() if state else None
    return OfflineReplaySummary(
        session_id,
        machine.depth_counters.as_dict(),
        machine.aggtrades.counters.as_dict(),
        causal_trade_count,
        trades_without_depth,
        state is not None,
        state.last_update_id if state else None,
        final_depth_hash,
        timeline.hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class SessionPaths:
    session_id: str
    directory: Path
    depth_raw: Path
    aggtrades_raw: Path
    manifest: Path
    summary: Path


def session_paths(root: str | Path, *, started_at: datetime) -> SessionPaths:
    started = _utc(started_at)
    session_id = started.strftime("%Y%m%dT%H%M%S%fZ")
    directory = (
        Path(root)
        / SYMBOL
        / f"{started:%Y}"
        / f"{started:%m}"
        / f"{started:%d}"
        / session_id
    )
    return SessionPaths(
        session_id,
        directory,
        directory / "depth.jsonl",
        directory / "aggtrades.jsonl",
        directory / "session.manifest.json",
        directory / "closure.summary.json",
    )


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise MicrostructureIntegrityError(f"immutable artifact could not be written: {path}") from exc


class ProspectiveSession:
    def __init__(
        self,
        paths: SessionPaths,
        *,
        started_at: datetime,
        duration_seconds: int,
        source_commit_sha: str | None,
        snapshot_limit: int,
        max_levels_per_side: int,
        fsync_each_record: bool = True,
    ) -> None:
        if paths.directory.exists():
            raise MicrostructureIntegrityError("session directory already exists")
        paths.directory.mkdir(parents=True)
        self.paths = paths
        self.started_at = _utc(started_at)
        self.duration_seconds = duration_seconds
        self.max_levels_per_side = max_levels_per_side
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "collector_version": COLLECTOR_VERSION,
            "session_id": paths.session_id,
            "symbol": SYMBOL,
            "started_at_utc": _utc_text(self.started_at),
            "requested_duration_seconds": duration_seconds,
            "source_commit_sha": source_commit_sha,
            "source": "BINANCE_PUBLIC_SPOT",
            "uses_authentication": False,
            "orders_enabled": False,
            "research_eligibility": "UNASSESSED",
            "streams": {
                "depth": {
                    "identity": DEPTH_STREAM,
                    "websocket_url": DEPTH_URL,
                    "rest_snapshot": "PUBLIC_API_V3_DEPTH",
                    "snapshot_limit": snapshot_limit,
                    "max_levels_per_side": max_levels_per_side,
                    "raw_file": paths.depth_raw.name,
                },
                "aggtrades": {
                    "identity": AGGTRADE_STREAM,
                    "websocket_url": AGGTRADE_URL,
                    "raw_file": paths.aggtrades_raw.name,
                },
            },
        }
        _write_immutable_json(paths.manifest, manifest)
        self.depth_store = DepthRawStore(
            paths.depth_raw,
            session_id=paths.session_id,
            fsync_each_record=fsync_each_record,
        )
        self.aggtrade_store = AggTradeRawStore(
            paths.aggtrades_raw,
            session_id=paths.session_id,
            fsync_each_record=fsync_each_record,
        )
        self._closed = False

    def close_raw(self) -> None:
        if not self._closed:
            self.depth_store.close()
            self.aggtrade_store.close()
            self._closed = True

    def finalize(
        self,
        *,
        depth_counters: DepthIntegrityCounters,
        aggtrade_counters: AggTradeIntegrityCounters,
        ended_at: datetime,
    ) -> dict[str, Any]:
        self.close_raw()
        ended = _utc(ended_at)
        if ended < self.started_at:
            raise MicrostructureIntegrityError("session end precedes start")
        offline = replay_synchronized_session(
            self.paths.depth_raw,
            self.paths.aggtrades_raw,
            session_id=self.paths.session_id,
            max_levels_per_side=self.max_levels_per_side,
        )
        replay_check = replay_synchronized_session(
            self.paths.depth_raw,
            self.paths.aggtrades_raw,
            session_id=self.paths.session_id,
            max_levels_per_side=self.max_levels_per_side,
        )
        deterministic_replay_passed = offline.to_dict() == replay_check.to_dict()
        depth = offline.depth_counters
        agg = offline.aggtrade_counters
        live_depth = depth_counters.as_dict()
        live_agg = aggtrade_counters.as_dict()
        depth_accounting_match = all(
            live_depth[key] == depth[key]
            for key in live_depth
            if key != "reconnect_count"
        )
        aggtrade_accounting_match = all(
            live_agg[key] == agg[key]
            for key in live_agg
            if key != "reconnect_count"
        )
        raw_record_accounting_match = bool(
            self.depth_store.record_count
            == depth["snapshots"] + depth["diff_events"] + depth["resync_count"]
            and self.aggtrade_store.record_count == agg["raw_events"]
        )
        integrity_accounting_match = bool(
            depth_accounting_match
            and aggtrade_accounting_match
            and raw_record_accounting_match
        )
        passed = bool(
            integrity_accounting_match
            and deterministic_replay_passed
            and depth["snapshots"] >= 1
            and depth["diff_events"] >= 1
            and depth["reconstructed_updates"] >= 1
            and depth["sequence_gaps"] == 0
            and depth["invalid_events"] == 0
            and depth["crossed_invalid_book_states"] == 0
            and offline.final_depth_valid
            and agg["accepted_events"] >= 1
            and agg["invalid_events"] == 0
            and agg["aggregate_id_gap_events"] == 0
            and agg["underlying_trade_id_gap_events"] == 0
        )
        payload = {
            "schema_version": SUMMARY_SCHEMA,
            "collector_version": COLLECTOR_VERSION,
            "session_id": self.paths.session_id,
            "symbol": SYMBOL,
            "started_at_utc": _utc_text(self.started_at),
            "ended_at_utc": _utc_text(ended),
            "duration_seconds_actual": (ended - self.started_at).total_seconds(),
            "closure_status": "CLOSED_NORMAL",
            "collector_left_running": False,
            "collection_classification": (
                "CLOSED_INTEGRITY_PASSED" if passed else "CLOSED_INTEGRITY_FAILED"
            ),
            "research_eligibility": "UNASSESSED",
            "raw_artifacts": {
                "depth": {
                    "file": self.paths.depth_raw.name,
                    "schema_version": DEPTH_SCHEMA,
                    "stream_identity": DEPTH_STREAM,
                    "record_count": self.depth_store.record_count,
                    "size_bytes": self.paths.depth_raw.stat().st_size,
                    "sha256": sha256_file(self.paths.depth_raw),
                },
                "aggtrades": {
                    "file": self.paths.aggtrades_raw.name,
                    "schema_version": AGGTRADE_SCHEMA,
                    "stream_identity": AGGTRADE_STREAM,
                    "record_count": self.aggtrade_store.record_count,
                    "size_bytes": self.paths.aggtrades_raw.stat().st_size,
                    "sha256": sha256_file(self.paths.aggtrades_raw),
                },
            },
            "manifest": {
                "file": self.paths.manifest.name,
                "sha256": sha256_file(self.paths.manifest),
            },
            "live_integrity": {
                "depth": live_depth,
                "aggtrades": live_agg,
            },
            "integrity_accounting": {
                "depth_live_matches_replay": depth_accounting_match,
                "aggtrades_live_matches_replay": aggtrade_accounting_match,
                "raw_record_counts_match_replay": raw_record_accounting_match,
                "passed": integrity_accounting_match,
            },
            "offline_replay": offline.to_dict(),
            "deterministic_replay_hash_check_passed": deterministic_replay_passed,
            "uses_authentication": False,
            "orders_enabled": False,
            "predictive_outcomes_evaluated": False,
        }
        _write_immutable_json(self.paths.summary, payload)
        return payload

    def __enter__(self) -> ProspectiveSession:
        return self

    def __exit__(self, *_: object) -> None:
        self.close_raw()


def classify_session(manifest_path: str | Path) -> str:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise MicrostructureIntegrityError("session manifest schema mismatch")
    summary_path = path.parent / "closure.summary.json"
    if not summary_path.is_file():
        return "INTERRUPTED"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA
        or summary.get("session_id") != manifest.get("session_id")
        or summary.get("closure_status") != "CLOSED_NORMAL"
        or summary.get("collector_left_running") is not False
    ):
        raise MicrostructureIntegrityError("closure summary identity mismatch")
    manifest_artifact = summary.get("manifest")
    raw_artifacts = summary.get("raw_artifacts")
    if not isinstance(manifest_artifact, dict) or not isinstance(raw_artifacts, dict):
        raise MicrostructureIntegrityError("closure artifact metadata is invalid")
    if (
        manifest_artifact.get("file") != path.name
        or manifest_artifact.get("sha256") != sha256_file(path)
    ):
        raise MicrostructureIntegrityError("session manifest hash mismatch")
    expected_raw = {
        "depth": ("depth.jsonl", DEPTH_SCHEMA),
        "aggtrades": ("aggtrades.jsonl", AGGTRADE_SCHEMA),
    }
    for name, (file_name, schema) in expected_raw.items():
        artifact = raw_artifacts.get(name)
        raw_path = path.parent / file_name
        if (
            not isinstance(artifact, dict)
            or artifact.get("file") != file_name
            or artifact.get("schema_version") != schema
            or not raw_path.is_file()
            or artifact.get("size_bytes") != raw_path.stat().st_size
            or artifact.get("sha256") != sha256_file(raw_path)
        ):
            raise MicrostructureIntegrityError(f"{name} raw artifact hash mismatch")
    classification = summary.get("collection_classification")
    if classification not in {"CLOSED_INTEGRITY_PASSED", "CLOSED_INTEGRITY_FAILED"}:
        raise MicrostructureIntegrityError("closure classification is invalid")
    return classification


async def collect_live_session(
    *,
    duration_seconds: int,
    output_root: str | Path,
    snapshot_limit: int,
    max_levels_per_side: int,
    workspace: str | Path = ".",
) -> tuple[dict[str, Any], SessionPaths]:
    started = datetime.now(UTC)
    paths = session_paths(output_root, started_at=started)
    with ProspectiveSession(
        paths,
        started_at=started,
        duration_seconds=duration_seconds,
        source_commit_sha=source_commit(workspace),
        snapshot_limit=snapshot_limit,
        max_levels_per_side=max_levels_per_side,
    ) as session:
        depth_sync = V10DepthSynchronizer(
            session.depth_store,
            max_levels_per_side=max_levels_per_side,
        )
        aggtrade_sync = AggTradeSynchronizer(session.aggtrade_store)
        collector = DualStreamCollector(
            depth=V9DepthRecorder(
                synchronizer=depth_sync,
                snapshot_client=PublicMarketDataClient(),
                snapshot_limit=snapshot_limit,
            ),
            aggtrades=AggTradeRecorder(aggtrade_sync),
        )
        depth_counters, aggtrade_counters = await collector.run(
            duration_seconds=duration_seconds
        )
        payload = session.finalize(
            depth_counters=depth_counters,
            aggtrade_counters=aggtrade_counters,
            ended_at=datetime.now(UTC),
        )
    return payload, paths
