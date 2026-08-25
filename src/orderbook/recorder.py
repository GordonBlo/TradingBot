"""Bounded public Binance Spot diff-depth recorder and synchronizer."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from websockets.asyncio.client import connect

from src.exchange.public_market_client import PublicMarketDataClient
from src.orderbook.book import ApplyStatus, ReconstructedOrderBook
from src.orderbook.models import DepthDataError, DepthDiffEvent, DepthSnapshot
from src.orderbook.storage import RawDepthEventStore


UTC = timezone.utc


class ResyncRequired(RuntimeError):
    pass


@dataclass(slots=True)
class DepthIntegrityCounters:
    snapshots: int = 0
    diff_events: int = 0
    stale_events: int = 0
    sequence_gaps: int = 0
    resync_count: int = 0
    reconnect_count: int = 0
    reconstructed_updates: int = 0
    crossed_invalid_book_states: int = 0
    invalid_events: int = 0
    first_event_timestamp: str | None = None
    last_event_timestamp: str | None = None

    def as_dict(self) -> dict[str, int | str | None]:
        return asdict(self)


class DepthSynchronizer:
    """Persist raw inputs, then apply Binance U/u snapshot-bridging semantics."""

    def __init__(self, store: RawDepthEventStore, *, max_levels_per_side: int = 5_000) -> None:
        self.store = store
        self.book = ReconstructedOrderBook(max_levels_per_side=max_levels_per_side)
        self.counters = DepthIntegrityCounters()

    def record_diff(self, payload: dict[str, Any], *, received_at: datetime) -> DepthDiffEvent:
        self.store.append(record_type="DIFF_DEPTH", received_at=received_at, payload=payload)
        self.counters.diff_events += 1
        try:
            event = DepthDiffEvent.from_payload(payload)
        except DepthDataError:
            self.counters.invalid_events += 1
            self.request_resync()
            raise
        timestamp = event.event_time.isoformat()
        if self.counters.first_event_timestamp is None:
            self.counters.first_event_timestamp = timestamp
        self.counters.last_event_timestamp = timestamp
        return event

    def install_snapshot(self, payload: dict[str, Any], *, received_at: datetime) -> None:
        self.store.append(record_type="REST_SNAPSHOT", received_at=received_at, payload=payload)
        snapshot = DepthSnapshot.from_payload(payload, received_at=received_at)
        self.book.reset(snapshot)
        self.counters.snapshots += 1
        if not self.book.is_valid:
            self.counters.crossed_invalid_book_states += 1
            self.request_resync()
            raise ResyncRequired("Snapshot produced an invalid or crossed book.")

    def apply_recorded(self, event: DepthDiffEvent) -> ApplyStatus:
        status = self.book.apply(event)
        if status is ApplyStatus.STALE:
            self.counters.stale_events += 1
        elif status is ApplyStatus.SEQUENCE_GAP:
            self.counters.sequence_gaps += 1
            self.request_resync()
        elif status is ApplyStatus.INVALID_BOOK:
            self.counters.reconstructed_updates += 1
            self.counters.crossed_invalid_book_states += 1
            self.request_resync()
        else:
            self.counters.reconstructed_updates += 1
        return status

    def request_resync(self) -> None:
        self.counters.resync_count += 1
        self.book.last_update_id = None


class V9DepthRecorder:
    """Always-time-bounded BTCUSDC public depth recording session."""

    WEBSOCKET_URL = "wss://data-stream.binance.vision/ws/btcusdc@depth@100ms"

    def __init__(
        self, *, synchronizer: DepthSynchronizer,
        snapshot_client: PublicMarketDataClient | None = None,
        snapshot_limit: int = 1_000,
        connect_factory: Callable[..., Any] = connect,
        bridge_buffer_limit: int = 4_096,
    ) -> None:
        if bridge_buffer_limit < 1:
            raise ValueError("Depth bridge buffer limit must be positive.")
        self.synchronizer = synchronizer
        self.snapshot_client = snapshot_client or PublicMarketDataClient()
        self.snapshot_limit = snapshot_limit
        self.connect_factory = connect_factory
        self.bridge_buffer_limit = bridge_buffer_limit

    async def _message(self, websocket: Any, timeout: float) -> dict[str, Any] | None:
        try:
            raw = await asyncio.wait_for(websocket.recv(), timeout=max(0.01, timeout))
        except TimeoutError:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = json.loads(raw)
        if isinstance(payload.get("data"), dict):
            payload = payload["data"]
        if not isinstance(payload, dict):
            raise DepthDataError("Depth WebSocket payload is not an object.")
        return payload

    async def _connected_cycle(self, websocket: Any, *, deadline: float) -> None:
        snapshot_task = asyncio.create_task(asyncio.to_thread(
            self.snapshot_client.get_order_book_snapshot,
            "BTCUSDC", limit=self.snapshot_limit,
        ))
        buffered: list[DepthDiffEvent] = []
        while not snapshot_task.done() and asyncio.get_running_loop().time() < deadline:
            payload = await self._message(
                websocket, min(0.05, deadline - asyncio.get_running_loop().time())
            )
            if payload is None:
                continue
            event = self.synchronizer.record_diff(payload, received_at=datetime.now(UTC))
            buffered.append(event)
            if len(buffered) > self.bridge_buffer_limit:
                self.synchronizer.request_resync()
                snapshot_task.cancel()
                raise ResyncRequired("Snapshot bridge buffer exhausted.")
        if asyncio.get_running_loop().time() >= deadline:
            snapshot_task.cancel()
            return
        snapshot_payload = await snapshot_task
        self.synchronizer.install_snapshot(snapshot_payload, received_at=datetime.now(UTC))
        for event in buffered:
            status = self.synchronizer.apply_recorded(event)
            if status in (ApplyStatus.SEQUENCE_GAP, ApplyStatus.INVALID_BOOK):
                raise ResyncRequired(status.value)

        while asyncio.get_running_loop().time() < deadline:
            payload = await self._message(websocket, min(1.0, deadline - asyncio.get_running_loop().time()))
            if payload is None:
                continue
            event = self.synchronizer.record_diff(payload, received_at=datetime.now(UTC))
            status = self.synchronizer.apply_recorded(event)
            if status in (ApplyStatus.SEQUENCE_GAP, ApplyStatus.INVALID_BOOK):
                raise ResyncRequired(status.value)

    async def run(self, *, duration_seconds: float) -> DepthIntegrityCounters:
        if not 0 < duration_seconds <= 86_400:
            raise ValueError("V9 recording duration must be within one day.")
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
                    max_queue=32,
                ) as websocket:
                    await self._connected_cycle(websocket, deadline=deadline)
            except ResyncRequired:
                continue
            except asyncio.CancelledError:
                raise
            except Exception:
                if loop.time() >= deadline:
                    break
                self.synchronizer.counters.reconnect_count += 1
                if self.synchronizer.book.last_update_id is not None:
                    self.synchronizer.request_resync()
                await asyncio.sleep(min(1.0, max(0.0, deadline - loop.time())))
        return self.synchronizer.counters


def session_paths(root: str | Path, *, started_at: datetime) -> tuple[str, Path, Path]:
    started = started_at.astimezone(UTC)
    session_id = started.strftime("%Y%m%dT%H%M%S%fZ")
    base = Path(root) / "BTCUSDC" / f"{started:%Y}" / f"{started:%m}" / f"{started:%d}"
    return session_id, base / f"{session_id}.jsonl", base / f"{session_id}.summary.json"


def reconstruct_raw_log(
    path: str | Path, *, max_levels_per_side: int = 5_000
) -> ReconstructedOrderBook:
    """Deterministically rebuild the latest synchronized state from a raw session."""

    book = ReconstructedOrderBook(max_levels_per_side=max_levels_per_side)
    pending: list[DepthDiffEvent] = []
    for record in RawDepthEventStore.records(path):
        received_at = datetime.fromisoformat(record["received_at_utc"])
        if record["record_type"] == "DIFF_DEPTH":
            event = DepthDiffEvent.from_payload(record["payload"])
            if book.last_update_id is None:
                pending.append(event)
                continue
            status = book.apply(event)
            if status in (ApplyStatus.SEQUENCE_GAP, ApplyStatus.INVALID_BOOK):
                book.last_update_id = None
                pending = [event]
        elif record["record_type"] == "REST_SNAPSHOT":
            book.reset(DepthSnapshot.from_payload(record["payload"], received_at=received_at))
            buffered = pending
            pending = []
            for event in buffered:
                status = book.apply(event)
                if status in (ApplyStatus.SEQUENCE_GAP, ApplyStatus.INVALID_BOOK):
                    book.last_update_id = None
                    pending = [event]
                    break
        else:
            raise DepthDataError("Unknown raw depth record type.")
    return book
