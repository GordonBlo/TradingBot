from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.exchange.public_market_client import PublicMarketDataClient
from src.orderbook.book import ApplyStatus, ReconstructedOrderBook
from src.orderbook.models import DepthDiffEvent, DepthSnapshot
from src.orderbook.recorder import (
    DepthSynchronizer,
    V9DepthRecorder,
    reconstruct_raw_log,
)
from src.orderbook.storage import RawDepthEventStore


UTC = timezone.utc
NOW = datetime(2026, 8, 25, 12, tzinfo=UTC)


def snapshot(last_id: int = 100) -> dict:
    return {
        "lastUpdateId": last_id,
        "bids": [["100.00", "1.0"], ["99.00", "2.0"]],
        "asks": [["101.00", "1.5"], ["102.00", "2.5"]],
    }


def diff(
    first: int, final: int, *, bids: list | None = None,
    asks: list | None = None, event_ms: int = 1_788_000_000_000,
) -> dict:
    return {
        "e": "depthUpdate", "E": event_ms, "s": "BTCUSDC",
        "U": first, "u": final, "b": bids or [], "a": asks or [],
    }


def synchronizer(tmp_path):
    store = RawDepthEventStore(
        tmp_path / "raw.jsonl", session_id="test", fsync_each_record=False
    )
    return DepthSynchronizer(store), store


def test_snapshot_bridge_accepts_u_range_containing_snapshot_plus_one(tmp_path) -> None:
    sync, store = synchronizer(tmp_path)
    try:
        event = sync.record_diff(
            diff(99, 101, bids=[["100", "3"]]), received_at=NOW
        )
        sync.install_snapshot(snapshot(100), received_at=NOW)
        assert sync.apply_recorded(event) is ApplyStatus.APPLIED
        assert sync.book.last_update_id == 101
        assert sync.book.best_bid == (Decimal("100"), Decimal("3"))
    finally:
        store.close()


def test_stale_event_is_rejected_without_mutation(tmp_path) -> None:
    sync, store = synchronizer(tmp_path)
    try:
        sync.install_snapshot(snapshot(100), received_at=NOW)
        event = sync.record_diff(
            diff(90, 100, bids=[["100", "9"]]), received_at=NOW
        )
        assert sync.apply_recorded(event) is ApplyStatus.STALE
        assert sync.book.best_bid == (Decimal("100.00"), Decimal("1.0"))
        assert sync.counters.stale_events == 1
    finally:
        store.close()


def test_sequence_gap_forces_resync_and_new_snapshot_recovers(tmp_path) -> None:
    sync, store = synchronizer(tmp_path)
    try:
        sync.install_snapshot(snapshot(100), received_at=NOW)
        gap = sync.record_diff(diff(102, 102), received_at=NOW)
        assert sync.apply_recorded(gap) is ApplyStatus.SEQUENCE_GAP
        assert sync.counters.sequence_gaps == 1
        assert sync.counters.resync_count == 1
        assert sync.book.last_update_id is None

        sync.install_snapshot(snapshot(200), received_at=NOW)
        recovered = sync.record_diff(
            diff(201, 201, asks=[["101", "2"]]), received_at=NOW
        )
        assert sync.apply_recorded(recovered) is ApplyStatus.APPLIED
        assert sync.book.last_update_id == 201
        assert sync.counters.snapshots == 2
    finally:
        store.close()


def test_price_levels_are_updated_inserted_and_deleted() -> None:
    book = ReconstructedOrderBook(max_levels_per_side=10)
    book.reset(DepthSnapshot.from_payload(snapshot(), received_at=NOW))
    event = DepthDiffEvent.from_payload(diff(
        101, 101,
        bids=[["100", "0"], ["100.5", "4.25"]],
        asks=[["101", "3.75"], ["103", "5"]],
    ))
    assert book.apply(event) is ApplyStatus.APPLIED
    bids, asks = book.top_levels(3)
    assert bids[0] == (Decimal("100.5"), Decimal("4.25"))
    assert Decimal("100") not in book.bids
    assert asks[0] == (Decimal("101"), Decimal("3.75"))
    assert asks[-1] == (Decimal("103"), Decimal("5"))


def test_crossed_book_is_counted_and_forces_resync(tmp_path) -> None:
    sync, store = synchronizer(tmp_path)
    try:
        sync.install_snapshot(snapshot(), received_at=NOW)
        crossed = sync.record_diff(
            diff(101, 101, bids=[["102", "1"]]), received_at=NOW
        )
        assert sync.apply_recorded(crossed) is ApplyStatus.INVALID_BOOK
        assert sync.counters.crossed_invalid_book_states == 1
        assert sync.counters.resync_count == 1
    finally:
        store.close()


def test_raw_log_reconstruction_is_deterministic_with_pre_snapshot_buffer(tmp_path) -> None:
    sync, store = synchronizer(tmp_path)
    event = sync.record_diff(
        diff(100, 101, bids=[["100", "2"]]), received_at=NOW
    )
    sync.install_snapshot(snapshot(100), received_at=NOW)
    sync.apply_recorded(event)
    next_event = sync.record_diff(
        diff(102, 102, asks=[["101", "0"], ["101.5", "1"]]),
        received_at=NOW,
    )
    sync.apply_recorded(next_event)
    expected = (sync.book.last_update_id, sync.book.top_levels(5))
    store.close()

    first = reconstruct_raw_log(tmp_path / "raw.jsonl")
    second = reconstruct_raw_log(tmp_path / "raw.jsonl")
    assert (first.last_update_id, first.top_levels(5)) == expected
    assert (second.last_update_id, second.top_levels(5)) == expected
    records = tuple(RawDepthEventStore.records(tmp_path / "raw.jsonl"))
    assert [item["record_index"] for item in records] == [0, 1, 2]
    assert [item["record_type"] for item in records] == [
        "DIFF_DEPTH", "REST_SNAPSHOT", "DIFF_DEPTH",
    ]


def test_memory_bound_keeps_nearest_levels() -> None:
    payload = {
        "lastUpdateId": 1,
        "bids": [[str(price), "1"] for price in range(90, 101)],
        "asks": [[str(price), "1"] for price in range(101, 112)],
    }
    book = ReconstructedOrderBook(max_levels_per_side=3)
    book.reset(DepthSnapshot.from_payload(payload, received_at=NOW))
    assert tuple(price for price, _ in book.top_levels(3)[0]) == (
        Decimal("100"), Decimal("99"), Decimal("98"),
    )
    assert tuple(price for price, _ in book.top_levels(3)[1]) == (
        Decimal("101"), Decimal("102"), Decimal("103"),
    )


def test_public_snapshot_network_response_is_mocked_and_decimal_text_preserved() -> None:
    class Response:
        def data(self):
            return {"lastUpdateId": 5, "bids": [["1.10", "2.20"]], "asks": [["1.20", "3.30"]]}

    class Rest:
        def depth(self, symbol: str, *, limit: int):
            assert symbol == "BTCUSDC" and limit == 1000
            return Response()

    client = PublicMarketDataClient(rest_api=Rest())
    assert client.get_order_book_snapshot()["bids"] == [["1.10", "2.20"]]
    assert client.uses_authentication is False


def test_mocked_stream_buffers_before_snapshot_and_reconstructs(tmp_path) -> None:
    sync, store = synchronizer(tmp_path)

    class SnapshotClient:
        def get_order_book_snapshot(self, symbol: str, *, limit: int):
            time.sleep(0.02)
            return snapshot(100)

    messages = [
        json.dumps(diff(90, 100)),
        json.dumps(diff(99, 101, bids=[["100", "2"]])),
        json.dumps(diff(102, 102, asks=[["101", "2"]])),
    ]

    class WebSocket:
        async def recv(self):
            if messages:
                return messages.pop(0)
            await asyncio.sleep(1)

    class Connection:
        async def __aenter__(self):
            return WebSocket()

        async def __aexit__(self, *_):
            return None

    recorder = V9DepthRecorder(
        synchronizer=sync,
        snapshot_client=SnapshotClient(),
        connect_factory=lambda *args, **kwargs: Connection(),
    )
    try:
        counters = asyncio.run(recorder.run(duration_seconds=0.08))
        assert counters.snapshots == 1
        assert counters.diff_events == 3
        assert counters.stale_events == 1
        assert counters.reconstructed_updates == 2
        assert counters.sequence_gaps == counters.resync_count == 0
        assert sync.book.last_update_id == 102
    finally:
        store.close()


@pytest.mark.parametrize("symbol", ("BTCUSDT", "ETHUSDC"))
def test_v9_rejects_non_target_symbols(symbol: str) -> None:
    with pytest.raises(ValueError, match="BTCUSDC"):
        PublicMarketDataClient(rest_api=object()).get_order_book_snapshot(symbol)
