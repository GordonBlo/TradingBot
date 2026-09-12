from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from src.cli.record_v10_microstructure import _duration, build_parser
from src.microstructure.v10 import (
    AGGTRADE_STREAM,
    DEPTH_STREAM,
    AggTradeIntegrityCounters,
    AggTradeIntegrityTracker,
    AggTradeRawStore,
    AggTradeRecorder,
    AggTradeSynchronizer,
    DualStreamCollector,
    DepthRawStore,
    MicrostructureIntegrityError,
    ProspectiveSession,
    V10DepthSynchronizer,
    classify_session,
    iter_causal_trade_contexts,
    parse_aggtrade_payload,
    replay_synchronized_session,
    session_paths,
)
from src.orderbook.book import ApplyStatus
from src.orderbook.recorder import DepthIntegrityCounters


NOW = datetime(2026, 9, 12, 12, tzinfo=UTC)


def snapshot(last_id: int = 100) -> dict:
    return {
        "lastUpdateId": last_id,
        "bids": [["100.00", "2.0"], ["99.00", "3.0"]],
        "asks": [["101.00", "2.5"], ["102.00", "4.0"]],
    }


def diff(
    first: int,
    final: int,
    *,
    event_at: datetime = NOW,
    bids: list[list[str]] | None = None,
    asks: list[list[str]] | None = None,
) -> dict:
    return {
        "e": "depthUpdate",
        "E": int(event_at.timestamp() * 1000),
        "s": "BTCUSDC",
        "U": first,
        "u": final,
        "b": bids or [],
        "a": asks or [],
    }


def aggtrade(
    aggregate_id: int,
    *,
    event_at: datetime = NOW,
    first_trade_id: int | None = None,
    last_trade_id: int | None = None,
    price: str = "100.50",
    quantity: str = "0.125",
    buyer_is_maker: bool = False,
) -> dict:
    first = aggregate_id if first_trade_id is None else first_trade_id
    last = first if last_trade_id is None else last_trade_id
    milliseconds = int(event_at.timestamp() * 1000)
    return {
        "e": "aggTrade",
        "E": milliseconds,
        "s": "BTCUSDC",
        "a": aggregate_id,
        "p": price,
        "q": quantity,
        "f": first,
        "l": last,
        "T": milliseconds,
        "m": buyer_is_maker,
        "M": True,
    }


def raw_pair(tmp_path, session_id: str = "session"):
    return (
        DepthRawStore(tmp_path / "depth.jsonl", session_id=session_id, fsync_each_record=False),
        AggTradeRawStore(
            tmp_path / "aggtrades.jsonl",
            session_id=session_id,
            fsync_each_record=False,
        ),
    )


def test_aggtrade_parser_preserves_exact_public_semantics() -> None:
    payload = aggtrade(10, first_trade_id=20, last_trade_id=22, buyer_is_maker=True)
    observed = parse_aggtrade_payload(payload, received_at=NOW + timedelta(milliseconds=3))
    assert observed.trade.aggregate_trade_id == 10
    assert observed.trade.price == Decimal("100.50")
    assert observed.trade.quantity == Decimal("0.125")
    assert observed.trade.buyer_is_maker is True
    assert observed.trade.best_price_match is True
    assert observed.exchange_event_at == NOW
    assert observed.received_at == NOW + timedelta(milliseconds=3)


@pytest.mark.parametrize(
    ("field", "value"),
    (("p", "0"), ("q", "NaN"), ("m", "false"), ("s", "BTCUSDT"), ("a", -1)),
)
def test_invalid_aggtrades_are_rejected(field: str, value: object) -> None:
    payload = aggtrade(10)
    payload[field] = value
    with pytest.raises((MicrostructureIntegrityError, ValueError)):
        parse_aggtrade_payload(payload, received_at=NOW)


def test_duplicate_gap_and_regression_accounting() -> None:
    tracker = AggTradeIntegrityTracker()
    first = aggtrade(10, first_trade_id=100, last_trade_id=101)
    assert tracker.process(first, received_at=NOW) is not None
    assert tracker.process(first, received_at=NOW + timedelta(milliseconds=1)) is None
    assert tracker.process(
        aggtrade(12, first_trade_id=104, last_trade_id=104),
        received_at=NOW + timedelta(milliseconds=2),
    ) is not None
    assert tracker.counters.duplicate_events == 1
    assert tracker.counters.missing_aggregate_trade_ids == 1
    assert tracker.counters.missing_underlying_trade_ids == 2
    with pytest.raises(MicrostructureIntegrityError, match="regressed"):
        tracker.process(aggtrade(11), received_at=NOW + timedelta(milliseconds=3))
    assert tracker.counters.id_regressions == 1
    assert tracker.counters.invalid_events == 1


def test_conflicting_duplicate_is_an_integrity_failure() -> None:
    tracker = AggTradeIntegrityTracker()
    tracker.process(aggtrade(10), received_at=NOW)
    with pytest.raises(MicrostructureIntegrityError, match="conflicting duplicate"):
        tracker.process(aggtrade(10, price="101.00"), received_at=NOW)
    assert tracker.counters.conflicting_duplicates == 1


def test_raw_artifacts_preserve_stream_identity_payload_and_strict_order(tmp_path) -> None:
    depth_store, trade_store = raw_pair(tmp_path)
    try:
        depth_payload = diff(101, 101)
        trade_payload = aggtrade(1)
        depth_store.append(record_type="DIFF_DEPTH", received_at=NOW, payload=depth_payload)
        trade_store.append(received_at=NOW, payload=trade_payload)
        with pytest.raises(MicrostructureIntegrityError, match="regress"):
            trade_store.append(received_at=NOW - timedelta(seconds=1), payload=aggtrade(2))
    finally:
        depth_store.close()
        trade_store.close()
    depth_record = json.loads((tmp_path / "depth.jsonl").read_text().splitlines()[0])
    trade_record = json.loads((tmp_path / "aggtrades.jsonl").read_text().splitlines()[0])
    assert depth_record["stream_identity"] == DEPTH_STREAM
    assert trade_record["stream_identity"] == AGGTRADE_STREAM
    assert depth_record["payload"] == depth_payload
    assert trade_record["payload"] == trade_payload


def test_causal_timeline_orders_equal_receipts_depth_first_and_never_looks_ahead(tmp_path) -> None:
    depth_store, trade_store = raw_pair(tmp_path)
    depth_store.append(record_type="REST_SNAPSHOT", received_at=NOW, payload=snapshot())
    trade_store.append(received_at=NOW, payload=aggtrade(1))
    depth_store.append(
        record_type="DIFF_DEPTH",
        received_at=NOW + timedelta(seconds=2),
        payload=diff(101, 101, bids=[["100.00", "7.0"]]),
    )
    trade_store.append(
        received_at=NOW + timedelta(seconds=1),
        payload=aggtrade(2, event_at=NOW + timedelta(seconds=1)),
    )
    depth_store.close()
    trade_store.close()

    contexts = list(
        iter_causal_trade_contexts(
            tmp_path / "depth.jsonl", tmp_path / "aggtrades.jsonl", session_id="session"
        )
    )
    assert [item.depth.last_update_id for item in contexts if item.depth] == [100, 100]
    assert contexts[0].depth is not None
    assert contexts[0].depth.available_at == NOW
    assert contexts[1].depth is not None
    assert contexts[1].depth.bids[0][1] == Decimal("2.0")


def test_trade_before_first_causally_available_book_has_no_depth(tmp_path) -> None:
    depth_store, trade_store = raw_pair(tmp_path)
    trade_store.append(received_at=NOW, payload=aggtrade(1))
    depth_store.append(
        record_type="REST_SNAPSHOT",
        received_at=NOW + timedelta(seconds=1),
        payload=snapshot(),
    )
    depth_store.close()
    trade_store.close()
    contexts = list(
        iter_causal_trade_contexts(
            tmp_path / "depth.jsonl", tmp_path / "aggtrades.jsonl", session_id="session"
        )
    )
    assert len(contexts) == 1
    assert contexts[0].depth is None


def test_depth_gap_persists_resync_boundary_and_replay_recovers(tmp_path) -> None:
    depth_store, trade_store = raw_pair(tmp_path)
    clock_values = iter((NOW + timedelta(seconds=2),))
    sync = V10DepthSynchronizer(depth_store, clock=lambda: next(clock_values))
    sync.install_snapshot(snapshot(), received_at=NOW)
    gap = sync.record_diff(diff(102, 102), received_at=NOW + timedelta(seconds=1))
    assert sync.apply_recorded(gap) is ApplyStatus.SEQUENCE_GAP
    sync.install_snapshot(snapshot(200), received_at=NOW + timedelta(seconds=3))
    recovered = sync.record_diff(diff(201, 201), received_at=NOW + timedelta(seconds=4))
    assert sync.apply_recorded(recovered) is ApplyStatus.APPLIED
    trade_store.append(
        received_at=NOW + timedelta(seconds=5),
        payload=aggtrade(1, event_at=NOW + timedelta(seconds=5)),
    )
    depth_store.close()
    trade_store.close()
    replay = replay_synchronized_session(
        tmp_path / "depth.jsonl", tmp_path / "aggtrades.jsonl", session_id="session"
    )
    assert replay.depth_counters["sequence_gaps"] == 1
    assert replay.depth_counters["resync_count"] == 1
    assert replay.final_depth_update_id == 201


def test_replay_and_hashes_are_deterministic(tmp_path) -> None:
    depth_store, trade_store = raw_pair(tmp_path)
    depth_store.append(record_type="REST_SNAPSHOT", received_at=NOW, payload=snapshot())
    depth_store.append(
        record_type="DIFF_DEPTH",
        received_at=NOW + timedelta(seconds=1),
        payload=diff(101, 101),
    )
    trade_store.append(
        received_at=NOW + timedelta(seconds=2),
        payload=aggtrade(1, event_at=NOW + timedelta(seconds=2)),
    )
    depth_store.close()
    trade_store.close()
    arguments = (tmp_path / "depth.jsonl", tmp_path / "aggtrades.jsonl")
    first = replay_synchronized_session(*arguments, session_id="session")
    second = replay_synchronized_session(*arguments, session_id="session")
    assert first.to_dict() == second.to_dict()
    assert first.synchronized_timeline_sha256 == second.synchronized_timeline_sha256
    assert first.final_depth_sha256 == second.final_depth_sha256


def test_clean_shared_session_closure_is_hash_bound(tmp_path) -> None:
    paths = session_paths(tmp_path, started_at=NOW)
    with ProspectiveSession(
        paths,
        started_at=NOW,
        duration_seconds=10,
        source_commit_sha="a" * 40,
        snapshot_limit=5000,
        max_levels_per_side=5000,
        fsync_each_record=False,
    ) as session:
        depth = V10DepthSynchronizer(session.depth_store)
        trades = AggTradeSynchronizer(session.aggtrade_store)
        depth.install_snapshot(snapshot(), received_at=NOW)
        event = depth.record_diff(diff(101, 101), received_at=NOW + timedelta(seconds=1))
        assert depth.apply_recorded(event) is ApplyStatus.APPLIED
        trades.record(
            aggtrade(1, event_at=NOW + timedelta(seconds=2)),
            received_at=NOW + timedelta(seconds=2),
        )
        summary = session.finalize(
            depth_counters=depth.counters,
            aggtrade_counters=trades.counters,
            ended_at=NOW + timedelta(seconds=10),
        )
    assert summary["session_id"] == paths.session_id
    assert summary["collection_classification"] == "CLOSED_INTEGRITY_PASSED"
    assert summary["integrity_accounting"]["passed"] is True
    assert summary["deterministic_replay_hash_check_passed"] is True
    assert summary["uses_authentication"] is False
    assert summary["orders_enabled"] is False
    assert summary["predictive_outcomes_evaluated"] is False
    assert classify_session(paths.manifest) == "CLOSED_INTEGRITY_PASSED"
    assert {path.name for path in paths.directory.iterdir()} == {
        "depth.jsonl",
        "aggtrades.jsonl",
        "session.manifest.json",
        "closure.summary.json",
    }


def test_tampering_after_closure_is_rejected(tmp_path) -> None:
    paths = session_paths(tmp_path, started_at=NOW)
    with ProspectiveSession(
        paths,
        started_at=NOW,
        duration_seconds=1,
        source_commit_sha=None,
        snapshot_limit=100,
        max_levels_per_side=10,
        fsync_each_record=False,
    ) as session:
        depth = V10DepthSynchronizer(session.depth_store, max_levels_per_side=10)
        trades = AggTradeSynchronizer(session.aggtrade_store)
        depth.install_snapshot(snapshot(), received_at=NOW)
        event = depth.record_diff(diff(101, 101), received_at=NOW)
        depth.apply_recorded(event)
        trades.record(aggtrade(1), received_at=NOW)
        session.finalize(
            depth_counters=depth.counters,
            aggtrade_counters=trades.counters,
            ended_at=NOW + timedelta(seconds=1),
        )
    with paths.aggtrades_raw.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(MicrostructureIntegrityError, match="hash mismatch"):
        classify_session(paths.manifest)


def test_unclosed_shared_session_is_classified_interrupted(tmp_path) -> None:
    paths = session_paths(tmp_path, started_at=NOW)
    with ProspectiveSession(
        paths,
        started_at=NOW,
        duration_seconds=10,
        source_commit_sha=None,
        snapshot_limit=100,
        max_levels_per_side=10,
        fsync_each_record=False,
    ):
        pass
    assert classify_session(paths.manifest) == "INTERRUPTED"
    assert not paths.summary.exists()


def test_dual_stream_collector_starts_both_tasks_concurrently() -> None:
    entered: list[str] = []
    both_entered = asyncio.Event()

    class Runner:
        def __init__(self, name: str, result: object) -> None:
            self.name = name
            self.result = result

        async def run(self, *, duration_seconds: float):
            assert duration_seconds == 3
            entered.append(self.name)
            if len(entered) == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), timeout=1)
            return self.result

    async def exercise():
        depth_result = DepthIntegrityCounters()
        trade_result = AggTradeIntegrityCounters()
        collector = DualStreamCollector(
            depth=Runner("depth", depth_result),  # type: ignore[arg-type]
            aggtrades=Runner("aggtrades", trade_result),  # type: ignore[arg-type]
        )
        assert await collector.run(duration_seconds=3) == (depth_result, trade_result)

    asyncio.run(exercise())
    assert set(entered) == {"depth", "aggtrades"}


def test_aggtrade_recorder_counts_reconnect_without_network(tmp_path) -> None:
    _, store = raw_pair(tmp_path)
    calls = 0

    class WebSocket:
        async def recv(self):
            await asyncio.sleep(1)

    class Connection:
        async def __aenter__(self):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("synthetic disconnect")
            return WebSocket()

        async def __aexit__(self, *_: object):
            return None

    recorder = AggTradeRecorder(
        AggTradeSynchronizer(store),
        connect_factory=lambda *args, **kwargs: Connection(),
        reconnect_delay_seconds=0,
    )
    try:
        counters = asyncio.run(recorder.run(duration_seconds=0.03))
    finally:
        store.close()
    assert counters.reconnect_count == 1
    assert calls == 2


def test_cli_accepts_smoke_and_multi_hour_bounded_durations() -> None:
    parser = build_parser()
    assert _duration(parser.parse_args(["--validate-seconds", "30"])) == 30
    assert _duration(parser.parse_args(["--duration-seconds", "10800"])) == 10800
    with pytest.raises(ValueError, match="Validation duration"):
        _duration(parser.parse_args(["--validate-seconds", "4"]))
