from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from src.orderbook.book import ReconstructedOrderBook
from src.orderbook.features import FeatureExtractor, event_flow_features, floor_utc, state_features
from src.orderbook.models import DepthDiffEvent, DepthSnapshot
from src.orderbook.storage import RawDepthEventStore


UTC = timezone.utc
T = datetime(2026, 8, 25, 12, tzinfo=UTC)


def snapshot(last_update_id: int = 100) -> dict:
    return {
        "lastUpdateId": last_update_id,
        "bids": [["100", "2"], ["99", "4"], ["98", "8"]],
        "asks": [["102", "1"], ["103", "3"], ["104", "6"]],
    }


def diff(first: int, final: int, at: datetime, *, bids: list | None = None, asks: list | None = None) -> dict:
    return {
        "e": "depthUpdate", "E": int(at.timestamp() * 1000), "s": "BTCUSDC",
        "U": first, "u": final, "b": bids or [], "a": asks or [],
    }


def write_raw(path: Path, records: list[tuple[str, datetime, dict]]) -> Path:
    with RawDepthEventStore(path, session_id="test", fsync_each_record=False) as store:
        for record_type, received_at, payload in records:
            store.append(record_type=record_type, received_at=received_at, payload=payload)
    return path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_spread_mid_microprice_and_concentration_math() -> None:
    book = ReconstructedOrderBook()
    book.reset(DepthSnapshot.from_payload(snapshot(), received_at=T))
    values = state_features(book)

    assert values["best_bid"] == Decimal("100")
    assert values["best_ask"] == Decimal("102")
    assert values["mid_price"] == Decimal("101")
    assert values["spread"] == Decimal("2")
    assert values["spread_bps"] == Decimal("2") / Decimal("101") * Decimal("10000")
    assert values["microprice"] == Decimal("304") / Decimal("3")
    assert values["microprice_minus_mid_bps"] > 0
    assert values["bid_depth_concentration_top1_over_top20"] == Decimal("2") / Decimal("14")
    assert values["ask_depth_concentration_top1_over_top20"] == Decimal("1") / Decimal("10")


def test_depth_imbalance_uses_exact_top_n_depth() -> None:
    book = ReconstructedOrderBook()
    book.reset(DepthSnapshot.from_payload(snapshot(), received_at=T))
    values = state_features(book)

    assert values["bid_depth_top_1"] == Decimal("2")
    assert values["ask_depth_top_1"] == Decimal("1")
    assert values["depth_imbalance_1"] == Decimal("1") / Decimal("3")
    assert values["depth_imbalance_5"] == Decimal("4") / Decimal("24")


def test_add_remove_depth_accounting_uses_pre_update_book() -> None:
    book = ReconstructedOrderBook()
    book.reset(DepthSnapshot.from_payload(snapshot(), received_at=T))
    event = DepthDiffEvent.from_payload(diff(
        101, 101, T,
        bids=[["100", "5"], ["99", "0"]],
        asks=[["102", "0"], ["101", "2"]],
    ))

    flow = event_flow_features(book, event)

    assert flow == {
        "bid_depth_added": Decimal("3"),
        "bid_depth_removed": Decimal("4"),
        "ask_depth_added": Decimal("2"),
        "ask_depth_removed": Decimal("1"),
        "bid_net_depth_change": Decimal("-1"),
        "ask_net_depth_change": Decimal("1"),
        "update_count": Decimal("4"),
    }


def test_exact_one_second_and_fifteen_minute_utc_boundaries(tmp_path) -> None:
    raw = write_raw(tmp_path / "raw.jsonl", [
        ("REST_SNAPSHOT", T, snapshot()),
        ("DIFF_DEPTH", T + timedelta(milliseconds=999), diff(101, 101, T + timedelta(milliseconds=999))),
        ("DIFF_DEPTH", T + timedelta(seconds=1), diff(102, 102, T + timedelta(seconds=1))),
        ("DIFF_DEPTH", T + timedelta(minutes=15), diff(103, 103, T + timedelta(minutes=15))),
    ])
    report = FeatureExtractor().extract(raw, tmp_path / "out")
    second = rows(Path(report["files"]["features_1s"]))
    fifteen = rows(Path(report["files"]["features_15m"]))
    second_by_open = {row["bucket_open_utc"]: row for row in second}
    fifteen_by_open = {row["bucket_open_utc"]: row for row in fifteen}

    assert second_by_open[T.isoformat()]["event_count"] == "1"
    assert second_by_open[(T + timedelta(seconds=1)).isoformat()]["event_count"] == "1"
    assert fifteen_by_open[T.isoformat()]["event_count"] == "2"
    assert fifteen_by_open[(T + timedelta(minutes=15)).isoformat()]["event_count"] == "1"
    assert report["counters"]["one_second_bucket_count"] == len(second)
    assert report["counters"]["fifteen_minute_bucket_count"] == len(fifteen)
    assert floor_utc(T + timedelta(minutes=15), timedelta(minutes=15)) == T + timedelta(minutes=15)


def test_pre_snapshot_event_is_not_available_before_snapshot_receipt(tmp_path) -> None:
    event_time = T
    snapshot_receipt = T + timedelta(seconds=1)
    raw = write_raw(tmp_path / "raw.jsonl", [
        ("DIFF_DEPTH", T, diff(100, 101, event_time)),
        ("REST_SNAPSHOT", snapshot_receipt, snapshot()),
    ])
    report = FeatureExtractor().extract(raw, tmp_path / "out")
    event = rows(Path(report["files"]["event_features"]))[0]
    second = rows(Path(report["files"]["features_1s"]))

    assert event["feature_available_at_utc"] == snapshot_receipt.isoformat()
    assert [row["bucket_open_utc"] for row in second] == [snapshot_receipt.isoformat()]
    assert report["counters"]["first_feature_timestamp"] == snapshot_receipt.isoformat()


def test_stale_event_remains_explicit_missing_state(tmp_path) -> None:
    raw = write_raw(tmp_path / "raw.jsonl", [
        ("REST_SNAPSHOT", T, snapshot()),
        ("DIFF_DEPTH", T + timedelta(seconds=1), diff(90, 100, T + timedelta(seconds=1))),
    ])
    report = FeatureExtractor().extract(raw, tmp_path / "out")
    event = rows(Path(report["files"]["event_features"]))[0]

    assert event["status"] == "STALE"
    assert event["valid_book_state"] == "False"
    assert event["spread"] == ""
    assert report["counters"]["stale_event_count"] == 1
    assert report["counters"]["missing_feature_state_count"] == 1


def test_replay_serialization_is_deterministic(tmp_path) -> None:
    raw = write_raw(tmp_path / "raw.jsonl", [
        ("DIFF_DEPTH", T, diff(100, 101, T)),
        ("REST_SNAPSHOT", T + timedelta(milliseconds=1), snapshot()),
        ("DIFF_DEPTH", T + timedelta(seconds=1), diff(102, 102, T + timedelta(seconds=1), bids=[["100", "3"]])),
    ])
    first = FeatureExtractor().extract(raw, tmp_path / "first")
    second = FeatureExtractor().extract(raw, tmp_path / "second")

    assert first["file_sha256"] == second["file_sha256"]
    assert first["counters"] == second["counters"]
