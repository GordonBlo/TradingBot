"""Streaming, causal V9 L2 microstructure features from persisted raw depth logs."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from src.orderbook.book import ApplyStatus, ReconstructedOrderBook
from src.orderbook.models import DepthDataError, DepthDiffEvent, DepthSnapshot
from src.orderbook.storage import RawDepthEventStore


UTC = timezone.utc
DEPTH_LEVELS = (1, 5, 10, 20)
STATE_FIELDS = (
    "best_bid", "best_ask", "mid_price", "spread", "spread_bps",
    "bid_depth_top_1", "bid_depth_top_5", "bid_depth_top_10", "bid_depth_top_20",
    "ask_depth_top_1", "ask_depth_top_5", "ask_depth_top_10", "ask_depth_top_20",
    "depth_imbalance_1", "depth_imbalance_5", "depth_imbalance_10", "depth_imbalance_20",
    "microprice", "microprice_minus_mid_bps",
    "bid_depth_concentration_top1_over_top20",
    "ask_depth_concentration_top1_over_top20",
)
FLOW_FIELDS = (
    "bid_depth_added", "bid_depth_removed", "ask_depth_added", "ask_depth_removed",
    "bid_net_depth_change", "ask_net_depth_change", "update_count",
)
EVENT_FIELDS = (
    "feature_available_at_utc", "source_event_timestamp_utc", "status", "valid_book_state",
    *STATE_FIELDS, *FLOW_FIELDS,
)


class FeatureExtractionError(RuntimeError):
    pass


def decimal_string(value: Decimal | None) -> str | None:
    if value is None:
        return None
    if not value.is_finite():
        raise FeatureExtractionError("Feature Decimal must be finite.")
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def floor_utc(timestamp: datetime, interval: timedelta) -> datetime:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise FeatureExtractionError("Feature timestamp must be UTC-aware.")
    seconds = int(interval.total_seconds())
    if seconds < 1:
        raise ValueError("Feature interval must be at least one second.")
    value = timestamp.astimezone(UTC)
    epoch_seconds = int(value.timestamp())
    return datetime.fromtimestamp(epoch_seconds - epoch_seconds % seconds, tz=UTC)


def _depth_change(
    levels: tuple[tuple[Decimal, Decimal], ...], book: dict[Decimal, Decimal]
) -> tuple[Decimal, Decimal]:
    added = Decimal("0")
    removed = Decimal("0")
    for price, new_quantity in levels:
        old_quantity = book.get(price, Decimal("0"))
        if new_quantity > old_quantity:
            added += new_quantity - old_quantity
        elif new_quantity < old_quantity:
            removed += old_quantity - new_quantity
    return added, removed


def event_flow_features(
    book: ReconstructedOrderBook, event: DepthDiffEvent
) -> dict[str, Decimal]:
    """Describe reported depth changes before applying the event to the local book."""

    bid_added, bid_removed = _depth_change(event.bids, book.bids)
    ask_added, ask_removed = _depth_change(event.asks, book.asks)
    return {
        "bid_depth_added": bid_added,
        "bid_depth_removed": bid_removed,
        "ask_depth_added": ask_added,
        "ask_depth_removed": ask_removed,
        "bid_net_depth_change": bid_added - bid_removed,
        "ask_net_depth_change": ask_added - ask_removed,
        "update_count": Decimal(len(event.bids) + len(event.asks)),
    }


def state_features(book: ReconstructedOrderBook) -> dict[str, Decimal | None]:
    """Return only book-state quantities; no forward-filled values are manufactured."""

    output: dict[str, Decimal | None] = {name: None for name in STATE_FIELDS}
    if not book.is_valid:
        return output
    bid = book.best_bid
    ask = book.best_ask
    assert bid is not None and ask is not None
    best_bid, best_bid_quantity = bid
    best_ask, best_ask_quantity = ask
    mid = (best_bid + best_ask) / Decimal("2")
    spread = best_ask - best_bid
    output.update({
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid,
        "spread": spread,
        "spread_bps": spread / mid * Decimal("10000"),
    })
    bids, asks = book.top_levels(max(DEPTH_LEVELS))
    for count in DEPTH_LEVELS:
        bid_depth = sum((quantity for _, quantity in bids[:count]), Decimal("0"))
        ask_depth = sum((quantity for _, quantity in asks[:count]), Decimal("0"))
        output[f"bid_depth_top_{count}"] = bid_depth
        output[f"ask_depth_top_{count}"] = ask_depth
        total_depth = bid_depth + ask_depth
        output[f"depth_imbalance_{count}"] = (
            (bid_depth - ask_depth) / total_depth if total_depth > 0 else None
        )
    best_total = best_bid_quantity + best_ask_quantity
    microprice = (
        (best_ask * best_bid_quantity + best_bid * best_ask_quantity) / best_total
        if best_total > 0 else None
    )
    output["microprice"] = microprice
    output["microprice_minus_mid_bps"] = (
        (microprice - mid) / mid * Decimal("10000") if microprice is not None else None
    )
    for side in ("bid", "ask"):
        top_1 = output[f"{side}_depth_top_1"]
        top_20 = output[f"{side}_depth_top_20"]
        output[f"{side}_depth_concentration_top1_over_top20"] = (
            top_1 / top_20 if top_1 is not None and top_20 is not None and top_20 > 0 else None
        )
    return output


@dataclass(frozen=True, slots=True)
class FeatureEvent:
    available_at: datetime
    source_event_at: datetime | None
    status: str
    valid_book_state: bool
    state: dict[str, Decimal | None]
    flow: dict[str, Decimal]

    def row(self) -> dict[str, str | int | bool | None]:
        return {
            "feature_available_at_utc": self.available_at.isoformat(),
            "source_event_timestamp_utc": (
                self.source_event_at.isoformat() if self.source_event_at else None
            ),
            "status": self.status,
            "valid_book_state": self.valid_book_state,
            **{name: decimal_string(self.state[name]) for name in STATE_FIELDS},
            **{name: decimal_string(self.flow[name]) for name in FLOW_FIELDS},
        }


@dataclass(slots=True)
class NumericStats:
    count: int = 0
    total: Decimal = Decimal("0")
    total_squares: Decimal = Decimal("0")
    minimum: Decimal | None = None
    maximum: Decimal | None = None
    last: Decimal | None = None

    def add(self, value: Decimal) -> None:
        self.count += 1
        self.total += value
        self.total_squares += value * value
        self.minimum = value if self.minimum is None or value < self.minimum else self.minimum
        self.maximum = value if self.maximum is None or value > self.maximum else self.maximum
        self.last = value

    def values(self) -> dict[str, Decimal | None]:
        if not self.count:
            return {"last": None, "mean": None, "min": None, "max": None, "std": None}
        mean = self.total / Decimal(self.count)
        variance = self.total_squares / Decimal(self.count) - mean * mean
        return {
            "last": self.last,
            "mean": mean,
            "min": self.minimum,
            "max": self.maximum,
            "std": max(variance, Decimal("0")).sqrt(),
        }


@dataclass(slots=True)
class BucketStats:
    bucket_open: datetime
    interval: timedelta
    event_count: int = 0
    valid_state_count: int = 0
    invalid_state_count: int = 0
    update_count: Decimal = Decimal("0")
    flows: dict[str, Decimal] = field(default_factory=lambda: {
        name: Decimal("0") for name in FLOW_FIELDS if name != "update_count"
    })
    values: dict[str, NumericStats] = field(default_factory=lambda: {
        name: NumericStats() for name in STATE_FIELDS
    })
    first_event_at: datetime | None = None
    last_event_at: datetime | None = None

    def add(self, item: FeatureEvent) -> None:
        self.event_count += 1
        self.first_event_at = self.first_event_at or item.available_at
        self.last_event_at = item.available_at
        self.update_count += item.flow["update_count"]
        for name, value in self.flows.items():
            self.flows[name] = value + item.flow[name]
        if item.valid_book_state:
            self.valid_state_count += 1
            for name, value in item.state.items():
                if value is not None:
                    self.values[name].add(value)
        else:
            self.invalid_state_count += 1

    def row(self) -> dict[str, str | int | None]:
        row: dict[str, str | int | None] = {
            "bucket_open_utc": self.bucket_open.isoformat(),
            "bucket_close_utc": (self.bucket_open + self.interval).isoformat(),
            "first_feature_available_at_utc": (
                self.first_event_at.isoformat() if self.first_event_at else None
            ),
            "last_feature_available_at_utc": (
                self.last_event_at.isoformat() if self.last_event_at else None
            ),
            "event_count": self.event_count,
            "valid_state_count": self.valid_state_count,
            "invalid_state_count": self.invalid_state_count,
            "update_count": decimal_string(self.update_count),
            "update_intensity_per_second": decimal_string(
                self.update_count / Decimal(int(self.interval.total_seconds()))
            ),
            **{name: decimal_string(value) for name, value in self.flows.items()},
        }
        for name, stats in self.values.items():
            for statistic, value in stats.values().items():
                row[f"{name}_{statistic}"] = decimal_string(value)
        return row


BUCKET_FIELDS = tuple(BucketStats(datetime(1970, 1, 1, tzinfo=UTC), timedelta(seconds=1)).row())


class StreamingBucketWriter:
    def __init__(self, path: Path, *, interval: timedelta) -> None:
        self.path = path
        self.interval = interval
        self.stream = path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.stream, fieldnames=BUCKET_FIELDS)
        self.writer.writeheader()
        self.current: BucketStats | None = None
        self.bucket_count = 0

    def _write_current(self) -> None:
        assert self.current is not None
        self.writer.writerow(self.current.row())
        self.bucket_count += 1

    def add(self, item: FeatureEvent) -> None:
        bucket_open = floor_utc(item.available_at, self.interval)
        if self.current is None:
            self.current = BucketStats(bucket_open, self.interval)
        if bucket_open < self.current.bucket_open:
            raise FeatureExtractionError("Feature availability timestamps are not chronological.")
        while self.current.bucket_open < bucket_open:
            self._write_current()
            self.current = BucketStats(self.current.bucket_open + self.interval, self.interval)
        self.current.add(item)

    def close(self) -> int:
        if self.current is None:
            self.stream.close()
            return 0
        while self.current is not None:
            self._write_current()
            self.current = None
        self.stream.close()
        return self.bucket_count


@dataclass(slots=True)
class ExtractionCounters:
    raw_record_count: int = 0
    raw_diff_event_count: int = 0
    snapshots: int = 0
    reconstructed_update_count: int = 0
    stale_event_count: int = 0
    sequence_gap_count: int = 0
    crossed_book_state_count: int = 0
    invalid_event_count: int = 0
    unreconstructed_event_count: int = 0
    missing_feature_state_count: int = 0
    first_feature_timestamp: str | None = None
    last_feature_timestamp: str | None = None


def _empty_state() -> dict[str, Decimal | None]:
    return {name: None for name in STATE_FIELDS}


def _empty_flow() -> dict[str, Decimal]:
    return {name: Decimal("0") for name in FLOW_FIELDS}


def _available_at(*timestamps: datetime) -> datetime:
    return max(item.astimezone(UTC) for item in timestamps)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FeatureExtractor:
    """Replay raw events once, emitting event, second, and fifteen-minute files."""

    def __init__(self, *, max_levels_per_side: int = 5_000) -> None:
        self.max_levels_per_side = max_levels_per_side

    def extract(self, raw_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
        raw_path = Path(raw_path)
        output_dir = Path(output_dir)
        if output_dir.exists():
            raise FileExistsError(f"V9 feature output already exists: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=False)
        event_path = output_dir / "event_features.csv"
        second_path = output_dir / "features_1s.csv"
        fifteen_path = output_dir / "features_15m.csv"
        counters = ExtractionCounters()
        global_stats = {name: NumericStats() for name in STATE_FIELDS}
        book = ReconstructedOrderBook(max_levels_per_side=self.max_levels_per_side)
        pending: list[tuple[DepthDiffEvent, datetime]] = []
        snapshot_received_at: datetime | None = None

        with event_path.open("w", encoding="utf-8", newline="") as event_stream:
            event_writer = csv.DictWriter(event_stream, fieldnames=EVENT_FIELDS)
            event_writer.writeheader()
            second_writer = StreamingBucketWriter(second_path, interval=timedelta(seconds=1))
            fifteen_writer = StreamingBucketWriter(fifteen_path, interval=timedelta(minutes=15))

            def emit(item: FeatureEvent) -> None:
                event_writer.writerow(item.row())
                second_writer.add(item)
                fifteen_writer.add(item)
                if item.valid_book_state:
                    counters.reconstructed_update_count += 1
                    if counters.first_feature_timestamp is None:
                        counters.first_feature_timestamp = item.available_at.isoformat()
                    counters.last_feature_timestamp = item.available_at.isoformat()
                    for name, value in item.state.items():
                        if value is not None:
                            global_stats[name].add(value)
                else:
                    counters.missing_feature_state_count += 1

            def process(event: DepthDiffEvent, received_at: datetime) -> bool:
                nonlocal pending
                assert snapshot_received_at is not None
                available_at = _available_at(received_at, event.event_time, snapshot_received_at)
                if book.last_update_id is None:
                    pending.append((event, received_at))
                    return False
                if event.final_update_id <= book.last_update_id:
                    counters.stale_event_count += 1
                    emit(FeatureEvent(available_at, event.event_time, "STALE", False, _empty_state(), _empty_flow()))
                    return True
                expected = book.last_update_id + 1
                if event.first_update_id > expected or event.final_update_id < expected:
                    counters.sequence_gap_count += 1
                    book.last_update_id = None
                    pending = [(event, received_at)]
                    return False
                flow = event_flow_features(book, event)
                status = book.apply(event)
                if status is ApplyStatus.INVALID_BOOK:
                    counters.crossed_book_state_count += 1
                    book.last_update_id = None
                    emit(FeatureEvent(available_at, event.event_time, "INVALID_BOOK", False, _empty_state(), _empty_flow()))
                    return True
                if status is not ApplyStatus.APPLIED:
                    raise FeatureExtractionError(f"Unexpected depth replay status: {status.value}")
                emit(FeatureEvent(available_at, event.event_time, "APPLIED", True, state_features(book), flow))
                return True

            for record in RawDepthEventStore.records(raw_path):
                counters.raw_record_count += 1
                try:
                    received_at = datetime.fromisoformat(record["received_at_utc"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise FeatureExtractionError("Raw depth receipt timestamp is invalid.") from exc
                if received_at.tzinfo is None or received_at.utcoffset() is None:
                    raise FeatureExtractionError("Raw depth receipt timestamp must be UTC-aware.")
                received_at = received_at.astimezone(UTC)
                if record["record_type"] == "DIFF_DEPTH":
                    counters.raw_diff_event_count += 1
                    try:
                        event = DepthDiffEvent.from_payload(record["payload"])
                    except DepthDataError:
                        counters.invalid_event_count += 1
                        emit(FeatureEvent(received_at, None, "INVALID_EVENT", False, _empty_state(), _empty_flow()))
                        continue
                    if book.last_update_id is None or snapshot_received_at is None:
                        pending.append((event, received_at))
                    else:
                        process(event, received_at)
                elif record["record_type"] == "REST_SNAPSHOT":
                    try:
                        snapshot = DepthSnapshot.from_payload(record["payload"], received_at=received_at)
                    except DepthDataError as exc:
                        raise FeatureExtractionError("Raw depth snapshot is invalid.") from exc
                    counters.snapshots += 1
                    book.reset(snapshot)
                    snapshot_received_at = received_at
                    if not book.is_valid:
                        counters.crossed_book_state_count += 1
                        book.last_update_id = None
                        continue
                    buffered = pending
                    pending = []
                    for event, event_received_at in buffered:
                        process(event, event_received_at)
                else:
                    raise FeatureExtractionError("Unknown raw depth record type.")

            for event, received_at in pending:
                counters.unreconstructed_event_count += 1
                emit(FeatureEvent(
                    received_at, event.event_time, "UNRECONSTRUCTED_AWAITING_SNAPSHOT",
                    False, _empty_state(), _empty_flow(),
                ))
            second_bucket_count = second_writer.close()
            fifteen_bucket_count = fifteen_writer.close()

        files = {
            "event_features": event_path,
            "features_1s": second_path,
            "features_15m": fifteen_path,
        }
        return {
            "version": "V9_L2_FEATURE_FOUNDATION_1",
            "raw_path": str(raw_path),
            "raw_sha256": _file_sha256(raw_path),
            "output_dir": str(output_dir),
            "files": {name: str(path) for name, path in files.items()},
            "file_sha256": {name: _file_sha256(path) for name, path in files.items()},
            "counters": {
                **asdict(counters),
                "event_feature_count": counters.raw_diff_event_count,
                "one_second_bucket_count": second_bucket_count,
                "fifteen_minute_bucket_count": fifteen_bucket_count,
            },
            "feature_sanity_min_max": {
                name: {
                    "min": decimal_string(stats.minimum),
                    "max": decimal_string(stats.maximum),
                }
                for name, stats in global_stats.items()
            },
        }


def deterministic_replay_check(
    raw_path: str | Path, *, extractor_factory: Callable[[], FeatureExtractor] = FeatureExtractor,
    temporary_output_root: str | Path,
) -> dict[str, Any]:
    """Run a second replay into an empty directory and compare serialized hashes."""

    temporary_output_root = Path(temporary_output_root)
    report = extractor_factory().extract(raw_path, temporary_output_root)
    return report
