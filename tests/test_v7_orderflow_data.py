from __future__ import annotations

import csv
import hashlib
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from io import StringIO
from pathlib import Path

import pytest

from src.orderflow.aggregation import aggregate_15m, completed_bucket_at
from src.orderflow.archive import (
    AggregateTradeArchiveDownloader,
    build_archive_location,
)
from src.orderflow.integrity import (
    AggregateTradeIntegrityError,
    validate_archive_boundaries,
    validate_trade_segment,
    verify_sha256,
)
from src.orderflow.models import AggregateTrade, AggressorSide
from src.orderflow.parser import AggregateTradeParseError, parse_aggtrade_csv
from src.orderflow.storage import OrderFlowBucketStore
from src.orderflow.timestamps import (
    BinanceTimestampError,
    normalize_binance_timestamp,
)


BASE = datetime(2025, 2, 1, tzinfo=timezone.utc)


def micros(value: datetime) -> int:
    delta = value - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def trade(
    aggregate_id: int,
    timestamp: datetime,
    *,
    price: str = "100.25",
    quantity: str = "2.5",
    first_id: int | None = None,
    last_id: int | None = None,
    buyer_is_maker: bool = False,
) -> AggregateTrade:
    first = aggregate_id if first_id is None else first_id
    last = first if last_id is None else last_id
    return AggregateTrade(
        aggregate_trade_id=aggregate_id,
        price=Decimal(price),
        quantity=Decimal(quantity),
        first_trade_id=first,
        last_trade_id=last,
        timestamp=timestamp,
        buyer_is_maker=buyer_is_maker,
        best_price_match=True,
    )


def row(
    aggregate_id: int,
    timestamp: datetime,
    *,
    price: str = "100.25",
    quantity: str = "2.5",
    first_id: int | None = None,
    last_id: int | None = None,
    buyer_is_maker: str = "false",
    best_match: str = "true",
) -> str:
    first = aggregate_id if first_id is None else first_id
    last = first if last_id is None else last_id
    return ",".join(
        (
            str(aggregate_id),
            price,
            quantity,
            str(first),
            str(last),
            str(micros(timestamp)),
            buyer_is_maker,
            best_match,
        )
    )


def test_raw_model_is_decimal_exact_and_maker_direction_is_not_reversed() -> None:
    taker_buy = trade(1, BASE, price="100.10", quantity="0.30")
    taker_sell = trade(2, BASE, buyer_is_maker=True)

    assert taker_buy.price == Decimal("100.10")
    assert taker_buy.quantity == Decimal("0.30")
    assert taker_buy.quote_quantity == Decimal("30.0300")
    assert taker_buy.aggressor_side is AggressorSide.TAKER_BUY
    assert taker_sell.aggressor_side is AggressorSide.TAKER_SELL


def test_timestamp_normalization_supports_transition_ms_and_us_in_utc() -> None:
    before_transition_ms = 1_735_689_599_999
    transition_us = 1_735_689_600_000_000

    before = normalize_binance_timestamp(before_transition_ms)
    at_transition = normalize_binance_timestamp(transition_us)

    assert before == datetime(2024, 12, 31, 23, 59, 59, 999000, tzinfo=timezone.utc)
    assert at_transition == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert before.tzinfo is timezone.utc
    assert at_transition.tzinfo is timezone.utc


@pytest.mark.parametrize(
    "value",
    (
        1_735_689_600,
        1_735_689_600_000,
        1_735_689_599_999_000,
        "not-a-timestamp",
    ),
)
def test_timestamp_normalization_rejects_invalid_or_transition_wrong_scale(
    value: int | str,
) -> None:
    with pytest.raises(BinanceTimestampError):
        normalize_binance_timestamp(value)


def test_parser_supports_headerless_and_header_csv_with_binance_booleans() -> None:
    headerless = parse_aggtrade_csv(StringIO(row(1, BASE)), source="daily.csv")
    header = (
        "agg_trade_id,price,quantity,first_trade_id,last_trade_id,"
        "transact_time,is_buyer_maker,is_best_match\n"
    )
    with_header = parse_aggtrade_csv(
        StringIO(header + row(2, BASE, buyer_is_maker="True", best_match="False")),
        source="daily-header.csv",
    )

    assert headerless[0].quote_quantity == Decimal("250.625")
    assert headerless[0].aggressor_side is AggressorSide.TAKER_BUY
    assert with_header[0].aggressor_side is AggressorSide.TAKER_SELL
    assert with_header[0].best_price_match is False


@pytest.mark.parametrize(
    ("csv_row", "message"),
    (
        ("1,100,2,1,1,bad,false,true", "timestamp"),
        (row(1, BASE, price="0"), "price"),
        (row(1, BASE, quantity="-1"), "quantity"),
        (row(1, BASE, first_id=3, last_id=2), "first_trade_id"),
        (row(1, BASE).rsplit(",", 2)[0], "7 or 8 fields"),
    ),
)
def test_parser_rejects_malformed_financial_rows_with_source_context(
    csv_row: str, message: str
) -> None:
    with pytest.raises(AggregateTradeParseError, match=message) as error:
        parse_aggtrade_csv(StringIO(csv_row), source="bad.csv")
    assert "bad.csv: row 1" in str(error.value)


def test_parser_allows_source_without_optional_best_match_field() -> None:
    parsed = parse_aggtrade_csv(
        StringIO(row(1, BASE).rsplit(",", 1)[0]), source="seven-columns.csv"
    )
    assert parsed[0].best_price_match is None


def test_integrity_rejects_duplicate_ids_and_decreasing_timestamps() -> None:
    with pytest.raises(AggregateTradeIntegrityError, match="Duplicate"):
        validate_trade_segment((trade(1, BASE), trade(1, BASE + timedelta(seconds=1))))
    with pytest.raises(AggregateTradeIntegrityError, match="timestamp decreases"):
        validate_trade_segment((trade(1, BASE), trade(2, BASE - timedelta(seconds=1))))


def test_cross_archive_integrity_allows_id_gaps_but_rejects_exact_duplicates() -> None:
    first = trade(1, BASE)
    later_with_gap = trade(10, BASE + timedelta(days=1))
    validate_archive_boundaries(((first,), (later_with_gap,)))

    with pytest.raises(AggregateTradeIntegrityError, match="duplicated across"):
        validate_archive_boundaries(((first,), (first,)))


def test_sha256_verification_accepts_official_checksum_shape_and_rejects_mismatch(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "sample.zip"
    archive.write_bytes(b"synthetic-safe-data")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()

    assert verify_sha256(archive, f"{digest}  sample.zip\n") == digest
    with pytest.raises(AggregateTradeIntegrityError, match="does not match"):
        verify_sha256(archive, f"{'0' * 64}  sample.zip\n")


def test_utc_boundary_bucketing_has_no_cross_bucket_leakage_and_exact_math() -> None:
    trades = (
        trade(1, BASE, price="100", quantity="2", first_id=10, last_id=12),
        trade(
            2,
            BASE + timedelta(minutes=14, seconds=59, microseconds=999999),
            price="50",
            quantity="2",
            first_id=13,
            last_id=13,
            buyer_is_maker=True,
        ),
        trade(3, BASE + timedelta(minutes=15), price="25", quantity="4"),
        trade(
            4,
            BASE + timedelta(minutes=30),
            price="20",
            quantity="1",
            buyer_is_maker=True,
        ),
    )

    first, second, third = aggregate_15m(trades)

    assert first.bucket_open_time == BASE
    assert first.bucket_close_time == BASE + timedelta(minutes=15)
    assert first.aggregate_trade_count == 2
    assert first.underlying_trade_count == 4
    assert first.total_base_volume == Decimal("4")
    assert first.total_quote_volume == Decimal("300")
    assert first.taker_buy_base_volume == Decimal("2")
    assert first.taker_buy_quote_volume == Decimal("200")
    assert first.taker_sell_base_volume == Decimal("2")
    assert first.taker_sell_quote_volume == Decimal("100")
    assert first.taker_buy_aggtrade_count == 1
    assert first.taker_sell_aggtrade_count == 1
    assert first.signed_base_volume == Decimal("0")
    assert first.signed_quote_volume == Decimal("100")
    assert first.taker_buy_base_ratio == Decimal("0.5")
    assert first.taker_buy_quote_ratio == Decimal("2") / Decimal("3")
    assert first.base_volume_imbalance == Decimal("0")
    assert first.quote_volume_imbalance == Decimal("1") / Decimal("3")
    assert first.average_aggtrade_base_size == Decimal("2")
    assert first.average_aggtrade_quote_size == Decimal("150")
    assert second.bucket_open_time == BASE + timedelta(minutes=15)
    assert second.base_volume_imbalance == Decimal("1")
    assert second.quote_volume_imbalance == Decimal("1")
    assert third.bucket_open_time == BASE + timedelta(minutes=30)
    assert third.base_volume_imbalance == Decimal("-1")
    assert third.quote_volume_imbalance == Decimal("-1")


def test_completed_bucket_lookup_exposes_only_exactly_completed_bucket_once() -> None:
    buckets = aggregate_15m(
        (
            trade(1, BASE),
            trade(2, BASE + timedelta(minutes=15)),
        )
    )

    assert completed_bucket_at(buckets, BASE + timedelta(minutes=14)) is None
    assert completed_bucket_at(buckets, BASE + timedelta(minutes=15)) == buckets[0]
    assert completed_bucket_at(buckets, BASE + timedelta(minutes=20)) is None
    assert completed_bucket_at(buckets, BASE + timedelta(minutes=30)) == buckets[1]


def test_bucket_storage_round_trip_has_deterministic_columns_and_decimals(
    tmp_path: Path,
) -> None:
    buckets = aggregate_15m((trade(1, BASE, price="100.00", quantity="2.00"),))
    store = OrderFlowBucketStore(tmp_path / "aggregated" / "15m")

    path = store.save(buckets)
    loaded = store.load()
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert loaded == buckets
    assert tuple(rows[0]) == store.CSV_FIELDS
    assert rows[0]["total_base_volume"] == "2"
    assert rows[0]["total_quote_volume"] == "200"
    assert rows[0]["bucket_open_time"].endswith("+00:00")


def test_archive_locations_and_downloader_are_official_bounded_and_checksum_verified(
    tmp_path: Path,
) -> None:
    location = build_archive_location(
        date(2025, 2, 3), cadence="daily", root=tmp_path / "raw"
    )
    archive_bytes = b"one-explicit-synthetic-archive"
    digest = hashlib.sha256(archive_bytes).hexdigest()
    calls: list[str] = []

    def fetcher(url: str, timeout: float) -> bytes:
        assert timeout == 3
        calls.append(url)
        if url.endswith(".CHECKSUM"):
            return f"{digest}  {location.destination.name}\n".encode()
        return archive_bytes

    downloader = AggregateTradeArchiveDownloader(fetcher=fetcher, timeout_seconds=3)
    first = downloader.download(location)
    second = downloader.download(location)

    assert location.url == (
        "https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDC/"
        "BTCUSDC-aggTrades-2025-02-03.zip"
    )
    assert location.destination == (
        tmp_path
        / "raw"
        / "BTCUSDC"
        / "daily"
        / "2025"
        / "02"
        / "BTCUSDC-aggTrades-2025-02-03.zip"
    )
    assert first.checksum_verified and not first.skipped_existing
    assert second.checksum_verified and second.skipped_existing
    assert calls.count(location.url) == 1


def test_monthly_archive_location_uses_official_pattern(tmp_path: Path) -> None:
    location = build_archive_location(
        date(2024, 12, 1), cadence="monthly", root=tmp_path
    )
    assert location.url.endswith(
        "/spot/monthly/aggTrades/BTCUSDC/BTCUSDC-aggTrades-2024-12.zip"
    )
    assert location.checksum_url == f"{location.url}.CHECKSUM"
