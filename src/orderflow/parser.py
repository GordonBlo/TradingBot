"""Strict parser for official Binance Spot aggTrades CSV files."""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterable, Iterator
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TextIO

from src.orderflow.models import AggregateTrade
from src.orderflow.timestamps import BinanceTimestampError, normalize_binance_timestamp


class AggregateTradeParseError(ValueError):
    """Raised with row/source context when an aggTrades CSV is malformed."""


_REQUIRED_COLUMNS = (
    "aggregate_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "timestamp",
    "buyer_is_maker",
)
_OPTIONAL_COLUMN = "best_price_match"
_ALIASES = {
    "agg_trade_id": "aggregate_trade_id",
    "aggregate_trade_id": "aggregate_trade_id",
    "price": "price",
    "quantity": "quantity",
    "qty": "quantity",
    "first_trade_id": "first_trade_id",
    "last_trade_id": "last_trade_id",
    "timestamp": "timestamp",
    "transact_time": "timestamp",
    "is_buyer_maker": "buyer_is_maker",
    "buyer_is_maker": "buyer_is_maker",
    "is_best_match": "best_price_match",
    "best_price_match": "best_price_match",
    "aggtradeid": "aggregate_trade_id",
    "firsttradeid": "first_trade_id",
    "lasttradeid": "last_trade_id",
    "transacttime": "timestamp",
    "isbuyermaker": "buyer_is_maker",
    "isbestmatch": "best_price_match",
}


def _boolean(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{name} must be a Binance boolean.")


def _header_mapping(row: list[str]) -> tuple[int, ...] | None:
    normalized = [item.strip().lower().lstrip("\ufeff") for item in row]
    if not any(item in _ALIASES for item in normalized):
        return None
    mapped = [_ALIASES.get(item) for item in normalized]
    if any(name is None for name in mapped):
        raise ValueError("aggTrades header contains an unknown column.")
    missing = [name for name in _REQUIRED_COLUMNS if name not in mapped]
    if missing:
        raise ValueError(f"aggTrades header is missing {', '.join(missing)}.")
    required = tuple(mapped.index(name) for name in _REQUIRED_COLUMNS)
    optional = (
        mapped.index(_OPTIONAL_COLUMN) if _OPTIONAL_COLUMN in mapped else -1
    )
    return (*required, optional)


def _parse_row(values: list[str]) -> AggregateTrade:
    if len(values) not in {7, 8}:
        raise ValueError("aggTrades row must contain 7 or 8 fields.")
    try:
        return AggregateTrade(
            aggregate_trade_id=int(values[0]),
            price=Decimal(values[1]),
            quantity=Decimal(values[2]),
            first_trade_id=int(values[3]),
            last_trade_id=int(values[4]),
            timestamp=normalize_binance_timestamp(values[5]),
            buyer_is_maker=_boolean(values[6], name="buyer_is_maker"),
            best_price_match=(
                _boolean(values[7], name="best_price_match")
                if len(values) == 8
                else None
            ),
        )
    except (InvalidOperation, BinanceTimestampError, ValueError) as exc:
        raise ValueError(str(exc)) from exc


def parse_aggtrade_csv(
    stream: TextIO | Iterable[str], *, source: str = "<stream>"
) -> tuple[AggregateTrade, ...]:
    return tuple(iter_aggtrade_csv(stream, source=source))


def iter_aggtrade_csv(
    stream: TextIO | Iterable[str], *, source: str = "<stream>"
) -> Iterator[AggregateTrade]:
    """Yield validated rows without retaining an archive-sized trade tuple."""

    reader = csv.reader(stream)
    mapping: tuple[int, ...] | None = None
    first_content_seen = False
    previous: AggregateTrade | None = None
    for row_number, row in enumerate(reader, start=1):
        if not row or all(not item.strip() for item in row):
            continue
        try:
            if not first_content_seen:
                first_content_seen = True
                mapping = _header_mapping(row)
                if mapping is not None:
                    continue
            if mapping:
                ordered = [row[index] for index in mapping[:-1]]
                if mapping[-1] >= 0:
                    ordered.append(row[mapping[-1]])
            else:
                ordered = row
            trade = _parse_row(ordered)
            if previous is not None:
                if trade.timestamp < previous.timestamp:
                    raise ValueError("Aggregate-trade timestamp decreases.")
                if trade.aggregate_trade_id <= previous.aggregate_trade_id:
                    raise ValueError("aggregate_trade_id is not increasing.")
            previous = trade
            yield trade
        except (IndexError, ValueError) as exc:
            raise AggregateTradeParseError(
                f"{source}: row {row_number}: {exc}"
            ) from exc


def parse_aggtrade_file(path: str | Path) -> tuple[AggregateTrade, ...]:
    source_path = Path(path)
    try:
        with source_path.open("r", encoding="utf-8-sig", newline="") as stream:
            return parse_aggtrade_csv(stream, source=str(source_path))
    except AggregateTradeParseError:
        raise
    except OSError as exc:
        raise AggregateTradeParseError(f"{source_path}: could not read archive CSV.") from exc


def parse_aggtrade_archive(path: str | Path) -> tuple[AggregateTrade, ...]:
    """Parse the single CSV member of a checksum-verified Binance ZIP archive."""

    return tuple(iter_aggtrade_archive(path))


def iter_aggtrade_archive(path: str | Path) -> Iterator[AggregateTrade]:
    archive_path = Path(path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = [
                name
                for name in archive.namelist()
                if not name.endswith("/") and name.lower().endswith(".csv")
            ]
            if len(members) != 1:
                raise AggregateTradeParseError(
                    f"{archive_path}: archive must contain exactly one CSV."
                )
            with archive.open(members[0]) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as stream:
                    yield from iter_aggtrade_csv(
                        stream, source=f"{archive_path}!{members[0]}"
                    )
    except AggregateTradeParseError:
        raise
    except (OSError, UnicodeError, zipfile.BadZipFile) as exc:
        raise AggregateTradeParseError(
            f"{archive_path}: could not read aggTrades ZIP."
        ) from exc
