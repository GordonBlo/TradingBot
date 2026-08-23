"""Strict parsers for confirmed Binance public USD-M archive schemas."""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Iterable
from pathlib import Path

from src.derivatives.models import (
    DerivativesPriceKline,
    DerivativesSource,
    FundingRateRecord,
    FuturesMetricsRecord,
    decimal_value,
    utc_timestamp,
)
from src.derivatives.sources import KLINE_SCHEMA


class DerivativesParseError(ValueError):
    pass


def _archive_rows(path: str | Path) -> tuple[tuple[str, ...], ...]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = tuple(name for name in archive.namelist() if name.lower().endswith(".csv"))
            if len(members) != 1:
                raise DerivativesParseError("Derivatives archive must contain exactly one CSV.")
            text = archive.read(members[0]).decode("utf-8-sig")
    except DerivativesParseError:
        raise
    except (OSError, UnicodeDecodeError, zipfile.BadZipFile, KeyError) as exc:
        raise DerivativesParseError("Derivatives archive could not be read safely.") from exc
    return tuple(tuple(cell.strip() for cell in row) for row in csv.reader(io.StringIO(text)) if row)


def _timestamp(value: str, *, preferred_unit: str) -> object:
    unit = "milliseconds" if value.lstrip("-").isdigit() else preferred_unit
    return utc_timestamp(value, unit=unit)


def parse_funding_archive(path: str | Path) -> tuple[FundingRateRecord, ...]:
    rows = _archive_rows(path)
    if not rows:
        return ()
    header = tuple(name.lower() for name in rows[0])
    expected = ("calc_time", "funding_interval_hours", "last_funding_rate")
    if header != expected:
        raise DerivativesParseError("Funding archive schema is unsupported.")
    output = []
    for row in rows[1:]:
        if len(row) != 3:
            raise DerivativesParseError("Funding archive row width changed.")
        output.append(
            FundingRateRecord(
                timestamp=_timestamp(row[0], preferred_unit="iso8601"),
                funding_interval_hours=decimal_value("funding_interval_hours", row[1]),
                funding_rate=decimal_value("last_funding_rate", row[2]),
            )
        )
    return tuple(output)


METRICS_SCHEMA = (
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)


def _optional_decimal(value: str):
    return decimal_value("optional metric", value) if value != "" else None


def parse_metrics_archive(path: str | Path) -> tuple[FuturesMetricsRecord, ...]:
    rows = _archive_rows(path)
    if not rows:
        return ()
    if tuple(name.lower() for name in rows[0]) != METRICS_SCHEMA:
        raise DerivativesParseError("Metrics archive schema is unsupported.")
    output = []
    for row in rows[1:]:
        if len(row) != len(METRICS_SCHEMA):
            raise DerivativesParseError("Metrics archive row width changed.")
        output.append(
            FuturesMetricsRecord(
                timestamp=_timestamp(row[0], preferred_unit="iso8601"),
                symbol=row[1],
                sum_open_interest=decimal_value("sum_open_interest", row[2]),
                sum_open_interest_value=decimal_value("sum_open_interest_value", row[3]),
                count_toptrader_long_short_ratio=_optional_decimal(row[4]),
                sum_toptrader_long_short_ratio=_optional_decimal(row[5]),
                count_long_short_ratio=_optional_decimal(row[6]),
                sum_taker_long_short_vol_ratio=_optional_decimal(row[7]),
            )
        )
    return tuple(output)


def parse_price_kline_archive(
    path: str | Path, *, source: DerivativesSource
) -> tuple[DerivativesPriceKline, ...]:
    if source not in {
        DerivativesSource.MARK_PRICE,
        DerivativesSource.INDEX_PRICE,
        DerivativesSource.PREMIUM_INDEX,
    }:
        raise DerivativesParseError("Price archive source is invalid.")
    rows = list(_archive_rows(path))
    if rows and tuple(name.lower().replace(" ", "_") for name in rows[0]) == KLINE_SCHEMA:
        rows.pop(0)
    output = []
    for row in rows:
        if len(row) != len(KLINE_SCHEMA):
            raise DerivativesParseError("Price kline archive row width changed.")
        output.append(
            DerivativesPriceKline(
                source=source,
                open_time=utc_timestamp(row[0], unit="milliseconds"),
                open=decimal_value("open", row[1]),
                high=decimal_value("high", row[2]),
                low=decimal_value("low", row[3]),
                close=decimal_value("close", row[4]),
                close_time=utc_timestamp(row[6], unit="milliseconds"),
            )
        )
    return tuple(output)


def exact_records_duplicated(records: Iterable[object]) -> int:
    values = tuple(records)
    return len(values) - len(set(values))

