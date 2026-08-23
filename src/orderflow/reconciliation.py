"""One-day, non-predictive aggTrades-to-kline integrity reconciliation."""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import median

from src.market.intervals import is_interval_aligned
from src.models.candle import Candle
from src.orderflow.aggregation import OrderFlowBucket
from src.orderflow.timestamps import normalize_binance_timestamp


EXPECTED_BUCKETS = 96
HOLDOUT_START = datetime(2025, 8, 1, tzinfo=timezone.utc)
HOLDOUT_END = datetime(2026, 2, 1, tzinfo=timezone.utc)


class OrderFlowValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BinanceKlineFlow:
    open_time: datetime
    base_volume: Decimal
    quote_volume: Decimal
    trade_count: int
    taker_buy_base_volume: Decimal
    taker_buy_quote_volume: Decimal


@dataclass(frozen=True, slots=True)
class DifferenceSummary:
    maximum_absolute_difference: Decimal
    maximum_relative_difference: Decimal | None
    median_absolute_difference: Decimal
    exact_decimal_matches: int
    compared_buckets: int
    maximum_allowed_absolute_difference: Decimal
    within_serialization_tolerance: bool


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    classification: str
    bucket_count: int
    kline_count: int
    financial_metrics: dict[str, DifferenceSummary]
    daily_financial_metrics: dict[str, DifferenceSummary]
    trade_counts_exact: bool
    daily_trade_count_exact: bool
    taker_buy_base_reconciled: bool
    taker_buy_quote_reconciled: bool
    reversed_side_fails: bool
    conservation_identities_pass: bool
    local_kline_base_volume_reconciled: bool
    discrepancy_categories: tuple[str, ...]
    bucket_rows: tuple[dict[str, object], ...]


def _day_bounds(day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time.min, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def require_safe_day(day: date) -> None:
    start, end = _day_bounds(day)
    if start < HOLDOUT_END and HOLDOUT_START < end:
        raise OrderFlowValidationError("Selected day overlaps the locked blind holdout.")


def select_earliest_complete_consumed_day(
    candles: tuple[Candle, ...],
    consumed_ranges: tuple[tuple[datetime, datetime], ...],
) -> date:
    """Select by chronology/completeness only, never by market values."""

    by_day: dict[date, list[datetime]] = {}
    for candle in candles:
        by_day.setdefault(candle.timestamp.date(), []).append(candle.timestamp)
    for day in sorted(by_day):
        require_safe_day(day)
        start, end = _day_bounds(day)
        if not any(range_start <= start and end <= range_end for range_start, range_end in consumed_ranges):
            continue
        expected = tuple(start + timedelta(minutes=15 * index) for index in range(96))
        if tuple(sorted(by_day[day])) == expected:
            return day
    raise OrderFlowValidationError("No complete safe consumed-data UTC day exists.")


def _parse_kline_rows(rows: list[list[str]], *, source: str) -> tuple[BinanceKlineFlow, ...]:
    parsed: list[BinanceKlineFlow] = []
    for row_number, row in enumerate(rows, start=1):
        if not row or all(not item.strip() for item in row):
            continue
        if row_number == 1 and row[0].strip().lower().lstrip("\ufeff") in {
            "open_time",
            "open time",
        }:
            continue
        if len(row) < 11:
            raise OrderFlowValidationError(
                f"{source}: kline row {row_number} has fewer than 11 fields."
            )
        try:
            item = BinanceKlineFlow(
                open_time=normalize_binance_timestamp(row[0]),
                base_volume=Decimal(row[5]),
                quote_volume=Decimal(row[7]),
                trade_count=int(row[8]),
                taker_buy_base_volume=Decimal(row[9]),
                taker_buy_quote_volume=Decimal(row[10]),
            )
        except (InvalidOperation, ValueError) as exc:
            raise OrderFlowValidationError(
                f"{source}: invalid kline row {row_number}."
            ) from exc
        if (
            not is_interval_aligned(item.open_time, "15m")
            or item.trade_count < 0
            or any(
                value < 0
                for value in (
                    item.base_volume,
                    item.quote_volume,
                    item.taker_buy_base_volume,
                    item.taker_buy_quote_volume,
                )
            )
        ):
            raise OrderFlowValidationError(
                f"{source}: invalid kline metrics at row {row_number}."
            )
        parsed.append(item)
    if any(later.open_time <= earlier.open_time for earlier, later in zip(parsed, parsed[1:])):
        raise OrderFlowValidationError(f"{source}: klines are not chronological.")
    return tuple(parsed)


def parse_kline_archive(path: str | Path) -> tuple[BinanceKlineFlow, ...]:
    archive_path = Path(path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = [
                name
                for name in archive.namelist()
                if not name.endswith("/") and name.lower().endswith(".csv")
            ]
            if len(members) != 1:
                raise OrderFlowValidationError(
                    f"{archive_path}: archive must contain exactly one kline CSV."
                )
            with archive.open(members[0]) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as stream:
                    return _parse_kline_rows(
                        list(csv.reader(stream)),
                        source=f"{archive_path}!{members[0]}",
                    )
    except OrderFlowValidationError:
        raise
    except (OSError, UnicodeError, zipfile.BadZipFile) as exc:
        raise OrderFlowValidationError(
            f"{archive_path}: could not read kline ZIP."
        ) from exc


def _quantum(reference: Decimal) -> Decimal:
    serialized = Decimal(1).scaleb(min(0, reference.as_tuple().exponent))
    return min(serialized, Decimal("0.00000001"))


def _difference_summary(
    actual: tuple[Decimal, ...], reference: tuple[Decimal, ...]
) -> DifferenceSummary:
    absolute = tuple(abs(left - right) for left, right in zip(actual, reference, strict=True))
    relative = tuple(
        difference / abs(right)
        for difference, right in zip(absolute, reference, strict=True)
        if right != 0
    )
    tolerances = tuple(_quantum(value) for value in reference)
    return DifferenceSummary(
        maximum_absolute_difference=max(absolute, default=Decimal("0")),
        maximum_relative_difference=max(relative) if relative else None,
        median_absolute_difference=median(absolute) if absolute else Decimal("0"),
        exact_decimal_matches=sum(value == 0 for value in absolute),
        compared_buckets=len(absolute),
        maximum_allowed_absolute_difference=max(tolerances, default=Decimal("0")),
        within_serialization_tolerance=all(
            difference <= tolerance
            for difference, tolerance in zip(absolute, tolerances, strict=True)
        ),
    )


def _daily_summary(actual: tuple[Decimal, ...], reference: tuple[Decimal, ...]) -> DifferenceSummary:
    actual_total = sum(actual, start=Decimal("0"))
    reference_total = sum(reference, start=Decimal("0"))
    difference = abs(actual_total - reference_total)
    tolerance = sum((_quantum(value) for value in reference), start=Decimal("0"))
    relative = difference / abs(reference_total) if reference_total else None
    return DifferenceSummary(
        maximum_absolute_difference=difference,
        maximum_relative_difference=relative,
        median_absolute_difference=difference,
        exact_decimal_matches=int(difference == 0),
        compared_buckets=1,
        maximum_allowed_absolute_difference=tolerance,
        within_serialization_tolerance=difference <= tolerance,
    )


def reconcile_orderflow_day(
    buckets: tuple[OrderFlowBucket, ...],
    klines: tuple[BinanceKlineFlow, ...],
    local_candles: tuple[Candle, ...],
) -> ReconciliationResult:
    if len(buckets) != EXPECTED_BUCKETS or len(klines) != EXPECTED_BUCKETS:
        raise OrderFlowValidationError("Expected exactly 96 order-flow and kline buckets.")
    if len(local_candles) != EXPECTED_BUCKETS:
        raise OrderFlowValidationError("Expected exactly 96 existing local candles.")
    return reconcile_orderflow_dataset(buckets, klines, local_candles)


def reconcile_orderflow_dataset(
    buckets: tuple[OrderFlowBucket, ...],
    klines: tuple[BinanceKlineFlow, ...],
    local_candles: tuple[Candle, ...],
    *,
    include_bucket_rows: bool = True,
) -> ReconciliationResult:
    if not buckets or len(buckets) != len(klines) or len(buckets) != len(local_candles):
        raise OrderFlowValidationError(
            "Order-flow, kline, and local-candle coverage counts differ."
        )
    bucket_times = tuple(item.bucket_open_time for item in buckets)
    if bucket_times != tuple(item.open_time for item in klines) or bucket_times != tuple(
        item.timestamp for item in local_candles
    ):
        raise OrderFlowValidationError("Exact UTC timestamp join failed.")

    field_pairs = {
        "total_base_volume": (
            tuple(item.total_base_volume for item in buckets),
            tuple(item.base_volume for item in klines),
        ),
        "total_quote_volume": (
            tuple(item.total_quote_volume for item in buckets),
            tuple(item.quote_volume for item in klines),
        ),
        "taker_buy_base_volume": (
            tuple(item.taker_buy_base_volume for item in buckets),
            tuple(item.taker_buy_base_volume for item in klines),
        ),
        "taker_buy_quote_volume": (
            tuple(item.taker_buy_quote_volume for item in buckets),
            tuple(item.taker_buy_quote_volume for item in klines),
        ),
    }
    financial = {
        name: _difference_summary(*values) for name, values in field_pairs.items()
    }
    daily = {name: _daily_summary(*values) for name, values in field_pairs.items()}
    trade_counts_exact = all(
        bucket.underlying_trade_count == kline.trade_count
        for bucket, kline in zip(buckets, klines, strict=True)
    )
    daily_trade_count_exact = sum(item.underlying_trade_count for item in buckets) == sum(
        item.trade_count for item in klines
    )
    local_base = _difference_summary(
        tuple(item.volume for item in local_candles),
        tuple(item.base_volume for item in klines),
    ).within_serialization_tolerance
    reversed_base = _difference_summary(
        tuple(item.taker_sell_base_volume for item in buckets),
        tuple(item.taker_buy_base_volume for item in klines),
    )
    reversed_quote = _difference_summary(
        tuple(item.taker_sell_quote_volume for item in buckets),
        tuple(item.taker_buy_quote_volume for item in klines),
    )
    reversed_fails = not (
        reversed_base.within_serialization_tolerance
        and reversed_quote.within_serialization_tolerance
    )
    ratio_tolerance = Decimal("1e-27")
    conservation = all(
        bucket.taker_buy_base_volume + bucket.taker_sell_base_volume
        == bucket.total_base_volume
        and bucket.taker_buy_quote_volume + bucket.taker_sell_quote_volume
        == bucket.total_quote_volume
        and Decimal("-1") <= bucket.base_volume_imbalance <= Decimal("1")
        and Decimal("-1") <= bucket.quote_volume_imbalance <= Decimal("1")
        and abs(
            bucket.taker_buy_base_ratio
            + bucket.taker_sell_base_volume / bucket.total_base_volume
            - Decimal("1")
        )
        <= ratio_tolerance
        and abs(
            bucket.taker_buy_quote_ratio
            + bucket.taker_sell_quote_volume / bucket.total_quote_volume
            - Decimal("1")
        )
        <= ratio_tolerance
        for bucket in buckets
    )
    discrepancies: list[str] = []
    discrepancies.extend(
        name
        for name, result in financial.items()
        if not result.within_serialization_tolerance
    )
    discrepancies.extend(
        f"daily_{name}"
        for name, result in daily.items()
        if not result.within_serialization_tolerance
    )
    if not trade_counts_exact or not daily_trade_count_exact:
        discrepancies.append("underlying_trade_count")
    if not local_base:
        discrepancies.append("existing_local_kline_base_volume")
    if not reversed_fails:
        discrepancies.append("taker_side_not_independently_distinguishable")
    if not conservation:
        discrepancies.append("conservation_identity")
    rows = tuple(
        {
            "bucket_open_time": bucket.bucket_open_time,
            "aggtrade_underlying_trade_count": bucket.underlying_trade_count,
            "kline_trade_count": kline.trade_count,
            **{
                f"{name}_{suffix}": value
                for name, (actual_values, reference_values) in field_pairs.items()
                for suffix, value in (
                    ("aggtrade", actual_values[index]),
                    ("kline", reference_values[index]),
                    ("absolute_difference", abs(actual_values[index] - reference_values[index])),
                    (
                        "relative_difference",
                        abs(actual_values[index] - reference_values[index])
                        / abs(reference_values[index])
                        if reference_values[index]
                        else None,
                    ),
                )
            },
        }
        for index, (bucket, kline) in enumerate(zip(buckets, klines, strict=True))
    ) if include_bucket_rows else ()
    return ReconciliationResult(
        classification="VALIDATED" if not discrepancies else "VALIDATION_FAILED",
        bucket_count=len(buckets),
        kline_count=len(klines),
        financial_metrics=financial,
        daily_financial_metrics=daily,
        trade_counts_exact=trade_counts_exact,
        daily_trade_count_exact=daily_trade_count_exact,
        taker_buy_base_reconciled=financial[
            "taker_buy_base_volume"
        ].within_serialization_tolerance,
        taker_buy_quote_reconciled=financial[
            "taker_buy_quote_volume"
        ].within_serialization_tolerance,
        reversed_side_fails=reversed_fails,
        conservation_identities_pass=conservation,
        local_kline_base_volume_reconciled=local_base,
        discrepancy_categories=tuple(discrepancies),
        bucket_rows=rows,
    )
