from __future__ import annotations

import zipfile
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from src.cli.validate_v7_orderflow_sample import main
from src.models.candle import Candle
from src.orderflow.aggregation import OrderFlowBucket
from src.orderflow.reconciliation import (
    BinanceKlineFlow,
    OrderFlowValidationError,
    parse_kline_archive,
    reconcile_orderflow_day,
    require_safe_day,
    select_earliest_complete_consumed_day,
)


DAY = datetime(2024, 1, 2, tzinfo=timezone.utc)


def candle(timestamp: datetime, *, volume: str = "10.00000000") -> Candle:
    return Candle(
        timestamp=timestamp,
        symbol="BTCUSDC",
        interval="15m",
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100"),
        volume=Decimal(volume),
        is_closed=True,
    )


def bucket(index: int) -> OrderFlowBucket:
    open_time = DAY + timedelta(minutes=15 * index)
    return OrderFlowBucket(
        bucket_open_time=open_time,
        bucket_close_time=open_time + timedelta(minutes=15),
        aggregate_trade_count=2,
        underlying_trade_count=3,
        total_base_volume=Decimal("10.00000000"),
        total_quote_volume=Decimal("1000.00000000"),
        taker_buy_base_volume=Decimal("6.00000000"),
        taker_buy_quote_volume=Decimal("600.00000000"),
        taker_buy_aggtrade_count=1,
        taker_sell_base_volume=Decimal("4.00000000"),
        taker_sell_quote_volume=Decimal("400.00000000"),
        taker_sell_aggtrade_count=1,
    )


def kline(index: int) -> BinanceKlineFlow:
    return BinanceKlineFlow(
        open_time=DAY + timedelta(minutes=15 * index),
        base_volume=Decimal("10.00000000"),
        quote_volume=Decimal("1000.00000000"),
        trade_count=3,
        taker_buy_base_volume=Decimal("6.00000000"),
        taker_buy_quote_volume=Decimal("600.00000000"),
    )


def complete_inputs() -> tuple[
    tuple[OrderFlowBucket, ...],
    tuple[BinanceKlineFlow, ...],
    tuple[Candle, ...],
]:
    return (
        tuple(bucket(index) for index in range(96)),
        tuple(kline(index) for index in range(96)),
        tuple(candle(DAY + timedelta(minutes=15 * index)) for index in range(96)),
    )


def test_exact_96_bucket_timestamp_join_trade_counts_taker_flow_and_totals() -> None:
    result = reconcile_orderflow_day(*complete_inputs())

    assert result.classification == "VALIDATED"
    assert result.bucket_count == result.kline_count == 96
    assert result.trade_counts_exact
    assert result.daily_trade_count_exact
    assert result.taker_buy_base_reconciled
    assert result.taker_buy_quote_reconciled
    assert result.conservation_identities_pass
    assert result.local_kline_base_volume_reconciled
    assert all(
        item.within_serialization_tolerance
        for item in result.daily_financial_metrics.values()
    )


def test_deliberately_reversed_taker_convention_fails_sanity_check() -> None:
    result = reconcile_orderflow_day(*complete_inputs())

    assert result.reversed_side_fails
    assert result.taker_buy_base_reconciled
    assert result.taker_buy_quote_reconciled


def test_decimal_absolute_relative_and_exact_match_statistics_are_reported() -> None:
    buckets, klines, candles = complete_inputs()
    modified = (replace(klines[0], base_volume=Decimal("9.99999999")), *klines[1:])

    result = reconcile_orderflow_day(buckets, modified, candles)
    summary = result.financial_metrics["total_base_volume"]

    assert result.classification == "VALIDATED"
    assert summary.maximum_absolute_difference == Decimal("0.00000001")
    assert summary.maximum_relative_difference is not None
    assert summary.maximum_relative_difference > 0
    assert summary.median_absolute_difference == 0
    assert summary.exact_decimal_matches == 95
    assert summary.within_serialization_tolerance


def test_material_volume_discrepancy_fails_validation() -> None:
    buckets, klines, candles = complete_inputs()
    modified = (replace(klines[0], quote_volume=Decimal("999.99")), *klines[1:])

    result = reconcile_orderflow_day(buckets, modified, candles)

    assert result.classification == "VALIDATION_FAILED"
    assert "total_quote_volume" in result.discrepancy_categories
    assert "daily_total_quote_volume" in result.discrepancy_categories


def test_wrong_bucket_count_and_timestamp_join_are_rejected() -> None:
    buckets, klines, candles = complete_inputs()
    with pytest.raises(OrderFlowValidationError, match="exactly 96"):
        reconcile_orderflow_day(buckets[:-1], klines, candles)
    shifted = (replace(klines[0], open_time=DAY + timedelta(seconds=1)), *klines[1:])
    with pytest.raises(OrderFlowValidationError, match="timestamp join"):
        reconcile_orderflow_day(buckets, shifted, candles)


def test_conservation_identity_is_enforced_by_bucket_model() -> None:
    valid = bucket(0)
    with pytest.raises(ValueError, match="base volume does not reconcile"):
        replace(valid, taker_sell_base_volume=Decimal("3"))


def test_holdout_day_is_rejected() -> None:
    with pytest.raises(OrderFlowValidationError, match="blind holdout"):
        require_safe_day(date(2025, 8, 1))


def test_safe_date_selection_is_earliest_complete_consumed_day() -> None:
    incomplete = tuple(
        candle(DAY - timedelta(days=1) + timedelta(minutes=15 * index))
        for index in range(95)
    )
    complete = tuple(
        candle(DAY + timedelta(minutes=15 * index)) for index in range(96)
    )
    ranges = ((DAY - timedelta(days=1), DAY + timedelta(days=1)),)

    assert select_earliest_complete_consumed_day(incomplete + complete, ranges) == DAY.date()


def test_kline_archive_parser_uses_named_binance_native_flow_fields(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "BTCUSDC-15m-2024-01-02.zip"
    timestamp_ms = int(DAY.timestamp() * 1_000)
    row = (
        f"{timestamp_ms},100,101,99,100,10.00000000,"
        f"{timestamp_ms + 899999},1000.00000000,3,"
        "6.00000000,600.00000000,0\n"
    )
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("BTCUSDC-15m-2024-01-02.csv", row)

    parsed = parse_kline_archive(archive)

    assert parsed == (kline(0),)


def test_cli_defaults_to_no_network_validation(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "NO VALIDATION EXECUTED" in capsys.readouterr().out
