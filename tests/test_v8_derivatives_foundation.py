from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.cli.validate_v8_derivatives_sample import (
    _plain,
    reject_holdout_day,
    select_safe_sample_date,
)
from src.derivatives.context import build_context_15m
from src.derivatives.integrity import (
    CoverageClassification,
    analyze_coverage,
    validate_archive_boundary,
)
from src.derivatives.models import (
    DerivativesContext15m,
    DerivativesPriceKline,
    DerivativesSource,
    FundingRateRecord,
    FuturesMetricsRecord,
    utc_timestamp,
)
from src.derivatives.parser import (
    DerivativesParseError,
    parse_funding_archive,
    parse_metrics_archive,
    parse_price_kline_archive,
)
from src.derivatives.sources import OFFICIAL_SOURCES, source_definition
from src.models.candle import Candle
from src.orderflow.integrity import verify_sha256


BASE = datetime(2023, 3, 13, tzinfo=timezone.utc)


def candle(index: int) -> Candle:
    return Candle(
        timestamp=BASE + index * timedelta(minutes=15),
        symbol="BTCUSDC",
        interval="15m",
        open=Decimal("20000"),
        high=Decimal("20001"),
        low=Decimal("19999"),
        close=Decimal("20000"),
        volume=Decimal("1"),
        is_closed=True,
    )


def price_kline(index: int, source=DerivativesSource.MARK_PRICE, close="20000"):
    open_time = BASE + index * timedelta(minutes=15)
    value = Decimal(close)
    return DerivativesPriceKline(
        source=source,
        open_time=open_time,
        close_time=open_time + timedelta(minutes=15) - timedelta(milliseconds=1),
        open=value,
        high=value,
        low=value,
        close=value,
    )


def archive(tmp_path, name: str, text: str):
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as output:
        output.writestr(name.replace(".zip", ".csv"), text)
    return path


def test_supported_source_inventory_and_unsupported_behavior() -> None:
    assert {item.source for item in OFFICIAL_SOURCES} == set(DerivativesSource)
    assert source_definition(DerivativesSource.METRICS).daily_available is True
    assert source_definition(DerivativesSource.FUNDING_RATE).monthly_available is True
    with pytest.raises(ValueError, match="Unsupported"):
        source_definition("unknown")  # type: ignore[arg-type]


def test_funding_decimal_and_timestamp_parsing(tmp_path) -> None:
    path = archive(
        tmp_path,
        "funding.zip",
        "calc_time,funding_interval_hours,last_funding_rate\n"
        "1678665600000,8,0.00010000\n",
    )
    row = parse_funding_archive(path)[0]
    assert row.timestamp == BASE
    assert row.funding_interval_hours == Decimal("8")
    assert row.funding_rate == Decimal("0.00010000")


def test_metrics_decimal_parsing_and_raw_optional_fields(tmp_path) -> None:
    path = archive(
        tmp_path,
        "metrics.zip",
        "create_time,symbol,sum_open_interest,sum_open_interest_value,"
        "count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,"
        "count_long_short_ratio,sum_taker_long_short_vol_ratio\n"
        "2023-03-13,BTCUSDT,100.5,200.25,0.8,1.1,0.9,1.2\n",
    )
    row = parse_metrics_archive(path)[0]
    assert row.timestamp == BASE
    assert row.sum_open_interest == Decimal("100.5")
    assert row.sum_taker_long_short_vol_ratio == Decimal("1.2")


@pytest.mark.parametrize(
    "source,value",
    (
        (DerivativesSource.MARK_PRICE, "20000.1"),
        (DerivativesSource.INDEX_PRICE, "19999.9"),
        (DerivativesSource.PREMIUM_INDEX, "0.00001"),
    ),
)
def test_price_archive_decimal_parsing(tmp_path, source, value) -> None:
    close_ms = 1678666499999
    text = f"1678665600000,{value},{value},{value},{value},0,{close_ms},0,0,0,0,0\n"
    row = parse_price_kline_archive(archive(tmp_path, f"{source.value}.zip", text), source=source)[0]
    assert row.close == Decimal(value)
    assert row.open_time == BASE


def test_official_special_kline_header_schema(tmp_path) -> None:
    header = (
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
        "taker_buy_volume,taker_buy_quote_volume,ignore\n"
    )
    data = "1678665600000,1,1,1,1,0,1678666499999,0,900,0,0,0\n"
    rows = parse_price_kline_archive(
        archive(tmp_path, "mark.zip", header + data),
        source=DerivativesSource.MARK_PRICE,
    )
    assert len(rows) == 1


def test_timestamp_units_are_explicit_and_invalid_values_rejected() -> None:
    assert utc_timestamp("1678665600000", unit="milliseconds") == BASE
    assert utc_timestamp("2023-03-13", unit="iso8601") == BASE
    with pytest.raises(ValueError, match="explicit"):
        utc_timestamp("1678665600000", unit="seconds")
    with pytest.raises(ValueError, match="impossible"):
        utc_timestamp("1", unit="milliseconds")


def test_unknown_archive_schema_rejected(tmp_path) -> None:
    path = archive(tmp_path, "bad.zip", "wrong,columns\n1,2\n")
    with pytest.raises(DerivativesParseError, match="schema"):
        parse_funding_archive(path)


def test_checksum_verification(tmp_path) -> None:
    path = tmp_path / "sample.zip"
    path.write_bytes(b"official")
    digest = hashlib.sha256(b"official").hexdigest()
    assert verify_sha256(path, f"{digest}  sample.zip\n") == digest
    with pytest.raises(ValueError, match="does not match"):
        verify_sha256(path, f"{'0' * 64}  sample.zip\n")


def test_duplicate_gap_and_archive_boundary_diagnostics() -> None:
    records = (price_kline(0), price_kline(0), price_kline(2))
    result = analyze_coverage(
        records,
        source="mark",
        timestamp_field="open_time",
        expected_step=timedelta(minutes=15),
        expected_count=3,
    )
    assert result.classification is CoverageClassification.DUPLICATED
    assert result.duplicate_timestamps == 1
    assert result.missing_intervals == 1
    issues = validate_archive_boundary(
        (price_kline(0),), (price_kline(0),), timestamp_field="open_time"
    )
    assert issues[0].issue_type == "DUPLICATE_TIMESTAMP"


def test_causal_context_allows_at_close_preserves_timestamp_and_blocks_future() -> None:
    funding = FundingRateRecord(BASE + timedelta(minutes=15), Decimal("0.001"))
    future = FundingRateRecord(BASE + timedelta(minutes=16), Decimal("0.002"))
    contexts = build_context_15m(
        spot_candles=(candle(0),), funding=(funding, future), mark=(price_kline(0),)
    )
    context = contexts[0]
    assert context.latest_known_funding_rate == Decimal("0.001")
    assert context.funding_timestamp == BASE + timedelta(minutes=15)
    assert context.mark_price_timestamp < context.bucket_close_time
    with pytest.raises(ValueError, match="future"):
        DerivativesContext15m(
            bucket_open_time=BASE,
            bucket_close_time=BASE + timedelta(minutes=15),
            latest_known_funding_rate=Decimal("0.1"),
            funding_timestamp=BASE + timedelta(minutes=16),
            open_interest=None,
            open_interest_value=None,
            open_interest_timestamp=None,
            mark_price=None,
            mark_price_timestamp=None,
            index_price=None,
            index_price_timestamp=None,
            premium_index=None,
            premium_index_timestamp=None,
            premium_fraction=None,
        )


def test_missing_context_fields_remain_missing() -> None:
    context = build_context_15m(spot_candles=(candle(0),))[0]
    assert context.latest_known_funding_rate is None
    assert context.open_interest is None
    assert context.mark_price is None
    assert context.premium_fraction is None


def test_exact_96_bucket_safe_day_and_holdout_exclusion() -> None:
    candles = tuple(candle(index) for index in range(96))
    ranges = (
        {
            "start": BASE.isoformat(),
            "end": (BASE + timedelta(days=1)).isoformat(),
            "data_status": "CONSUMED_RESEARCH_DATA",
            "symbol": "BTCUSDC",
            "interval": "15m",
        },
    )
    assert select_safe_sample_date(candles, ranges) == date(2023, 3, 13)
    assert len(build_context_15m(spot_candles=candles)) == 96
    with pytest.raises(ValueError, match="holdout"):
        reject_holdout_day(date(2025, 8, 1))


def test_premium_math_and_zero_index_rejection() -> None:
    assert DerivativesContext15m.derive_premium(
        Decimal("101"), Decimal("100")
    ) == Decimal("0.01")
    with pytest.raises(ValueError, match="positive"):
        DerivativesContext15m.derive_premium(Decimal("1"), Decimal("0"))


def test_context_serialization_is_deterministic() -> None:
    context = build_context_15m(spot_candles=(candle(0),))[0]
    first = json.dumps(_plain(context.__dict__ if hasattr(context, "__dict__") else {
        name: getattr(context, name) for name in context.__slots__
    }), sort_keys=True)
    second = json.dumps(_plain({name: getattr(context, name) for name in context.__slots__}), sort_keys=True)
    assert first == second

def test_optimized_context_lookup_matches_reference_scan() -> None:
    spot = tuple(candle(index) for index in range(8))

    funding = (
        FundingRateRecord(
            BASE + timedelta(minutes=15),
            Decimal("0.001"),
        ),
        FundingRateRecord(
            BASE + timedelta(minutes=45),
            Decimal("0.002"),
        ),
        FundingRateRecord(
            BASE + timedelta(minutes=90),
            Decimal("0.003"),
        ),
    )

    mark = tuple(
        price_kline(
            index,
            DerivativesSource.MARK_PRICE,
            str(20000 + index),
        )
        for index in range(8)
    )

    contexts = build_context_15m(
        spot_candles=spot,
        funding=funding,
        mark=mark,
    )

    for candle_row, context in zip(
        spot,
        contexts,
        strict=True,
    ):
        cutoff = (
            candle_row.timestamp
            + timedelta(minutes=15)
        )

        eligible_funding = tuple(
            row
            for row in funding
            if row.timestamp <= cutoff
        )

        expected_funding = (
            eligible_funding[-1]
            if eligible_funding
            else None
        )

        eligible_mark = tuple(
            row
            for row in mark
            if row.close_time <= cutoff
        )

        expected_mark = (
            eligible_mark[-1]
            if eligible_mark
            else None
        )

        assert context.latest_known_funding_rate == (
            expected_funding.funding_rate
            if expected_funding
            else None
        )

        assert context.funding_timestamp == (
            expected_funding.timestamp
            if expected_funding
            else None
        )

        assert context.mark_price == (
            expected_mark.close
            if expected_mark
            else None
        )

        assert context.mark_price_timestamp == (
            expected_mark.close_time
            if expected_mark
            else None
        )