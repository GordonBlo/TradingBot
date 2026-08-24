"""Frozen official Binance USD-M public archive inventory and sample locations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.derivatives.models import DerivativesSource
from src.orderflow.archive import ArchiveLocation


@dataclass(frozen=True, slots=True)
class OfficialSourceDefinition:
    source: DerivativesSource
    official_path_name: str
    daily_available: bool
    monthly_available: bool
    schema: tuple[str, ...]
    timestamp_unit: str
    earliest_safe_date: str | None
    checksum_available: bool = True


KLINE_SCHEMA = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)

OFFICIAL_SOURCES = (
    OfficialSourceDefinition(
        DerivativesSource.FUNDING_RATE,
        "data/futures/um/monthly/fundingRate/BTCUSDT",
        False,
        True,
        ("calc_time", "funding_interval_hours", "last_funding_rate"),
        "milliseconds or archived ISO UTC",
        "2020-01-01",
    ),
    OfficialSourceDefinition(
        DerivativesSource.METRICS,
        "data/futures/um/daily/metrics/BTCUSDT",
        True,
        False,
        (
            "create_time",
            "symbol",
            "sum_open_interest",
            "sum_open_interest_value",
            "count_toptrader_long_short_ratio",
            "sum_toptrader_long_short_ratio",
            "count_long_short_ratio",
            "sum_taker_long_short_vol_ratio",
        ),
        "archived ISO UTC/date",
        None,
    ),
    *(
        OfficialSourceDefinition(
            source,
            f"data/futures/um/daily/{source.value}/BTCUSDT/15m",
            True,
            True,
            KLINE_SCHEMA,
            "milliseconds",
            "2020-01-01",
        )
        for source in (
            DerivativesSource.MARK_PRICE,
            DerivativesSource.INDEX_PRICE,
            DerivativesSource.PREMIUM_INDEX,
        )
    ),
)


def source_definition(source: DerivativesSource) -> OfficialSourceDefinition:
    try:
        return next(item for item in OFFICIAL_SOURCES if item.source is source)
    except StopIteration as exc:
        raise ValueError(f"Unsupported official derivatives source: {source}") from exc


def build_sample_locations(
    sample_date: date, *, root: str | Path = "data/derivatives/raw"
) -> tuple[tuple[DerivativesSource, ArchiveLocation], ...]:
    root = Path(root)
    base = "https://data.binance.vision/data/futures/um"
    output = []
    for source in DerivativesSource:
        if source is DerivativesSource.FUNDING_RATE:
            cadence = "monthly"
            period = sample_date.strftime("%Y-%m")
            filename = f"BTCUSDT-fundingRate-{period}.zip"
            remote = f"{base}/{cadence}/{source.value}/BTCUSDT/{filename}"
            local = root / "um" / source.value / "BTCUSDT" / cadence / str(sample_date.year) / filename
        elif source is DerivativesSource.METRICS:
            cadence = "daily"
            filename = f"BTCUSDT-metrics-{sample_date.isoformat()}.zip"
            remote = f"{base}/{cadence}/{source.value}/BTCUSDT/{filename}"
            local = root / "um" / source.value / "BTCUSDT" / cadence / f"{sample_date.year:04d}" / f"{sample_date.month:02d}" / filename
        else:
            cadence = "daily"
            filename = f"BTCUSDT-15m-{sample_date.isoformat()}.zip"
            remote = f"{base}/{cadence}/{source.value}/BTCUSDT/15m/{filename}"
            local = root / "um" / source.value / "BTCUSDT" / "15m" / cadence / f"{sample_date.year:04d}" / f"{sample_date.month:02d}" / filename
        output.append(
            (
                source,
                ArchiveLocation(
                    url=remote,
                    checksum_url=f"{remote}.CHECKSUM",
                    destination=local,
                    checksum_destination=local.with_suffix(".zip.CHECKSUM"),
                ),
            )
        )
    return tuple(output)


def build_archive_location(
    source: DerivativesSource,
    period: date,
    *,
    cadence: str,
    root: str | Path = "data/derivatives/raw",
) -> ArchiveLocation:
    """Build one fixed official USD-M archive location."""

    definition = source_definition(source)
    if cadence == "daily" and not definition.daily_available:
        raise ValueError(f"{source.value} has no official daily archive.")
    if cadence == "monthly" and not definition.monthly_available:
        raise ValueError(f"{source.value} has no official monthly archive.")
    if cadence not in {"daily", "monthly"}:
        raise ValueError("Derivatives archive cadence is invalid.")
    period_text = period.isoformat() if cadence == "daily" else period.strftime("%Y-%m")
    base = "https://data.binance.vision/data/futures/um"
    if source is DerivativesSource.FUNDING_RATE:
        filename = f"BTCUSDT-fundingRate-{period_text}.zip"
        remote = f"{base}/{cadence}/{source.value}/BTCUSDT/{filename}"
        local = Path(root) / "um" / source.value / "BTCUSDT" / cadence / f"{period.year:04d}" / filename
    elif source is DerivativesSource.METRICS:
        filename = f"BTCUSDT-metrics-{period_text}.zip"
        remote = f"{base}/{cadence}/{source.value}/BTCUSDT/{filename}"
        local = Path(root) / "um" / source.value / "BTCUSDT" / cadence / f"{period.year:04d}" / f"{period.month:02d}" / filename
    else:
        filename = f"BTCUSDT-15m-{period_text}.zip"
        remote = f"{base}/{cadence}/{source.value}/BTCUSDT/15m/{filename}"
        tail = (f"{period.year:04d}",) if cadence == "monthly" else (f"{period.year:04d}", f"{period.month:02d}")
        local = Path(root) / "um" / source.value / "BTCUSDT" / "15m" / cadence / Path(*tail) / filename
    return ArchiveLocation(
        url=remote,
        checksum_url=f"{remote}.CHECKSUM",
        destination=local,
        checksum_destination=local.with_suffix(".zip.CHECKSUM"),
    )
