"""Bounded unauthenticated acquisition of official Binance aggTrades archives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from src.orderflow.integrity import AggregateTradeIntegrityError, verify_sha256


class AggregateTradeArchiveError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ArchiveLocation:
    url: str
    checksum_url: str
    destination: Path
    checksum_destination: Path


@dataclass(frozen=True, slots=True)
class ArchiveDownloadResult:
    path: Path
    checksum_verified: bool
    skipped_existing: bool


def build_archive_location(
    period: date,
    *,
    cadence: str,
    symbol: str = "BTCUSDC",
    root: str | Path = "data/orderflow/raw",
) -> ArchiveLocation:
    symbol = symbol.strip().upper()
    if symbol != "BTCUSDC":
        raise ValueError("V7 order-flow foundation supports BTCUSDC only.")
    if cadence == "daily":
        period_text = period.isoformat()
        destination_dir = Path(root) / symbol / "daily" / f"{period.year:04d}" / f"{period.month:02d}"
    elif cadence == "monthly":
        period_text = f"{period.year:04d}-{period.month:02d}"
        destination_dir = Path(root) / symbol / "monthly" / f"{period.year:04d}"
    else:
        raise ValueError("Archive cadence must be 'daily' or 'monthly'.")
    filename = f"{symbol}-aggTrades-{period_text}.zip"
    url = (
        "https://data.binance.vision/data/spot/"
        f"{cadence}/aggTrades/{symbol}/{filename}"
    )
    destination = destination_dir / filename
    return ArchiveLocation(
        url=url,
        checksum_url=f"{url}.CHECKSUM",
        destination=destination,
        checksum_destination=destination.with_suffix(".zip.CHECKSUM"),
    )


def build_kline_archive_location(
    period: date,
    *,
    interval: str = "15m",
    symbol: str = "BTCUSDC",
    root: str | Path = "data/orderflow/raw",
) -> ArchiveLocation:
    """Build the official one-day kline reference used only for reconciliation."""

    symbol = symbol.strip().upper()
    if symbol != "BTCUSDC" or interval != "15m":
        raise ValueError("V7 validation supports BTCUSDC 15m only.")
    filename = f"{symbol}-{interval}-{period.isoformat()}.zip"
    url = (
        "https://data.binance.vision/data/spot/daily/klines/"
        f"{symbol}/{interval}/{filename}"
    )
    destination = (
        Path(root)
        / symbol
        / "validation_klines"
        / "daily"
        / f"{period.year:04d}"
        / f"{period.month:02d}"
        / filename
    )
    return ArchiveLocation(
        url=url,
        checksum_url=f"{url}.CHECKSUM",
        destination=destination,
        checksum_destination=destination.with_suffix(".zip.CHECKSUM"),
    )


Fetcher = Callable[[str, float], bytes]


def _fetch(url: str, timeout: float) -> bytes:
    try:
        with urlopen(url, timeout=timeout) as response:  # nosec: official fixed host
            return response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise AggregateTradeArchiveError(
            f"Public Binance archive request failed: {url}"
        ) from exc


class AggregateTradeArchiveDownloader:
    """Download one explicitly requested public archive; never scans periods."""

    def __init__(self, *, fetcher: Fetcher = _fetch, timeout_seconds: float = 30.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Archive timeout must be positive.")
        self._fetcher = fetcher
        self.timeout_seconds = timeout_seconds

    def download(
        self, location: ArchiveLocation, *, verify_checksum: bool = True
    ) -> ArchiveDownloadResult:
        checksum_bytes: bytes | None = None
        if verify_checksum:
            try:
                checksum_bytes = self._fetcher(
                    location.checksum_url, self.timeout_seconds
                )
            except AggregateTradeArchiveError:
                raise
            except Exception as exc:
                raise AggregateTradeArchiveError(
                    "Public Binance CHECKSUM request failed."
                ) from exc
            try:
                checksum_text = checksum_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise AggregateTradeArchiveError("CHECKSUM is not UTF-8 text.") from exc
            if location.destination.is_file():
                try:
                    verify_sha256(location.destination, checksum_text)
                except AggregateTradeIntegrityError:
                    pass
                else:
                    location.checksum_destination.parent.mkdir(
                        parents=True, exist_ok=True
                    )
                    location.checksum_destination.write_bytes(checksum_bytes)
                    return ArchiveDownloadResult(location.destination, True, True)
        elif location.destination.exists():
            raise AggregateTradeArchiveError(
                "Existing archive cannot be skipped without checksum verification."
            )

        try:
            archive_bytes = self._fetcher(location.url, self.timeout_seconds)
        except AggregateTradeArchiveError:
            raise
        except Exception as exc:
            raise AggregateTradeArchiveError(
                "Public Binance archive request failed."
            ) from exc
        location.destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = location.destination.with_suffix(".zip.tmp")
        try:
            temporary.write_bytes(archive_bytes)
            if verify_checksum:
                assert checksum_bytes is not None
                checksum_text = checksum_bytes.decode("utf-8")
                verify_sha256(
                    temporary,
                    checksum_text,
                    expected_filename=location.destination.name,
                )
            temporary.replace(location.destination)
            if checksum_bytes is not None:
                location.checksum_destination.write_bytes(checksum_bytes)
        except (OSError, AggregateTradeIntegrityError) as exc:
            raise AggregateTradeArchiveError(
                "Downloaded Binance archive failed integrity validation."
            ) from exc
        finally:
            if temporary.exists():
                temporary.unlink()
        return ArchiveDownloadResult(
            location.destination, verify_checksum, False
        )
