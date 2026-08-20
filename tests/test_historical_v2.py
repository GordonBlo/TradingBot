from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.exchange.historical_data import HistoricalDataError
from src.exchange.public_market_client import PublicMarketDataError
from src.historical.dataset import HistoricalDataset
from src.historical.downloader import HistoricalRangeDownloader
from src.historical.manager import HistoricalDatasetManager
from src.historical.storage import HistoricalDatasetStore
from src.historical.validator import (
    DatasetValidationError,
    DatasetValidator,
    HistoricalDataGapError,
    InvalidCandleError,
)
from src.models.candle import Candle
from src.models.market_data_source import MarketDataSource


BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)
INTERVAL = timedelta(minutes=15)


@pytest.fixture
def v2_tmp_path() -> Iterator[Path]:
    root = Path(__file__).resolve().parents[1] / ".test_artifacts" / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="v2-", dir=root) as directory:
        yield Path(directory)


def raw_kline(index: int, *, volume: str = "5") -> list[object]:
    open_time = int((BASE + INTERVAL * index).timestamp() * 1_000)
    return [
        open_time,
        str(100 + index),
        str(102 + index),
        str(99 + index),
        str(101 + index),
        volume,
        open_time + 899_999,
        "0",
        1,
        "0",
        "0",
        "0",
    ]


def candle(index: int, **changes: object) -> Candle:
    values: dict[str, object] = {
        "timestamp": BASE + INTERVAL * index,
        "symbol": "BTCUSDC",
        "interval": "15m",
        "open": Decimal("100"),
        "high": Decimal("105"),
        "low": Decimal("95"),
        "close": Decimal("102"),
        "volume": Decimal("10"),
        "is_closed": True,
    }
    values.update(changes)
    return Candle(**values)  # type: ignore[arg-type]


def unsafe_candle(index: int, **changes: object) -> Candle:
    item = candle(index)
    for name, value in changes.items():
        object.__setattr__(item, name, value)
    return item


class PagingClient:
    source = MarketDataSource.BINANCE_PUBLIC

    def __init__(self, pages: list[list[list[object]]]) -> None:
        self.pages = list(pages)
        self.starts: list[int | None] = []
        self.ends: list[int | None] = []

    def get_server_time(self) -> int:
        return int((BASE + timedelta(days=10)).timestamp() * 1_000)

    def get_klines(
        self,
        symbol: str,
        interval: str,
        *,
        limit: int,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> list[list[object]]:
        assert symbol == "BTCUSDC"
        assert interval == "15m"
        self.starts.append(start_time_ms)
        self.ends.append(end_time_ms)
        return self.pages.pop(0) if self.pages else []


def test_historical_pagination_single_partial_page_and_boundaries() -> None:
    client = PagingClient([[raw_kline(0), raw_kline(1), raw_kline(2)]])
    downloader = HistoricalRangeDownloader(client, page_limit=5)

    result = downloader.download("BTCUSDC", "15m", BASE, BASE + INTERVAL * 3)

    assert len(result.candles) == 3
    assert result.request_count == 1
    assert [item.timestamp for item in result.candles] == [
        BASE,
        BASE + INTERVAL,
        BASE + INTERVAL * 2,
    ]
    assert client.starts == [int(BASE.timestamp() * 1_000)]
    assert client.ends == [int((BASE + INTERVAL * 3).timestamp() * 1_000) - 1]


def test_historical_pagination_multiple_pages_uses_next_open_and_empty_final_page() -> None:
    client = PagingClient(
        [
            [raw_kline(0), raw_kline(1)],
            [raw_kline(2), raw_kline(3)],
            [],
        ]
    )
    downloader = HistoricalRangeDownloader(client, page_limit=2)

    result = downloader.download("BTCUSDC", "15m", BASE, BASE + INTERVAL * 5)

    assert len(result.candles) == 4
    assert result.request_count == 3
    assert client.starts == [
        int(BASE.timestamp() * 1_000),
        int((BASE + INTERVAL * 2).timestamp() * 1_000),
        int((BASE + INTERVAL * 4).timestamp() * 1_000),
    ]
    assert len({item.timestamp for item in result.candles}) == 4


def test_historical_pagination_deduplicates_overlap() -> None:
    client = PagingClient(
        [
            [raw_kline(0), raw_kline(1)],
            [raw_kline(1), raw_kline(2)],
            [raw_kline(3)],
        ]
    )
    result = HistoricalRangeDownloader(client, page_limit=2).download(
        "BTCUSDC", "15m", BASE, BASE + INTERVAL * 4
    )

    assert [item.timestamp for item in result.candles] == [
        BASE + INTERVAL * index for index in range(4)
    ]


def test_historical_pagination_rejects_non_advancing_page() -> None:
    client = PagingClient([[raw_kline(0)], [raw_kline(0)]])
    downloader = HistoricalRangeDownloader(client, page_limit=1)

    with pytest.raises(HistoricalDataError, match="no forward progress"):
        downloader.download("BTCUSDC", "15m", BASE, BASE + INTERVAL * 3)


def test_historical_429_respects_retry_after_and_is_bounded() -> None:
    sleeps: list[float] = []

    class RateLimitedClient(PagingClient):
        def __init__(self) -> None:
            super().__init__([[raw_kline(0)]])
            self.calls = 0

        def get_klines(self, *args: object, **kwargs: object) -> list[list[object]]:
            self.calls += 1
            if self.calls == 1:
                raise PublicMarketDataError(
                    "limited", status_code=429, retry_after_seconds=2.5
                )
            return super().get_klines(*args, **kwargs)  # type: ignore[arg-type]

    client = RateLimitedClient()
    result = HistoricalRangeDownloader(
        client,
        page_limit=2,
        max_retries=1,
        sleeper=sleeps.append,
    ).download("BTCUSDC", "15m", BASE, BASE + INTERVAL)

    assert len(result.candles) == 1
    assert sleeps == [2.5]
    assert client.calls == 2


def test_dataset_validator_accepts_valid_closed_utc_data_and_diagnostic_gaps() -> None:
    validator = DatasetValidator()
    valid = [candle(0), candle(1), candle(2)]

    result = validator.validate(valid, symbol="BTCUSDC", interval="15m")
    diagnostic = validator.validate(
        [candle(0), candle(2)],
        symbol="BTCUSDC",
        interval="15m",
        allow_gaps=True,
    )

    assert result.candle_count == 3
    assert result.gaps == ()
    assert diagnostic.gaps == ((BASE + INTERVAL, BASE + INTERVAL * 2),)


def test_dataset_validator_rejects_duplicates_out_of_order_and_gaps() -> None:
    validator = DatasetValidator()
    with pytest.raises(DatasetValidationError, match="Duplicate"):
        validator.validate([candle(0), candle(0)])
    with pytest.raises(DatasetValidationError, match="chronological"):
        validator.validate([candle(1), candle(0)])
    with pytest.raises(HistoricalDataGapError, match="expected candle"):
        validator.validate([candle(0), candle(2)])


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"high": Decimal("94")}, "high is below low"),
        ({"high": Decimal("99")}, "high is below open"),
        ({"high": Decimal("101")}, "high is below close"),
        ({"low": Decimal("101")}, "low is above open"),
        (
            {"open": Decimal("104"), "low": Decimal("103")},
            "low is above close",
        ),
        ({"open": Decimal("0")}, "non-positive"),
        ({"close": Decimal("-1")}, "non-positive"),
        ({"volume": Decimal("-1")}, "negative volume"),
        ({"is_closed": False}, "not closed"),
    ),
)
def test_dataset_validator_rejects_invalid_candles(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(InvalidCandleError, match=message):
        DatasetValidator().validate([unsafe_candle(0, **changes)])


def test_dataset_validator_rejects_misaligned_or_non_utc_timestamp() -> None:
    with pytest.raises(InvalidCandleError, match="not aligned"):
        DatasetValidator().validate(
            [unsafe_candle(0, timestamp=BASE + timedelta(minutes=1))]
        )
    local = timezone(timedelta(hours=2))
    with pytest.raises(InvalidCandleError, match="UTC"):
        DatasetValidator().validate(
            [unsafe_candle(0, timestamp=BASE.astimezone(local))]
        )


def test_historical_store_round_trip_and_metadata(v2_tmp_path: Path) -> None:
    fixed_now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = HistoricalDatasetStore(v2_tmp_path, clock=lambda: fixed_now)
    dataset = HistoricalDataset(
        "BTCUSDC",
        "15m",
        MarketDataSource.BINANCE_PUBLIC,
        (candle(0), candle(1)),
    )

    csv_path, metadata_path = store.save(dataset)
    loaded = store.load("BTCUSDC", "15m")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert csv_path.is_file()
    assert loaded == dataset
    assert metadata == {
        "candle_count": 2,
        "download_update_timestamp": fixed_now.isoformat(),
        "first_candle_timestamp": BASE.isoformat(),
        "interval": "15m",
        "last_candle_timestamp": (BASE + INTERVAL).isoformat(),
        "source": "BINANCE_PUBLIC_SPOT",
        "source_id": "binance_public",
        "symbol": "BTCUSDC",
    }


def test_explicit_source_gap_cache_records_gaps_without_synthesizing(
    v2_tmp_path: Path,
) -> None:
    dataset = HistoricalDataset(
        "BTCUSDC",
        "15m",
        MarketDataSource.BINANCE_PUBLIC,
        (candle(0), candle(2)),
    )
    permissive = HistoricalDatasetStore(v2_tmp_path, allow_source_gaps=True)
    _, metadata_path = permissive.save(dataset)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert permissive.load("BTCUSDC", "15m") == dataset
    assert metadata["source_gap_count"] == 1
    assert metadata["source_gaps"] == [
        {
            "start": (BASE + INTERVAL).isoformat(),
            "end": (BASE + INTERVAL * 2).isoformat(),
        }
    ]
    with pytest.raises((HistoricalDataGapError, RuntimeError)):
        HistoricalDatasetStore(v2_tmp_path).load("BTCUSDC", "15m")


def test_historical_manager_incrementally_downloads_only_missing_tail(
    v2_tmp_path: Path,
) -> None:
    class FilteringClient:
        source = MarketDataSource.BINANCE_PUBLIC

        def __init__(self) -> None:
            self.starts: list[int | None] = []

        def get_server_time(self) -> int:
            return int((BASE + timedelta(days=10)).timestamp() * 1_000)

        def get_klines(
            self,
            symbol: str,
            interval: str,
            *,
            limit: int,
            start_time_ms: int | None = None,
            end_time_ms: int | None = None,
        ) -> list[list[object]]:
            self.starts.append(start_time_ms)
            return [
                row
                for row in (raw_kline(index) for index in range(6))
                if start_time_ms <= int(row[0]) <= end_time_ms
            ][:limit]

    client = FilteringClient()
    validator = DatasetValidator()
    manager = HistoricalDatasetManager(
        HistoricalRangeDownloader(client, page_limit=1000),
        HistoricalDatasetStore(v2_tmp_path, validator=validator),
        validator=validator,
    )

    first = manager.update("BTCUSDC", "15m", BASE, BASE + INTERVAL * 3)
    second = manager.update("BTCUSDC", "15m", BASE, BASE + INTERVAL * 5)

    assert first.downloaded_candle_count == 3
    assert second.downloaded_candle_count == 2
    assert len(second.dataset.candles) == 5
    assert client.starts == [
        int(BASE.timestamp() * 1_000),
        int((BASE + INTERVAL * 3).timestamp() * 1_000),
    ]


def test_historical_dataset_slice_preserves_chronology() -> None:
    dataset = HistoricalDataset(
        "BTCUSDC",
        "15m",
        MarketDataSource.BINANCE_PUBLIC,
        tuple(candle(index) for index in range(6)),
    )

    sliced = dataset.slice(BASE + INTERVAL * 2, BASE + INTERVAL * 5)

    assert [item.timestamp for item in sliced.candles] == [
        BASE + INTERVAL * index for index in range(2, 5)
    ]
