"""Transparent CSV and JSON persistence for historical candle caches."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Callable

from src.historical.dataset import HistoricalDataset
from src.historical.validator import DatasetValidationError, DatasetValidator
from src.models.candle import Candle
from src.models.market_data_source import MarketDataSource


class DatasetStorageError(RuntimeError):
    """Raised when a cached historical dataset cannot be trusted or persisted."""


class HistoricalDatasetStore:
    """Persist one CSV plus metadata JSON per symbol and interval."""

    CSV_FIELDS = (
        "timestamp",
        "symbol",
        "interval",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "is_closed",
    )

    def __init__(
        self,
        root: str | Path = "data/historical",
        *,
        validator: DatasetValidator | None = None,
        allow_source_gaps: bool = False,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.root = Path(root)
        self._validator = validator or DatasetValidator()
        self.allow_source_gaps = allow_source_gaps
        self._clock = clock

    def directory(self, symbol: str, interval: str) -> Path:
        return self.root / symbol.strip().upper() / interval.strip()

    def candles_path(self, symbol: str, interval: str) -> Path:
        return self.directory(symbol, interval) / "candles.csv"

    def metadata_path(self, symbol: str, interval: str) -> Path:
        return self.directory(symbol, interval) / "metadata.json"

    def load(self, symbol: str, interval: str) -> HistoricalDataset | None:
        csv_path = self.candles_path(symbol, interval)
        metadata_path = self.metadata_path(symbol, interval)
        if not csv_path.exists() and not metadata_path.exists():
            return None
        if not csv_path.is_file() or not metadata_path.is_file():
            raise DatasetStorageError(
                "Historical cache is incomplete; both candles.csv and metadata.json are required."
            )

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source = MarketDataSource(str(metadata["source_id"]))
            candles = self._read_csv(csv_path)
            dataset = HistoricalDataset(
                symbol=str(metadata["symbol"]),
                interval=str(metadata["interval"]),
                source=source,
                candles=tuple(candles),
            )
            result = self._validator.validate(
                dataset.candles,
                symbol=dataset.symbol,
                interval=dataset.interval,
                allow_gaps=self.allow_source_gaps,
            )
            if dataset.symbol != symbol.strip().upper() or dataset.interval != interval:
                raise DatasetStorageError("Historical metadata path does not match its dataset.")
            if source is not MarketDataSource.BINANCE_PUBLIC:
                raise DatasetStorageError("Historical cache source is not Binance public Spot.")
            expected = {
                "candle_count": result.candle_count,
                "first_candle_timestamp": result.first_timestamp.isoformat(),
                "last_candle_timestamp": result.last_timestamp.isoformat(),
            }
            for name, value in expected.items():
                if metadata.get(name) != value:
                    raise DatasetStorageError(
                        f"Historical metadata {name} does not match candles.csv."
                    )
            recorded_gaps = tuple(
                (datetime.fromisoformat(item["start"]), datetime.fromisoformat(item["end"]))
                for item in metadata.get("source_gaps", ())
            )
            if recorded_gaps != result.gaps:
                raise DatasetStorageError(
                    "Historical metadata source gaps do not match candles.csv."
                )
            if result.gaps and not self.allow_source_gaps:
                raise DatasetStorageError(
                    "Historical cache contains source gaps but strict loading was requested."
                )
            return dataset
        except DatasetStorageError:
            raise
        except (DatasetValidationError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DatasetStorageError("Historical cache validation failed.") from exc

    def _read_csv(self, path: Path) -> list[Candle]:
        candles: list[Candle] = []
        try:
            with path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if tuple(reader.fieldnames or ()) != self.CSV_FIELDS:
                    raise DatasetStorageError("Historical CSV columns are invalid.")
                for row in reader:
                    candles.append(
                        Candle(
                            timestamp=datetime.fromisoformat(row["timestamp"]),
                            symbol=row["symbol"],
                            interval=row["interval"],
                            open=Decimal(row["open"]),
                            high=Decimal(row["high"]),
                            low=Decimal(row["low"]),
                            close=Decimal(row["close"]),
                            volume=Decimal(row["volume"]),
                            is_closed=row["is_closed"].lower() == "true",
                        )
                    )
        except DatasetStorageError:
            raise
        except (OSError, KeyError, ValueError, InvalidOperation) as exc:
            raise DatasetStorageError("Historical CSV could not be read safely.") from exc
        return candles

    def save(self, dataset: HistoricalDataset) -> tuple[Path, Path]:
        result = self._validator.validate(
            dataset.candles,
            symbol=dataset.symbol,
            interval=dataset.interval,
            allow_gaps=self.allow_source_gaps,
        )
        if dataset.source is not MarketDataSource.BINANCE_PUBLIC:
            raise DatasetStorageError("Only Binance public Spot data may be cached.")

        destination = self.directory(dataset.symbol, dataset.interval)
        destination.mkdir(parents=True, exist_ok=True)
        csv_path = self.candles_path(dataset.symbol, dataset.interval)
        metadata_path = self.metadata_path(dataset.symbol, dataset.interval)
        csv_temp = csv_path.with_suffix(".csv.tmp")
        metadata_temp = metadata_path.with_suffix(".json.tmp")
        try:
            with csv_temp.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.CSV_FIELDS)
                writer.writeheader()
                for candle in dataset.candles:
                    writer.writerow(
                        {
                            "timestamp": candle.timestamp.isoformat(),
                            "symbol": candle.symbol,
                            "interval": candle.interval,
                            "open": str(candle.open),
                            "high": str(candle.high),
                            "low": str(candle.low),
                            "close": str(candle.close),
                            "volume": str(candle.volume),
                            "is_closed": str(candle.is_closed).lower(),
                        }
                    )
            metadata = {
                "symbol": dataset.symbol,
                "interval": dataset.interval,
                "source": "BINANCE_PUBLIC_SPOT",
                "source_id": dataset.source.value,
                "first_candle_timestamp": result.first_timestamp.isoformat(),
                "last_candle_timestamp": result.last_timestamp.isoformat(),
                "candle_count": result.candle_count,
                "download_update_timestamp": self._clock()
                .astimezone(timezone.utc)
                .isoformat(),
            }
            if self.allow_source_gaps:
                metadata["source_gap_count"] = len(result.gaps)
                metadata["source_gaps"] = [
                    {"start": start.isoformat(), "end": end.isoformat()}
                    for start, end in result.gaps
                ]
            metadata_temp.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            csv_temp.replace(csv_path)
            metadata_temp.replace(metadata_path)
        except OSError as exc:
            raise DatasetStorageError("Historical cache could not be saved.") from exc
        finally:
            for temporary in (csv_temp, metadata_temp):
                if temporary.exists():
                    temporary.unlink()
        return csv_path, metadata_path
