"""Deterministic CSV persistence for completed 15m order-flow buckets."""

from __future__ import annotations

import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from src.orderflow.aggregation import OrderFlowBucket


class OrderFlowStorageError(RuntimeError):
    pass


def decimal_string(value: Decimal) -> str:
    if not value.is_finite():
        raise OrderFlowStorageError("Order-flow Decimal must be finite.")
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


class OrderFlowBucketStore:
    CSV_FIELDS = (
        "bucket_open_time",
        "bucket_close_time",
        "aggregate_trade_count",
        "underlying_trade_count",
        "total_base_volume",
        "total_quote_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "taker_buy_aggtrade_count",
        "taker_sell_base_volume",
        "taker_sell_quote_volume",
        "taker_sell_aggtrade_count",
        "signed_base_volume",
        "signed_quote_volume",
        "taker_buy_base_ratio",
        "taker_buy_quote_ratio",
        "base_volume_imbalance",
        "quote_volume_imbalance",
        "average_aggtrade_base_size",
        "average_aggtrade_quote_size",
    )

    def __init__(self, root: str | Path = "data/orderflow/aggregated/15m") -> None:
        self.root = Path(root)

    def path(self, symbol: str = "BTCUSDC") -> Path:
        symbol = symbol.strip().upper()
        if symbol != "BTCUSDC":
            raise ValueError("V7 order-flow foundation supports BTCUSDC only.")
        return self.root / symbol / "orderflow.csv"

    @staticmethod
    def _row(bucket: OrderFlowBucket) -> dict[str, str | int]:
        values: dict[str, str | int] = {
            "bucket_open_time": bucket.bucket_open_time.isoformat(),
            "bucket_close_time": bucket.bucket_close_time.isoformat(),
            "aggregate_trade_count": bucket.aggregate_trade_count,
            "underlying_trade_count": bucket.underlying_trade_count,
            "taker_buy_aggtrade_count": bucket.taker_buy_aggtrade_count,
            "taker_sell_aggtrade_count": bucket.taker_sell_aggtrade_count,
        }
        for name in OrderFlowBucketStore.CSV_FIELDS:
            if name not in values:
                values[name] = decimal_string(getattr(bucket, name))
        return values

    def save(
        self,
        buckets: list[OrderFlowBucket] | tuple[OrderFlowBucket, ...],
        *,
        symbol: str = "BTCUSDC",
    ) -> Path:
        path = self.path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".csv.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.CSV_FIELDS)
                writer.writeheader()
                for bucket in buckets:
                    writer.writerow(self._row(bucket))
            temporary.replace(path)
        except OSError as exc:
            raise OrderFlowStorageError("Order-flow CSV could not be saved.") from exc
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    def load(self, *, symbol: str = "BTCUSDC") -> tuple[OrderFlowBucket, ...]:
        path = self.path(symbol)
        try:
            with path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if tuple(reader.fieldnames or ()) != self.CSV_FIELDS:
                    raise OrderFlowStorageError("Order-flow CSV columns are invalid.")
                buckets: list[OrderFlowBucket] = []
                for row_number, row in enumerate(reader, start=2):
                    bucket = OrderFlowBucket(
                        bucket_open_time=datetime.fromisoformat(row["bucket_open_time"]),
                        bucket_close_time=datetime.fromisoformat(row["bucket_close_time"]),
                        aggregate_trade_count=int(row["aggregate_trade_count"]),
                        underlying_trade_count=int(row["underlying_trade_count"]),
                        total_base_volume=Decimal(row["total_base_volume"]),
                        total_quote_volume=Decimal(row["total_quote_volume"]),
                        taker_buy_base_volume=Decimal(row["taker_buy_base_volume"]),
                        taker_buy_quote_volume=Decimal(row["taker_buy_quote_volume"]),
                        taker_buy_aggtrade_count=int(row["taker_buy_aggtrade_count"]),
                        taker_sell_base_volume=Decimal(row["taker_sell_base_volume"]),
                        taker_sell_quote_volume=Decimal(row["taker_sell_quote_volume"]),
                        taker_sell_aggtrade_count=int(row["taker_sell_aggtrade_count"]),
                    )
                    persisted = {name: row[name] for name in self.CSV_FIELDS}
                    expected = {name: str(value) for name, value in self._row(bucket).items()}
                    if persisted != expected:
                        raise OrderFlowStorageError(
                            f"Order-flow derived fields do not reconcile at row {row_number}."
                        )
                    buckets.append(bucket)
        except OrderFlowStorageError:
            raise
        except (OSError, KeyError, ValueError, InvalidOperation) as exc:
            raise OrderFlowStorageError("Order-flow CSV could not be loaded safely.") from exc
        if any(
            later.bucket_open_time <= earlier.bucket_open_time
            for earlier, later in zip(buckets, buckets[1:])
        ):
            raise OrderFlowStorageError("Order-flow buckets are not chronological.")
        return tuple(buckets)
