"""Public Binance Spot aggregate-trade data infrastructure."""

from src.orderflow.aggregation import (
    OrderFlowBucket,
    aggregate_15m,
    completed_bucket_at,
)
from src.orderflow.archive import (
    AggregateTradeArchiveDownloader,
    ArchiveDownloadResult,
    ArchiveLocation,
    build_archive_location,
)
from src.orderflow.integrity import (
    AggregateTradeIntegrityError,
    verify_sha256,
)
from src.orderflow.models import AggregateTrade, AggressorSide
from src.orderflow.parser import (
    AggregateTradeParseError,
    parse_aggtrade_csv,
    parse_aggtrade_file,
)
from src.orderflow.storage import OrderFlowBucketStore, OrderFlowStorageError

__all__ = (
    "AggregateTrade",
    "AggregateTradeArchiveDownloader",
    "AggregateTradeIntegrityError",
    "AggregateTradeParseError",
    "AggressorSide",
    "ArchiveDownloadResult",
    "ArchiveLocation",
    "OrderFlowBucket",
    "OrderFlowBucketStore",
    "OrderFlowStorageError",
    "aggregate_15m",
    "build_archive_location",
    "completed_bucket_at",
    "parse_aggtrade_csv",
    "parse_aggtrade_file",
    "verify_sha256",
)
