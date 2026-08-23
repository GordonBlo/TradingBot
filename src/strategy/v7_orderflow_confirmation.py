"""Frozen V7-H0 exact-bucket order-flow filter for V6-H0 entries."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping

from src.orderflow.aggregation import OrderFlowBucket
from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext
from src.strategy.models import (
    DecisionReason,
    StrategyAction,
    StrategyDecision,
    TrendMomentumConfig,
)
from src.strategy.v6_mtf_continuation import V6MTFContinuationStrategy


class V7OrderFlowConfirmationStrategy(BaseStrategy):
    """Retain only V6-H0 BUY decisions with positive current-bucket flow."""

    NAME = "V7OrderFlowConfirmationStrategy"
    VERSION = "7.0"
    V6_H0_RUN_ID = "e1eef7bdd0c37ad4"
    ORDERFLOW_DATASET_ID = "8fdfcee8d68b6f48"
    ORDERFLOW_DATASET_DEFINITION_SHA256 = (
        "dbca5d298f870068cfdc99ad0d550eda3d0af52b452c484c2e6d1e80e3c79567"
    )
    SYMBOL = "BTCUSDC"
    INTERVAL = "15m"
    QUOTE_IMBALANCE_THRESHOLD = Decimal("0")

    def __init__(
        self,
        config: TrendMomentumConfig,
        *,
        buckets: Sequence[OrderFlowBucket],
        dataset_id: str,
        dataset_definition_sha256: str,
        v6_strategy: V6MTFContinuationStrategy | None = None,
    ) -> None:
        self._validate_dataset_identity(dataset_id, dataset_definition_sha256)
        if v6_strategy is not None and not isinstance(
            v6_strategy, V6MTFContinuationStrategy
        ):
            raise TypeError("V7-H0 requires a V6MTFContinuationStrategy instance.")
        self.config = config
        self._v6_strategy = v6_strategy or V6MTFContinuationStrategy(config)
        by_open_time: dict[datetime, OrderFlowBucket] = {}
        for bucket in buckets:
            if bucket.bucket_open_time in by_open_time:
                raise ValueError("V7-H0 order-flow bucket timestamp is duplicated.")
            by_open_time[bucket.bucket_open_time] = bucket
        self._buckets_by_open_time: Mapping[datetime, OrderFlowBucket] = (
            MappingProxyType(by_open_time)
        )
        self._missing_bucket_timestamps: set[datetime] = set()

    @staticmethod
    def _validate_dataset_identity(
        dataset_id: str,
        dataset_definition_sha256: str,
    ) -> None:
        if dataset_id != V7OrderFlowConfirmationStrategy.ORDERFLOW_DATASET_ID:
            raise ValueError("V7-H0 requires the frozen order-flow dataset ID.")
        if (
            dataset_definition_sha256
            != V7OrderFlowConfirmationStrategy.ORDERFLOW_DATASET_DEFINITION_SHA256
        ):
            raise ValueError(
                "V7-H0 requires the frozen order-flow dataset definition SHA-256."
            )

    @property
    def required_history_bars(self) -> int:
        return self._v6_strategy.required_history_bars

    @property
    def requires_full_history(self) -> bool:
        return self._v6_strategy.requires_full_history

    @property
    def missing_bucket_timestamps(self) -> frozenset[datetime]:
        """Exact V6 signal buckets unavailable to V7, for integrity reporting."""

        return frozenset(self._missing_bucket_timestamps)

    @staticmethod
    def quote_flow_confirms(
        *,
        taker_buy_quote_volume: Decimal,
        taker_sell_quote_volume: Decimal,
        total_quote_volume: Decimal,
    ) -> bool:
        """Apply the one frozen V7-H0 feature without any magnitude threshold."""

        return (
            total_quote_volume > 0
            and taker_buy_quote_volume > taker_sell_quote_volume
            and (
                (taker_buy_quote_volume - taker_sell_quote_volume)
                / total_quote_volume
                > V7OrderFlowConfirmationStrategy.QUOTE_IMBALANCE_THRESHOLD
            )
        )

    def evaluate(self, context: StrategyContext) -> StrategyDecision:
        v6_decision = self._v6_strategy.evaluate(context)
        if v6_decision.action is not StrategyAction.ENTER_LONG:
            return v6_decision

        signal_open_time = self._utc(context.current_candle.timestamp)
        bucket = self._buckets_by_open_time.get(signal_open_time)
        if bucket is None:
            self._missing_bucket_timestamps.add(signal_open_time)
            return self._hold(
                "Exact completed order-flow bucket is unavailable",
                status="MISSING_EXACT_BUCKET",
            )
        if bucket.bucket_close_time > self._utc(context.timestamp):
            return self._hold(
                "Order-flow bucket was not complete at V6 signal time",
                status="INCOMPLETE_BUCKET",
            )
        if not self.quote_flow_confirms(
            taker_buy_quote_volume=bucket.taker_buy_quote_volume,
            taker_sell_quote_volume=bucket.taker_sell_quote_volume,
            total_quote_volume=bucket.total_quote_volume,
        ):
            return self._hold(
                "Current signal-candle taker-buy quote volume does not dominate",
                status="NON_POSITIVE_QUOTE_IMBALANCE",
            )
        return v6_decision

    @staticmethod
    def _utc(timestamp: datetime) -> datetime:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("V7-H0 timestamps must be timezone-aware.")
        return timestamp.astimezone(timezone.utc)

    @staticmethod
    def _hold(reason: str, *, status: str) -> StrategyDecision:
        return StrategyDecision(
            StrategyAction.HOLD,
            DecisionReason.NO_ENTRY,
            reason,
            metadata={"orderflow_integrity_status": status},
        )


def assert_v7_entries_are_v6_subset(
    v7_entry_timestamps: Iterable[datetime],
    v6_entry_timestamps: Iterable[datetime],
) -> None:
    """Reject any V7 entry timestamp not emitted by frozen V6-H0."""

    v6_entries = {
        V7OrderFlowConfirmationStrategy._utc(timestamp)
        for timestamp in v6_entry_timestamps
    }
    non_v6_entries = {
        V7OrderFlowConfirmationStrategy._utc(timestamp)
        for timestamp in v7_entry_timestamps
    }.difference(v6_entries)
    if non_v6_entries:
        raise ValueError("V7-H0 created entries absent from the V6-H0 reference.")
