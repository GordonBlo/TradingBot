"""Deterministic bounded reconstruction of Binance Spot L2 depth."""

from __future__ import annotations

from decimal import Decimal
from enum import Enum

from src.orderbook.models import DepthDiffEvent, DepthLevel, DepthSnapshot


class ApplyStatus(str, Enum):
    APPLIED = "APPLIED"
    STALE = "STALE"
    SEQUENCE_GAP = "SEQUENCE_GAP"
    INVALID_BOOK = "INVALID_BOOK"


class ReconstructedOrderBook:
    """Maintain the nearest bounded levels while raw data remains lossless on disk."""

    def __init__(self, *, max_levels_per_side: int = 5_000) -> None:
        if max_levels_per_side < 1:
            raise ValueError("Order-book level bound must be positive.")
        self.max_levels_per_side = max_levels_per_side
        self.bids: dict[Decimal, Decimal] = {}
        self.asks: dict[Decimal, Decimal] = {}
        self.last_update_id: int | None = None

    def reset(self, snapshot: DepthSnapshot) -> None:
        self.bids = {price: quantity for price, quantity in snapshot.bids if quantity > 0}
        self.asks = {price: quantity for price, quantity in snapshot.asks if quantity > 0}
        self.last_update_id = snapshot.last_update_id
        self._trim()

    def _trim(self) -> None:
        if len(self.bids) > self.max_levels_per_side:
            keep = set(sorted(self.bids, reverse=True)[: self.max_levels_per_side])
            self.bids = {price: quantity for price, quantity in self.bids.items() if price in keep}
        if len(self.asks) > self.max_levels_per_side:
            keep = set(sorted(self.asks)[: self.max_levels_per_side])
            self.asks = {price: quantity for price, quantity in self.asks.items() if price in keep}

    @staticmethod
    def _apply_levels(book: dict[Decimal, Decimal], levels: tuple[DepthLevel, ...]) -> None:
        for price, quantity in levels:
            if quantity == 0:
                book.pop(price, None)
            else:
                book[price] = quantity

    def apply(self, event: DepthDiffEvent) -> ApplyStatus:
        if self.last_update_id is None:
            raise RuntimeError("A depth snapshot is required before diff reconstruction.")
        if event.final_update_id <= self.last_update_id:
            return ApplyStatus.STALE
        expected = self.last_update_id + 1
        if event.first_update_id > expected or event.final_update_id < expected:
            return ApplyStatus.SEQUENCE_GAP
        self._apply_levels(self.bids, event.bids)
        self._apply_levels(self.asks, event.asks)
        self.last_update_id = event.final_update_id
        self._trim()
        return ApplyStatus.APPLIED if self.is_valid else ApplyStatus.INVALID_BOOK

    @property
    def best_bid(self) -> DepthLevel | None:
        if not self.bids:
            return None
        price = max(self.bids)
        return price, self.bids[price]

    @property
    def best_ask(self) -> DepthLevel | None:
        if not self.asks:
            return None
        price = min(self.asks)
        return price, self.asks[price]

    @property
    def is_valid(self) -> bool:
        bid = self.best_bid
        ask = self.best_ask
        return bool(bid and ask and bid[0] < ask[0])

    def top_levels(self, count: int) -> tuple[tuple[DepthLevel, ...], tuple[DepthLevel, ...]]:
        if count < 1:
            raise ValueError("Top-level count must be positive.")
        bids = tuple((price, self.bids[price]) for price in sorted(self.bids, reverse=True)[:count])
        asks = tuple((price, self.asks[price]) for price in sorted(self.asks)[:count])
        return bids, asks
