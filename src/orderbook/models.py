"""Strict Decimal-safe models for Binance Spot depth data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any


UTC = timezone.utc
SYMBOL = "BTCUSDC"
DepthLevel = tuple[Decimal, Decimal]


class DepthDataError(ValueError):
    pass


def _utc_from_ms(value: object) -> datetime:
    try:
        milliseconds = int(value)
    except (TypeError, ValueError) as exc:
        raise DepthDataError("Depth event timestamp is invalid.") from exc
    if milliseconds < 0:
        raise DepthDataError("Depth event timestamp is negative.")
    return datetime.fromtimestamp(milliseconds / 1_000, tz=UTC)


def _levels(value: object, *, side: str) -> tuple[DepthLevel, ...]:
    if not isinstance(value, list):
        raise DepthDataError(f"Depth {side} levels must be a list.")
    levels = []
    for raw in value:
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            raise DepthDataError(f"Depth {side} level is malformed.")
        try:
            price = Decimal(str(raw[0]))
            quantity = Decimal(str(raw[1]))
        except (InvalidOperation, ValueError) as exc:
            raise DepthDataError(f"Depth {side} level is not Decimal-safe.") from exc
        if not price.is_finite() or not quantity.is_finite() or price <= 0 or quantity < 0:
            raise DepthDataError(f"Depth {side} level is invalid.")
        levels.append((price, quantity))
    return tuple(levels)


@dataclass(frozen=True, slots=True)
class DepthSnapshot:
    last_update_id: int
    bids: tuple[DepthLevel, ...]
    asks: tuple[DepthLevel, ...]
    received_at: datetime

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, received_at: datetime) -> "DepthSnapshot":
        if received_at.tzinfo is None or received_at.utcoffset() is None:
            raise DepthDataError("Snapshot receipt timestamp must be UTC-aware.")
        try:
            last_update_id = int(payload["lastUpdateId"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DepthDataError("Snapshot lastUpdateId is invalid.") from exc
        if last_update_id < 0:
            raise DepthDataError("Snapshot lastUpdateId is negative.")
        return cls(
            last_update_id=last_update_id,
            bids=_levels(payload.get("bids"), side="bid"),
            asks=_levels(payload.get("asks"), side="ask"),
            received_at=received_at.astimezone(UTC),
        )


@dataclass(frozen=True, slots=True)
class DepthDiffEvent:
    first_update_id: int
    final_update_id: int
    event_time: datetime
    bids: tuple[DepthLevel, ...]
    asks: tuple[DepthLevel, ...]
    symbol: str = SYMBOL

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DepthDiffEvent":
        if payload.get("e") != "depthUpdate" or str(payload.get("s", "")).upper() != SYMBOL:
            raise DepthDataError("Only BTCUSDC Spot diff-depth events are supported.")
        try:
            first = int(payload["U"])
            final = int(payload["u"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DepthDataError("Depth update sequence is invalid.") from exc
        if first < 0 or final < first:
            raise DepthDataError("Depth update sequence range is invalid.")
        return cls(
            first_update_id=first,
            final_update_id=final,
            event_time=_utc_from_ms(payload.get("E")),
            bids=_levels(payload.get("b"), side="bid"),
            asks=_levels(payload.get("a"), side="ask"),
        )
