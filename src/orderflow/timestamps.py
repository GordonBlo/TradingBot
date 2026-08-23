"""Deterministic Binance Spot archive timestamp normalization."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


class BinanceTimestampError(ValueError):
    """Raised when an archive timestamp has an unsupported or unsafe scale."""


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_EARLIEST = datetime(2017, 1, 1, tzinfo=timezone.utc)
_MICROSECOND_TRANSITION = datetime(2025, 1, 1, tzinfo=timezone.utc)
_LATEST = datetime(2100, 1, 1, tzinfo=timezone.utc)


def normalize_binance_timestamp(raw: int | str) -> datetime:
    """Normalize supported archive milliseconds/microseconds to exact UTC.

    Official Spot archives use milliseconds before 2025-01-01 and microseconds
    from that date onward. Magnitude selects the unit; the date/schema transition
    then validates it, so values with an ambiguous or inconsistent scale fail.
    """

    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise BinanceTimestampError("Binance timestamp must be an integer.") from exc
    if value < 0 or str(raw).strip().lstrip("+").lstrip("-") != str(value):
        raise BinanceTimestampError("Binance timestamp must be a canonical integer.")

    if 10**12 <= value < 10**14:
        microseconds = value * 1_000
        unit = "milliseconds"
    elif 10**15 <= value < 10**17:
        microseconds = value
        unit = "microseconds"
    else:
        raise BinanceTimestampError("Binance timestamp scale is unsupported.")

    try:
        result = _EPOCH + timedelta(microseconds=microseconds)
    except OverflowError as exc:
        raise BinanceTimestampError("Binance timestamp is outside datetime range.") from exc
    if not _EARLIEST <= result < _LATEST:
        raise BinanceTimestampError("Binance timestamp is implausible.")
    if result < _MICROSECOND_TRANSITION and unit != "milliseconds":
        raise BinanceTimestampError("Pre-2025 Spot archive timestamp must be milliseconds.")
    if result >= _MICROSECOND_TRANSITION and unit != "microseconds":
        raise BinanceTimestampError(
            "Spot archive timestamp from 2025 onward must be microseconds."
        )
    return result
