"""Binance candle-interval arithmetic and UTC alignment helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


_FIXED_INTERVALS: dict[str, timedelta] = {
    "1s": timedelta(seconds=1),
    "1m": timedelta(minutes=1),
    "3m": timedelta(minutes=3),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h": timedelta(hours=1),
    "2h": timedelta(hours=2),
    "4h": timedelta(hours=4),
    "6h": timedelta(hours=6),
    "8h": timedelta(hours=8),
    "12h": timedelta(hours=12),
    "1d": timedelta(days=1),
    "3d": timedelta(days=3),
    "1w": timedelta(weeks=1),
}

SUPPORTED_INTERVALS = frozenset((*_FIXED_INTERVALS, "1M"))


def require_utc(value: datetime, *, name: str = "datetime") -> datetime:
    """Return a UTC datetime or reject naive/non-UTC values."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware UTC.")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use UTC.")
    return value.astimezone(timezone.utc)


def next_open_time(timestamp: datetime, interval: str) -> datetime:
    """Return the next chronological candle-open timestamp."""

    timestamp = require_utc(timestamp, name="timestamp")
    if interval in _FIXED_INTERVALS:
        return timestamp + _FIXED_INTERVALS[interval]
    if interval == "1M":
        year = timestamp.year + (1 if timestamp.month == 12 else 0)
        month = 1 if timestamp.month == 12 else timestamp.month + 1
        return timestamp.replace(year=year, month=month, day=1)
    raise ValueError(f"Unsupported candle interval: {interval}")


def is_interval_aligned(timestamp: datetime, interval: str) -> bool:
    """Return whether a UTC timestamp lies on Binance's interval grid."""

    timestamp = require_utc(timestamp, name="timestamp")
    if timestamp.microsecond != 0:
        return False
    if interval == "1s":
        return True
    if interval.endswith("m") and interval != "1M":
        minutes = int(interval[:-1])
        return timestamp.second == 0 and timestamp.minute % minutes == 0
    if interval.endswith("h"):
        hours = int(interval[:-1])
        return (
            timestamp.second == 0
            and timestamp.minute == 0
            and timestamp.hour % hours == 0
        )
    if interval in {"1d", "3d"}:
        if any((timestamp.hour, timestamp.minute, timestamp.second)):
            return False
        if interval == "1d":
            return True
        epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        return (timestamp - epoch).days % 3 == 0
    if interval == "1w":
        return (
            timestamp.weekday() == 0
            and not any((timestamp.hour, timestamp.minute, timestamp.second))
        )
    if interval == "1M":
        return timestamp.day == 1 and not any(
            (timestamp.hour, timestamp.minute, timestamp.second)
        )
    raise ValueError(f"Unsupported candle interval: {interval}")


def to_unix_milliseconds(timestamp: datetime) -> int:
    """Convert a UTC datetime to exact Unix milliseconds."""

    timestamp = require_utc(timestamp, name="timestamp")
    return int(timestamp.timestamp() * 1_000)
