"""Shared CLI parsing helpers."""

from __future__ import annotations

from datetime import datetime, timezone


def parse_utc_datetime(value: str) -> datetime:
    """Parse ISO date/datetime; naive values are explicitly interpreted as UTC."""

    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO date/datetime: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
