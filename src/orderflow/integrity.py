"""Checksum and stream-integrity checks for aggregate trades."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path

from src.orderflow.models import AggregateTrade


class AggregateTradeIntegrityError(ValueError):
    """Raised when archive bytes or parsed trade ordering cannot be trusted."""


def parse_checksum(text: str, *, expected_filename: str | None = None) -> str:
    parts = text.strip().split()
    if not parts or len(parts[0]) != 64:
        raise AggregateTradeIntegrityError("CHECKSUM does not contain a SHA-256 digest.")
    digest = parts[0].lower()
    if any(character not in "0123456789abcdef" for character in digest):
        raise AggregateTradeIntegrityError("CHECKSUM SHA-256 digest is invalid.")
    if expected_filename is not None and len(parts) >= 2:
        recorded = parts[-1].lstrip("*")
        if Path(recorded).name != expected_filename:
            raise AggregateTradeIntegrityError("CHECKSUM filename does not match archive.")
    return digest


def verify_sha256(
    path: str | Path,
    checksum: str | Path,
    *,
    expected_filename: str | None = None,
) -> str:
    archive_path = Path(path)
    checksum_text = (
        Path(checksum).read_text(encoding="utf-8")
        if isinstance(checksum, Path)
        else str(checksum)
    )
    expected = parse_checksum(
        checksum_text, expected_filename=expected_filename or archive_path.name
    )
    hasher = hashlib.sha256()
    try:
        with archive_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
    except OSError as exc:
        raise AggregateTradeIntegrityError("Archive could not be checksummed.") from exc
    actual = hasher.hexdigest()
    if actual != expected:
        raise AggregateTradeIntegrityError("Archive SHA-256 does not match CHECKSUM.")
    return actual


def validate_trade_segment(trades: Sequence[AggregateTrade]) -> None:
    """Validate ordering guarantees within one official source segment."""

    seen_ids: set[int] = set()
    previous: AggregateTrade | None = None
    for index, trade in enumerate(trades):
        if trade.aggregate_trade_id in seen_ids:
            raise AggregateTradeIntegrityError(
                f"Duplicate aggregate_trade_id at row {index}."
            )
        seen_ids.add(trade.aggregate_trade_id)
        if previous is not None:
            if trade.timestamp < previous.timestamp:
                raise AggregateTradeIntegrityError(
                    f"Aggregate-trade timestamp decreases at row {index}."
                )
            if trade.aggregate_trade_id <= previous.aggregate_trade_id:
                raise AggregateTradeIntegrityError(
                    f"aggregate_trade_id is not increasing at row {index}."
                )
        previous = trade


def validate_archive_boundaries(
    segments: Iterable[Sequence[AggregateTrade]],
) -> None:
    """Detect exact duplicate records without assuming consecutive cross-file IDs."""

    seen_records: set[AggregateTrade] = set()
    for segment in segments:
        validate_trade_segment(segment)
        for trade in segment:
            if trade in seen_records:
                raise AggregateTradeIntegrityError(
                    "Exact aggregate-trade record is duplicated across source segments."
                )
            seen_records.add(trade)
