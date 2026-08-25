"""Append-only durable raw Binance depth-event persistence."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


UTC = timezone.utc


class RawDepthStorageError(RuntimeError):
    pass


class RawDepthEventStore:
    """Write one canonical JSON record per durable append-only line."""

    def __init__(
        self, path: str | Path, *, session_id: str, fsync_each_record: bool = True
    ) -> None:
        self.path = Path(path)
        self.session_id = session_id
        self.fsync_each_record = fsync_each_record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8", newline="\n")
        with self.path.open("r", encoding="utf-8") as existing:
            self._record_index = sum(1 for _ in existing)

    def append(
        self, *, record_type: str, received_at: datetime, payload: dict[str, Any]
    ) -> None:
        if received_at.tzinfo is None or received_at.utcoffset() is None:
            raise RawDepthStorageError("Raw depth receipt timestamp must be UTC-aware.")
        record = {
            "schema_version": "V9_DEPTH_RAW_1",
            "session_id": self.session_id,
            "record_index": self._record_index,
            "record_type": record_type,
            "received_at_utc": received_at.astimezone(UTC).isoformat(),
            "payload": payload,
        }
        try:
            self._stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            self._stream.flush()
            if self.fsync_each_record:
                os.fsync(self._stream.fileno())
        except OSError as exc:
            raise RawDepthStorageError("Raw depth event could not be persisted.") from exc
        self._record_index += 1

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> "RawDepthEventStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def records(path: str | Path) -> Iterator[dict[str, Any]]:
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RawDepthStorageError(
                        f"Raw depth log is invalid at line {line_number}."
                    ) from exc
                if item.get("schema_version") != "V9_DEPTH_RAW_1":
                    raise RawDepthStorageError("Raw depth schema changed.")
                yield item
