from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable


STATE_SCHEMA = "LIVE_SHADOW_STATE_1"
SHADOW_SCHEMA = "LIVE_SHADOW_DECISION_1"


class RuntimeIntegrityError(RuntimeError):
    pass


class RuntimeState(str, Enum):
    STARTING = "STARTING"
    SYNCING = "SYNCING"
    READY = "READY"
    STALE = "STALE"
    RECOVERING = "RECOVERING"
    STOPPED = "STOPPED"


ALLOWED_TRANSITIONS = {
    RuntimeState.STARTING: {
        RuntimeState.SYNCING,
        RuntimeState.RECOVERING,
        RuntimeState.STOPPED,
    },
    RuntimeState.SYNCING: {
        RuntimeState.STARTING,
        RuntimeState.READY,
        RuntimeState.STALE,
        RuntimeState.STOPPED,
    },
    RuntimeState.READY: {
        RuntimeState.STALE,
        RuntimeState.RECOVERING,
        RuntimeState.STARTING,
        RuntimeState.STOPPED,
    },
    RuntimeState.STALE: {
        RuntimeState.STARTING,
        RuntimeState.RECOVERING,
        RuntimeState.STOPPED,
    },
    RuntimeState.RECOVERING: {
        RuntimeState.READY,
        RuntimeState.STARTING,
        RuntimeState.STALE,
        RuntimeState.STOPPED,
    },
    RuntimeState.STOPPED: {RuntimeState.STARTING},
}


def utc_text(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("runtime timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_utc(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeIntegrityError("persisted runtime timestamp is not UTC-aware")
    return parsed.astimezone(UTC)


@dataclass(slots=True)
class RuntimeCounters:
    market_events: int = 0
    open_candle_updates: int = 0
    closed_candles: int = 0
    evaluations: int = 0
    shadow_entry_signals: int = 0
    duplicate_candles_suppressed: int = 0
    blocked_evaluations: int = 0
    stale_transitions: int = 0
    disconnects: int = 0
    recovery_attempts: int = 0
    successful_recoveries: int = 0
    candle_gaps: int = 0
    integrity_failures: int = 0
    evaluation_errors: int = 0
    orders_attempted: int = 0


@dataclass(slots=True)
class RuntimeSnapshot:
    run_identity: str
    strategy_name: str
    strategy_version: str
    state: RuntimeState = RuntimeState.STOPPED
    transition_sequence: int = 0
    last_transition_at_utc: str | None = None
    last_transition_reason: str = "not started"
    last_market_event_at_utc: str | None = None
    last_observed_candle_at_utc: str | None = None
    last_observed_candle_sha256: str | None = None
    last_observed_candle_closed: bool | None = None
    last_closed_candle_at_utc: str | None = None
    last_evaluated_candle_at_utc: str | None = None
    last_evaluated_candle_sha256: str | None = None
    shadow_sequence: int = 0
    shadow_last_hash: str | None = None
    counters: RuntimeCounters = field(default_factory=RuntimeCounters)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = STATE_SCHEMA
        payload["state"] = self.state.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RuntimeSnapshot":
        if payload.get("schema_version") != STATE_SCHEMA:
            raise RuntimeIntegrityError("runtime state schema mismatch")
        values = dict(payload)
        values.pop("schema_version")
        values["state"] = RuntimeState(values["state"])
        values["counters"] = RuntimeCounters(**values.get("counters", {}))
        return cls(**values)


class RuntimeStateStore:
    """Atomic local persistence for restart-safe runtime watermarks."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> RuntimeSnapshot | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeIntegrityError("runtime state is unreadable") from exc
        if not isinstance(payload, dict):
            raise RuntimeIntegrityError("runtime state is not an object")
        return RuntimeSnapshot.from_dict(payload)

    def save(self, snapshot: RuntimeSnapshot) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        encoded = json.dumps(
            snapshot.to_dict(), sort_keys=True, separators=(",", ":")
        ) + "\n"
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except OSError as exc:
            raise RuntimeIntegrityError("runtime state could not be persisted") from exc


def _canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


class ShadowDecisionJournal:
    """Durable append-only, hash-chained deterministic shadow decisions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        records = tuple(self.records())
        self.sequence = len(records)
        self.last_hash = records[-1]["event_hash"] if records else None

    def records(self) -> Iterable[dict[str, Any]]:
        if not self.path.exists():
            return ()
        output: list[dict[str, Any]] = []
        previous_hash: str | None = None
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeIntegrityError(
                        f"shadow journal is invalid at line {line_number}"
                    ) from exc
                if record.get("schema_version") != SHADOW_SCHEMA:
                    raise RuntimeIntegrityError("shadow journal schema mismatch")
                if record.get("sequence") != line_number - 1:
                    raise RuntimeIntegrityError("shadow journal sequence mismatch")
                if record.get("previous_hash") != previous_hash:
                    raise RuntimeIntegrityError("shadow journal hash chain mismatch")
                event_hash = record.get("event_hash")
                unsigned = dict(record)
                unsigned.pop("event_hash", None)
                expected = hashlib.sha256(_canonical(unsigned).encode("utf-8")).hexdigest()
                if event_hash != expected:
                    raise RuntimeIntegrityError("shadow journal event hash mismatch")
                previous_hash = event_hash
                output.append(record)
        return tuple(output)

    def append(self, payload: dict[str, Any]) -> dict[str, Any]:
        record = {
            "schema_version": SHADOW_SCHEMA,
            "sequence": self.sequence,
            "previous_hash": self.last_hash,
            **payload,
        }
        record["event_hash"] = hashlib.sha256(
            _canonical(record).encode("utf-8")
        ).hexdigest()
        try:
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(_canonical(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise RuntimeIntegrityError("shadow decision could not be persisted") from exc
        self.sequence += 1
        self.last_hash = record["event_hash"]
        return record
