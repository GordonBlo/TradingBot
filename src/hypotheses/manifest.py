"""Persistent V3.2 research-integrity and blind-holdout lifecycle metadata."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from src.diagnostics.models import DiagnosticRunInput
from src.hypotheses.models import (
    DatasetStatus,
    HypothesisSuiteConfig,
    HypothesisStatus,
    hypothesis_registry,
)
from src.market.intervals import next_open_time, require_utc
from src.strategy.baseline_trend import TrendMomentumBaselineStrategy


class HoldoutStatus(str, Enum):
    UNSEEN = "UNSEEN"
    LOCKED_BLIND_HOLDOUT = "LOCKED_BLIND_HOLDOUT"
    REVEALED = "REVEALED"
    CONSUMED = "CONSUMED"


@dataclass(frozen=True, slots=True)
class ConsumedDatasetRange:
    symbol: str
    interval: str
    start: datetime
    end: datetime
    source_period: str
    data_status: DatasetStatus = DatasetStatus.CONSUMED_RESEARCH_DATA

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "interval", self.interval.strip())
        object.__setattr__(self, "start", require_utc(self.start, name="start"))
        object.__setattr__(self, "end", require_utc(self.end, name="end"))
        if self.start >= self.end:
            raise ValueError("Consumed dataset range must be non-empty.")
        if self.data_status is not DatasetStatus.CONSUMED_RESEARCH_DATA:
            raise ValueError("Inspected ranges must be CONSUMED_RESEARCH_DATA.")


@dataclass(frozen=True, slots=True)
class BlindHoldoutRegistration:
    symbol: str
    interval: str
    start: datetime
    end: datetime
    status: HoldoutStatus
    registered_at: datetime
    reveal_timestamp: datetime | None = None
    consumed_timestamp: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "interval", self.interval.strip())
        object.__setattr__(self, "start", require_utc(self.start, name="start"))
        object.__setattr__(self, "end", require_utc(self.end, name="end"))
        object.__setattr__(
            self, "registered_at", require_utc(self.registered_at, name="registered_at")
        )
        if self.start >= self.end:
            raise ValueError("Blind holdout start must be before end.")
        if self.reveal_timestamp is not None:
            object.__setattr__(
                self,
                "reveal_timestamp",
                require_utc(self.reveal_timestamp, name="reveal_timestamp"),
            )
        if self.consumed_timestamp is not None:
            object.__setattr__(
                self,
                "consumed_timestamp",
                require_utc(self.consumed_timestamp, name="consumed_timestamp"),
            )


@dataclass(frozen=True, slots=True)
class ResearchManifest:
    project_version: str
    baseline_name: str
    baseline_version: str
    source_research_run_id: str
    creation_timestamp: datetime
    hypotheses: tuple[dict[str, Any], ...]
    consumed_dataset_ranges: tuple[ConsumedDatasetRange, ...]
    holdout_status: HoldoutStatus = HoldoutStatus.UNSEEN
    blind_holdout: BlindHoldoutRegistration | None = None
    multiregime_runs: tuple[dict[str, Any], ...] = ()
    mechanism_runs: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "creation_timestamp",
            require_utc(self.creation_timestamp, name="creation_timestamp"),
        )
        if self.blind_holdout is None and self.holdout_status is not HoldoutStatus.UNSEEN:
            raise ValueError("A non-UNSEEN holdout status requires a registration.")
        if self.blind_holdout is not None and self.blind_holdout.status is not self.holdout_status:
            raise ValueError("Manifest and registered holdout statuses must match.")


def _plain(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


class ResearchManifestStore:
    """Atomically persist holdout knowledge; state can only move forward."""

    def __init__(
        self,
        path: str | Path = "research/hypothesis_manifest.json",
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.path = Path(path)
        self._clock = clock

    def create_from_research_run(
        self,
        run: DiagnosticRunInput,
        suite_config: HypothesisSuiteConfig | None = None,
    ) -> ResearchManifest:
        if self.path.exists():
            raise ValueError(f"Research manifest already exists: {self.path}")
        fixed = suite_config or HypothesisSuiteConfig()
        consumed = tuple(
            ConsumedDatasetRange(
                symbol=run.symbol,
                interval=run.interval,
                start=period.dataset.candles[0].timestamp,
                end=next_open_time(
                    period.dataset.candles[-1].timestamp, period.dataset.interval
                ),
                source_period=period.period.value,
            )
            for period in run.periods
        )
        hypotheses = tuple(
            {
                **asdict(item),
                "hypothesis_id": item.hypothesis_id.value,
                "status": HypothesisStatus.REGISTERED.value,
            }
            for item in hypothesis_registry(fixed)
        )
        manifest = ResearchManifest(
            project_version="3.2",
            baseline_name=TrendMomentumBaselineStrategy.NAME,
            baseline_version=TrendMomentumBaselineStrategy.VERSION,
            source_research_run_id=run.research_run_id,
            creation_timestamp=self._now(),
            hypotheses=hypotheses,
            consumed_dataset_ranges=consumed,
        )
        self.save(manifest)
        return manifest

    def ensure_for_research_run(
        self,
        run: DiagnosticRunInput,
        suite_config: HypothesisSuiteConfig | None = None,
    ) -> ResearchManifest:
        if not self.path.exists():
            return self.create_from_research_run(run, suite_config)
        manifest = self.load()
        if manifest.source_research_run_id != run.research_run_id:
            raise ValueError(
                "Existing manifest is registered to a different source research run."
            )
        expected = tuple(
            (
                period.dataset.candles[0].timestamp,
                next_open_time(
                    period.dataset.candles[-1].timestamp, period.dataset.interval
                ),
            )
            for period in run.periods
        )
        source_names = {period.period.value for period in run.periods}
        recorded = tuple(
            (item.start, item.end)
            for item in manifest.consumed_dataset_ranges
            if item.source_period in source_names
        )
        if recorded != expected:
            raise ValueError("Manifest consumed ranges do not match source research data.")
        return manifest

    def register_holdout(
        self,
        *,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> ResearchManifest:
        manifest = self.load()
        start = require_utc(start, name="start")
        end = require_utc(end, name="end")
        if start >= end:
            raise ValueError("Blind holdout start must be before end.")
        if manifest.blind_holdout is not None:
            raise ValueError(
                "A blind holdout is already registered; it cannot be replaced or reset."
            )
        normalized_symbol = symbol.strip().upper()
        normalized_interval = interval.strip()
        registered_markets = {
            (item.symbol, item.interval) for item in manifest.consumed_dataset_ranges
        }
        if (normalized_symbol, normalized_interval) not in registered_markets:
            raise ValueError(
                "Blind holdout market must match the manifest's consumed research market."
            )
        for consumed in manifest.consumed_dataset_ranges:
            same_market = (
                consumed.symbol == normalized_symbol
                and consumed.interval == normalized_interval
            )
            if same_market and start < consumed.end and consumed.start < end:
                raise ValueError(
                    "Refusing blind holdout: range overlaps CONSUMED_RESEARCH_DATA "
                    f"[{consumed.start.isoformat()}, {consumed.end.isoformat()})."
                )
        registration = BlindHoldoutRegistration(
            symbol=normalized_symbol,
            interval=normalized_interval,
            start=start,
            end=end,
            status=HoldoutStatus.LOCKED_BLIND_HOLDOUT,
            registered_at=self._now(),
        )
        updated = replace(
            manifest,
            holdout_status=HoldoutStatus.LOCKED_BLIND_HOLDOUT,
            blind_holdout=registration,
        )
        self.save(updated)
        return updated

    def mark_hypotheses_tested_on_consumed_data(self) -> ResearchManifest:
        """Record completion without profit-based promotion or selection."""

        manifest = self.load()
        hypotheses = tuple(
            {**item, "status": HypothesisStatus.TESTED_RESEARCH_DATA.value}
            for item in manifest.hypotheses
        )
        updated = replace(manifest, hypotheses=hypotheses)
        self.save(updated)
        return updated

    def record_multiregime_run(
        self,
        *,
        run_id: str,
        manifest_path: str,
        configuration_sha256: str,
    ) -> ResearchManifest:
        """Append an immutable V3.2.1 preregistration reference."""

        manifest = self.load()
        existing = next(
            (item for item in manifest.multiregime_runs if item["run_id"] == run_id),
            None,
        )
        record = {
            "run_id": run_id,
            "manifest_path": manifest_path,
            "configuration_sha256": configuration_sha256,
        }
        if existing is not None and existing != record:
            raise ValueError("Multi-regime run ID conflicts with existing manifest history.")
        runs = manifest.multiregime_runs if existing is not None else (*manifest.multiregime_runs, record)
        updated = replace(manifest, project_version="3.2.1", multiregime_runs=runs)
        self.save(updated)
        return updated

    def record_mechanism_run(
        self,
        *,
        run_id: str,
        manifest_path: str,
        configuration_sha256: str,
        source_multiregime_run_id: str,
    ) -> ResearchManifest:
        """Append an immutable V3.2.2 preregistration reference."""

        manifest = self.load()
        record = {
            "run_id": run_id,
            "manifest_path": manifest_path,
            "configuration_sha256": configuration_sha256,
            "source_multiregime_run_id": source_multiregime_run_id,
        }
        existing = next(
            (item for item in manifest.mechanism_runs if item["run_id"] == run_id),
            None,
        )
        if existing is not None and existing != record:
            raise ValueError("Mechanism run ID conflicts with existing manifest history.")
        runs = (
            manifest.mechanism_runs
            if existing is not None
            else (*manifest.mechanism_runs, record)
        )
        updated = replace(
            manifest,
            project_version="3.2.2",
            mechanism_runs=runs,
        )
        self.save(updated)
        return updated

    def mark_expansion_consumed(
        self,
        *,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        run_id: str,
    ) -> ResearchManifest:
        """Permanently consume expansion data immediately before evaluation."""

        manifest = self.load()
        start = require_utc(start, name="start")
        end = require_utc(end, name="end")
        holdout = manifest.blind_holdout
        if holdout is None or holdout.status is not HoldoutStatus.LOCKED_BLIND_HOLDOUT:
            raise ValueError("Expansion evaluation requires a locked blind holdout.")
        if start < holdout.end and holdout.start < end:
            raise ValueError("Research expansion overlaps the locked blind holdout.")
        source_period = f"multiregime_expansion:{run_id}"
        existing = next(
            (
                item
                for item in manifest.consumed_dataset_ranges
                if item.source_period == source_period
            ),
            None,
        )
        new_range = ConsumedDatasetRange(
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            source_period=source_period,
        )
        if existing is not None and existing != new_range:
            raise ValueError("Recorded expansion range conflicts with this run.")
        ranges = (
            manifest.consumed_dataset_ranges
            if existing is not None
            else (*manifest.consumed_dataset_ranges, new_range)
        )
        updated = replace(
            manifest,
            project_version="3.2.1",
            consumed_dataset_ranges=ranges,
        )
        self.save(updated)
        return updated

    def reveal(self) -> ResearchManifest:
        manifest = self.load()
        holdout = manifest.blind_holdout
        if holdout is None or holdout.status is not HoldoutStatus.LOCKED_BLIND_HOLDOUT:
            raise ValueError("Only a LOCKED_BLIND_HOLDOUT can be revealed.")
        now = self._now()
        revealed = replace(
            holdout,
            status=HoldoutStatus.REVEALED,
            reveal_timestamp=now,
        )
        updated = replace(
            manifest,
            holdout_status=HoldoutStatus.REVEALED,
            blind_holdout=revealed,
        )
        self.save(updated)
        return updated

    def consume(self) -> ResearchManifest:
        manifest = self.load()
        holdout = manifest.blind_holdout
        if holdout is None or holdout.status is not HoldoutStatus.REVEALED:
            raise ValueError("Only a REVEALED holdout can become CONSUMED.")
        consumed = replace(
            holdout,
            status=HoldoutStatus.CONSUMED,
            consumed_timestamp=self._now(),
        )
        updated = replace(
            manifest,
            holdout_status=HoldoutStatus.CONSUMED,
            blind_holdout=consumed,
        )
        self.save(updated)
        return updated

    def save(self, manifest: ResearchManifest) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = _plain(asdict(manifest))
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def load(self) -> ResearchManifest:
        if not self.path.is_file():
            raise ValueError(f"Research manifest not found: {self.path}")
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        consumed = tuple(
            ConsumedDatasetRange(
                symbol=item["symbol"],
                interval=item["interval"],
                start=datetime.fromisoformat(item["start"]),
                end=datetime.fromisoformat(item["end"]),
                source_period=item["source_period"],
                data_status=DatasetStatus(item["data_status"]),
            )
            for item in raw["consumed_dataset_ranges"]
        )
        holdout_raw = raw.get("blind_holdout")
        holdout = (
            BlindHoldoutRegistration(
                symbol=holdout_raw["symbol"],
                interval=holdout_raw["interval"],
                start=datetime.fromisoformat(holdout_raw["start"]),
                end=datetime.fromisoformat(holdout_raw["end"]),
                status=HoldoutStatus(holdout_raw["status"]),
                registered_at=datetime.fromisoformat(holdout_raw["registered_at"]),
                reveal_timestamp=(
                    datetime.fromisoformat(holdout_raw["reveal_timestamp"])
                    if holdout_raw.get("reveal_timestamp")
                    else None
                ),
                consumed_timestamp=(
                    datetime.fromisoformat(holdout_raw["consumed_timestamp"])
                    if holdout_raw.get("consumed_timestamp")
                    else None
                ),
            )
            if holdout_raw is not None
            else None
        )
        return ResearchManifest(
            project_version=raw["project_version"],
            baseline_name=raw["baseline_name"],
            baseline_version=raw["baseline_version"],
            source_research_run_id=raw["source_research_run_id"],
            creation_timestamp=datetime.fromisoformat(raw["creation_timestamp"]),
            hypotheses=tuple(raw["hypotheses"]),
            consumed_dataset_ranges=consumed,
            holdout_status=HoldoutStatus(raw["holdout_status"]),
            blind_holdout=holdout,
            multiregime_runs=tuple(raw.get("multiregime_runs", ())),
            mechanism_runs=tuple(raw.get("mechanism_runs", ())),
        )

    def _now(self) -> datetime:
        return require_utc(self._clock(), name="clock")
