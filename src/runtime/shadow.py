from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol

from src.analysis.indicators import IndicatorEngine
from src.backtest.execution import basis_points_rate
from src.market.candle_history import CandleHistory
from src.market.intervals import is_interval_aligned, next_open_time
from src.models.candle import Candle
from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext
from src.strategy.models import StrategyAction, StrategyDecision, TrendMomentumConfig
from src.runtime.state import (
    ALLOWED_TRANSITIONS,
    RuntimeIntegrityError,
    RuntimeSnapshot,
    RuntimeState,
    RuntimeStateStore,
    ShadowDecisionJournal,
    parse_utc,
    utc_text,
)


def _candle_hash(candle: Candle) -> str:
    payload = {
        "timestamp_utc": utc_text(candle.timestamp),
        "symbol": candle.symbol,
        "interval": candle.interval,
        "open": str(candle.open),
        "high": str(candle.high),
        "low": str(candle.low),
        "close": str(candle.close),
        "volume": str(candle.volume),
        "is_closed": candle.is_closed,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise RuntimeIntegrityError(f"unsupported shadow metadata type: {type(value).__name__}")


class PublicCandleStream(Protocol):
    def subscribe(self, handler: Callable[[Candle], Any]) -> Callable[[], None]: ...

    def subscribe_connection(
        self, handler: Callable[[bool], Any]
    ) -> Callable[[], None]: ...

    async def run(self, stop_event: asyncio.Event | None = None) -> None: ...


class ShadowRuntime:
    """Deterministic closed-candle strategy evaluation with no execution boundary."""

    SYMBOL = "BTCUSDC"
    INTERVAL = "15m"

    def __init__(
        self,
        *,
        strategy: BaseStrategy,
        strategy_config: TrendMomentumConfig,
        run_identity: str,
        state_store: RuntimeStateStore,
        journal: ShadowDecisionJournal,
        history_limit: int = 1_000,
        stale_after: timedelta = timedelta(seconds=30),
        equity: Decimal = Decimal("1000"),
        cash_usdc: Decimal = Decimal("1000"),
        fee_bps: Decimal = Decimal("10"),
    ) -> None:
        if not run_identity.strip():
            raise ValueError("shadow run identity must not be empty")
        if not 1 <= history_limit <= 1_000:
            raise ValueError("shadow history limit must be between 1 and 1000")
        if stale_after <= timedelta(0):
            raise ValueError("stale threshold must be positive")
        self.strategy = strategy
        self.strategy_config = strategy_config
        self.run_identity = run_identity.strip()
        self.strategy_name = str(getattr(strategy, "NAME", type(strategy).__name__))
        self.strategy_version = str(getattr(strategy, "VERSION", "UNVERSIONED"))
        self.state_store = state_store
        self.journal = journal
        self.history_limit = history_limit
        self.stale_after = stale_after
        self.equity = Decimal(str(equity))
        self.cash_usdc = Decimal(str(cash_usdc))
        self.entry_fee_rate = basis_points_rate(Decimal(str(fee_bps)))
        self.history = CandleHistory(self.SYMBOL, self.INTERVAL, max_size=history_limit)
        self._indicator_engine = IndicatorEngine()

        loaded = state_store.load()
        self.snapshot = loaded or RuntimeSnapshot(
            run_identity=self.run_identity,
            strategy_name=self.strategy_name,
            strategy_version=self.strategy_version,
        )
        if (
            self.snapshot.run_identity != self.run_identity
            or self.snapshot.strategy_name != self.strategy_name
            or self.snapshot.strategy_version != self.strategy_version
        ):
            raise RuntimeIntegrityError("runtime identity does not match persisted state")
        self._reconcile_journal()

    @property
    def state(self) -> RuntimeState:
        return self.snapshot.state

    def _reconcile_journal(self) -> None:
        records = tuple(self.journal.records())
        if self.snapshot.shadow_sequence > len(records):
            raise RuntimeIntegrityError("runtime state is ahead of the durable shadow journal")
        for record in records:
            if (
                record.get("run_identity") != self.run_identity
                or record.get("strategy", {}).get("name") != self.strategy_name
                or record.get("strategy", {}).get("version") != self.strategy_version
            ):
                raise RuntimeIntegrityError("shadow journal identity mismatch")
        if records:
            latest = records[-1]
            journal_time = str(latest["candle_open_time_utc"])
            state_time = self.snapshot.last_evaluated_candle_at_utc
            if state_time is not None and parse_utc(state_time) > parse_utc(journal_time):
                raise RuntimeIntegrityError("runtime evaluation watermark is ahead of journal")
            self.snapshot.last_evaluated_candle_at_utc = journal_time
            self.snapshot.last_closed_candle_at_utc = journal_time
            self.snapshot.last_evaluated_candle_sha256 = latest["candle_sha256"]
        self.snapshot.shadow_sequence = len(records)
        self.snapshot.shadow_last_hash = self.journal.last_hash

    def _persist(self) -> None:
        self.state_store.save(self.snapshot)

    def _transition(self, target: RuntimeState, *, at: datetime, reason: str) -> None:
        if target is self.state:
            return
        if target not in ALLOWED_TRANSITIONS[self.state]:
            raise RuntimeIntegrityError(
                f"invalid runtime transition {self.state.value}->{target.value}"
            )
        self.snapshot.state = target
        self.snapshot.transition_sequence += 1
        self.snapshot.last_transition_at_utc = utc_text(at)
        self.snapshot.last_transition_reason = reason
        self._persist()

    def start(self, *, at: datetime) -> None:
        if self.state is RuntimeState.STARTING:
            return
        self._transition(RuntimeState.STARTING, at=at, reason="runtime process started")

    def stop(self, *, at: datetime) -> None:
        if self.state is not RuntimeState.STOPPED:
            self._transition(RuntimeState.STOPPED, at=at, reason="bounded runtime stopped")

    def mark_disconnected(self, *, at: datetime) -> None:
        self.snapshot.counters.disconnects += 1
        if self.state is RuntimeState.READY:
            self._transition(
                RuntimeState.RECOVERING,
                at=at,
                reason="public candle stream disconnected",
            )
        else:
            self._persist()

    def recovery_failed(self, *, at: datetime, reason: str) -> None:
        self.snapshot.counters.integrity_failures += 1
        if self.state is not RuntimeState.STALE:
            self._transition(RuntimeState.STALE, at=at, reason=reason)
        else:
            self._persist()

    def check_stale(self, *, now: datetime) -> bool:
        if self.state is not RuntimeState.READY:
            return False
        last_event = parse_utc(self.snapshot.last_market_event_at_utc)
        transition_at = parse_utc(self.snapshot.last_transition_at_utc)
        reference = last_event or transition_at
        if reference is None or now.astimezone(UTC) - reference > self.stale_after:
            self.snapshot.counters.stale_transitions += 1
            self._transition(RuntimeState.STALE, at=now, reason="public candle data became stale")
            return True
        return False

    @staticmethod
    def _validated_candles(candles: Sequence[Candle]) -> tuple[Candle, ...]:
        by_timestamp: dict[datetime, Candle] = {}
        for candle in candles:
            if (
                candle.symbol != ShadowRuntime.SYMBOL
                or candle.interval != ShadowRuntime.INTERVAL
                or not candle.is_closed
                or not is_interval_aligned(candle.timestamp, candle.interval)
            ):
                raise RuntimeIntegrityError(
                    "runtime synchronization requires aligned closed BTCUSDC 15m candles"
                )
            previous = by_timestamp.get(candle.timestamp)
            if previous is not None and previous != candle:
                raise RuntimeIntegrityError("conflicting duplicate recovery candle")
            by_timestamp[candle.timestamp] = candle
        if not by_timestamp:
            raise RuntimeIntegrityError("runtime synchronization returned no closed candles")
        chronological = tuple(
            by_timestamp[timestamp] for timestamp in sorted(by_timestamp)
        )
        for previous, current in zip(chronological, chronological[1:]):
            if current.timestamp != next_open_time(previous.timestamp, previous.interval):
                raise RuntimeIntegrityError("public closed-candle history is not contiguous")
        return chronological

    def synchronize(
        self,
        candles: Sequence[Candle],
        *,
        at: datetime,
        recovery: bool,
    ) -> tuple[StrategyDecision, ...]:
        last_evaluated = parse_utc(self.snapshot.last_evaluated_candle_at_utc)
        effective_recovery = recovery or last_evaluated is not None
        target = RuntimeState.RECOVERING if effective_recovery else RuntimeState.SYNCING
        if self.state is not target:
            self._transition(target, at=at, reason="loading public closed-candle history")
        if effective_recovery:
            self.snapshot.counters.recovery_attempts += 1
            self._persist()
        try:
            source = self._validated_candles(candles)
        except RuntimeIntegrityError as exc:
            if "not contiguous" in str(exc):
                self.snapshot.counters.candle_gaps += 1
            self.recovery_failed(at=at, reason=f"public history integrity failed: {exc}")
            raise
        if last_evaluated is None:
            self.history = CandleHistory(
                self.SYMBOL, self.INTERVAL, max_size=self.history_limit, candles=source
            )
            latest = source[-1]
            self.snapshot.last_closed_candle_at_utc = utc_text(latest.timestamp)
            self.snapshot.last_evaluated_candle_at_utc = utc_text(latest.timestamp)
            self.snapshot.last_evaluated_candle_sha256 = _candle_hash(latest)
            self.snapshot.last_market_event_at_utc = utc_text(at)
            self._transition(RuntimeState.READY, at=at, reason="initial public history synchronized")
            return ()

        known = {candle.timestamp: candle for candle in source}
        watermark = known.get(last_evaluated)
        if watermark is None:
            self.recovery_failed(
                at=at, reason="recovery history does not contain evaluation watermark"
            )
            raise RuntimeIntegrityError("recovery history cannot bridge persisted watermark")
        if _candle_hash(watermark) != self.snapshot.last_evaluated_candle_sha256:
            self.recovery_failed(at=at, reason="persisted candle changed during recovery")
            raise RuntimeIntegrityError("persisted evaluated candle does not match Binance history")

        base = tuple(candle for candle in source if candle.timestamp <= last_evaluated)
        pending = tuple(candle for candle in source if candle.timestamp > last_evaluated)
        expected = next_open_time(last_evaluated, self.INTERVAL)
        for candle in pending:
            if candle.timestamp != expected:
                self.snapshot.counters.candle_gaps += 1
                self.recovery_failed(at=at, reason="closed-candle recovery gap detected")
                raise RuntimeIntegrityError("closed-candle recovery is not contiguous")
            expected = next_open_time(expected, self.INTERVAL)
        self.history = CandleHistory(
            self.SYMBOL, self.INTERVAL, max_size=self.history_limit, candles=base
        )
        self.snapshot.last_market_event_at_utc = utc_text(at)
        if effective_recovery:
            self.snapshot.counters.successful_recoveries += 1
        self._transition(RuntimeState.READY, at=at, reason="public history recovery completed")
        decisions = []
        for candle in pending:
            decision = self._process_closed_candle(candle, source="RECOVERY")
            if decision is not None:
                decisions.append(decision)
        return tuple(decisions)

    def accept_market_candle(
        self, candle: Candle, *, received_at: datetime
    ) -> StrategyDecision | None:
        if (
            candle.symbol != self.SYMBOL
            or candle.interval != self.INTERVAL
            or not is_interval_aligned(candle.timestamp, candle.interval)
        ):
            self.snapshot.counters.integrity_failures += 1
            self._persist()
            raise RuntimeIntegrityError("live runtime received a mismatched candle")
        self.snapshot.counters.market_events += 1
        self.snapshot.last_market_event_at_utc = utc_text(received_at)
        self.snapshot.last_observed_candle_at_utc = utc_text(candle.timestamp)
        self.snapshot.last_observed_candle_sha256 = _candle_hash(candle)
        self.snapshot.last_observed_candle_closed = candle.is_closed
        if not candle.is_closed:
            self.snapshot.counters.open_candle_updates += 1
            self._persist()
            return None
        return self._process_closed_candle(candle, source="LIVE")

    def _process_closed_candle(
        self, candle: Candle, *, source: str
    ) -> StrategyDecision | None:
        if self.state is not RuntimeState.READY:
            self.snapshot.counters.blocked_evaluations += 1
            self._persist()
            return None
        if (
            candle.symbol != self.SYMBOL
            or candle.interval != self.INTERVAL
            or not candle.is_closed
            or not is_interval_aligned(candle.timestamp, candle.interval)
        ):
            self.snapshot.counters.integrity_failures += 1
            self._transition(RuntimeState.STALE, at=datetime.now(UTC), reason="invalid live candle")
            raise RuntimeIntegrityError("live runtime received an invalid closed candle")

        last_evaluated = parse_utc(self.snapshot.last_evaluated_candle_at_utc)
        candle_sha = _candle_hash(candle)
        if last_evaluated is not None and candle.timestamp <= last_evaluated:
            if (
                candle.timestamp == last_evaluated
                and candle_sha != self.snapshot.last_evaluated_candle_sha256
            ):
                self.snapshot.counters.integrity_failures += 1
                self._transition(
                    RuntimeState.STALE,
                    at=next_open_time(candle.timestamp, candle.interval),
                    reason="conflicting evaluated candle duplicate",
                )
                raise RuntimeIntegrityError("evaluated closed candle changed")
            self.snapshot.counters.duplicate_candles_suppressed += 1
            self._persist()
            return None
        if last_evaluated is not None:
            expected = next_open_time(last_evaluated, self.INTERVAL)
            if candle.timestamp != expected:
                self.snapshot.counters.candle_gaps += 1
                self._transition(
                    RuntimeState.RECOVERING,
                    at=next_open_time(last_evaluated, self.INTERVAL),
                    reason="live closed-candle gap detected",
                )
                return None

        self.history.upsert(candle)
        candles = self.history.candles
        snapshots = self._indicator_engine.calculate_research_series(
            candles,
            fast_ema_period=self.strategy_config.fast_ema_period,
            slow_ema_period=self.strategy_config.slow_ema_period,
            rsi_period=self.strategy_config.rsi_period,
            atr_period=self.strategy_config.atr_period,
            volume_sma_period=self.strategy_config.volume_sma_period,
        )
        current = snapshots[-1]
        previous = snapshots[-2] if len(snapshots) > 1 else None
        history_limit = max(
            self.strategy_config.fast_ema_period,
            self.strategy_config.slow_ema_period,
            self.strategy_config.rsi_period + 1,
            self.strategy_config.atr_period,
            self.strategy_config.volume_sma_period,
            self.strategy.required_history_bars,
        ) + 1
        recent_history = (
            candles if self.strategy.requires_full_history else candles[-history_limit:]
        )
        context = StrategyContext(
            timestamp=next_open_time(candle.timestamp, candle.interval),
            current_candle=candle,
            recent_history=recent_history,
            indicators=current,
            previous_indicators=previous,
            has_position=False,
            bars_in_position=0,
            equity=self.equity,
            cash_usdc=self.cash_usdc,
            completed_trade_count=0,
            bars_since_exit=None,
            entry_fee_rate=self.entry_fee_rate,
        )
        try:
            decision = self.strategy.evaluate(context)
        except Exception:
            self.snapshot.counters.evaluation_errors += 1
            self._transition(
                RuntimeState.STALE,
                at=context.timestamp,
                reason="strategy evaluation failed",
            )
            raise

        record = self.journal.append(
            {
                "run_identity": self.run_identity,
                "strategy": {
                    "name": self.strategy_name,
                    "version": self.strategy_version,
                },
                "source": source,
                "signal_time_utc": utc_text(context.timestamp),
                "candle_open_time_utc": utc_text(candle.timestamp),
                "candle_sha256": candle_sha,
                "reference_price": str(candle.close),
                "decision": {
                    "action": decision.action.value,
                    "reason_code": decision.reason_code.value,
                    "reason": decision.reason,
                    "risk_budget": _json_value(decision.risk_budget),
                    "stop_distance": _json_value(decision.stop_distance),
                    "reward_risk_ratio": _json_value(decision.reward_risk_ratio),
                    "max_quote_amount": _json_value(decision.max_quote_amount),
                    "minimum_stop_distance_fraction": _json_value(
                        decision.minimum_stop_distance_fraction
                    ),
                    "metadata": _json_value(dict(decision.metadata)),
                },
                "orders_enabled": False,
                "authentication_used": False,
            }
        )
        self.snapshot.counters.closed_candles += 1
        self.snapshot.counters.evaluations += 1
        if decision.action is StrategyAction.ENTER_LONG:
            self.snapshot.counters.shadow_entry_signals += 1
        self.snapshot.last_closed_candle_at_utc = utc_text(candle.timestamp)
        self.snapshot.last_evaluated_candle_at_utc = utc_text(candle.timestamp)
        self.snapshot.last_evaluated_candle_sha256 = candle_sha
        self.snapshot.shadow_sequence = self.journal.sequence
        self.snapshot.shadow_last_hash = record["event_hash"]
        self._persist()
        return decision


class LiveShadowService:
    """Coordinate public REST recovery, public WebSocket data, and shadow runtime."""

    def __init__(
        self,
        *,
        runtime: ShadowRuntime,
        stream: PublicCandleStream,
        load_closed_history: Callable[[], Awaitable[Sequence[Candle]]],
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.runtime = runtime
        self.stream = stream
        self.load_closed_history = load_closed_history
        self.now = now
        self._lock = asyncio.Lock()
        self._stopping = False

    async def _recover_locked(self) -> None:
        try:
            candles = await self.load_closed_history()
            self.runtime.synchronize(candles, at=self.now(), recovery=True)
        except Exception as exc:
            self.runtime.recovery_failed(
                at=self.now(), reason=f"public history recovery failed: {type(exc).__name__}"
            )
            raise

    async def _on_connection(self, connected: bool) -> None:
        if self._stopping:
            return
        async with self._lock:
            if not connected:
                self.runtime.mark_disconnected(at=self.now())
            elif self.runtime.state in (RuntimeState.STALE, RuntimeState.RECOVERING):
                await self._recover_locked()

    async def _on_candle(self, candle: Candle) -> None:
        async with self._lock:
            if self.runtime.state in (RuntimeState.STALE, RuntimeState.RECOVERING):
                await self._recover_locked()
            self.runtime.accept_market_candle(candle, received_at=self.now())

    async def _monitor_staleness(self, stop_event: asyncio.Event) -> None:
        interval = max(0.1, min(1.0, self.runtime.stale_after.total_seconds() / 2))
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except TimeoutError:
                async with self._lock:
                    if self.runtime.check_stale(now=self.now()):
                        await self._recover_locked()

    async def run_until_stopped(self, stop_event: asyncio.Event) -> None:
        """Run continuously until the caller-owned stop event is set."""

        self.runtime.start(at=self.now())
        initial = await self.load_closed_history()
        self.runtime.synchronize(initial, at=self.now(), recovery=False)
        unsubscribe_candles = self.stream.subscribe(self._on_candle)
        unsubscribe_connections = self.stream.subscribe_connection(self._on_connection)
        stream_task = asyncio.create_task(self.stream.run(stop_event))
        stale_task = asyncio.create_task(self._monitor_staleness(stop_event))
        try:
            await stop_event.wait()
        finally:
            self._stopping = True
            stop_event.set()
            await asyncio.gather(stream_task, stale_task, return_exceptions=True)
            unsubscribe_candles()
            unsubscribe_connections()
            self.runtime.stop(at=self.now())

    async def run(self, *, duration_seconds: float) -> None:
        """Run the same service in an always-bounded smoke/validation mode."""

        if not 0 < duration_seconds <= 86_400:
            raise ValueError("live shadow duration must be within one day")
        stop_event = asyncio.Event()
        task = asyncio.create_task(self.run_until_stopped(stop_event))
        try:
            await asyncio.sleep(duration_seconds)
        finally:
            stop_event.set()
            await task
