from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from src.models.candle import Candle
from src.runtime.shadow import LiveShadowService, ShadowRuntime
from src.runtime.state import (
    RuntimeIntegrityError,
    RuntimeState,
    RuntimeStateStore,
    ShadowDecisionJournal,
)
from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext
from src.strategy.models import (
    DecisionReason,
    StrategyAction,
    StrategyDecision,
    TrendMomentumConfig,
)


BASE = datetime(2026, 8, 1, tzinfo=UTC)


def candle(index: int, *, closed: bool = True, close: str | None = None) -> Candle:
    value = Decimal(close or str(100 + index))
    return Candle(
        timestamp=BASE + timedelta(minutes=15 * index),
        symbol="BTCUSDC",
        interval="15m",
        open=value,
        high=value + Decimal("1"),
        low=value - Decimal("1"),
        close=value,
        volume=Decimal("10"),
        is_closed=closed,
    )


class RecordingStrategy(BaseStrategy):
    NAME = "RecordingStrategy"
    VERSION = "1"

    def __init__(self, *, enter: bool = False) -> None:
        self.contexts: list[StrategyContext] = []
        self.enter = enter

    def evaluate(self, context: StrategyContext) -> StrategyDecision:
        self.contexts.append(context)
        if self.enter:
            return StrategyDecision(
                StrategyAction.ENTER_LONG,
                DecisionReason.PRICE_BREAKOUT_ENTRY,
                "synthetic shadow entry",
                risk_budget=Decimal("1"),
                stop_distance=Decimal("2"),
                reward_risk_ratio=Decimal("2"),
                max_quote_amount=Decimal("50"),
            )
        return StrategyDecision(
            StrategyAction.HOLD,
            DecisionReason.NO_ENTRY,
            "synthetic hold",
        )


def runtime(
    tmp_path,
    strategy: RecordingStrategy,
    *,
    run_id: str = "test",
    storage_id: str | None = None,
) -> ShadowRuntime:
    root = tmp_path / (storage_id or run_id)
    return ShadowRuntime(
        strategy=strategy,
        strategy_config=TrendMomentumConfig(),
        run_identity=run_id,
        state_store=RuntimeStateStore(root / "state.json"),
        journal=ShadowDecisionJournal(root / "decisions.jsonl"),
        stale_after=timedelta(seconds=10),
    )


def bootstrap(instance: ShadowRuntime, count: int = 60) -> None:
    now = candle(count - 1).timestamp + timedelta(minutes=15)
    instance.start(at=now)
    assert instance.synchronize(
        tuple(candle(index) for index in range(count)), at=now, recovery=False
    ) == ()
    assert instance.state is RuntimeState.READY


def test_only_closed_candles_evaluate_and_duplicate_is_suppressed(tmp_path) -> None:
    strategy = RecordingStrategy()
    instance = runtime(tmp_path, strategy)
    bootstrap(instance)
    received = candle(60).timestamp + timedelta(minutes=15)
    assert instance.accept_market_candle(candle(60, closed=False), received_at=received) is None
    assert strategy.contexts == []
    assert instance.snapshot.last_observed_candle_at_utc == candle(60).timestamp.isoformat().replace(
        "+00:00", "Z"
    )
    assert instance.snapshot.last_observed_candle_closed is False
    decision = instance.accept_market_candle(candle(60), received_at=received)
    assert decision is not None
    assert len(strategy.contexts) == 1
    assert strategy.contexts[0].current_candle.is_closed
    assert instance.accept_market_candle(candle(60), received_at=received) is None
    assert len(strategy.contexts) == 1
    assert instance.snapshot.counters.duplicate_candles_suppressed == 1


def test_stale_state_blocks_strategy_evaluation(tmp_path) -> None:
    strategy = RecordingStrategy()
    instance = runtime(tmp_path, strategy)
    bootstrap(instance)
    transition = datetime.fromisoformat(instance.snapshot.last_transition_at_utc.replace("Z", "+00:00"))
    assert instance.check_stale(now=transition + timedelta(seconds=11))
    assert instance.state is RuntimeState.STALE
    assert instance.accept_market_candle(
        candle(60), received_at=transition + timedelta(seconds=12)
    ) is None
    assert strategy.contexts == []
    assert instance.snapshot.counters.blocked_evaluations == 1


def test_disconnect_recovery_bridges_and_evaluates_each_missing_close_once(tmp_path) -> None:
    strategy = RecordingStrategy()
    instance = runtime(tmp_path, strategy)
    bootstrap(instance)
    at = candle(60).timestamp + timedelta(minutes=15)
    instance.mark_disconnected(at=at)
    assert instance.state is RuntimeState.RECOVERING
    decisions = instance.synchronize(
        tuple(candle(index) for index in range(62)), at=at, recovery=True
    )
    assert len(decisions) == 2
    assert [context.current_candle.timestamp for context in strategy.contexts] == [
        candle(60).timestamp,
        candle(61).timestamp,
    ]
    assert instance.state is RuntimeState.READY
    assert instance.snapshot.counters.disconnects == 1
    assert instance.snapshot.counters.recovery_attempts == 1
    assert instance.snapshot.counters.successful_recoveries == 1


def test_restart_recovers_watermark_and_hash_chained_journal(tmp_path) -> None:
    first_strategy = RecordingStrategy()
    first = runtime(tmp_path, first_strategy, run_id="restart")
    bootstrap(first)
    first.accept_market_candle(
        candle(60), received_at=candle(60).timestamp + timedelta(minutes=15)
    )
    first.stop(at=candle(61).timestamp)

    second_strategy = RecordingStrategy()
    second = runtime(tmp_path, second_strategy, run_id="restart")
    second.start(at=candle(61).timestamp)
    second.synchronize(
        tuple(candle(index) for index in range(61)),
        at=candle(61).timestamp,
        recovery=False,
    )
    assert second.accept_market_candle(
        candle(60), received_at=candle(61).timestamp
    ) is None
    second.accept_market_candle(
        candle(61), received_at=candle(62).timestamp
    )
    records = tuple(second.journal.records())
    assert len(records) == 2
    assert records[1]["previous_hash"] == records[0]["event_hash"]
    assert len(second_strategy.contexts) == 1
    assert second.snapshot.counters.recovery_attempts == 1
    assert second.snapshot.counters.successful_recoveries == 1


def test_shadow_logging_is_deterministic_and_has_no_order_boundary(tmp_path) -> None:
    lines = []
    for storage_id in ("left", "right"):
        strategy = RecordingStrategy(enter=True)
        instance = runtime(
            tmp_path,
            strategy,
            run_id="deterministic",
            storage_id=storage_id,
        )
        bootstrap(instance)
        instance.accept_market_candle(
            candle(60), received_at=candle(61).timestamp
        )
        record = tuple(instance.journal.records())[0]
        lines.append(record)
        assert record["orders_enabled"] is False
        assert record["authentication_used"] is False
        assert record["decision"]["action"] == "ENTER_LONG"
        assert record["signal_time_utc"] == candle(61).timestamp.isoformat().replace(
            "+00:00", "Z"
        )
        assert record["reference_price"] == str(candle(60).close)
        assert record["strategy"] == {"name": "RecordingStrategy", "version": "1"}
        assert record["run_identity"] == "deterministic"
        assert instance.snapshot.counters.orders_attempted == 0
    assert lines[0] == lines[1]


def test_recovery_gap_is_explicit_integrity_failure(tmp_path) -> None:
    instance = runtime(tmp_path, RecordingStrategy())
    bootstrap(instance)
    instance.mark_disconnected(at=candle(60).timestamp)
    with pytest.raises(RuntimeIntegrityError, match="not contiguous"):
        instance.synchronize(
            (*tuple(candle(index) for index in range(60)), candle(61)),
            at=candle(62).timestamp,
            recovery=True,
        )
    assert instance.state is RuntimeState.STALE
    assert instance.snapshot.counters.candle_gaps == 1


def test_service_reconnect_uses_public_history_recovery(tmp_path) -> None:
    strategy = RecordingStrategy()
    instance = runtime(tmp_path, strategy, run_id="service")
    histories = [
        tuple(candle(index) for index in range(60)),
        tuple(candle(index) for index in range(61)),
    ]

    async def load_history():
        return histories.pop(0)

    class FakeStream:
        def subscribe(self, handler):
            self.candle_handler = handler
            return lambda: None

        def subscribe_connection(self, handler):
            self.connection_handler = handler
            return lambda: None

        async def run(self, stop_event=None):
            await self.connection_handler(True)
            await self.connection_handler(False)
            await self.connection_handler(True)
            await stop_event.wait()

    clock = iter(
        BASE + timedelta(days=3, seconds=index) for index in range(20)
    )
    service = LiveShadowService(
        runtime=instance,
        stream=FakeStream(),
        load_closed_history=load_history,
        now=lambda: next(clock),
    )
    asyncio.run(service.run(duration_seconds=0.02))
    assert instance.state is RuntimeState.STOPPED
    assert len(strategy.contexts) == 1
    assert instance.snapshot.counters.disconnects == 1
    assert instance.snapshot.counters.successful_recoveries == 1
