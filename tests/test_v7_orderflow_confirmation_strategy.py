from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.backtest.models import BacktestConfig, ExitReason
from src.historical.dataset import HistoricalDataset
from src.models.candle import Candle
from src.models.market_data_source import MarketDataSource
from src.models.research_indicator_snapshot import ResearchIndicatorSnapshot
from src.orderflow.aggregation import OrderFlowBucket
from src.research.evaluation import evaluate_strategy_period
from src.strategy.context import StrategyContext
from src.strategy.models import (
    DecisionReason,
    StrategyAction,
    StrategyDecision,
    TrendMomentumConfig,
)
from src.strategy.v6_mtf_continuation import V6MTFContinuationStrategy
from src.strategy.v7_orderflow_confirmation import (
    V7OrderFlowConfirmationStrategy,
    assert_v7_entries_are_v6_subset,
)


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
CONFIG = TrendMomentumConfig()


def candle(timestamp: datetime, *, open: Decimal = Decimal("100"), high: Decimal | None = None, low: Decimal | None = None, close: Decimal = Decimal("100")) -> Candle:
    return Candle(
        timestamp=timestamp,
        symbol="BTCUSDC",
        interval="15m",
        open=open,
        high=max(open, close) if high is None else high,
        low=min(open, close) if low is None else low,
        close=close,
        volume=Decimal("1"),
        is_closed=True,
    )


def strategy_context(
    timestamp: datetime = BASE,
    *,
    decision_time: datetime | None = None,
) -> StrategyContext:
    current = candle(timestamp)
    indicators = ResearchIndicatorSnapshot(
        timestamp=current.timestamp,
        symbol="BTCUSDC",
        interval="15m",
        close=current.close,
        volume=current.volume,
        ema_fast=None,
        ema_slow=None,
        rsi=None,
        atr=None,
        volume_sma=None,
        volume_ratio=None,
    )
    return StrategyContext(
        timestamp=decision_time or timestamp + timedelta(minutes=15),
        current_candle=current,
        recent_history=(current,),
        indicators=indicators,
        previous_indicators=None,
        has_position=False,
        bars_in_position=0,
        equity=Decimal("1000"),
        cash_usdc=Decimal("1000"),
        completed_trade_count=0,
        bars_since_exit=None,
    )


def bucket(
    timestamp: datetime = BASE,
    *,
    buy: Decimal = Decimal("60"),
    sell: Decimal = Decimal("40"),
) -> OrderFlowBucket:
    return OrderFlowBucket(
        bucket_open_time=timestamp.astimezone(timezone.utc),
        bucket_close_time=timestamp.astimezone(timezone.utc) + timedelta(minutes=15),
        aggregate_trade_count=2,
        underlying_trade_count=2,
        total_base_volume=Decimal("2"),
        total_quote_volume=buy + sell,
        taker_buy_base_volume=Decimal("1"),
        taker_buy_quote_volume=buy,
        taker_buy_aggtrade_count=1,
        taker_sell_base_volume=Decimal("1"),
        taker_sell_quote_volume=sell,
        taker_sell_aggtrade_count=1,
    )


def enter_decision(stop_distance: Decimal = Decimal("5")) -> StrategyDecision:
    return StrategyDecision(
        StrategyAction.ENTER_LONG,
        DecisionReason.MTF_CONTINUATION_ENTRY,
        "Frozen V6 BUY",
        risk_budget=Decimal("1"),
        stop_distance=stop_distance,
        reward_risk_ratio=Decimal("2"),
        max_quote_amount=Decimal("50"),
        minimum_stop_distance_fraction=Decimal("0.0096"),
    )


class ControlledV6(V6MTFContinuationStrategy):
    def __init__(self, decision: StrategyDecision) -> None:
        super().__init__(CONFIG)
        self.decision = decision

    def evaluate(self, context: StrategyContext) -> StrategyDecision:
        return self.decision


def v7(
    decision: StrategyDecision,
    buckets: tuple[OrderFlowBucket, ...] = (bucket(),),
    **identity: str,
) -> V7OrderFlowConfirmationStrategy:
    return V7OrderFlowConfirmationStrategy(
        CONFIG,
        buckets=buckets,
        dataset_id=identity.get("dataset_id", V7OrderFlowConfirmationStrategy.ORDERFLOW_DATASET_ID),
        dataset_definition_sha256=identity.get(
            "dataset_definition_sha256",
            V7OrderFlowConfirmationStrategy.ORDERFLOW_DATASET_DEFINITION_SHA256,
        ),
        v6_strategy=ControlledV6(decision),
    )


@pytest.mark.parametrize(
    ("buy", "sell", "total", "expected"),
    (
        (Decimal("60"), Decimal("40"), Decimal("100"), True),
        (Decimal("50"), Decimal("50"), Decimal("100"), False),
        (Decimal("40"), Decimal("60"), Decimal("100"), False),
        (Decimal("0"), Decimal("0"), Decimal("0"), False),
    ),
)
def test_frozen_quote_flow_confirmation_is_strict(
    buy: Decimal, sell: Decimal, total: Decimal, expected: bool
) -> None:
    assert V7OrderFlowConfirmationStrategy.quote_flow_confirms(
        taker_buy_quote_volume=buy,
        taker_sell_quote_volume=sell,
        total_quote_volume=total,
    ) is expected


def test_v6_hold_stays_hold_despite_positive_flow() -> None:
    decision = StrategyDecision(StrategyAction.HOLD, DecisionReason.NO_ENTRY, "V6 hold")

    assert v7(decision).evaluate(strategy_context()) is decision


@pytest.mark.parametrize(
    ("flow_bucket", "expected_action"),
    (
        (bucket(buy=Decimal("60"), sell=Decimal("40")), StrategyAction.ENTER_LONG),
        (bucket(buy=Decimal("50"), sell=Decimal("50")), StrategyAction.HOLD),
        (bucket(buy=Decimal("40"), sell=Decimal("60")), StrategyAction.HOLD),
    ),
)
def test_v7_retains_only_confirmed_v6_buys(
    flow_bucket: OrderFlowBucket, expected_action: StrategyAction
) -> None:
    assert v7(enter_decision(), (flow_bucket,)).evaluate(strategy_context()).action is expected_action


def test_exact_signal_bucket_timestamp_is_required_and_missing_is_reported() -> None:
    strategy = v7(enter_decision(), (bucket(BASE - timedelta(minutes=15)),))

    decision = strategy.evaluate(strategy_context())

    assert decision.action is StrategyAction.HOLD
    assert decision.metadata["orderflow_integrity_status"] == "MISSING_EXACT_BUCKET"
    assert strategy.missing_bucket_timestamps == frozenset({BASE})


def test_next_bucket_cannot_satisfy_current_signal() -> None:
    strategy = v7(enter_decision(), (bucket(BASE + timedelta(minutes=15)),))

    assert strategy.evaluate(strategy_context()).action is StrategyAction.HOLD


def test_timezone_normalized_exact_utc_timestamp_match_succeeds() -> None:
    offset = timezone(timedelta(hours=2))
    local_signal = BASE.astimezone(offset)
    local_close = (BASE + timedelta(minutes=15)).astimezone(offset)

    assert v7(enter_decision()).evaluate(
        strategy_context(local_signal, decision_time=local_close)
    ).action is StrategyAction.ENTER_LONG


def test_incomplete_current_bucket_is_not_exposed() -> None:
    assert v7(enter_decision()).evaluate(
        strategy_context(decision_time=BASE)
    ).action is StrategyAction.HOLD


def test_subset_invariant_accepts_only_v6_entry_timestamps() -> None:
    assert_v7_entries_are_v6_subset(
        (BASE.astimezone(timezone(timedelta(hours=2))),), (BASE,)
    )
    with pytest.raises(ValueError, match="absent from the V6"):
        assert_v7_entries_are_v6_subset((BASE + timedelta(minutes=15),), (BASE,))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("dataset_id", "wrong-dataset", "dataset ID"),
        ("dataset_definition_sha256", "wrong-sha", "SHA-256"),
    ),
)
def test_non_frozen_orderflow_dataset_is_rejected(
    field: str, value: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        v7(enter_decision(), **{field: value})


def test_retained_v6_buy_uses_existing_next_open_stop_first_execution() -> None:
    signal = candle(BASE)
    actual_fill = Decimal("110")
    stop_distance = actual_fill * Decimal("0.0096")
    target = actual_fill + Decimal("2") * stop_distance
    stop = actual_fill - stop_distance
    entry = candle(
        BASE + timedelta(minutes=15),
        open=actual_fill,
        close=actual_fill,
    )
    exit_candle = candle(
        BASE + timedelta(minutes=30),
        open=actual_fill,
        high=target,
        low=stop,
        close=actual_fill,
    )
    dataset = HistoricalDataset(
        symbol="BTCUSDC",
        interval="15m",
        source=MarketDataSource.BINANCE_PUBLIC,
        candles=(signal, entry, exit_candle),
    )
    result = evaluate_strategy_period(
        dataset,
        strategy=v7(enter_decision(Decimal("0.1"))),
        strategy_config=CONFIG,
        backtest_config=BacktestConfig(fee_bps=Decimal("0"), slippage_bps=Decimal("0")),
    ).backtest

    trade = result.trades[0]
    assert trade.entry_price == actual_fill
    assert trade.exit_price == stop
    assert trade.exit_reason is ExitReason.STOP_LOSS
    assert target == actual_fill + Decimal("2") * stop_distance


def test_v7_returns_the_unmodified_v6_risk_decision_when_confirmed() -> None:
    decision = enter_decision()

    assert v7(decision).evaluate(strategy_context()) is decision
