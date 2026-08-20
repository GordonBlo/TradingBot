from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.backtest.models import BacktestConfig
from src.historical.dataset import HistoricalDataset
from src.hypotheses.candidates import build_candidate
from src.hypotheses.mechanisms import (
    CostOpportunityGuardStrategy,
    MechanismClassification,
    MechanismHypothesisId,
    MechanismSuiteConfig,
    OneBarPersistenceStrategy,
    VolatilityBandStrategy,
    build_mechanism_candidate,
    mechanism_registry,
    round_trip_cost_bps,
    target_distance_bps,
)
from src.hypotheses.models import HypothesisId, HypothesisSuiteConfig, JournalEvent
from src.hypotheses.runner import HypothesisMetrics
from src.models.candle import Candle
from src.models.market_data_source import MarketDataSource
from src.models.research_indicator_snapshot import ResearchIndicatorSnapshot
from src.research.evaluation import evaluate_strategy_period
from src.research.multiregime.mechanism_runner import MechanismResearchRunner
from src.research.multiregime.models import (
    MultiRegimeConfig,
    PartitionKind,
    PartitionStatus,
    ResearchPartitionMetadata,
    ResearchRegion,
    SupportGateResult,
)
from src.research.multiregime.stability import classify_mechanism_candidate
from src.strategy.context import StrategyContext
from src.strategy.models import StrategyAction, TrendMomentumConfig


UTC = timezone.utc
BASE = datetime(2024, 1, 1, tzinfo=UTC)


def _candle(
    index: int,
    close: str = "100",
    *,
    open_price: str | None = None,
) -> Candle:
    close_value = Decimal(close)
    open_value = Decimal(open_price) if open_price is not None else close_value
    return Candle(
        timestamp=BASE + timedelta(minutes=15 * index),
        symbol="BTCUSDC",
        interval="15m",
        open=open_value,
        high=max(open_value, close_value) + Decimal("1"),
        low=min(open_value, close_value) - Decimal("1"),
        close=close_value,
        volume=Decimal("10"),
        is_closed=True,
    )


def _snapshot(
    index: int,
    *,
    close: str = "100",
    fast: str | None = "101",
    slow: str | None = "100",
    atr: str | None = "1",
    rsi: str | None = "58",
    volume_ratio: str | None = "1",
) -> ResearchIndicatorSnapshot:
    return ResearchIndicatorSnapshot(
        timestamp=BASE + timedelta(minutes=15 * index),
        symbol="BTCUSDC",
        interval="15m",
        close=Decimal(close),
        volume=Decimal("10"),
        ema_fast=Decimal(fast) if fast is not None else None,
        ema_slow=Decimal(slow) if slow is not None else None,
        rsi=Decimal(rsi) if rsi is not None else None,
        atr=Decimal(atr) if atr is not None else None,
        volume_sma=Decimal("10") if volume_ratio is not None else None,
        volume_ratio=Decimal(volume_ratio) if volume_ratio is not None else None,
    )


def _context(
    index: int,
    *,
    current: ResearchIndicatorSnapshot | None = None,
    previous: ResearchIndicatorSnapshot | None = None,
) -> StrategyContext:
    current = current or _snapshot(index)
    previous = previous or _snapshot(index - 1, fast="99", slow="100")
    current_candle = _candle(index, str(current.close))
    return StrategyContext(
        timestamp=current_candle.timestamp + timedelta(minutes=15),
        current_candle=current_candle,
        recent_history=(current_candle,),
        indicators=current,
        previous_indicators=previous,
        has_position=False,
        bars_in_position=0,
        equity=Decimal("1000"),
        cash_usdc=Decimal("1000"),
        completed_trade_count=0,
        bars_since_exit=None,
        entry_fee_rate=Decimal("0.001"),
    )


def _h5_with_prior_values() -> VolatilityBandStrategy:
    strategy = VolatilityBandStrategy(
        TrendMomentumConfig(),
        lookback=100,
        minimum_percentile=Decimal("25"),
        maximum_percentile=Decimal("75"),
    )
    for index in range(1, 101):
        decision = strategy.evaluate(
            _context(
                index,
                current=_snapshot(index, atr=str(index), fast="101", slow="100"),
                previous=_snapshot(index - 1, atr=str(index), fast="101", slow="100"),
            )
        )
        assert decision.action is StrategyAction.HOLD
    return strategy


@pytest.mark.parametrize(
    ("atr", "expected_action"),
    (
        ("25", StrategyAction.HOLD),
        ("50", StrategyAction.ENTER_LONG),
        ("76", StrategyAction.HOLD),
    ),
)
def test_h5_causal_q25_q75_band_rejects_below_and_above_and_accepts_between(
    atr: str, expected_action: StrategyAction
) -> None:
    strategy = _h5_with_prior_values()
    decision = strategy.evaluate(
        _context(
            101,
            current=_snapshot(101, atr=atr, fast="101", slow="99"),
            previous=_snapshot(100, fast="98", slow="99"),
        )
    )
    assert strategy.last_thresholds == (Decimal("25.75"), Decimal("75.25"))
    assert decision.action is expected_action
    assert strategy.observation_count == 100


def test_h5_current_and_future_values_cannot_change_recorded_thresholds() -> None:
    strategy = _h5_with_prior_values()
    strategy.evaluate(
        _context(
            101,
            current=_snapshot(101, atr="1000", fast="101", slow="99"),
            previous=_snapshot(100, fast="98", slow="99"),
        )
    )
    threshold_at_signal = strategy.last_thresholds
    strategy.evaluate(
        _context(
            102,
            current=_snapshot(102, atr="0.001", fast="101", slow="100"),
            previous=_snapshot(101, atr="1000", fast="101", slow="100"),
        )
    )
    assert threshold_at_signal == (Decimal("25.75"), Decimal("75.25"))
    assert strategy.observation_count == 100


def test_h6_arms_without_entry_then_confirms_exactly_one_closed_bar_later() -> None:
    strategy = OneBarPersistenceStrategy(TrendMomentumConfig())
    armed = strategy.evaluate(
        _context(1, current=_snapshot(1, close="110", fast="101", slow="100"))
    )
    assert armed.action is StrategyAction.HOLD
    assert strategy.is_pending
    assert strategy.journal[-1].event is JournalEvent.ARMED

    confirmed = strategy.evaluate(
        _context(
            2,
            current=_snapshot(2, close="110", fast="102", slow="100"),
            previous=_snapshot(1, fast="101", slow="100"),
        )
    )
    assert confirmed.action is StrategyAction.ENTER_LONG
    assert not strategy.is_pending
    assert strategy.journal[-1].event is JournalEvent.CONFIRMED


def test_h6_failed_confirmation_cancels_and_requires_a_new_crossover() -> None:
    strategy = OneBarPersistenceStrategy(TrendMomentumConfig())
    strategy.evaluate(
        _context(1, current=_snapshot(1, close="110", fast="101", slow="100"))
    )
    failed = strategy.evaluate(
        _context(
            2,
            current=_snapshot(2, close="110", fast="99", slow="100"),
            previous=_snapshot(1, fast="101", slow="100"),
        )
    )
    assert failed.action is StrategyAction.HOLD
    assert strategy.journal[-1].event is JournalEvent.INVALIDATED

    no_cross = strategy.evaluate(
        _context(
            3,
            current=_snapshot(3, close="110", fast="102", slow="100"),
            previous=_snapshot(2, fast="101", slow="100"),
        )
    )
    assert no_cross.action is StrategyAction.HOLD
    assert not strategy.is_pending

    rearmed = strategy.evaluate(
        _context(
            4,
            current=_snapshot(4, close="110", fast="102", slow="100"),
            previous=_snapshot(3, fast="99", slow="100"),
        )
    )
    assert rearmed.action is StrategyAction.HOLD
    assert strategy.is_pending


def test_h6_confirmed_signal_executes_at_the_following_bar_open() -> None:
    closes = ("100", "90", "80", "100", "95", "96", "97", "110", "111", "112")
    candles = tuple(
        _candle(index, close, open_price="123" if index == 7 else None)
        for index, close in enumerate(closes)
    )
    dataset = HistoricalDataset(
        "BTCUSDC", "15m", MarketDataSource.BINANCE_PUBLIC, candles
    )
    config = TrendMomentumConfig(
        fast_ema_period=2,
        slow_ema_period=3,
        rsi_period=2,
        rsi_min=Decimal("0"),
        rsi_max=Decimal("100"),
        volume_sma_period=2,
        minimum_volume_ratio=Decimal("0"),
        atr_period=2,
    )
    evaluation = evaluate_strategy_period(
        dataset,
        strategy=OneBarPersistenceStrategy(config),
        strategy_config=config,
        backtest_config=BacktestConfig(fee_bps=Decimal("0"), slippage_bps=Decimal("0")),
    )
    entry = next(
        signal
        for signal in evaluation.signals
        if signal.action is StrategyAction.ENTER_LONG
    )
    trade = evaluation.backtest.trades[0]
    assert entry.timestamp == candles[7].timestamp
    assert trade.entry_signal_time == candles[7].timestamp
    assert trade.entry_time == candles[7].timestamp
    assert trade.entry_price == Decimal("123")
    assert trade.entry_price != candles[6].close


def test_h7_formula_and_inclusive_five_times_cost_boundary() -> None:
    assert target_distance_bps(Decimal("0.3"), Decimal("100")) == Decimal("120")
    assert round_trip_cost_bps(Decimal("10"), Decimal("2")) == Decimal("24")
    strategy = CostOpportunityGuardStrategy(
        TrendMomentumConfig(),
        fee_bps=Decimal("10"),
        slippage_bps=Decimal("2"),
        multiplier=Decimal("5"),
    )
    assert strategy.required_target_distance_bps == Decimal("120")
    accepted = strategy.evaluate(
        _context(
            1,
            current=_snapshot(1, atr="0.3", fast="101", slow="99"),
            previous=_snapshot(0, fast="98", slow="99"),
        )
    )
    assert accepted.action is StrategyAction.ENTER_LONG


def test_h7_rejects_target_below_five_times_round_trip_cost() -> None:
    strategy = CostOpportunityGuardStrategy(
        TrendMomentumConfig(),
        fee_bps=Decimal("10"),
        slippage_bps=Decimal("2"),
        multiplier=Decimal("5"),
    )
    rejected = strategy.evaluate(
        _context(
            1,
            current=_snapshot(1, atr="0.299", fast="101", slow="99"),
            previous=_snapshot(0, fast="98", slow="99"),
        )
    )
    assert rejected.action is StrategyAction.HOLD
    assert strategy.journal[-1].values["target_distance_bps"] == Decimal("119.60000")


def test_v322_registry_and_parameters_are_fixed_without_combinations() -> None:
    registry = mechanism_registry()
    assert tuple(item.hypothesis_id for item in registry) == tuple(
        MechanismHypothesisId
    )
    assert registry[2].fixed_parameters == {
        "H5_LOOKBACK": 100,
        "H5_MIN_PERCENTILE": "25",
        "H5_MAX_PERCENTILE": "75",
    }
    assert registry[3].fixed_parameters == {"H6_CONFIRMATION_BARS": 1}
    assert registry[4].fixed_parameters == {"H7_COST_OPPORTUNITY_MULTIPLIER": "5"}
    with pytest.raises(ValueError, match="tuning is refused"):
        MechanismSuiteConfig(h6_confirmation_bars=2)
    with pytest.raises(ValueError, match="tuning is refused"):
        MechanismSuiteConfig(h7_cost_opportunity_multiplier=Decimal("4"))


def test_h0_and_h1_factories_reuse_unchanged_v32_candidates() -> None:
    strategy_config = TrendMomentumConfig()
    v32_config = HypothesisSuiteConfig()
    mechanism_config = MechanismSuiteConfig()
    backtest_config = BacktestConfig()
    for mechanism_id, old_id in (
        (MechanismHypothesisId.H0, HypothesisId.H0),
        (MechanismHypothesisId.H1, HypothesisId.H1),
    ):
        old = build_candidate(old_id, strategy_config, v32_config)
        new = build_mechanism_candidate(
            mechanism_id,
            strategy_config,
            mechanism_config,
            v32_config,
            backtest_config,
        )
        decisions = []
        for index in range(1, 102):
            candidate_context = _context(
                index,
                current=_snapshot(index, fast="101", slow="100"),
                previous=_snapshot(index - 1, fast="101", slow="100"),
            )
            decisions.append((old.evaluate(candidate_context), new.evaluate(candidate_context)))
        crossover = _context(
            102,
            current=_snapshot(102, fast="101", slow="99"),
            previous=_snapshot(101, fast="98", slow="99"),
        )
        decisions.append((old.evaluate(crossover), new.evaluate(crossover)))
        assert all(before == after for before, after in decisions)


def _metrics(
    *,
    trades: int = 100,
    frictionless_expectancy: str = "1",
) -> HypothesisMetrics:
    zero = Decimal("0")
    return HypothesisMetrics(
        trades=trades,
        frictionless_pnl=zero,
        gross_after_slippage=zero,
        fee_drag=zero,
        slippage_drag=zero,
        net_pnl=zero,
        return_percent=zero,
        win_rate_percent=zero,
        profit_factor=Decimal("1"),
        expectancy=zero,
        average_winner=zero,
        average_loser=zero,
        payoff_ratio=None,
        maximum_drawdown_percent=zero,
        total_fees=zero,
        market_exposure_percent=zero,
        stop_loss_percent=zero,
        take_profit_percent=zero,
        trend_exit_percent=zero,
        time_exit_percent=zero,
        median_mfe_r=None,
        median_mae_r=None,
        reached_one_r_percent=zero,
        reached_two_r_percent=zero,
        average_holding_bars=zero,
        average_frictionless_pnl_per_trade=Decimal(frictionless_expectancy),
        average_fee_per_trade=zero,
        average_slippage_drag_per_trade=zero,
        average_total_friction_per_trade=zero,
        friction_to_absolute_frictionless_percent=None,
    )


def test_new_classification_requires_positive_gross_edge_for_next_stage() -> None:
    all_met = SupportGateResult(*(True for _ in range(9)))
    common = {
        "hypothesis_id": MechanismHypothesisId.H5,
        "gate": all_met,
        "baseline": _metrics(frictionless_expectancy="-1"),
        "eligible_windows": 10,
        "minimum_trades_warning": 20,
        "frictionless_better_windows": 8,
        "net_better_windows": 8,
        "config": MultiRegimeConfig(),
    }
    assert classify_mechanism_candidate(
        combined=_metrics(frictionless_expectancy="0.01"), **common
    ) is MechanismClassification.NEXT_STAGE_ELIGIBLE
    assert classify_mechanism_candidate(
        combined=_metrics(frictionless_expectancy="0"), **common
    ) is MechanismClassification.MECHANISM_SUPPORTED


def _synthetic_runner_inputs() -> tuple[
    MechanismResearchRunner,
    tuple[ResearchRegion, ...],
    tuple[ResearchPartitionMetadata, ...],
]:
    pattern = tuple(map(Decimal, ("100", "90", "80", "100", "110", "111", "112", "113")))
    candles = tuple(
        Candle(
            timestamp=BASE + timedelta(days=index),
            symbol="BTCUSDC",
            interval="1d",
            open=pattern[index % len(pattern)],
            high=pattern[index % len(pattern)] + Decimal("1"),
            low=pattern[index % len(pattern)] - Decimal("1"),
            close=pattern[index % len(pattern)],
            volume=Decimal("10"),
            is_closed=True,
        )
        for index in range(630)
    )
    dataset = HistoricalDataset(
        "BTCUSDC", "1d", MarketDataSource.BINANCE_PUBLIC, candles
    )
    metadata = ResearchPartitionMetadata(
        kind=PartitionKind.RESEARCH_EXPANSION,
        symbol="BTCUSDC",
        interval="1d",
        start=BASE,
        end=BASE + timedelta(days=630),
        status=PartitionStatus.CONSUMED_RESEARCH_DATA,
        candle_count=len(candles),
        created_at=datetime(2026, 8, 19, tzinfo=UTC),
        source="SYNTHETIC_TEST_ONLY",
        dataset_sha256="synthetic",
    )
    strategy_config = TrendMomentumConfig(
        fast_ema_period=2,
        slow_ema_period=3,
        rsi_period=2,
        rsi_min=Decimal("0"),
        rsi_max=Decimal("100"),
        volume_sma_period=2,
        minimum_volume_ratio=Decimal("0"),
        atr_period=2,
        maximum_bars_in_position=2,
        cooldown_bars=1,
    )
    runner = MechanismResearchRunner(
        multiregime_config=MultiRegimeConfig(),
        v32_config=HypothesisSuiteConfig(),
        mechanism_config=MechanismSuiteConfig(),
        strategy_config=strategy_config,
        backtest_config=BacktestConfig(
            fee_bps=Decimal("10"), slippage_bps=Decimal("2")
        ),
        minimum_trades_warning=1,
    )
    return runner, (ResearchRegion(metadata, dataset),), (metadata,)


def test_v322_reruns_are_deterministic_with_independent_accounts_and_real_cost_stress() -> None:
    runner, regions, partitions = _synthetic_runner_inputs()
    arguments = {
        "run_id": "synthetic-v322",
        "regions": regions,
        "partitions": partitions,
        "cost_stress": True,
        "previous_h0_reproduction_verified": True,
        "previous_h1_reproduction_verified": True,
    }
    first = runner.run(**arguments)
    second = runner.run(**arguments)
    assert first == second
    assert not first.holdout_revealed
    assert not first.holdout_consumed
    assert first.stress_backtest_config == BacktestConfig(
        fee_bps=Decimal("20"), slippage_bps=Decimal("4")
    )
    for window in first.windows:
        for candidate in window.candidates:
            assert candidate.backtest.metrics.initial_capital == Decimal("1000")
            assert candidate.backtest.equity_curve[0].cash == Decimal("1000")
            assert candidate.backtest.equity_curve[0].position_value == Decimal("0")
            assert candidate.stress_metrics is not None
    h0 = first.stability[0]
    assert h0.stress_combined_metrics is not None
    assert h0.combined_metrics.trades > 0
    assert h0.stress_combined_metrics.total_fees > h0.combined_metrics.total_fees
    assert abs(h0.stress_combined_metrics.average_slippage_drag_per_trade) > abs(
        h0.combined_metrics.average_slippage_drag_per_trade
    )
