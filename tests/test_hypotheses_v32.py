from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.backtest.engine import BacktestEngine
from src.backtest.models import BacktestConfig, ExitReason, OrderAction, OrderIntent
from src.diagnostics.models import DiagnosticPeriodInput, DiagnosticRunInput
from src.historical.dataset import HistoricalDataset
from src.hypotheses.candidates import (
    EntryGuardStrategy,
    OneRTargetStrategy,
    PullbackConfirmationStrategy,
    build_candidate,
)
from src.hypotheses.causal import CausalRollingPercentile
from src.hypotheses.manifest import (
    ConsumedDatasetRange,
    HoldoutStatus,
    ResearchManifest,
    ResearchManifestStore,
)
from src.hypotheses.models import (
    DatasetStatus,
    HypothesisId,
    HypothesisSuiteConfig,
    JournalEvent,
    hypothesis_registry,
)
from src.hypotheses.report import HypothesisReportWriter
from src.hypotheses.runner import (
    HypothesisMetrics,
    HypothesisResearchRunner,
    metric_delta,
)
from src.models.candle import Candle
from src.models.market_data_source import MarketDataSource
from src.models.research_indicator_snapshot import ResearchIndicatorSnapshot
from src.research.dataset_split import ResearchPeriod
from src.research.evaluation import evaluate_strategy_period
from src.strategy.baseline_trend import TrendMomentumBaselineStrategy
from src.strategy.context import StrategyContext
from src.strategy.models import StrategyAction, TrendMomentumConfig


BASE = datetime(2026, 2, 1, tzinfo=timezone.utc)


def candle(
    index: int,
    close: str = "100",
    *,
    open: str | None = None,
    high: str | None = None,
    low: str | None = None,
) -> Candle:
    close_value = Decimal(close)
    open_value = Decimal(open) if open is not None else close_value
    return Candle(
        timestamp=BASE + timedelta(minutes=15 * index),
        symbol="BTCUSDC",
        interval="15m",
        open=open_value,
        high=Decimal(high) if high is not None else max(open_value, close_value) + 1,
        low=Decimal(low) if low is not None else min(open_value, close_value) - 1,
        close=close_value,
        volume=Decimal("10"),
        is_closed=True,
    )


def snapshot(
    index: int,
    *,
    close: str = "110",
    fast: str | None = "105",
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


def context(
    index: int,
    *,
    current: ResearchIndicatorSnapshot | None = None,
    previous: ResearchIndicatorSnapshot | None = None,
    low: str | None = None,
    has_position: bool = False,
) -> StrategyContext:
    current = current or snapshot(index)
    previous = previous or snapshot(index - 1, fast="99", slow="100")
    current_candle = candle(index, str(current.close), low=low)
    return StrategyContext(
        timestamp=current_candle.timestamp + timedelta(minutes=15),
        current_candle=current_candle,
        recent_history=(current_candle,),
        indicators=current,
        previous_indicators=previous,
        has_position=has_position,
        bars_in_position=1 if has_position else 0,
        equity=Decimal("1000"),
        cash_usdc=Decimal("950") if has_position else Decimal("1000"),
        completed_trade_count=0,
        bars_since_exit=None,
        entry_fee_rate=Decimal("0.001"),
    )


def populate_guard(strategy: EntryGuardStrategy, count: int = 100) -> None:
    for index in range(1, count + 1):
        decision = strategy.evaluate(
            context(
                index,
                current=snapshot(index, close="100", fast="101", slow="100", atr="1"),
                previous=snapshot(index - 1, fast="101", slow="100", atr="1"),
            )
        )
        assert decision.action is StrategyAction.HOLD


def test_causal_percentile_excludes_current_and_future_observations() -> None:
    rolling = CausalRollingPercentile(lookback=4, percentile_rank=Decimal("75"))
    for value in map(Decimal, ("1", "2", "3", "4")):
        assert rolling.evaluate_then_observe(value) is None

    threshold_at_t = rolling.evaluate_then_observe(Decimal("1000"))
    assert threshold_at_t == Decimal("3.25")
    assert rolling.evaluate_then_observe(Decimal("-1000")) == Decimal("253")
    assert threshold_at_t == Decimal("3.25")


def test_h1_guard_rejects_above_q75_and_accepts_exact_boundary() -> None:
    config = TrendMomentumConfig()
    rejected = build_candidate(HypothesisId.H1, config)
    assert isinstance(rejected, EntryGuardStrategy)
    populate_guard(rejected)
    high = rejected.evaluate(
        context(
            101,
            current=snapshot(101, close="100", fast="101", slow="99", atr="2"),
            previous=snapshot(100, fast="98", slow="99", atr="1"),
        )
    )
    assert high.action is StrategyAction.HOLD
    assert rejected.journal[-1].event is JournalEvent.SKIPPED_ENTRY
    assert rejected.journal[-1].values["atr_percent"] == Decimal("2")
    assert rejected.journal[-1].values["atr_percentile_threshold"] == Decimal("1")

    accepted = build_candidate(HypothesisId.H1, config)
    assert isinstance(accepted, EntryGuardStrategy)
    populate_guard(accepted)
    boundary = accepted.evaluate(
        context(
            101,
            current=snapshot(101, close="100", fast="101", slow="99", atr="1"),
            previous=snapshot(100, fast="98", slow="99", atr="1"),
        )
    )
    assert boundary.action is StrategyAction.ENTER_LONG


def test_h2_guard_is_causal_and_boundary_is_inclusive() -> None:
    config = TrendMomentumConfig()
    rejected = build_candidate(HypothesisId.H2, config)
    assert isinstance(rejected, EntryGuardStrategy)
    populate_guard(rejected)
    high = rejected.evaluate(
        context(
            101,
            current=snapshot(101, close="110", fast="102", slow="100", atr="1"),
            previous=snapshot(100, fast="99", slow="100", atr="1"),
        )
    )
    assert high.action is StrategyAction.HOLD
    assert rejected.journal[-1].values["ema_spread_percent"] == Decimal("2")
    assert rejected.journal[-1].values["ema_percentile_threshold"] == Decimal("1")

    accepted = build_candidate(HypothesisId.H2, config)
    assert isinstance(accepted, EntryGuardStrategy)
    populate_guard(accepted)
    exact = accepted.evaluate(
        context(
            101,
            current=snapshot(101, close="110", fast="101", slow="100", atr="1"),
            previous=snapshot(100, fast="99", slow="100", atr="1"),
        )
    )
    assert exact.action is StrategyAction.ENTER_LONG


def test_h3_arms_then_confirms_on_a_later_closed_pullback() -> None:
    strategy = PullbackConfirmationStrategy(TrendMomentumConfig(), 8)
    armed = strategy.evaluate(context(1))
    assert armed.action is StrategyAction.HOLD
    assert strategy.is_armed
    assert [item.event for item in strategy.journal] == [
        JournalEvent.CROSSOVER_DETECTED,
        JournalEvent.ARMED,
    ]

    confirmed = strategy.evaluate(
        context(
            2,
            current=snapshot(2, close="106", fast="105", slow="100"),
            previous=snapshot(1, fast="105", slow="100"),
            low="104",
        )
    )
    assert confirmed.action is StrategyAction.ENTER_LONG
    assert confirmed.reward_risk_ratio == Decimal("2.0")
    assert not strategy.is_armed
    assert strategy.journal[-1].event is JournalEvent.CONFIRMED
    assert strategy.journal[-1].values["bars_observed"] == 1


def test_h3_expires_after_exactly_eight_subsequent_closed_bars() -> None:
    strategy = PullbackConfirmationStrategy(TrendMomentumConfig(), 8)
    strategy.evaluate(context(1))
    for index in range(2, 10):
        decision = strategy.evaluate(
            context(
                index,
                current=snapshot(index, close="110", fast="105", slow="100"),
                previous=snapshot(index - 1, fast="105", slow="100"),
                low="106",
            )
        )
    assert decision.action is StrategyAction.HOLD
    assert strategy.journal[-1].event is JournalEvent.EXPIRED
    assert not strategy.is_armed


@pytest.mark.parametrize(
    "invalid",
    (
        snapshot(2, close="101", fast="100", slow="100"),
        snapshot(2, close="100", fast="105", slow="100"),
    ),
)
def test_h3_invalidates_broken_trend_or_close(invalid: ResearchIndicatorSnapshot) -> None:
    strategy = PullbackConfirmationStrategy(TrendMomentumConfig(), 8)
    strategy.evaluate(context(1))
    decision = strategy.evaluate(
        context(
            2,
            current=invalid,
            previous=snapshot(1, fast="105", slow="100"),
            low="99",
        )
    )
    assert decision.action is StrategyAction.HOLD
    assert strategy.journal[-1].event is JournalEvent.INVALIDATED
    assert not strategy.is_armed


def test_h3_does_not_rearm_without_a_new_crossover() -> None:
    strategy = PullbackConfirmationStrategy(TrendMomentumConfig(), 8)
    strategy.evaluate(context(1))
    strategy.evaluate(
        context(
            2,
            current=snapshot(2, close="100", fast="105", slow="100"),
            previous=snapshot(1, fast="105", slow="100"),
        )
    )
    journal_count = len(strategy.journal)
    persistent = strategy.evaluate(
        context(
            3,
            current=snapshot(3, fast="105", slow="100"),
            previous=snapshot(2, fast="105", slow="100"),
        )
    )
    assert persistent.action is StrategyAction.HOLD
    assert not strategy.is_armed
    assert len(strategy.journal) == journal_count


def test_h3_confirmation_executes_at_next_bar_open_not_confirmation_price() -> None:
    closes = ("100", "90", "80", "100", "95", "96", "97", "110", "111")
    candles = tuple(
        candle(
            index,
            close,
            open="108" if index == 7 else None,
            low="96" if index == 6 else None,
        )
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
        strategy=PullbackConfirmationStrategy(config, 8),
        strategy_config=config,
        backtest_config=BacktestConfig(fee_bps=Decimal("0"), slippage_bps=Decimal("0")),
    )
    entry = next(signal for signal in evaluation.signals if signal.action is StrategyAction.ENTER_LONG)
    trade = evaluation.backtest.trades[0]
    assert entry.timestamp == candles[7].timestamp
    assert trade.entry_time == candles[7].timestamp
    assert trade.entry_price == Decimal("108")
    assert trade.entry_price != candles[6].close


def test_h4_changes_only_target_to_exactly_one_r() -> None:
    config = TrendMomentumConfig()
    baseline = TrendMomentumBaselineStrategy(config).evaluate(context(1))
    h4 = build_candidate(HypothesisId.H4, config)
    assert isinstance(h4, OneRTargetStrategy)
    candidate = h4.evaluate(context(1))
    assert replace(candidate, reward_risk_ratio=baseline.reward_risk_ratio, metadata=baseline.metadata) == baseline
    assert candidate.stop_distance == Decimal("2.0")
    assert candidate.reward_risk_ratio == Decimal("1.0")


def test_h4_one_r_ambiguous_bar_remains_stop_first() -> None:
    data = HistoricalDataset(
        "BTCUSDC",
        "15m",
        MarketDataSource.BINANCE_PUBLIC,
        (
            candle(0, "100", high="101", low="99"),
            candle(1, "100", open="100", high="106", low="94"),
        ),
    )
    result = BacktestEngine(
        BacktestConfig(fee_bps=Decimal("0"), slippage_bps=Decimal("0"))
    ).run(
        data,
        lambda ctx: OrderIntent(
            OrderAction.BUY,
            risk_budget=Decimal("5"),
            stop_distance=Decimal("5"),
            reward_risk_ratio=Decimal("1"),
            max_quote_amount=Decimal("100"),
            reason="H4 1R",
        )
        if ctx.index == 0
        else None,
    )
    assert result.trades[0].exit_reason is ExitReason.STOP_LOSS
    assert result.trades[0].exit_price == Decimal("95")


def test_registry_contains_exactly_h0_through_h4_and_fixed_parameters() -> None:
    registry = hypothesis_registry()
    assert tuple(item.hypothesis_id for item in registry) == tuple(HypothesisId)
    assert len(registry) == 5
    assert registry[1].fixed_parameters == {
        "H1_ATR_PERCENTILE_LOOKBACK": 100,
        "H1_MAX_PERCENTILE": "75",
    }
    assert registry[4].fixed_parameters == {"H4_REWARD_RISK_RATIO": "1.0"}
    with pytest.raises(ValueError, match="pre-registered"):
        HypothesisSuiteConfig(h3_confirmation_window_bars=7)


def test_holdout_overlap_is_refused_and_lifecycle_never_resets() -> None:
    root = Path(__file__).resolve().parents[1] / ".test_artifacts" / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    fixed_time = datetime(2026, 8, 19, tzinfo=timezone.utc)
    with TemporaryDirectory(prefix="manifest-", dir=root) as directory:
        store = ResearchManifestStore(
            Path(directory) / "manifest.json", clock=lambda: fixed_time
        )
        store.save(
            ResearchManifest(
                project_version="3.2",
                baseline_name="TrendMomentumBaselineStrategy",
                baseline_version="3.0",
                source_research_run_id="source",
                creation_timestamp=fixed_time,
                hypotheses=tuple(),
                consumed_dataset_ranges=(
                    ConsumedDatasetRange(
                        "BTCUSDC",
                        "15m",
                        BASE,
                        BASE + timedelta(days=10),
                        "out_of_sample",
                    ),
                ),
            )
        )
        with pytest.raises(ValueError, match="CONSUMED_RESEARCH_DATA"):
            store.register_holdout(
                symbol="BTCUSDC",
                interval="15m",
                start=BASE + timedelta(days=9),
                end=BASE + timedelta(days=11),
            )
        locked = store.register_holdout(
            symbol="BTCUSDC",
            interval="15m",
            start=BASE + timedelta(days=10),
            end=BASE + timedelta(days=11),
        )
        assert locked.holdout_status is HoldoutStatus.LOCKED_BLIND_HOLDOUT
        assert store.reveal().holdout_status is HoldoutStatus.REVEALED
        assert store.consume().holdout_status is HoldoutStatus.CONSUMED
        with pytest.raises(ValueError, match="LOCKED"):
            store.reveal()
        with pytest.raises(ValueError, match="already registered"):
            store.register_holdout(
                symbol="BTCUSDC",
                interval="15m",
                start=BASE + timedelta(days=12),
                end=BASE + timedelta(days=13),
            )


def _small_strategy_config() -> TrendMomentumConfig:
    return TrendMomentumConfig(
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


def _runner_source() -> DiagnosticRunInput:
    pattern = tuple(map(Decimal, ("100", "90", "80", "100", "110", "111", "112", "113")))
    full = tuple(candle(index, str(pattern[index % 8])) for index in range(24))
    config = _small_strategy_config()
    backtest = BacktestConfig()
    inputs = []
    for period_index, period in enumerate(ResearchPeriod):
        period_dataset = HistoricalDataset(
            "BTCUSDC",
            "15m",
            MarketDataSource.BINANCE_PUBLIC,
            full[period_index * 8 : (period_index + 1) * 8],
        )
        evaluation = evaluate_strategy_period(
            period_dataset,
            strategy=TrendMomentumBaselineStrategy(config),
            strategy_config=config,
            backtest_config=backtest,
        )
        inputs.append(
            DiagnosticPeriodInput(
                period, period_dataset, evaluation.backtest.trades, evaluation.signals
            )
        )
    return DiagnosticRunInput(
        research_run_id="synthetic",
        symbol="BTCUSDC",
        interval="15m",
        backtest_config=backtest,
        strategy_config=config,
        minimum_trades_warning=2,
        periods=tuple(inputs),  # type: ignore[arg-type]
    )


def test_h0_reproduces_signals_trades_pnl_equity_and_metrics() -> None:
    source = _runner_source()
    result = HypothesisResearchRunner().run(source)
    h0 = result.experiments[0]
    for recorded, reproduced in zip(source.periods, h0.periods, strict=True):
        direct = evaluate_strategy_period(
            recorded.dataset,
            strategy=TrendMomentumBaselineStrategy(source.strategy_config),
            strategy_config=source.strategy_config,
            backtest_config=source.backtest_config,
        )
        assert reproduced.signals == recorded.signals == direct.signals
        assert reproduced.backtest.trades == recorded.trades == direct.backtest.trades
        assert reproduced.backtest.equity_curve == direct.backtest.equity_curve
        assert reproduced.backtest.metrics == direct.backtest.metrics
    assert result.baseline_reproduction_verified


def test_candidate_runs_are_isolated_deterministic_and_cost_stress_is_real() -> None:
    source = _runner_source()
    runner = HypothesisResearchRunner()
    first = runner.run(source, cost_stress=True)
    second = runner.run(source, cost_stress=True)
    assert first == second
    assert len(first.experiments) == 5
    assert all(
        period.backtest.config.initial_capital_usdc == Decimal("1000")
        for experiment in first.experiments
        for period in experiment.periods
    )
    assert first.stress_backtest_config is not None
    assert first.stress_backtest_config.fee_bps == Decimal("20")
    assert first.stress_backtest_config.slippage_bps == Decimal("4")


def test_manifest_and_report_mark_former_oos_consumed_and_are_reproducible() -> None:
    root = Path(__file__).resolve().parents[1] / ".test_artifacts" / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    source = _runner_source()
    result = HypothesisResearchRunner().run(source)
    with TemporaryDirectory(prefix="hypothesis-report-", dir=root) as directory:
        temporary = Path(directory)
        manifest_store = ResearchManifestStore(
            temporary / "research" / "hypothesis_manifest.json",
            clock=lambda: datetime(2026, 8, 19, tzinfo=timezone.utc),
        )
        manifest = manifest_store.create_from_research_run(source)
        former_oos = manifest.consumed_dataset_ranges[-1]
        assert former_oos.source_period == "out_of_sample"
        assert former_oos.data_status is DatasetStatus.CONSUMED_RESEARCH_DATA

        writer = HypothesisReportWriter(temporary / "reports")
        first = writer.write(result, manifest)
        first_files = {
            path.relative_to(first.directory): path.read_bytes()
            for path in first.directory.rglob("*")
            if path.is_file()
        }
        second = writer.write(result, manifest)
        assert first.directory == second.directory
        assert first_files == {
            path.relative_to(second.directory): path.read_bytes()
            for path in second.directory.rglob("*")
            if path.is_file()
        }
        assert "CONSUMED_RESEARCH_DATA" in first.manifest.read_text(encoding="utf-8")
        assert (first.directory / "H3_pullback_confirmation").is_dir()


def _metrics(**changes: Decimal | int | None) -> HypothesisMetrics:
    base = HypothesisMetrics(
        trades=10,
        frictionless_pnl=Decimal("10"),
        gross_after_slippage=Decimal("9"),
        fee_drag=Decimal("2"),
        slippage_drag=Decimal("-1"),
        net_pnl=Decimal("7"),
        return_percent=Decimal("0.7"),
        win_rate_percent=Decimal("40"),
        profit_factor=Decimal("1.2"),
        expectancy=Decimal("0.7"),
        average_winner=Decimal("2"),
        average_loser=Decimal("-1"),
        payoff_ratio=Decimal("2"),
        maximum_drawdown_percent=Decimal("3"),
        total_fees=Decimal("2"),
        market_exposure_percent=Decimal("5"),
        stop_loss_percent=Decimal("50"),
        take_profit_percent=Decimal("30"),
        trend_exit_percent=Decimal("10"),
        time_exit_percent=Decimal("10"),
        median_mfe_r=Decimal("1"),
        median_mae_r=Decimal("1"),
        reached_one_r_percent=Decimal("50"),
        reached_two_r_percent=Decimal("20"),
        average_holding_bars=Decimal("5"),
        average_frictionless_pnl_per_trade=Decimal("1"),
        average_fee_per_trade=Decimal("0.2"),
        average_slippage_drag_per_trade=Decimal("-0.1"),
        average_total_friction_per_trade=Decimal("0.3"),
        friction_to_absolute_frictionless_percent=Decimal("30"),
    )
    return replace(base, **changes)


def test_delta_calculation_is_candidate_minus_baseline() -> None:
    baseline = _metrics()
    candidate = _metrics(
        trades=8,
        net_pnl=Decimal("9"),
        frictionless_pnl=Decimal("11"),
        expectancy=Decimal("1.1"),
        profit_factor=Decimal("1.5"),
        maximum_drawdown_percent=Decimal("2"),
        total_fees=Decimal("1.5"),
        win_rate_percent=Decimal("50"),
        payoff_ratio=Decimal("1.8"),
    )
    delta = metric_delta(candidate, baseline)
    assert delta.net_pnl == Decimal("2")
    assert delta.frictionless_pnl == Decimal("1")
    assert delta.expectancy == Decimal("0.4")
    assert delta.profit_factor == Decimal("0.3")
    assert delta.maximum_drawdown_percent == Decimal("-1")
    assert delta.trade_count == -2
    assert delta.fees == Decimal("-0.5")
    assert delta.win_rate_percent == Decimal("10")
    assert delta.payoff_ratio == Decimal("-0.2")
