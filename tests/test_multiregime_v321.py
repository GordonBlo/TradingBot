from __future__ import annotations

import hashlib
import importlib
import csv
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from src.backtest.models import BacktestConfig
from src.diagnostics.loader import ResearchRunLoader
from src.historical.dataset import HistoricalDataset
from src.hypotheses.models import HypothesisId, HypothesisSuiteConfig, hypothesis_registry
from src.hypotheses.manifest import (
    BlindHoldoutRegistration,
    HoldoutStatus,
    ResearchManifest,
    ResearchManifestStore,
)
from src.hypotheses.runner import HypothesisMetrics
from src.models.candle import Candle
from src.models.market_data_source import MarketDataSource
from src.research.evaluation import evaluate_strategy_period
from src.research.multiregime.models import (
    MultiRegimeClassification,
    MultiRegimeConfig,
    PartitionKind,
    PartitionStatus,
    ResearchPartitionMetadata,
    ResearchRegion,
    ResearchWindow,
)
from src.research.multiregime.regime_context import build_regime_context
from src.research.multiregime.stability import classify_candidate, support_gate
from src.research.multiregime.runner import MultiRegimeResearchRunner
from src.research.multiregime.windows import construct_windows
from src.strategy.baseline_trend import TrendMomentumBaselineStrategy
from src.strategy.models import TrendMomentumConfig


UTC = timezone.utc
EXPANSION_START = datetime(2023, 1, 1, tzinfo=UTC)
HOLDOUT_START = datetime(2025, 8, 1, tzinfo=UTC)
HOLDOUT_END = datetime(2026, 2, 1, tzinfo=UTC)
CONSUMED_END = datetime(2026, 8, 18, tzinfo=UTC)


def _candle(timestamp: datetime, index: int) -> Candle:
    close = Decimal("100") + Decimal(index % 37) / Decimal("10")
    return Candle(
        timestamp=timestamp,
        symbol="BTCUSDC",
        interval="15m",
        open=close,
        high=close + Decimal("1"),
        low=close - Decimal("1"),
        close=close,
        volume=Decimal("10") + Decimal(index % 5),
        is_closed=True,
    )


def _dataset(start: datetime, end: datetime, *, step: timedelta = timedelta(days=1)) -> HistoricalDataset:
    candles = []
    timestamp = start
    while timestamp < end:
        candles.append(_candle(timestamp, len(candles)))
        timestamp += step
    return HistoricalDataset(
        "BTCUSDC", "15m", MarketDataSource.BINANCE_PUBLIC, tuple(candles)
    )


def _region(
    kind: PartitionKind,
    start: datetime,
    end: datetime,
    *,
    status: PartitionStatus = PartitionStatus.CONSUMED_RESEARCH_DATA,
) -> ResearchRegion:
    dataset = _dataset(start, end)
    return ResearchRegion(
        ResearchPartitionMetadata(
            kind=kind,
            symbol="BTCUSDC",
            interval="15m",
            start=start,
            end=end,
            status=status,
            candle_count=len(dataset.candles),
            created_at=datetime(2026, 8, 19, tzinfo=UTC),
            source="TEST",
            dataset_sha256="synthetic",
        ),
        dataset,
    )


def _metrics(**changes: Decimal | int | None) -> HypothesisMetrics:
    base = HypothesisMetrics(
        trades=100,
        frictionless_pnl=Decimal("100"),
        gross_after_slippage=Decimal("90"),
        fee_drag=Decimal("20"),
        slippage_drag=Decimal("-10"),
        net_pnl=Decimal("70"),
        return_percent=Decimal("7"),
        win_rate_percent=Decimal("50"),
        profit_factor=Decimal("1.20"),
        expectancy=Decimal("0.70"),
        average_winner=Decimal("2"),
        average_loser=Decimal("-1"),
        payoff_ratio=Decimal("2"),
        maximum_drawdown_percent=Decimal("10"),
        total_fees=Decimal("20"),
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


def test_fixed_windows_are_nonoverlapping_and_never_cross_blind_gap() -> None:
    regions = (
        _region(PartitionKind.RESEARCH_EXPANSION, EXPANSION_START, HOLDOUT_START),
        _region(PartitionKind.CONSUMED_RESEARCH, HOLDOUT_END, CONSUMED_END),
    )
    windows = construct_windows(regions, MultiRegimeConfig())

    assert all(window.duration_days <= Decimal("90") for window in windows)
    assert all(left.end <= right.start for left, right in zip(windows, windows[1:]))
    assert all(
        window.end <= HOLDOUT_START or window.start >= HOLDOUT_END for window in windows
    )
    assert not any(
        HOLDOUT_START <= candle.timestamp < HOLDOUT_END
        for window in windows
        for candle in window.replay_dataset.candles
    )
    assert windows[-1].partial_window
    assert windows[-1].duration_days == Decimal("18")
    assert not windows[-1].duration_eligible


def test_sixty_day_partial_window_is_eligible_but_shorter_is_not() -> None:
    sixty = construct_windows(
        (_region(PartitionKind.RESEARCH_EXPANSION, EXPANSION_START, EXPANSION_START + timedelta(days=60)),),
        MultiRegimeConfig(),
    )[0]
    short = construct_windows(
        (_region(PartitionKind.RESEARCH_EXPANSION, EXPANSION_START, EXPANSION_START + timedelta(days=59)),),
        MultiRegimeConfig(),
    )[0]
    assert sixty.partial_window and sixty.duration_eligible
    assert short.partial_window and not short.duration_eligible


def test_locked_holdout_region_is_rejected_even_without_candles() -> None:
    locked = _region(
        PartitionKind.LOCKED_BLIND_HOLDOUT,
        HOLDOUT_START,
        HOLDOUT_END,
        status=PartitionStatus.LOCKED_BLIND_HOLDOUT,
    )
    with pytest.raises(ValueError, match="blind-holdout"):
        construct_windows((locked,), MultiRegimeConfig())


def test_candle_outside_safe_partition_cannot_be_used_as_warmup() -> None:
    safe = _region(
        PartitionKind.CONSUMED_RESEARCH,
        HOLDOUT_END,
        HOLDOUT_END + timedelta(days=90),
    )
    contaminated = ResearchRegion(
        safe.metadata,
        HistoricalDataset(
            safe.dataset.symbol,
            safe.dataset.interval,
            safe.dataset.source,
            (_candle(HOLDOUT_END - timedelta(days=1), 0), *safe.dataset.candles),
        ),
    )
    with pytest.raises(ValueError, match="outside its safe partition"):
        construct_windows((contaminated,), MultiRegimeConfig())


def test_window_warmup_is_local_and_counted_dataset_excludes_it() -> None:
    start = EXPANSION_START
    region = _region(
        PartitionKind.RESEARCH_EXPANSION, start, start + timedelta(days=200)
    )
    first, second, *_ = construct_windows((region,), MultiRegimeConfig())
    assert first.evaluation_start_index == 0
    assert second.evaluation_start_index == 90
    assert second.replay_dataset.candles[0].timestamp == start
    assert second.evaluation_dataset.candles[0].timestamp == second.start
    assert all(candle.timestamp < second.start for candle in second.replay_dataset.candles[:90])


def test_diagnostic_regimes_are_causal_under_future_append() -> None:
    start = EXPANSION_START
    base = _dataset(start, start + timedelta(minutes=15 * 500), step=timedelta(minutes=15))
    window = ResearchWindow(
        window_id="W001",
        start=base.candles[250].timestamp,
        end=base.candles[-1].timestamp + timedelta(minutes=15),
        duration_days=Decimal("2.604166666666666666666666667"),
        partition_kind=PartitionKind.RESEARCH_EXPANSION,
        data_status=PartitionStatus.CONSUMED_RESEARCH_DATA,
        partial_window=True,
        duration_eligible=False,
        replay_dataset=base,
        evaluation_start_index=250,
    )
    _, first = build_regime_context(window, MultiRegimeConfig())
    extended_data = HistoricalDataset(
        base.symbol,
        base.interval,
        base.source,
        base.candles
        + tuple(
            _candle(base.candles[-1].timestamp + timedelta(minutes=15 * index), 500 + index)
            for index in range(1, 21)
        ),
    )
    extended = replace(
        window,
        end=extended_data.candles[-1].timestamp + timedelta(minutes=15),
        replay_dataset=extended_data,
    )
    _, second = build_regime_context(extended, MultiRegimeConfig())
    assert second[: len(first)] == first


def test_regime_context_is_diagnostic_only_and_cannot_change_h0() -> None:
    source = ResearchRunLoader().load("adeef00722e9704d")
    period = source.periods[0]
    before = evaluate_strategy_period(
        period.dataset,
        strategy=TrendMomentumBaselineStrategy(source.strategy_config),
        strategy_config=source.strategy_config,
        backtest_config=source.backtest_config,
    )
    window = ResearchWindow(
        window_id="IDENTITY",
        start=period.dataset.candles[0].timestamp,
        end=period.dataset.candles[-1].timestamp + timedelta(minutes=15),
        duration_days=Decimal("103"),
        partition_kind=PartitionKind.CONSUMED_RESEARCH,
        data_status=PartitionStatus.CONSUMED_RESEARCH_DATA,
        partial_window=False,
        duration_eligible=True,
        replay_dataset=period.dataset,
        evaluation_start_index=0,
    )
    build_regime_context(window, MultiRegimeConfig())
    after = evaluate_strategy_period(
        period.dataset,
        strategy=TrendMomentumBaselineStrategy(source.strategy_config),
        strategy_config=source.strategy_config,
        backtest_config=source.backtest_config,
    )
    assert before == after


def test_frozen_h0_reproduces_existing_19008_candle_research_exactly() -> None:
    source = ResearchRunLoader().load("adeef00722e9704d")
    assert sum(len(period.dataset.candles) for period in source.periods) == 19_008
    reproduced_trades = 0
    for period in source.periods:
        result = evaluate_strategy_period(
            period.dataset,
            strategy=TrendMomentumBaselineStrategy(source.strategy_config),
            strategy_config=source.strategy_config,
            backtest_config=source.backtest_config,
        )
        assert result.backtest.trades == period.trades
        assert result.signals == period.signals
        baseline_directory = (
            Path(__file__).resolve().parents[1]
            / "reports"
            / "research"
            / source.research_run_id
            / period.period.value
            / "baseline"
        )
        summary = json.loads(
            (baseline_directory / "summary.json").read_text(encoding="utf-8")
        )
        for name, expected in summary.items():
            actual = getattr(result.backtest.metrics, name)
            assert actual == (
                Decimal(str(expected)) if isinstance(actual, Decimal) else expected
            )
        with (baseline_directory / "equity.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            rows = tuple(csv.DictReader(stream))
        assert len(rows) == len(result.backtest.equity_curve)
        for point, row in zip(result.backtest.equity_curve, rows, strict=True):
            assert point.timestamp.isoformat() == row["timestamp"]
            assert point.cash == Decimal(row["cash"])
            assert point.position_value == Decimal(row["position_value"])
            assert point.total_equity == Decimal(row["total_equity"])
        reproduced_trades += len(result.backtest.trades)
    assert reproduced_trades == 94


def test_h0_h4_registry_and_baseline_source_are_frozen() -> None:
    assert tuple(item.hypothesis_id for item in hypothesis_registry()) == tuple(HypothesisId)
    assert HypothesisSuiteConfig() == HypothesisSuiteConfig()
    baseline_path = Path(__file__).resolve().parents[1] / "src" / "strategy" / "baseline_trend.py"
    assert hashlib.sha256(baseline_path.read_bytes()).hexdigest().upper() == (
        "FF3F7F54C59621D428E7865991FBABC8A3E7172F139798565A5204BA723BAEB9"
    )
    with pytest.raises(ValueError, match="frozen"):
        MultiRegimeConfig(window_days=91)


def test_support_gate_requires_every_preregistered_condition() -> None:
    config = MultiRegimeConfig()
    baseline = _metrics()
    candidate = _metrics(
        trades=60,
        profit_factor=Decimal("1.3"),
        expectancy=Decimal("0.8"),
        maximum_drawdown_percent=Decimal("12.5"),
        average_frictionless_pnl_per_trade=Decimal("1.1"),
    )
    baseline_stress = _metrics(expectancy=Decimal("0.4"))
    candidate_stress = _metrics(expectancy=Decimal("0.4"))
    gate = support_gate(
        candidate=candidate,
        baseline=baseline,
        candidate_stress=candidate_stress,
        baseline_stress=baseline_stress,
        eligible_windows=10,
        frictionless_better_windows=6,
        net_better_windows=6,
        config=config,
    )
    assert gate.all_met
    assert classify_candidate(
        hypothesis_id=HypothesisId.H1,
        gate=gate,
        combined=candidate,
        baseline=baseline,
        eligible_windows=10,
        minimum_trades_warning=30,
        frictionless_better_windows=6,
        net_better_windows=6,
        config=config,
    ) is MultiRegimeClassification.V3_3_ELIGIBLE

    assert not support_gate(
        candidate=replace(candidate, trades=39),
        baseline=baseline,
        candidate_stress=replace(candidate_stress, expectancy=Decimal("0.39")),
        baseline_stress=baseline_stress,
        eligible_windows=10,
        frictionless_better_windows=5,
        net_better_windows=5,
        config=config,
    ).all_met
    assert not support_gate(
        candidate=replace(candidate, maximum_drawdown_percent=Decimal("12.5001")),
        baseline=baseline,
        candidate_stress=candidate_stress,
        baseline_stress=baseline_stress,
        eligible_windows=10,
        frictionless_better_windows=6,
        net_better_windows=6,
        config=config,
    ).drawdown_limit_met


def test_nonpositive_supported_edge_is_not_v33_eligible() -> None:
    config = MultiRegimeConfig()
    baseline = _metrics(
        expectancy=Decimal("-1.0"),
        average_frictionless_pnl_per_trade=Decimal("-1.0"),
    )
    candidate = _metrics(
        trades=60,
        profit_factor=Decimal("1.2"),
        expectancy=Decimal("-0.5"),
        average_frictionless_pnl_per_trade=Decimal("-0.5"),
    )
    gate = support_gate(
        candidate=candidate,
        baseline=baseline,
        candidate_stress=replace(candidate, expectancy=Decimal("-0.7")),
        baseline_stress=replace(baseline, expectancy=Decimal("-0.8")),
        eligible_windows=10,
        frictionless_better_windows=10,
        net_better_windows=10,
        config=config,
    )
    assert gate.all_met
    assert classify_candidate(
        hypothesis_id=HypothesisId.H1,
        gate=gate,
        combined=candidate,
        baseline=baseline,
        eligible_windows=10,
        minimum_trades_warning=30,
        frictionless_better_windows=10,
        net_better_windows=10,
        config=config,
    ) is MultiRegimeClassification.MECHANISM_SUPPORTED


@pytest.mark.parametrize(
    ("change", "failed_field"),
    (
        ({"frictionless_better_windows": 5}, "frictionless_consistency_met"),
        ({"net_better_windows": 5}, "net_consistency_met"),
        ({"candidate_trades": 39}, "trade_count_ratio_met"),
        ({"candidate_stress_expectancy": "0.39"}, "stress_expectancy_not_worse"),
    ),
)
def test_support_gate_failure_modes_are_independent(
    change: dict[str, int | str], failed_field: str
) -> None:
    baseline = _metrics()
    candidate = _metrics(
        trades=int(change.get("candidate_trades", 60)),
        profit_factor=Decimal("1.3"),
        expectancy=Decimal("0.8"),
        maximum_drawdown_percent=Decimal("12.5"),
        average_frictionless_pnl_per_trade=Decimal("1.1"),
    )
    gate = support_gate(
        candidate=candidate,
        baseline=baseline,
        candidate_stress=_metrics(
            expectancy=Decimal(str(change.get("candidate_stress_expectancy", "0.4")))
        ),
        baseline_stress=_metrics(expectancy=Decimal("0.4")),
        eligible_windows=10,
        frictionless_better_windows=int(change.get("frictionless_better_windows", 6)),
        net_better_windows=int(change.get("net_better_windows", 6)),
        config=MultiRegimeConfig(),
    )
    assert not getattr(gate, failed_field)


def _synthetic_runner_inputs() -> tuple[MultiRegimeResearchRunner, tuple[ResearchRegion, ...], tuple[ResearchPartitionMetadata, ...]]:
    pattern = tuple(map(Decimal, ("100", "90", "80", "100", "110", "111", "112", "113")))
    start = EXPANSION_START
    end = start + timedelta(days=630)
    candles = tuple(
        Candle(
            timestamp=start + timedelta(days=index),
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
        start=start,
        end=end,
        status=PartitionStatus.CONSUMED_RESEARCH_DATA,
        candle_count=len(candles),
        created_at=datetime(2026, 8, 19, tzinfo=UTC),
        source="SYNTHETIC_TEST_ONLY",
        dataset_sha256="synthetic",
    )
    strategy = TrendMomentumConfig(
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
    backtest = BacktestConfig(fee_bps=Decimal("0"), slippage_bps=Decimal("0"))
    runner = MultiRegimeResearchRunner(
        multiregime_config=MultiRegimeConfig(),
        hypothesis_config=HypothesisSuiteConfig(),
        strategy_config=strategy,
        backtest_config=backtest,
        minimum_trades_warning=1,
    )
    return runner, (ResearchRegion(metadata, dataset),), (metadata,)


def test_candidate_window_accounts_are_independent_and_rerun_is_deterministic() -> None:
    runner, regions, partitions = _synthetic_runner_inputs()
    first = runner.run(
        run_id="synthetic",
        regions=regions,
        partitions=partitions,
        cost_stress=True,
        previous_h0_reproduction_verified=True,
    )
    second = runner.run(
        run_id="synthetic",
        regions=regions,
        partitions=partitions,
        cost_stress=True,
        previous_h0_reproduction_verified=True,
    )
    assert first == second
    assert len(first.windows) == 7
    for window in first.windows:
        for candidate in window.candidates:
            assert candidate.backtest.metrics.initial_capital == Decimal("1000")
            assert candidate.backtest.config == BacktestConfig(
                fee_bps=Decimal("0"), slippage_bps=Decimal("0")
            )
            assert candidate.backtest.equity_curve[0].cash == Decimal("1000")
            assert candidate.backtest.equity_curve[0].position_value == Decimal("0")
            assert candidate.stress_metrics is not None


def test_changing_diagnostic_labels_cannot_change_any_hypothesis_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, regions, partitions = _synthetic_runner_inputs()
    original = runner.run(
        run_id="labels-a",
        regions=regions,
        partitions=partitions,
        cost_stress=False,
        previous_h0_reproduction_verified=True,
    )
    module = importlib.import_module("src.research.multiregime.runner")
    real_builder = module.build_regime_context

    def relabeled(window: ResearchWindow, config: MultiRegimeConfig):
        market, contexts = real_builder(window, config)
        return market, tuple(
            replace(
                item,
                long_trend=(
                    item.long_trend.__class__.LONG_TREND_DOWN
                    if item.long_trend.value != "LONG_TREND_DOWN"
                    else item.long_trend.__class__.LONG_TREND_UP
                ),
                volatility=(
                    item.volatility.__class__.HIGH_VOLATILITY
                    if item.volatility.value != "HIGH_VOLATILITY"
                    else item.volatility.__class__.LOW_VOLATILITY
                ),
            )
            for item in contexts
        )

    monkeypatch.setattr(module, "build_regime_context", relabeled)
    changed = runner.run(
        run_id="labels-b",
        regions=regions,
        partitions=partitions,
        cost_stress=False,
        previous_h0_reproduction_verified=True,
    )
    for before_window, after_window in zip(original.windows, changed.windows, strict=True):
        for before, after in zip(before_window.candidates, after_window.candidates, strict=True):
            assert before.backtest == after.backtest
            assert before.diagnostics == after.diagnostics
            assert before.metrics == after.metrics


def test_multiregime_runner_has_no_holdout_lifecycle_side_effect() -> None:
    project_tmp = Path(__file__).resolve().parents[1] / ".test_artifacts" / "tmp"
    project_tmp.mkdir(parents=True, exist_ok=True)
    fixed = datetime(2026, 8, 19, tzinfo=UTC)
    with TemporaryDirectory(prefix="v321-holdout-", dir=project_tmp) as directory:
        store = ResearchManifestStore(Path(directory) / "manifest.json", clock=lambda: fixed)
        locked = BlindHoldoutRegistration(
            symbol="BTCUSDC",
            interval="15m",
            start=HOLDOUT_START,
            end=HOLDOUT_END,
            status=HoldoutStatus.LOCKED_BLIND_HOLDOUT,
            registered_at=fixed,
        )
        store.save(
            ResearchManifest(
                project_version="3.2.1",
                baseline_name="TrendMomentumBaselineStrategy",
                baseline_version="3.0",
                source_research_run_id="synthetic",
                creation_timestamp=fixed,
                hypotheses=tuple(),
                consumed_dataset_ranges=tuple(),
                holdout_status=HoldoutStatus.LOCKED_BLIND_HOLDOUT,
                blind_holdout=locked,
            )
        )
        before = store.path.read_bytes()
        runner, regions, partitions = _synthetic_runner_inputs()
        runner.run(
            run_id="no-lifecycle-access",
            regions=regions,
            partitions=partitions,
            cost_stress=True,
            previous_h0_reproduction_verified=True,
        )
        after = store.load()
        assert store.path.read_bytes() == before
        assert after.holdout_status is HoldoutStatus.LOCKED_BLIND_HOLDOUT
        assert after.blind_holdout == locked
