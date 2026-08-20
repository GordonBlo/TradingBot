from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from collections.abc import Iterator

import pytest

from src.backtest.models import BacktestConfig
from src.historical.dataset import HistoricalDataset
from src.models.candle import Candle
from src.models.market_data_source import MarketDataSource
from src.research.dataset_split import (
    ResearchPeriod,
    ResearchPeriodRange,
    split_dataset,
)
from src.research.evaluation import evaluate_strategy_period
from src.research.report import ResearchReportWriter
from src.research.runner import ResearchConfig, StrategyResearchRunner
from src.strategy.base import BaseStrategy
from src.strategy.context import StrategyContext
from src.strategy.models import (
    DecisionReason,
    StrategyAction,
    StrategyDecision,
    TrendMomentumConfig,
)


BASE = datetime(2026, 2, 1, tzinfo=timezone.utc)


@pytest.fixture
def research_tmp_path() -> Iterator[Path]:
    root = Path(__file__).resolve().parents[1] / ".test_artifacts" / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="research-", dir=root) as directory:
        yield Path(directory)


def candle(index: int, close: Decimal) -> Candle:
    return Candle(
        timestamp=BASE + timedelta(minutes=15 * index),
        symbol="BTCUSDC",
        interval="15m",
        open=close,
        high=close + Decimal("1"),
        low=close - Decimal("1"),
        close=close,
        volume=Decimal("10"),
        is_closed=True,
    )


def patterned_dataset(periods: int = 3) -> HistoricalDataset:
    pattern = tuple(map(Decimal, ("100", "90", "80", "100", "110", "111", "112", "113")))
    candles = tuple(
        candle(index, pattern[index % len(pattern)])
        for index in range(len(pattern) * periods)
    )
    return HistoricalDataset(
        "BTCUSDC", "15m", MarketDataSource.BINANCE_PUBLIC, candles
    )


def ranges() -> tuple[ResearchPeriodRange, ResearchPeriodRange, ResearchPeriodRange]:
    return (
        ResearchPeriodRange(
            ResearchPeriod.DEVELOPMENT, BASE, BASE + timedelta(hours=2)
        ),
        ResearchPeriodRange(
            ResearchPeriod.VALIDATION,
            BASE + timedelta(hours=2),
            BASE + timedelta(hours=4),
        ),
        ResearchPeriodRange(
            ResearchPeriod.OUT_OF_SAMPLE,
            BASE + timedelta(hours=4),
            BASE + timedelta(hours=6),
        ),
    )


def strategy_config() -> TrendMomentumConfig:
    return TrendMomentumConfig(
        fast_ema_period=2,
        slow_ema_period=3,
        rsi_period=2,
        rsi_min=Decimal("0"),
        rsi_max=Decimal("100"),
        volume_sma_period=2,
        minimum_volume_ratio=Decimal("0"),
        atr_period=2,
        atr_stop_multiplier=Decimal("2"),
        reward_risk_ratio=Decimal("2"),
        risk_per_trade_percent=Decimal("0.5"),
        maximum_bars_in_position=2,
        cooldown_bars=1,
        maximum_position_notional_usdc=Decimal("50"),
    )


def research_config(*, cost_stress: bool = False) -> ResearchConfig:
    development, validation, oos = ranges()
    return ResearchConfig(
        development=development,
        validation=validation,
        out_of_sample=oos,
        backtest=BacktestConfig(
            fee_bps=Decimal("10"), slippage_bps=Decimal("2")
        ),
        strategy=strategy_config(),
        minimum_trades_warning=30,
        cost_stress=cost_stress,
    )


def test_split_is_contiguous_nonoverlapping_and_end_exclusive() -> None:
    data = patterned_dataset()
    split = split_dataset(data, *ranges())

    assert [len(item.candles) for _, item in split.items()] == [8, 8, 8]
    assert split.development.candles[-1].timestamp + timedelta(minutes=15) == (
        split.validation.candles[0].timestamp
    )
    assert split.validation.candles[-1].timestamp + timedelta(minutes=15) == (
        split.out_of_sample.candles[0].timestamp
    )
    all_timestamps = [
        item.timestamp
        for _, period_dataset in split.items()
        for item in period_dataset.candles
    ]
    assert all_timestamps == sorted(all_timestamps)
    assert len(set(all_timestamps)) == len(all_timestamps)


def test_split_rejects_gap_or_overlap_between_periods() -> None:
    development, validation, oos = ranges()
    invalid_validation = ResearchPeriodRange(
        ResearchPeriod.VALIDATION,
        validation.start + timedelta(minutes=15),
        validation.end,
    )
    with pytest.raises(ValueError, match="contiguous"):
        split_dataset(patterned_dataset(), development, invalid_validation, oos)


def test_each_period_has_independent_initial_account_and_baseline_trade() -> None:
    result = StrategyResearchRunner().run(patterned_dataset(), research_config())

    assert len(result.periods) == 3
    assert all(
        period.baseline.metrics.initial_capital == Decimal("1000")
        for period in result.periods
    )
    assert all(period.baseline.metrics.total_trades == 1 for period in result.periods)
    assert all(period.signals for period in result.periods)
    assert all(row.low_sample_size for row in result.comparison)


def test_research_is_fully_deterministic() -> None:
    runner = StrategyResearchRunner()
    config = research_config()

    first = runner.run(patterned_dataset(), config)
    second = runner.run(patterned_dataset(), config)

    assert first == second
    assert first.comparison == second.comparison
    assert tuple(item.signals for item in first.periods) == tuple(
        item.signals for item in second.periods
    )


def test_cost_stress_reaches_backtest_execution_configuration() -> None:
    result = StrategyResearchRunner().run(
        patterned_dataset(), research_config(cost_stress=True)
    )

    for period in result.periods:
        assert period.cost_stress is not None
        assert period.cost_stress.config.fee_bps == Decimal("20")
        assert period.cost_stress.config.slippage_bps == Decimal("4")
        assert period.cost_stress.metrics.total_fees_paid > (
            period.baseline.metrics.total_fees_paid
        )


def test_research_report_has_stable_id_and_expected_files(
    research_tmp_path: Path,
) -> None:
    result = StrategyResearchRunner().run(patterned_dataset(), research_config())
    writer = ResearchReportWriter(research_tmp_path)

    first = writer.write(result)
    first_contents = {
        path.relative_to(first.directory): path.read_bytes()
        for path in first.directory.rglob("*")
        if path.is_file()
    }
    second = writer.write(result)

    assert first.directory == second.directory
    assert first_contents == {
        path.relative_to(second.directory): path.read_bytes()
        for path in second.directory.rglob("*")
        if path.is_file()
    }
    assert first.config.is_file()
    assert first.summary.is_file()
    assert first.comparison.is_file()
    assert (first.directory / "out_of_sample" / "signals.csv").is_file()


class _HistoryProbeStrategy(BaseStrategy):
    def __init__(self) -> None:
        self.observations: list[tuple[int, int, bool]] = []

    def evaluate(self, context: StrategyContext) -> StrategyDecision:
        index = int(
            (context.current_candle.timestamp - BASE).total_seconds() // (15 * 60)
        )
        future_blocked = False
        try:
            _ = context.recent_history[index + 1]
        except IndexError:
            future_blocked = True
        self.observations.append((index, len(context.recent_history), future_blocked))
        return StrategyDecision(
            StrategyAction.HOLD, DecisionReason.NO_ENTRY, "probe"
        )


def test_strategy_at_index_50_cannot_access_candle_51() -> None:
    data = HistoricalDataset(
        "BTCUSDC",
        "15m",
        MarketDataSource.BINANCE_PUBLIC,
        tuple(candle(index, Decimal("100") + index) for index in range(55)),
    )
    probe = _HistoryProbeStrategy()

    evaluate_strategy_period(
        data,
        strategy=probe,
        strategy_config=TrendMomentumConfig(),
        backtest_config=BacktestConfig(),
    )

    at_50 = next(item for item in probe.observations if item[0] == 50)
    assert at_50 == (50, 51, True)
    assert all(
        length == min(index + 1, 51)
        for index, length, _ in probe.observations
    )
