"""Independent, deterministic V3 research-period orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace

from src.backtest.benchmark import BuyAndHoldBenchmark
from src.backtest.engine import BacktestEngine
from src.backtest.models import BacktestConfig, BacktestResult
from src.historical.dataset import HistoricalDataset
from src.historical.validator import DatasetValidator
from src.research.comparison import (
    ComparisonRow,
    StabilityDiagnostics,
    comparison_row,
    stability_diagnostics,
)
from src.research.dataset_split import (
    ResearchPeriod,
    ResearchPeriodRange,
    split_dataset,
)
from src.research.evaluation import SignalRecord, evaluate_strategy_period
from src.strategy.baseline_trend import TrendMomentumBaselineStrategy
from src.strategy.models import TrendMomentumConfig


@dataclass(frozen=True, slots=True)
class ResearchConfig:
    development: ResearchPeriodRange
    validation: ResearchPeriodRange
    out_of_sample: ResearchPeriodRange
    backtest: BacktestConfig
    strategy: TrendMomentumConfig
    minimum_trades_warning: int = 30
    cost_stress: bool = False

    def __post_init__(self) -> None:
        if self.minimum_trades_warning < 1:
            raise ValueError("Research minimum-trades warning must be at least 1.")


@dataclass(frozen=True, slots=True)
class PeriodResearchResult:
    period: ResearchPeriod
    baseline: BacktestResult
    benchmark: BacktestResult
    signals: tuple[SignalRecord, ...]
    cost_stress: BacktestResult | None


@dataclass(frozen=True, slots=True)
class ResearchResult:
    config: ResearchConfig
    periods: tuple[PeriodResearchResult, PeriodResearchResult, PeriodResearchResult]
    comparison: tuple[ComparisonRow, ComparisonRow, ComparisonRow]
    stability: StabilityDiagnostics


class StrategyResearchRunner:
    """Run the same frozen baseline independently on all three periods."""

    def __init__(self, *, validator: DatasetValidator | None = None) -> None:
        self._validator = validator or DatasetValidator()

    def run(
        self, dataset: HistoricalDataset, config: ResearchConfig
    ) -> ResearchResult:
        self._validator.validate(
            dataset.candles, symbol=dataset.symbol, interval=dataset.interval
        )
        split = split_dataset(
            dataset,
            config.development,
            config.validation,
            config.out_of_sample,
        )
        period_results: list[PeriodResearchResult] = []
        rows: list[ComparisonRow] = []
        for period, period_dataset in split.items():
            evaluation = evaluate_strategy_period(
                period_dataset,
                strategy=TrendMomentumBaselineStrategy(config.strategy),
                strategy_config=config.strategy,
                backtest_config=config.backtest,
            )
            benchmark = BacktestEngine(config.backtest).run(
                period_dataset,
                BuyAndHoldBenchmark(
                    config.backtest.initial_capital_usdc,
                    config.backtest.fee_bps,
                ),
            )
            stress_result = None
            if config.cost_stress:
                stress_config = replace(
                    config.backtest,
                    fee_bps=config.backtest.fee_bps * 2,
                    slippage_bps=config.backtest.slippage_bps * 2,
                )
                stress_result = evaluate_strategy_period(
                    period_dataset,
                    strategy=TrendMomentumBaselineStrategy(config.strategy),
                    strategy_config=config.strategy,
                    backtest_config=stress_config,
                ).backtest
            result = PeriodResearchResult(
                period=period,
                baseline=evaluation.backtest,
                benchmark=benchmark,
                signals=evaluation.signals,
                cost_stress=stress_result,
            )
            period_results.append(result)
            rows.append(
                comparison_row(
                    period,
                    evaluation.backtest,
                    benchmark,
                    minimum_trades_warning=config.minimum_trades_warning,
                )
            )
        typed_periods = tuple(period_results)
        typed_rows = tuple(rows)
        assert len(typed_periods) == 3 and len(typed_rows) == 3
        return ResearchResult(
            config=config,
            periods=typed_periods,  # type: ignore[arg-type]
            comparison=typed_rows,  # type: ignore[arg-type]
            stability=stability_diagnostics(typed_rows),  # type: ignore[arg-type]
        )

