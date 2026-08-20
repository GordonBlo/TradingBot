"""Typed V3.2.1 partitions, windows, regimes, and stability outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from src.backtest.models import BacktestConfig, BacktestResult
from src.diagnostics.models import PeriodDiagnostics
from src.historical.dataset import HistoricalDataset
from src.hypotheses.models import HypothesisSuiteConfig
from src.hypotheses.runner import HypothesisMetrics
from src.strategy.models import TrendMomentumConfig


class PartitionKind(str, Enum):
    RESEARCH_EXPANSION = "RESEARCH_EXPANSION"
    LOCKED_BLIND_HOLDOUT = "LOCKED_BLIND_HOLDOUT"
    CONSUMED_RESEARCH = "CONSUMED_RESEARCH"


class PartitionStatus(str, Enum):
    READY_FOR_RESEARCH = "READY_FOR_RESEARCH"
    LOCKED_BLIND_HOLDOUT = "LOCKED_BLIND_HOLDOUT"
    CONSUMED_RESEARCH_DATA = "CONSUMED_RESEARCH_DATA"


class LongTrendRegime(str, Enum):
    LONG_TREND_UP = "LONG_TREND_UP"
    LONG_TREND_DOWN = "LONG_TREND_DOWN"
    LONG_TREND_MIXED = "LONG_TREND_MIXED"
    UNAVAILABLE = "UNAVAILABLE"


class DiagnosticVolatilityRegime(str, Enum):
    LOW_VOLATILITY = "LOW_VOLATILITY"
    MEDIUM_VOLATILITY = "MEDIUM_VOLATILITY"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    UNAVAILABLE = "UNAVAILABLE"


class MultiRegimeClassification(str, Enum):
    CONTROL = "CONTROL"
    V3_3_ELIGIBLE = "V3.3_ELIGIBLE"
    MECHANISM_SUPPORTED = "MECHANISM_SUPPORTED_ON_RESEARCH_DATA"
    MIXED = "MIXED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True, slots=True)
class MultiRegimeConfig:
    window_days: int = 90
    minimum_window_days: int = 60
    volatility_lookback: int = 200
    volatility_low_percentile: Decimal = Decimal("33")
    volatility_high_percentile: Decimal = Decimal("67")
    minimum_eligible_windows: int = 6
    consistency_percent_required: Decimal = Decimal("60")
    trade_count_ratio_required: Decimal = Decimal("0.40")
    maximum_drawdown_worse_ratio: Decimal = Decimal("1.25")
    minimum_expansion_days: int = 365
    warmup_candles: int = 250

    def __post_init__(self) -> None:
        decimal_fields = (
            "volatility_low_percentile",
            "volatility_high_percentile",
            "consistency_percent_required",
            "trade_count_ratio_required",
            "maximum_drawdown_worse_ratio",
        )
        for name in decimal_fields:
            value = Decimal(str(getattr(self, name)))
            if not value.is_finite():
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        fixed = {
            "window_days": 90,
            "minimum_window_days": 60,
            "volatility_lookback": 200,
            "volatility_low_percentile": Decimal("33"),
            "volatility_high_percentile": Decimal("67"),
            "minimum_eligible_windows": 6,
            "consistency_percent_required": Decimal("60"),
            "trade_count_ratio_required": Decimal("0.40"),
            "maximum_drawdown_worse_ratio": Decimal("1.25"),
            "minimum_expansion_days": 365,
            "warmup_candles": 250,
        }
        for name, expected in fixed.items():
            if getattr(self, name) != expected:
                raise ValueError(
                    f"{name} is frozen at {expected} for V3.2.1; tuning is refused."
                )


@dataclass(frozen=True, slots=True)
class ResearchPartitionMetadata:
    kind: PartitionKind
    symbol: str
    interval: str
    start: datetime
    end: datetime
    status: PartitionStatus
    candle_count: int | None
    created_at: datetime
    source: str
    dataset_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class ResearchRegion:
    metadata: ResearchPartitionMetadata
    dataset: HistoricalDataset


@dataclass(frozen=True, slots=True)
class ResearchWindow:
    window_id: str
    start: datetime
    end: datetime
    duration_days: Decimal
    partition_kind: PartitionKind
    data_status: PartitionStatus
    partial_window: bool
    duration_eligible: bool
    replay_dataset: HistoricalDataset
    evaluation_start_index: int

    @property
    def evaluation_dataset(self) -> HistoricalDataset:
        return HistoricalDataset(
            symbol=self.replay_dataset.symbol,
            interval=self.replay_dataset.interval,
            source=self.replay_dataset.source,
            candles=self.replay_dataset.candles[self.evaluation_start_index :],
        )


@dataclass(frozen=True, slots=True)
class CandleRegimeContext:
    signal_timestamp: datetime
    long_trend: LongTrendRegime
    volatility: DiagnosticVolatilityRegime
    atr_percent: Decimal | None


@dataclass(frozen=True, slots=True)
class MarketWindowContext:
    btc_return_percent: Decimal
    btc_maximum_drawdown_percent: Decimal
    realized_candle_volatility_percent: Decimal
    average_atr_percent: Decimal | None
    median_atr_percent: Decimal | None
    average_volume_ratio: Decimal | None


@dataclass(frozen=True, slots=True)
class WindowDelta:
    frictionless_expectancy: Decimal
    net_expectancy: Decimal
    profit_factor: Decimal | None
    return_percent: Decimal
    maximum_drawdown_percent: Decimal
    trades: int
    fees: Decimal
    win_rate_percent: Decimal
    payoff_ratio: Decimal | None


@dataclass(frozen=True, slots=True)
class WindowCandidateResult:
    hypothesis_id: Enum
    backtest: BacktestResult
    diagnostics: PeriodDiagnostics
    metrics: HypothesisMetrics
    delta_to_h0: WindowDelta | None
    low_sample_size: bool
    trade_count_collapse: bool
    stress_metrics: HypothesisMetrics | None = None
    stress_diagnostics: PeriodDiagnostics | None = None


@dataclass(frozen=True, slots=True)
class MultiRegimeWindowResult:
    window: ResearchWindow
    market: MarketWindowContext
    regimes: tuple[CandleRegimeContext, ...]
    eligible: bool
    candidates: tuple[WindowCandidateResult, ...]


@dataclass(frozen=True, slots=True)
class WindowDistribution:
    percentile_25: Decimal | None
    median: Decimal | None
    percentile_75: Decimal | None


@dataclass(frozen=True, slots=True)
class SupportGateResult:
    enough_eligible_windows: bool
    frictionless_expectancy_better: bool
    net_expectancy_better: bool
    profit_factor_not_worse: bool
    frictionless_consistency_met: bool
    net_consistency_met: bool
    trade_count_ratio_met: bool
    drawdown_limit_met: bool
    stress_expectancy_not_worse: bool

    @property
    def all_met(self) -> bool:
        return all(
            (
                self.enough_eligible_windows,
                self.frictionless_expectancy_better,
                self.net_expectancy_better,
                self.profit_factor_not_worse,
                self.frictionless_consistency_met,
                self.net_consistency_met,
                self.trade_count_ratio_met,
                self.drawdown_limit_met,
                self.stress_expectancy_not_worse,
            )
        )


@dataclass(frozen=True, slots=True)
class CandidateStability:
    hypothesis_id: Enum
    eligible_windows: int
    combined_metrics: HypothesisMetrics
    stress_combined_metrics: HypothesisMetrics | None
    baseline_combined_metrics: HypothesisMetrics
    baseline_stress_combined_metrics: HypothesisMetrics | None
    frictionless_better_windows: int
    net_better_windows: int
    profit_factor_better_windows: int
    lower_drawdown_windows: int
    frictionless_better_percent: Decimal
    net_better_percent: Decimal
    profit_factor_better_percent: Decimal
    lower_drawdown_percent: Decimal
    trade_count_ratio: Decimal
    trade_count_collapse: bool
    frictionless_expectancy_distribution: WindowDistribution
    net_expectancy_distribution: WindowDistribution
    profit_factor_distribution: WindowDistribution
    return_distribution: WindowDistribution
    drawdown_distribution: WindowDistribution
    gate: SupportGateResult
    classification: Enum
    gross_edge_costs_dominate: bool


@dataclass(frozen=True, slots=True)
class RegimePerformance:
    hypothesis_id: Enum
    dimension: str
    regime: str
    trades: int
    windows_represented: int
    frictionless_expectancy: Decimal
    net_expectancy: Decimal
    profit_factor: Decimal | None
    win_rate_percent: Decimal
    maximum_drawdown_percent: Decimal | None
    low_sample_size: bool


@dataclass(frozen=True, slots=True)
class MultiRegimeResearchResult:
    run_id: str
    symbol: str
    interval: str
    config: MultiRegimeConfig
    hypothesis_config: object
    strategy_config: TrendMomentumConfig
    base_backtest_config: BacktestConfig
    stress_backtest_config: BacktestConfig | None
    partitions: tuple[ResearchPartitionMetadata, ...]
    windows: tuple[MultiRegimeWindowResult, ...]
    stability: tuple[CandidateStability, ...]
    regimes: tuple[RegimePerformance, ...]
    previous_h0_reproduction_verified: bool
    holdout_revealed: bool
    holdout_consumed: bool
    previous_h1_reproduction_verified: bool | None = None
