"""Typed V3.1 diagnostic inputs and analytical outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

from src.backtest.models import BacktestConfig, ExitReason, Trade
from src.historical.dataset import HistoricalDataset
from src.research.dataset_split import ResearchPeriod
from src.research.evaluation import SignalRecord
from src.strategy.models import TrendMomentumConfig


class EvidenceStrength(str, Enum):
    DESCRIPTIVE_ONLY = "DESCRIPTIVE ONLY"
    LOW_SAMPLE_SIZE = "LOW SAMPLE SIZE"
    MODERATE_SAMPLE = "MODERATE SAMPLE"


class MarketRegime(str, Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE_OR_MIXED = "RANGE_OR_MIXED"
    UNAVAILABLE = "UNAVAILABLE"


class VolatilityRegime(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class DiagnosticPeriodInput:
    period: ResearchPeriod | str
    dataset: HistoricalDataset
    trades: tuple[Trade, ...]
    signals: tuple[SignalRecord, ...]


@dataclass(frozen=True, slots=True)
class DiagnosticRunInput:
    research_run_id: str
    symbol: str
    interval: str
    backtest_config: BacktestConfig
    strategy_config: TrendMomentumConfig
    minimum_trades_warning: int
    periods: tuple[
        DiagnosticPeriodInput, DiagnosticPeriodInput, DiagnosticPeriodInput
    ]


@dataclass(frozen=True, slots=True)
class TradeDiagnostic:
    trade_id: str
    entry_signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    entry_price: Decimal
    exit_price: Decimal
    reference_entry_price: Decimal
    reference_exit_price: Decimal
    quantity: Decimal
    exit_reason: ExitReason
    frictionless_pnl: Decimal
    gross_pnl: Decimal
    slippage_drag: Decimal
    fee_cost: Decimal
    net_pnl: Decimal
    return_percent: Decimal
    initial_stop_price: Decimal | None
    initial_risk_per_unit: Decimal | None
    initial_risk_amount: Decimal | None
    r_multiple: Decimal | None
    bars_held: int
    holding_minutes: int
    highest_observed_price: Decimal
    lowest_observed_price: Decimal
    mfe: Decimal
    mae: Decimal
    mfe_percent: Decimal
    mae_percent: Decimal
    mfe_r: Decimal | None
    mae_r: Decimal | None
    bars_to_mfe: int
    bars_to_mae: int
    first_four_bar_mae_r: Decimal | None
    capture_ratio: Decimal | None
    loss_efficiency: Decimal | None
    entry_close: Decimal | None
    entry_rsi: Decimal | None
    entry_atr: Decimal | None
    entry_atr_percent: Decimal | None
    entry_volume_ratio: Decimal | None
    entry_ema_fast: Decimal | None
    entry_ema_slow: Decimal | None
    entry_ema_spread_percent: Decimal | None
    close_vs_slow_ema_percent: Decimal | None
    market_regime: MarketRegime
    volatility_regime: VolatilityRegime
    entry_utc_hour: int
    entry_day_of_week: str
    diagnostic_status: str


@dataclass(frozen=True, slots=True)
class GroupDiagnostics:
    dimension: str
    group: str
    trade_count: int
    percentage_of_trades: Decimal
    win_rate_percent: Decimal
    gross_pnl: Decimal
    net_pnl: Decimal
    expectancy: Decimal
    profit_factor: Decimal | None
    average_r: Decimal | None
    maximum_trade_loss: Decimal | None
    average_bars_held: Decimal
    average_mfe_r: Decimal | None
    average_mae_r: Decimal | None
    evidence_strength: EvidenceStrength


@dataclass(frozen=True, slots=True)
class ExitReasonDiagnostics:
    exit_reason: ExitReason
    trade_count: int
    percentage_of_trades: Decimal
    win_rate_percent: Decimal
    gross_pnl: Decimal
    net_pnl: Decimal
    average_pnl: Decimal
    average_r: Decimal | None
    average_bars_held: Decimal
    average_mfe_r: Decimal | None
    average_mae_r: Decimal | None
    evidence_strength: EvidenceStrength


@dataclass(frozen=True, slots=True)
class OutcomeDiagnostics:
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    average_win: Decimal
    median_win: Decimal
    largest_win: Decimal
    average_loss: Decimal
    median_loss: Decimal
    largest_loss: Decimal
    average_winner_r: Decimal | None
    median_winner_r: Decimal | None
    average_loser_r: Decimal | None
    median_loser_r: Decimal | None
    pnl_percentile_25: Decimal | None
    pnl_percentile_50: Decimal | None
    pnl_percentile_75: Decimal | None
    expectancy_from_outcomes: Decimal
    average_trade_pnl: Decimal


@dataclass(frozen=True, slots=True)
class CostDiagnostics:
    frictionless_pnl: Decimal
    slippage_adjusted_gross_pnl: Decimal
    slippage_drag: Decimal
    fee_drag: Decimal
    net_pnl: Decimal
    fees_percent_initial_capital: Decimal
    average_fee_per_trade: Decimal
    median_fee_per_trade: Decimal
    fees_percent_gross_profitable_movement: Decimal | None
    average_slippage_drag_per_trade: Decimal
    slippage_percent_initial_capital: Decimal
    slippage_relative_to_frictionless_pnl: Decimal | None


@dataclass(frozen=True, slots=True)
class ExcursionDiagnostics:
    median_mfe_r: Decimal | None
    median_mae_r: Decimal | None
    average_bars_to_mfe_all: Decimal
    median_bars_to_mfe_all: Decimal
    average_bars_to_mfe_winners: Decimal
    median_bars_to_mfe_winners: Decimal
    average_bars_to_mfe_losers: Decimal
    median_bars_to_mfe_losers: Decimal
    average_bars_to_mae_all: Decimal
    median_bars_to_mae_all: Decimal
    average_bars_to_mae_winners: Decimal
    median_bars_to_mae_winners: Decimal
    average_bars_to_mae_losers: Decimal
    median_bars_to_mae_losers: Decimal
    mean_profitable_capture_ratio: Decimal | None
    median_profitable_capture_ratio: Decimal | None
    mean_losing_loss_efficiency: Decimal | None
    losing_mae_half_r_first_four_percent: Decimal | None
    losing_mae_one_r_first_four_percent: Decimal | None
    mfe_thresholds: tuple[GroupDiagnostics, ...]
    mae_thresholds: tuple[GroupDiagnostics, ...]


@dataclass(frozen=True, slots=True)
class StopLossDiagnostics:
    count: int
    percentage_of_trades: Decimal
    average_loss_r: Decimal | None
    average_bars_before_stop: Decimal
    median_bars_before_stop: Decimal
    average_mfe_before_stop: Decimal
    average_mfe_r_before_stop: Decimal | None


@dataclass(frozen=True, slots=True)
class TakeProfitDiagnostics:
    count: int
    percentage_of_trades: Decimal
    average_bars_to_target: Decimal
    average_mae_before_target: Decimal
    average_mae_r_before_target: Decimal | None


@dataclass(frozen=True, slots=True)
class HoldingDiagnostics:
    average_bars: Decimal
    median_bars: Decimal
    minimum_bars: int
    maximum_bars: int
    average_winner_bars: Decimal
    median_winner_bars: Decimal
    average_loser_bars: Decimal
    median_loser_bars: Decimal


@dataclass(frozen=True, slots=True)
class EntryFeatureComparison:
    feature: str
    winner_average: Decimal | None
    loser_average: Decimal | None
    all_average: Decimal | None


@dataclass(frozen=True, slots=True)
class DiagnosticObservation:
    observation: str
    evidence_strength: EvidenceStrength


@dataclass(frozen=True, slots=True)
class PeriodDiagnostics:
    period: ResearchPeriod | str
    start: datetime
    end: datetime
    initial_capital: Decimal
    trades: tuple[TradeDiagnostic, ...]
    total_trades: int
    win_rate_percent: Decimal
    profit_factor: Decimal | None
    expectancy: Decimal
    maximum_consecutive_losses: int
    loss_streak_distribution: tuple[tuple[str, int], ...]
    evidence_strength: EvidenceStrength
    outcomes: OutcomeDiagnostics
    costs: CostDiagnostics
    excursions: ExcursionDiagnostics
    stop_loss: StopLossDiagnostics
    take_profit: TakeProfitDiagnostics
    holding: HoldingDiagnostics
    exit_reasons: tuple[ExitReasonDiagnostics, ...]
    holding_buckets: tuple[GroupDiagnostics, ...]
    feature_buckets: tuple[GroupDiagnostics, ...]
    market_regimes: tuple[GroupDiagnostics, ...]
    volatility_regimes: tuple[GroupDiagnostics, ...]
    utc_hours: tuple[GroupDiagnostics, ...]
    utc_day_groups: tuple[GroupDiagnostics, ...]
    entry_features: tuple[EntryFeatureComparison, ...]
    observations: tuple[DiagnosticObservation, ...]


@dataclass(frozen=True, slots=True)
class StrategyDiagnosticsReport:
    research_run_id: str
    symbol: str
    interval: str
    strategy_name: str
    strategy_version: str
    minimum_trades_warning: int
    periods: tuple[PeriodDiagnostics, PeriodDiagnostics, PeriodDiagnostics]
    combined_label: str
    combined: PeriodDiagnostics
    cross_period_observations: tuple[DiagnosticObservation, ...]
