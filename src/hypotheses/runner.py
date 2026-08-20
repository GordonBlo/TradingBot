"""Deterministic H0-H4 replay on already-consumed research periods."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal

from src.backtest.models import BacktestConfig, BacktestResult, ExitReason
from src.diagnostics.analyzer import StrategyDiagnosticsAnalyzer
from src.diagnostics.models import (
    DiagnosticPeriodInput,
    DiagnosticRunInput,
    PeriodDiagnostics,
)
from src.hypotheses.candidates import build_candidate
from src.hypotheses.models import (
    DatasetStatus,
    HypothesisId,
    HypothesisJournalRecord,
    HypothesisSuiteConfig,
    SupportClassification,
    hypothesis_registry,
)
from src.research.evaluation import SignalRecord, evaluate_strategy_period
from src.strategy.models import StrategyAction


@dataclass(frozen=True, slots=True)
class HypothesisMetrics:
    trades: int
    frictionless_pnl: Decimal
    gross_after_slippage: Decimal
    fee_drag: Decimal
    slippage_drag: Decimal
    net_pnl: Decimal
    return_percent: Decimal
    win_rate_percent: Decimal
    profit_factor: Decimal | None
    expectancy: Decimal
    average_winner: Decimal
    average_loser: Decimal
    payoff_ratio: Decimal | None
    maximum_drawdown_percent: Decimal
    total_fees: Decimal
    market_exposure_percent: Decimal
    stop_loss_percent: Decimal
    take_profit_percent: Decimal
    trend_exit_percent: Decimal
    time_exit_percent: Decimal
    median_mfe_r: Decimal | None
    median_mae_r: Decimal | None
    reached_one_r_percent: Decimal
    reached_two_r_percent: Decimal
    average_holding_bars: Decimal
    average_frictionless_pnl_per_trade: Decimal
    average_fee_per_trade: Decimal
    average_slippage_drag_per_trade: Decimal
    average_total_friction_per_trade: Decimal
    friction_to_absolute_frictionless_percent: Decimal | None


@dataclass(frozen=True, slots=True)
class HypothesisDelta:
    net_pnl: Decimal
    frictionless_pnl: Decimal
    expectancy: Decimal
    profit_factor: Decimal | None
    maximum_drawdown_percent: Decimal
    trade_count: int
    fees: Decimal
    win_rate_percent: Decimal
    payoff_ratio: Decimal | None


@dataclass(frozen=True, slots=True)
class SignalAlignment:
    shared_signals: int
    baseline_only_signals: int
    candidate_only_signals: int


@dataclass(frozen=True, slots=True)
class HypothesisPeriodResult:
    source_period: str
    data_status: DatasetStatus
    backtest: BacktestResult
    signals: tuple[SignalRecord, ...]
    journals: tuple[HypothesisJournalRecord, ...]
    diagnostics: PeriodDiagnostics
    metrics: HypothesisMetrics


@dataclass(frozen=True, slots=True)
class HypothesisExperimentResult:
    hypothesis_id: HypothesisId
    name: str
    periods: tuple[HypothesisPeriodResult, ...]
    combined_metrics: HypothesisMetrics
    delta_to_baseline: HypothesisDelta | None
    alignment: SignalAlignment
    classification: SupportClassification
    verification_status: str
    warnings: tuple[str, ...]
    stress_combined_metrics: HypothesisMetrics | None = None


@dataclass(frozen=True, slots=True)
class HypothesisResearchResult:
    source_research_run_id: str
    symbol: str
    interval: str
    dataset_status: DatasetStatus
    baseline_reproduction_verified: bool
    base_backtest_config: BacktestConfig
    stress_backtest_config: BacktestConfig | None
    suite_config: HypothesisSuiteConfig
    experiments: tuple[HypothesisExperimentResult, ...]


def metric_delta(
    candidate: HypothesisMetrics, baseline: HypothesisMetrics
) -> HypothesisDelta:
    """Calculate candidate minus baseline; no composite score is produced."""

    return HypothesisDelta(
        net_pnl=candidate.net_pnl - baseline.net_pnl,
        frictionless_pnl=candidate.frictionless_pnl - baseline.frictionless_pnl,
        expectancy=candidate.expectancy - baseline.expectancy,
        profit_factor=(
            candidate.profit_factor - baseline.profit_factor
            if candidate.profit_factor is not None and baseline.profit_factor is not None
            else None
        ),
        maximum_drawdown_percent=(
            candidate.maximum_drawdown_percent - baseline.maximum_drawdown_percent
        ),
        trade_count=candidate.trades - baseline.trades,
        fees=candidate.total_fees - baseline.total_fees,
        win_rate_percent=candidate.win_rate_percent - baseline.win_rate_percent,
        payoff_ratio=(
            candidate.payoff_ratio - baseline.payoff_ratio
            if candidate.payoff_ratio is not None and baseline.payoff_ratio is not None
            else None
        ),
    )


def _exit_percent(diagnostics: PeriodDiagnostics, reason: ExitReason) -> Decimal:
    row = next(
        (item for item in diagnostics.exit_reasons if item.exit_reason is reason),
        None,
    )
    return row.percentage_of_trades if row is not None else Decimal("0")


def _reached_percent(diagnostics: PeriodDiagnostics, group: str) -> Decimal:
    row = next(
        (item for item in diagnostics.excursions.mfe_thresholds if item.group == group),
        None,
    )
    return row.percentage_of_trades if row is not None else Decimal("0")


def hypothesis_metrics_from_diagnostics(
    diagnostics: PeriodDiagnostics,
    *,
    return_percent: Decimal,
    maximum_drawdown_percent: Decimal,
    market_exposure_percent: Decimal,
) -> HypothesisMetrics:
    """Build V3.2 metrics from the established V2/V3.1 calculations."""

    count = diagnostics.total_trades
    costs = diagnostics.costs
    total_friction = costs.fee_drag + abs(costs.slippage_drag)
    payoff = (
        diagnostics.outcomes.average_win / abs(diagnostics.outcomes.average_loss)
        if diagnostics.outcomes.average_loss != 0
        else None
    )
    return HypothesisMetrics(
        trades=count,
        frictionless_pnl=costs.frictionless_pnl,
        gross_after_slippage=costs.slippage_adjusted_gross_pnl,
        fee_drag=costs.fee_drag,
        slippage_drag=costs.slippage_drag,
        net_pnl=costs.net_pnl,
        return_percent=return_percent,
        win_rate_percent=diagnostics.win_rate_percent,
        profit_factor=diagnostics.profit_factor,
        expectancy=diagnostics.expectancy,
        average_winner=diagnostics.outcomes.average_win,
        average_loser=diagnostics.outcomes.average_loss,
        payoff_ratio=payoff,
        maximum_drawdown_percent=maximum_drawdown_percent,
        total_fees=costs.fee_drag,
        market_exposure_percent=market_exposure_percent,
        stop_loss_percent=_exit_percent(diagnostics, ExitReason.STOP_LOSS),
        take_profit_percent=_exit_percent(diagnostics, ExitReason.TAKE_PROFIT),
        trend_exit_percent=_exit_percent(diagnostics, ExitReason.TREND_EXIT),
        time_exit_percent=_exit_percent(diagnostics, ExitReason.TIME_EXIT),
        median_mfe_r=diagnostics.excursions.median_mfe_r,
        median_mae_r=diagnostics.excursions.median_mae_r,
        reached_one_r_percent=_reached_percent(diagnostics, ">=1R"),
        reached_two_r_percent=_reached_percent(diagnostics, ">=2R"),
        average_holding_bars=diagnostics.holding.average_bars,
        average_frictionless_pnl_per_trade=(
            costs.frictionless_pnl / Decimal(count) if count else Decimal("0")
        ),
        average_fee_per_trade=(
            costs.fee_drag / Decimal(count) if count else Decimal("0")
        ),
        average_slippage_drag_per_trade=(
            costs.slippage_drag / Decimal(count) if count else Decimal("0")
        ),
        average_total_friction_per_trade=(
            total_friction / Decimal(count) if count else Decimal("0")
        ),
        friction_to_absolute_frictionless_percent=(
            total_friction / abs(costs.frictionless_pnl) * Decimal("100")
            if costs.frictionless_pnl != 0
            else None
        ),
    )


class HypothesisResearchRunner:
    """Replay each pre-registered mechanism in a clean independent V2 account."""

    def __init__(self, suite_config: HypothesisSuiteConfig | None = None) -> None:
        self.suite_config = suite_config or HypothesisSuiteConfig()
        self._analyzer = StrategyDiagnosticsAnalyzer()

    def run(
        self,
        source: DiagnosticRunInput,
        *,
        cost_stress: bool = False,
    ) -> HypothesisResearchResult:
        base_runs = {
            hypothesis_id: self._run_candidate(source, hypothesis_id, source.backtest_config)
            for hypothesis_id in HypothesisId
        }
        self._verify_frozen_baseline(source, base_runs[HypothesisId.H0][0])

        stress_config = (
            replace(
                source.backtest_config,
                fee_bps=source.backtest_config.fee_bps * Decimal("2"),
                slippage_bps=source.backtest_config.slippage_bps * Decimal("2"),
            )
            if cost_stress
            else None
        )
        stress_runs = (
            {
                hypothesis_id: self._run_candidate(source, hypothesis_id, stress_config)
                for hypothesis_id in HypothesisId
            }
            if stress_config is not None
            else {}
        )

        baseline_periods, baseline_combined = base_runs[HypothesisId.H0]
        registry = {item.hypothesis_id: item for item in hypothesis_registry(self.suite_config)}
        experiments: list[HypothesisExperimentResult] = []
        for hypothesis_id in HypothesisId:
            periods, combined = base_runs[hypothesis_id]
            warnings = self._warnings(
                periods,
                baseline_periods,
                source.minimum_trades_warning,
            )
            experiments.append(
                HypothesisExperimentResult(
                    hypothesis_id=hypothesis_id,
                    name=registry[hypothesis_id].name,
                    periods=periods,
                    combined_metrics=combined,
                    delta_to_baseline=(
                        None if hypothesis_id is HypothesisId.H0 else metric_delta(combined, baseline_combined)
                    ),
                    alignment=self._alignment(periods, baseline_periods),
                    classification=(
                        SupportClassification.CONTROL
                        if hypothesis_id is HypothesisId.H0
                        else self._classify(periods, baseline_periods, source.minimum_trades_warning)
                    ),
                    verification_status="NOT YET VERIFIED ON BLIND HOLDOUT",
                    warnings=warnings,
                    stress_combined_metrics=(
                        stress_runs[hypothesis_id][1] if stress_config is not None else None
                    ),
                )
            )
        return HypothesisResearchResult(
            source_research_run_id=source.research_run_id,
            symbol=source.symbol,
            interval=source.interval,
            dataset_status=DatasetStatus.CONSUMED_RESEARCH_DATA,
            baseline_reproduction_verified=True,
            base_backtest_config=source.backtest_config,
            stress_backtest_config=stress_config,
            suite_config=self.suite_config,
            experiments=tuple(experiments),
        )

    def _run_candidate(
        self,
        source: DiagnosticRunInput,
        hypothesis_id: HypothesisId,
        backtest_config: BacktestConfig,
    ) -> tuple[tuple[HypothesisPeriodResult, ...], HypothesisMetrics]:
        raw: list[
            tuple[DiagnosticPeriodInput, BacktestResult, tuple[SignalRecord, ...], tuple[HypothesisJournalRecord, ...]]
        ] = []
        for source_period in source.periods:
            strategy = build_candidate(
                hypothesis_id, source.strategy_config, self.suite_config
            )
            evaluation = evaluate_strategy_period(
                source_period.dataset,
                strategy=strategy,
                strategy_config=source.strategy_config,
                backtest_config=backtest_config,
            )
            journals = tuple(getattr(strategy, "journal", ()))
            raw.append(
                (
                    DiagnosticPeriodInput(
                        source_period.period,
                        source_period.dataset,
                        evaluation.backtest.trades,
                        evaluation.signals,
                    ),
                    evaluation.backtest,
                    evaluation.signals,
                    journals,
                )
            )
        diagnostic_input = DiagnosticRunInput(
            research_run_id=source.research_run_id,
            symbol=source.symbol,
            interval=source.interval,
            backtest_config=backtest_config,
            strategy_config=source.strategy_config,
            minimum_trades_warning=source.minimum_trades_warning,
            periods=tuple(item[0] for item in raw),  # type: ignore[arg-type]
        )
        diagnostics = self._analyzer.analyze(diagnostic_input)
        periods = tuple(
            HypothesisPeriodResult(
                source_period=item[0].period.value,
                data_status=DatasetStatus.CONSUMED_RESEARCH_DATA,
                backtest=item[1],
                signals=item[2],
                journals=item[3],
                diagnostics=diagnostic,
                metrics=self._period_metrics(item[1], diagnostic),
            )
            for item, diagnostic in zip(raw, diagnostics.periods, strict=True)
        )
        return periods, self._combined_metrics(periods, diagnostics.combined)

    @staticmethod
    def _verify_frozen_baseline(
        source: DiagnosticRunInput, periods: tuple[HypothesisPeriodResult, ...]
    ) -> None:
        for recorded, reproduced in zip(source.periods, periods, strict=True):
            if reproduced.backtest.trades != recorded.trades:
                raise RuntimeError(
                    f"H0 frozen-baseline trade mismatch in {recorded.period.value}."
                )
            if reproduced.signals != recorded.signals:
                raise RuntimeError(
                    f"H0 frozen-baseline signal mismatch in {recorded.period.value}."
                )

    def _period_metrics(
        self, backtest: BacktestResult, diagnostics: PeriodDiagnostics
    ) -> HypothesisMetrics:
        return self._metrics(
            diagnostics,
            return_percent=backtest.metrics.total_return_percent,
            maximum_drawdown_percent=backtest.metrics.maximum_drawdown_percent,
            market_exposure_percent=backtest.metrics.market_exposure_percent,
        )

    def _combined_metrics(
        self,
        periods: tuple[HypothesisPeriodResult, ...],
        diagnostics: PeriodDiagnostics,
    ) -> HypothesisMetrics:
        total_bars = sum(len(period.backtest.dataset.candles) for period in periods)
        exposure = (
            sum(
                (
                    period.backtest.metrics.market_exposure_percent
                    * Decimal(len(period.backtest.dataset.candles))
                    for period in periods
                ),
                start=Decimal("0"),
            )
            / Decimal(total_bars)
            if total_bars
            else Decimal("0")
        )
        return self._metrics(
            diagnostics,
            return_percent=(
                diagnostics.costs.net_pnl / diagnostics.initial_capital * Decimal("100")
            ),
            maximum_drawdown_percent=max(
                (period.backtest.metrics.maximum_drawdown_percent for period in periods),
                default=Decimal("0"),
            ),
            market_exposure_percent=exposure,
        )

    def _metrics(
        self,
        diagnostics: PeriodDiagnostics,
        *,
        return_percent: Decimal,
        maximum_drawdown_percent: Decimal,
        market_exposure_percent: Decimal,
    ) -> HypothesisMetrics:
        return hypothesis_metrics_from_diagnostics(
            diagnostics,
            return_percent=return_percent,
            maximum_drawdown_percent=maximum_drawdown_percent,
            market_exposure_percent=market_exposure_percent,
        )

    def _classify(
        self,
        candidate: tuple[HypothesisPeriodResult, ...],
        baseline: tuple[HypothesisPeriodResult, ...],
        minimum_trades_warning: int,
    ) -> SupportClassification:
        if sum(item.metrics.trades for item in candidate) < minimum_trades_warning:
            return SupportClassification.INSUFFICIENT_SAMPLE
        favorable = 0
        unfavorable = 0
        for candidate_period, baseline_period in zip(candidate, baseline, strict=True):
            comparisons = (
                candidate_period.metrics.expectancy - baseline_period.metrics.expectancy,
                candidate_period.metrics.net_pnl - baseline_period.metrics.net_pnl,
                candidate_period.metrics.frictionless_pnl - baseline_period.metrics.frictionless_pnl,
            )
            positive = sum(value > 0 for value in comparisons)
            negative = sum(value < 0 for value in comparisons)
            if positive >= 2:
                favorable += 1
            elif negative >= 2:
                unfavorable += 1
        if favorable >= 2 and unfavorable == 0:
            return SupportClassification.SUPPORTED_ON_CONSUMED_DATA
        if unfavorable >= 2 and favorable == 0:
            return SupportClassification.NOT_SUPPORTED
        return SupportClassification.MIXED

    def _warnings(
        self,
        candidate: tuple[HypothesisPeriodResult, ...],
        baseline: tuple[HypothesisPeriodResult, ...],
        minimum: int,
    ) -> tuple[str, ...]:
        warnings: list[str] = []
        for candidate_period, baseline_period in zip(candidate, baseline, strict=True):
            if candidate_period.metrics.trades < minimum:
                warnings.append(
                    f"LOW SAMPLE SIZE: {candidate_period.source_period} has {candidate_period.metrics.trades} trades (<{minimum})."
                )
            baseline_count = baseline_period.metrics.trades
            if (
                baseline_count
                and Decimal(candidate_period.metrics.trades) / Decimal(baseline_count)
                < self.suite_config.trade_count_ratio_warning
            ):
                warnings.append(
                    f"TRADE COUNT COLLAPSE: {candidate_period.source_period} retains "
                    f"{candidate_period.metrics.trades}/{baseline_count} baseline trades."
                )
        return tuple(warnings)

    @staticmethod
    def _alignment(
        candidate: tuple[HypothesisPeriodResult, ...],
        baseline: tuple[HypothesisPeriodResult, ...],
    ) -> SignalAlignment:
        candidate_entries = {
            (period.source_period, signal.timestamp)
            for period in candidate
            for signal in period.signals
            if signal.action is StrategyAction.ENTER_LONG
        }
        baseline_entries = {
            (period.source_period, signal.timestamp)
            for period in baseline
            for signal in period.signals
            if signal.action is StrategyAction.ENTER_LONG
        }
        return SignalAlignment(
            shared_signals=len(candidate_entries & baseline_entries),
            baseline_only_signals=len(baseline_entries - candidate_entries),
            candidate_only_signals=len(candidate_entries - baseline_entries),
        )
