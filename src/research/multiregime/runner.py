"""Offline V3.2.1 orchestration over fixed, independent research windows."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from enum import Enum
from collections.abc import Callable

from src.backtest.models import BacktestConfig
from src.backtest.metrics import maximum_drawdown_percent
from src.diagnostics.analyzer import StrategyDiagnosticsAnalyzer
from src.diagnostics.models import DiagnosticPeriodInput, TradeDiagnostic
from src.hypotheses.candidates import build_candidate
from src.hypotheses.models import HypothesisId, HypothesisSuiteConfig
from src.hypotheses.runner import (
    HypothesisMetrics,
    hypothesis_metrics_from_diagnostics,
)
from src.research.evaluation import evaluate_strategy_period
from src.research.multiregime.models import (
    CandidateStability,
    DiagnosticVolatilityRegime,
    LongTrendRegime,
    MultiRegimeConfig,
    MultiRegimeResearchResult,
    MultiRegimeWindowResult,
    RegimePerformance,
    ResearchPartitionMetadata,
    ResearchRegion,
    ResearchWindow,
    WindowCandidateResult,
    WindowDelta,
)
from src.research.multiregime.regime_context import build_regime_context
from src.research.multiregime.stability import (
    classify_candidate,
    distribution,
    support_gate,
)
from src.research.multiregime.windows import construct_windows
from src.strategy.models import TrendMomentumConfig
from src.strategy.base import BaseStrategy


ClassificationResolver = Callable[..., Enum]
CandidateFactory = Callable[[Enum], BaseStrategy]


class MultiRegimeResearchRunner:
    """Run H0-H4 unchanged, with one fresh V2 account per candidate/window."""

    def __init__(
        self,
        *,
        multiregime_config: MultiRegimeConfig,
        hypothesis_config: object,
        strategy_config: TrendMomentumConfig,
        backtest_config: BacktestConfig,
        minimum_trades_warning: int,
        candidate_ids: tuple[Enum, ...] | None = None,
        candidate_factory: CandidateFactory | None = None,
        classification_resolver: ClassificationResolver | None = None,
        eligible_classification_value: str = "V3.3_ELIGIBLE",
        trade_count_ratio_warning: Decimal | None = None,
    ) -> None:
        self.multiregime_config = multiregime_config
        self.hypothesis_config = hypothesis_config
        self.strategy_config = strategy_config
        self.backtest_config = backtest_config
        self.minimum_trades_warning = minimum_trades_warning
        self.candidate_ids = candidate_ids or tuple(HypothesisId)
        if not self.candidate_ids or self.candidate_ids[0].value != "H0":
            raise ValueError("The first multi-regime candidate must be H0.")
        self.control_id = self.candidate_ids[0]
        self._candidate_factory = candidate_factory or self._default_candidate_factory
        self._classification_resolver = classification_resolver
        self._eligible_classification_value = eligible_classification_value
        configured_ratio = getattr(
            hypothesis_config, "trade_count_ratio_warning", Decimal("0.40")
        )
        self.trade_count_ratio_warning = Decimal(
            str(trade_count_ratio_warning or configured_ratio)
        )
        self._analyzer = StrategyDiagnosticsAnalyzer()

    def _default_candidate_factory(self, hypothesis_id: Enum) -> BaseStrategy:
        if not isinstance(self.hypothesis_config, HypothesisSuiteConfig):
            raise ValueError("The default H0-H4 factory requires HypothesisSuiteConfig.")
        return build_candidate(
            HypothesisId(hypothesis_id.value),
            self.strategy_config,
            self.hypothesis_config,
        )

    def run(
        self,
        *,
        run_id: str,
        regions: tuple[ResearchRegion, ...],
        partitions: tuple[ResearchPartitionMetadata, ...],
        cost_stress: bool,
        previous_h0_reproduction_verified: bool,
        previous_h1_reproduction_verified: bool | None = None,
    ) -> MultiRegimeResearchResult:
        windows = construct_windows(regions, self.multiregime_config)
        stress_config = (
            replace(
                self.backtest_config,
                fee_bps=self.backtest_config.fee_bps * Decimal("2"),
                slippage_bps=self.backtest_config.slippage_bps * Decimal("2"),
            )
            if cost_stress
            else None
        )
        window_results = tuple(
            self._run_window(window, stress_config) for window in windows
        )
        stability = self._stability(window_results)
        regimes = self._regime_performance(window_results)
        return MultiRegimeResearchResult(
            run_id=run_id,
            symbol=regions[0].dataset.symbol,
            interval=regions[0].dataset.interval,
            config=self.multiregime_config,
            hypothesis_config=self.hypothesis_config,
            strategy_config=self.strategy_config,
            base_backtest_config=self.backtest_config,
            stress_backtest_config=stress_config,
            partitions=partitions,
            windows=window_results,
            stability=stability,
            regimes=regimes,
            previous_h0_reproduction_verified=previous_h0_reproduction_verified,
            holdout_revealed=False,
            holdout_consumed=False,
            previous_h1_reproduction_verified=previous_h1_reproduction_verified,
        )

    def _run_window(
        self,
        window: ResearchWindow,
        stress_config: BacktestConfig | None,
    ) -> MultiRegimeWindowResult:
        market, regimes = build_regime_context(window, self.multiregime_config)
        candidates = [
            self._evaluate(window, hypothesis_id, self.backtest_config)
            for hypothesis_id in self.candidate_ids
        ]
        if stress_config is not None:
            stress = {
                hypothesis_id: self._evaluate(window, hypothesis_id, stress_config)
                for hypothesis_id in self.candidate_ids
            }
            candidates = [
                replace(
                    item,
                    stress_metrics=stress[item.hypothesis_id].metrics,
                    stress_diagnostics=stress[item.hypothesis_id].diagnostics,
                )
                for item in candidates
            ]
        baseline = candidates[0]
        candidates = [
            replace(
                item,
                delta_to_h0=(
                    None if item.hypothesis_id is self.control_id else _window_delta(item.metrics, baseline.metrics)
                ),
                trade_count_collapse=(
                    item.hypothesis_id is not self.control_id
                    and baseline.metrics.trades > 0
                    and Decimal(item.metrics.trades) / Decimal(baseline.metrics.trades)
                    < self.trade_count_ratio_warning
                ),
            )
            for item in candidates
        ]
        eligible = (
            window.duration_eligible
            and baseline.metrics.trades >= self.minimum_trades_warning
        )
        return MultiRegimeWindowResult(
            window=window,
            market=market,
            regimes=regimes,
            eligible=eligible,
            candidates=tuple(candidates),
        )

    def _evaluate(
        self,
        window: ResearchWindow,
        hypothesis_id: Enum,
        backtest_config: BacktestConfig,
    ) -> WindowCandidateResult:
        strategy = self._candidate_factory(hypothesis_id)
        evaluation = evaluate_strategy_period(
            window.replay_dataset,
            strategy=strategy,
            strategy_config=self.strategy_config,
            backtest_config=backtest_config,
            evaluation_start_index=window.evaluation_start_index,
        )
        period = DiagnosticPeriodInput(
            period=window.window_id,
            dataset=evaluation.backtest.dataset,
            trades=evaluation.backtest.trades,
            signals=evaluation.signals,
        )
        diagnostics = self._analyzer.analyze_period(
            period,
            backtest_config=backtest_config,
            strategy_config=self.strategy_config,
            minimum_trades_warning=self.minimum_trades_warning,
        )
        metrics = hypothesis_metrics_from_diagnostics(
            diagnostics,
            return_percent=evaluation.backtest.metrics.total_return_percent,
            maximum_drawdown_percent=evaluation.backtest.metrics.maximum_drawdown_percent,
            market_exposure_percent=evaluation.backtest.metrics.market_exposure_percent,
        )
        return WindowCandidateResult(
            hypothesis_id=hypothesis_id,
            backtest=evaluation.backtest,
            diagnostics=diagnostics,
            metrics=metrics,
            delta_to_h0=None,
            low_sample_size=metrics.trades < self.minimum_trades_warning,
            trade_count_collapse=False,
        )

    def _stability(
        self, windows: tuple[MultiRegimeWindowResult, ...]
    ) -> tuple[CandidateStability, ...]:
        eligible = tuple(window for window in windows if window.eligible)
        if not eligible:
            raise ValueError("No eligible windows exist for stability analysis.")
        combined = {
            hypothesis_id: self._combined_metrics(eligible, hypothesis_id, stress=False)
            for hypothesis_id in self.candidate_ids
        }
        stress_available = self.backtest_config is not None and all(
            candidate.stress_metrics is not None
            for window in eligible
            for candidate in window.candidates
        )
        stress = (
            {
                hypothesis_id: self._combined_metrics(eligible, hypothesis_id, stress=True)
                for hypothesis_id in self.candidate_ids
            }
            if stress_available
            else {}
        )
        baseline = combined[self.control_id]
        baseline_stress = stress.get(self.control_id)
        results: list[CandidateStability] = []
        for hypothesis_id in self.candidate_ids:
            pairs = tuple(
                (
                    _candidate(window, hypothesis_id).metrics,
                    _candidate(window, self.control_id).metrics,
                )
                for window in eligible
            )
            frictionless_better = sum(
                candidate.average_frictionless_pnl_per_trade
                > control.average_frictionless_pnl_per_trade
                for candidate, control in pairs
            )
            net_better = sum(
                candidate.expectancy > control.expectancy
                for candidate, control in pairs
            )
            pf_better = sum(
                candidate.profit_factor is not None
                and control.profit_factor is not None
                and candidate.profit_factor > control.profit_factor
                for candidate, control in pairs
            )
            lower_dd = sum(
                candidate.maximum_drawdown_percent < control.maximum_drawdown_percent
                for candidate, control in pairs
            )
            count = len(eligible)
            current = combined[hypothesis_id]
            current_stress = stress.get(hypothesis_id)
            gate = support_gate(
                candidate=current,
                baseline=baseline,
                candidate_stress=current_stress,
                baseline_stress=baseline_stress,
                eligible_windows=count,
                frictionless_better_windows=frictionless_better,
                net_better_windows=net_better,
                config=self.multiregime_config,
            )
            classification_arguments = {
                "hypothesis_id": hypothesis_id,
                "gate": gate,
                "combined": current,
                "baseline": baseline,
                "eligible_windows": count,
                "minimum_trades_warning": self.minimum_trades_warning,
                "frictionless_better_windows": frictionless_better,
                "net_better_windows": net_better,
                "config": self.multiregime_config,
            }
            classification = (
                self._classification_resolver(**classification_arguments)
                if self._classification_resolver is not None
                else classify_candidate(
                    hypothesis_id=HypothesisId(hypothesis_id.value),
                    **{
                        key: value
                        for key, value in classification_arguments.items()
                        if key != "hypothesis_id"
                    },
                )
            )
            results.append(
                CandidateStability(
                    hypothesis_id=hypothesis_id,
                    eligible_windows=count,
                    combined_metrics=current,
                    stress_combined_metrics=current_stress,
                    baseline_combined_metrics=baseline,
                    baseline_stress_combined_metrics=baseline_stress,
                    frictionless_better_windows=frictionless_better,
                    net_better_windows=net_better,
                    profit_factor_better_windows=pf_better,
                    lower_drawdown_windows=lower_dd,
                    frictionless_better_percent=_percent(frictionless_better, count),
                    net_better_percent=_percent(net_better, count),
                    profit_factor_better_percent=_percent(pf_better, count),
                    lower_drawdown_percent=_percent(lower_dd, count),
                    trade_count_ratio=(
                        Decimal(current.trades) / Decimal(baseline.trades)
                        if baseline.trades
                        else Decimal("0")
                    ),
                    trade_count_collapse=(
                        baseline.trades > 0
                        and Decimal(current.trades) / Decimal(baseline.trades)
                        < self.trade_count_ratio_warning
                    ),
                    frictionless_expectancy_distribution=distribution(
                        tuple(candidate.average_frictionless_pnl_per_trade for candidate, _ in pairs)
                    ),
                    net_expectancy_distribution=distribution(
                        tuple(candidate.expectancy for candidate, _ in pairs)
                    ),
                    profit_factor_distribution=distribution(
                        tuple(
                            candidate.profit_factor
                            for candidate, _ in pairs
                            if candidate.profit_factor is not None
                        )
                    ),
                    return_distribution=distribution(
                        tuple(candidate.return_percent for candidate, _ in pairs)
                    ),
                    drawdown_distribution=distribution(
                        tuple(candidate.maximum_drawdown_percent for candidate, _ in pairs)
                    ),
                    gate=gate,
                    classification=classification,
                    gross_edge_costs_dominate=(
                        classification.value == self._eligible_classification_value
                        and current.expectancy < 0
                    ),
                )
            )
        return tuple(results)

    def _combined_metrics(
        self,
        windows: tuple[MultiRegimeWindowResult, ...],
        hypothesis_id: Enum,
        *,
        stress: bool,
    ) -> HypothesisMetrics:
        selected = tuple(_candidate(window, hypothesis_id) for window in windows)
        diagnostics = tuple(
            item.stress_diagnostics if stress else item.diagnostics for item in selected
        )
        if any(item is None for item in diagnostics):
            raise ValueError("Cost-stress diagnostics are incomplete.")
        typed = tuple(item for item in diagnostics if item is not None)
        all_trades = tuple(trade for item in typed for trade in item.trades)
        summary = self._analyzer.summarize_trades(
            period="ELIGIBLE_WINDOWS_COMBINED_DESCRIPTIVE",
            start=windows[0].window.start,
            end=windows[-1].window.end,
            initial_capital=(
                self.backtest_config.initial_capital_usdc * Decimal(len(windows))
            ),
            trades=all_trades,
            minimum_trades_warning=self.minimum_trades_warning,
        )
        metrics = tuple(
            item.stress_metrics if stress else item.metrics for item in selected
        )
        present = tuple(item for item in metrics if item is not None)
        total_bars = sum(len(item.backtest.dataset.candles) for item in selected)
        exposure = (
            sum(
                metric.market_exposure_percent
                * Decimal(len(item.backtest.dataset.candles))
                for item, metric in zip(selected, present, strict=True)
            )
            / Decimal(total_bars)
        )
        return hypothesis_metrics_from_diagnostics(
            summary,
            return_percent=(
                summary.costs.net_pnl / summary.initial_capital * Decimal("100")
            ),
            maximum_drawdown_percent=max(
                metric.maximum_drawdown_percent for metric in present
            ),
            market_exposure_percent=exposure,
        )

    def _regime_performance(
        self, windows: tuple[MultiRegimeWindowResult, ...]
    ) -> tuple[RegimePerformance, ...]:
        results: list[RegimePerformance] = []
        dimensions = (
            ("LONG_TERM_TREND", tuple(LongTrendRegime)),
            ("VOLATILITY", tuple(DiagnosticVolatilityRegime)),
        )
        for hypothesis_id in self.candidate_ids:
            for dimension, labels in dimensions:
                for label in labels:
                    selected: list[tuple[str, TradeDiagnostic]] = []
                    for window in windows:
                        context = {item.signal_timestamp: item for item in window.regimes}
                        candidate = _candidate(window, hypothesis_id)
                        for trade in candidate.diagnostics.trades:
                            regime = context.get(trade.entry_signal_time)
                            actual = (
                                regime.long_trend
                                if regime is not None and dimension == "LONG_TERM_TREND"
                                else regime.volatility
                                if regime is not None
                                else None
                            )
                            if actual is label:
                                selected.append((window.window.window_id, trade))
                    trades = tuple(item[1] for item in selected)
                    net_profit = sum(
                        (trade.net_pnl for trade in trades if trade.net_pnl > 0),
                        start=Decimal("0"),
                    )
                    net_loss = sum(
                        (trade.net_pnl for trade in trades if trade.net_pnl < 0),
                        start=Decimal("0"),
                    )
                    results.append(
                        RegimePerformance(
                            hypothesis_id=hypothesis_id,
                            dimension=dimension,
                            regime=label.value,
                            trades=len(trades),
                            windows_represented=len({item[0] for item in selected}),
                            frictionless_expectancy=_average(
                                tuple(trade.frictionless_pnl for trade in trades)
                            ),
                            net_expectancy=_average(
                                tuple(trade.net_pnl for trade in trades)
                            ),
                            profit_factor=(
                                net_profit / abs(net_loss) if net_loss else None
                            ),
                            win_rate_percent=(
                                Decimal(sum(trade.net_pnl > 0 for trade in trades))
                                / Decimal(len(trades))
                                * Decimal("100")
                                if trades
                                else Decimal("0")
                            ),
                            maximum_drawdown_percent=_regime_drawdown(
                                tuple(selected), self.backtest_config.initial_capital_usdc
                            ),
                            low_sample_size=len(trades) < self.minimum_trades_warning,
                        )
                    )
        return tuple(results)


def _candidate(
    window: MultiRegimeWindowResult, hypothesis_id: Enum
) -> WindowCandidateResult:
    return next(item for item in window.candidates if item.hypothesis_id is hypothesis_id)


def _window_delta(
    candidate: HypothesisMetrics, baseline: HypothesisMetrics
) -> WindowDelta:
    return WindowDelta(
        frictionless_expectancy=(
            candidate.average_frictionless_pnl_per_trade
            - baseline.average_frictionless_pnl_per_trade
        ),
        net_expectancy=candidate.expectancy - baseline.expectancy,
        profit_factor=(
            candidate.profit_factor - baseline.profit_factor
            if candidate.profit_factor is not None and baseline.profit_factor is not None
            else None
        ),
        return_percent=candidate.return_percent - baseline.return_percent,
        maximum_drawdown_percent=(
            candidate.maximum_drawdown_percent - baseline.maximum_drawdown_percent
        ),
        trades=candidate.trades - baseline.trades,
        fees=candidate.total_fees - baseline.total_fees,
        win_rate_percent=candidate.win_rate_percent - baseline.win_rate_percent,
        payoff_ratio=(
            candidate.payoff_ratio - baseline.payoff_ratio
            if candidate.payoff_ratio is not None and baseline.payoff_ratio is not None
            else None
        ),
    )


def _percent(count: int, total: int) -> Decimal:
    return Decimal(count) / Decimal(total) * Decimal("100") if total else Decimal("0")


def _average(values: tuple[Decimal, ...]) -> Decimal:
    return sum(values, start=Decimal("0")) / Decimal(len(values)) if values else Decimal("0")


def _regime_drawdown(
    selected: tuple[tuple[str, TradeDiagnostic], ...], initial_capital: Decimal
) -> Decimal | None:
    """Return the worst independent-window trade-close drawdown for a regime."""

    if len(selected) < 2:
        return None
    by_window: dict[str, list[TradeDiagnostic]] = {}
    for window_id, trade in selected:
        by_window.setdefault(window_id, []).append(trade)
    drawdowns: list[Decimal] = []
    for trades in by_window.values():
        equity = initial_capital
        curve = [equity]
        for trade in trades:
            equity += trade.net_pnl
            if equity <= 0:
                return Decimal("100")
            curve.append(equity)
        drawdowns.append(maximum_drawdown_percent(curve))
    return max(drawdowns, default=Decimal("0"))
