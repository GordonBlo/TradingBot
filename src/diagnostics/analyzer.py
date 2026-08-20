"""Top-level V3.1 period-isolated diagnostic orchestration."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from src.backtest.models import BacktestConfig
from src.diagnostics.cost_analysis import aggregate_costs
from src.diagnostics.distributions import (
    evidence_strength,
    holding_buckets,
    holding_diagnostics,
    loss_streaks,
    outcome_diagnostics,
    profit_factor,
)
from src.diagnostics.entry_analysis import (
    entry_feature_comparisons,
    feature_bucket_diagnostics,
)
from src.diagnostics.excursion_analysis import excursion_diagnostics
from src.diagnostics.exit_analysis import (
    exit_reason_diagnostics,
    stop_loss_diagnostics,
    take_profit_diagnostics,
)
from src.diagnostics.models import (
    DiagnosticObservation,
    DiagnosticPeriodInput,
    DiagnosticRunInput,
    EvidenceStrength,
    CostDiagnostics,
    ExitReasonDiagnostics,
    GroupDiagnostics,
    PeriodDiagnostics,
    StrategyDiagnosticsReport,
    TradeDiagnostic,
)
from src.diagnostics.regime_analysis import (
    market_regime_diagnostics,
    utc_day_diagnostics,
    utc_hour_diagnostics,
    volatility_regime_diagnostics,
)
from src.diagnostics.trade_metrics import build_trade_diagnostics
from src.market.intervals import next_open_time
from src.research.dataset_split import ResearchPeriod
from src.strategy.baseline_trend import TrendMomentumBaselineStrategy
from src.strategy.models import TrendMomentumConfig


class StrategyDiagnosticsAnalyzer:
    """Analyze completed results only; never participates in strategy replay."""

    def analyze(self, run: DiagnosticRunInput) -> StrategyDiagnosticsReport:
        periods = tuple(self._analyze_period(period, run) for period in run.periods)
        assert len(periods) == 3
        combined_trades = tuple(
            trade for period in periods for trade in period.trades
        )
        combined = self._summarize(
            period="COMBINED_DESCRIPTIVE_ONLY",
            start=periods[0].start,
            end=periods[-1].end,
            initial_capital=(
                run.backtest_config.initial_capital_usdc * Decimal(len(periods))
            ),
            trades=combined_trades,
            minimum=run.minimum_trades_warning,
            combined=True,
        )
        return StrategyDiagnosticsReport(
            research_run_id=run.research_run_id,
            symbol=run.symbol,
            interval=run.interval,
            strategy_name=TrendMomentumBaselineStrategy.NAME,
            strategy_version=TrendMomentumBaselineStrategy.VERSION,
            minimum_trades_warning=run.minimum_trades_warning,
            periods=periods,  # type: ignore[arg-type]
            combined_label="COMBINED — DESCRIPTIVE ONLY",
            combined=combined,
            cross_period_observations=self._cross_period_observations(periods),
        )

    def _analyze_period(
        self, period: DiagnosticPeriodInput, run: DiagnosticRunInput
    ) -> PeriodDiagnostics:
        return self.analyze_period(
            period,
            backtest_config=run.backtest_config,
            strategy_config=run.strategy_config,
            minimum_trades_warning=run.minimum_trades_warning,
        )

    def analyze_period(
        self,
        period: DiagnosticPeriodInput,
        *,
        backtest_config: BacktestConfig,
        strategy_config: TrendMomentumConfig,
        minimum_trades_warning: int,
    ) -> PeriodDiagnostics:
        """Analyze one isolated period for V3.2+ window orchestration."""

        trades = build_trade_diagnostics(
            period,
            strategy_config=strategy_config,
            slippage_bps=backtest_config.slippage_bps,
        )
        dataset = period.dataset
        return self._summarize(
            period=period.period,
            start=dataset.candles[0].timestamp,
            end=next_open_time(dataset.candles[-1].timestamp, dataset.interval),
            initial_capital=backtest_config.initial_capital_usdc,
            trades=trades,
            minimum=minimum_trades_warning,
            combined=False,
        )

    def summarize_trades(
        self,
        *,
        period: str,
        start: datetime,
        end: datetime,
        initial_capital: Decimal,
        trades: tuple[TradeDiagnostic, ...],
        minimum_trades_warning: int,
    ) -> PeriodDiagnostics:
        """Reuse established V3.1 aggregation for deterministic combined windows."""

        return self._summarize(
            period=period,
            start=start,
            end=end,
            initial_capital=initial_capital,
            trades=trades,
            minimum=minimum_trades_warning,
            combined=True,
        )

    def _summarize(
        self,
        *,
        period: ResearchPeriod | str,
        start: datetime,
        end: datetime,
        initial_capital: Decimal,
        trades: tuple[TradeDiagnostic, ...],
        minimum: int,
        combined: bool,
    ) -> PeriodDiagnostics:
        outcomes = outcome_diagnostics(trades)
        costs = aggregate_costs(trades, initial_capital)
        exits = exit_reason_diagnostics(trades, minimum)
        streak_max, streak_distribution = loss_streaks(trades)
        win_rate = (
            Decimal(outcomes.winning_trades)
            / Decimal(len(trades))
            * Decimal("100")
            if trades
            else Decimal("0")
        )
        volatility = volatility_regime_diagnostics(trades, minimum)
        observations = self._observations(
            trades,
            costs,
            exits,
            volatility,
            minimum,
            combined=combined,
        )
        return PeriodDiagnostics(
            period=period,
            start=start,
            end=end,
            initial_capital=initial_capital,
            trades=trades,
            total_trades=len(trades),
            win_rate_percent=win_rate,
            profit_factor=profit_factor(trades),
            expectancy=outcomes.average_trade_pnl,
            maximum_consecutive_losses=streak_max,
            loss_streak_distribution=streak_distribution,
            evidence_strength=(
                EvidenceStrength.DESCRIPTIVE_ONLY
                if combined
                else evidence_strength(len(trades), minimum)
            ),
            outcomes=outcomes,
            costs=costs,
            excursions=excursion_diagnostics(trades, minimum),
            stop_loss=stop_loss_diagnostics(trades),
            take_profit=take_profit_diagnostics(trades),
            holding=holding_diagnostics(trades),
            exit_reasons=exits,
            holding_buckets=holding_buckets(trades, minimum),
            feature_buckets=feature_bucket_diagnostics(trades, minimum),
            market_regimes=market_regime_diagnostics(trades, minimum),
            volatility_regimes=volatility,
            utc_hours=utc_hour_diagnostics(trades, minimum),
            utc_day_groups=utc_day_diagnostics(trades, minimum),
            entry_features=entry_feature_comparisons(trades),
            observations=observations,
        )

    @staticmethod
    def _observations(
        trades: tuple[TradeDiagnostic, ...],
        costs: CostDiagnostics,
        exits: tuple[ExitReasonDiagnostics, ...],
        volatility: tuple[GroupDiagnostics, ...],
        minimum: int,
        *,
        combined: bool,
    ) -> tuple[DiagnosticObservation, ...]:
        evidence = (
            EvidenceStrength.DESCRIPTIVE_ONLY
            if combined
            else evidence_strength(len(trades), minimum)
        )
        observations: list[DiagnosticObservation] = []
        frictionless = costs.frictionless_pnl
        net = costs.net_pnl
        if frictionless < 0:
            observations.append(
                DiagnosticObservation(
                    f"Frictionless PnL is negative ({frictionless}); losses exist before costs.",
                    evidence,
                )
            )
        elif frictionless > 0 and net < 0:
            observations.append(
                DiagnosticObservation(
                    "Frictionless PnL is positive but simulated slippage and fees make net PnL negative.",
                    evidence,
                )
            )
        else:
            observations.append(
                DiagnosticObservation(
                    f"Frictionless PnL is {frictionless} and net PnL is {net}.",
                    evidence,
                )
            )
        if net < 0:
            fee_share = costs.fee_drag / abs(net) * Decimal("100")
            slippage_share = abs(costs.slippage_drag) / abs(net) * Decimal("100")
            observations.append(
                DiagnosticObservation(
                    f"Fees equal {fee_share}% and slippage drag equals {slippage_share}% of the net loss magnitude.",
                    evidence,
                )
            )
        negative_exits = [row for row in exits if row.net_pnl < 0]
        if negative_exits:
            worst = min(negative_exits, key=lambda row: row.net_pnl)
            observations.append(
                DiagnosticObservation(
                    f"{worst.exit_reason.value} contributes the most negative exit-group PnL ({worst.net_pnl}).",
                    worst.evidence_strength,
                )
            )
        if len(trades) < minimum:
            observations.append(
                DiagnosticObservation(
                    f"Only {len(trades)} trades are available; subgroup findings are low-sample descriptions.",
                    EvidenceStrength.LOW_SAMPLE_SIZE,
                )
            )
        nonempty_volatility = [row for row in volatility if row.trade_count]
        if len(nonempty_volatility) > 1:
            best = max(nonempty_volatility, key=lambda row: row.expectancy)
            worst = min(nonempty_volatility, key=lambda row: row.expectancy)
            observations.append(
                DiagnosticObservation(
                    f"Causal volatility-regime expectancy ranges from {worst.expectancy} ({worst.group}, n={worst.trade_count}) to {best.expectancy} ({best.group}, n={best.trade_count}).",
                    EvidenceStrength.DESCRIPTIVE_ONLY,
                )
            )
        return tuple(observations)

    @staticmethod
    def _cross_period_observations(
        periods: tuple[PeriodDiagnostics, ...]
    ) -> tuple[DiagnosticObservation, ...]:
        negative_count = sum(period.costs.net_pnl < 0 for period in periods)
        frictionless_negative = sum(
            period.costs.frictionless_pnl < 0 for period in periods
        )
        return (
            DiagnosticObservation(
                f"Net PnL is negative in {negative_count} of 3 isolated periods; frictionless PnL is negative in {frictionless_negative} of 3.",
                EvidenceStrength.DESCRIPTIVE_ONLY,
            ),
            DiagnosticObservation(
                "Validation and out-of-sample remain isolated; combined diagnostics are descriptive only.",
                EvidenceStrength.DESCRIPTIVE_ONLY,
            ),
        )
