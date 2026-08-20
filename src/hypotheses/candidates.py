"""Small isolated H0-H4 behaviors layered around the frozen baseline."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Callable

from src.hypotheses.causal import CausalRollingPercentile
from src.hypotheses.models import (
    HypothesisId,
    HypothesisJournalRecord,
    HypothesisSuiteConfig,
    JournalEvent,
)
from src.strategy.base import BaseStrategy
from src.strategy.baseline_trend import TrendMomentumBaselineStrategy
from src.strategy.context import StrategyContext
from src.strategy.models import (
    DecisionReason,
    StrategyAction,
    StrategyDecision,
    TrendMomentumConfig,
)


def _hold(reason: str) -> StrategyDecision:
    return StrategyDecision(StrategyAction.HOLD, DecisionReason.NO_ENTRY, reason)


class EntryGuardStrategy(BaseStrategy):
    """Accept or reject an otherwise unchanged baseline entry decision."""

    def __init__(
        self,
        *,
        hypothesis_id: HypothesisId,
        baseline_config: TrendMomentumConfig,
        lookback: int,
        percentile_rank: Decimal,
        value_name: str,
        extractor: Callable[[StrategyContext], Decimal | None],
    ) -> None:
        self.baseline_config = baseline_config
        self._baseline = TrendMomentumBaselineStrategy(baseline_config)
        self._hypothesis_id = hypothesis_id
        self._value_name = value_name
        self._extractor = extractor
        self._rolling = CausalRollingPercentile(
            lookback=lookback, percentile_rank=percentile_rank
        )
        self.journal: list[HypothesisJournalRecord] = []

    def evaluate(self, context: StrategyContext) -> StrategyDecision:
        current_value = self._extractor(context)
        threshold = self._rolling.evaluate_then_observe(current_value)
        baseline = self._baseline.evaluate(context)
        if baseline.action is not StrategyAction.ENTER_LONG:
            return baseline
        if threshold is not None and current_value is not None and current_value <= threshold:
            return baseline
        reason = (
            f"{self._value_name} entry guard lacks {self._rolling.lookback} prior valid observations"
            if threshold is None
            else f"{self._value_name} {current_value} exceeds prior-window Q{self._rolling.percentile_rank} {threshold}"
        )
        self.journal.append(
            HypothesisJournalRecord(
                timestamp=context.timestamp,
                hypothesis_id=self._hypothesis_id,
                event=JournalEvent.SKIPPED_ENTRY,
                reason=reason,
                values={
                    "baseline_signal": baseline.reason_code.value,
                    "candidate": self._hypothesis_id.value,
                    "atr_percent": _atr_percent(context),
                    "atr_percentile_threshold": (
                        threshold if self._hypothesis_id is HypothesisId.H1 else None
                    ),
                    "ema_spread_percent": _ema_spread_percent(context),
                    "ema_percentile_threshold": (
                        threshold if self._hypothesis_id is HypothesisId.H2 else None
                    ),
                },
            )
        )
        return _hold(reason)


class OneRTargetStrategy(BaseStrategy):
    """Preserve the baseline decision and replace only an entry's R:R value."""

    def __init__(
        self, baseline_config: TrendMomentumConfig, reward_risk_ratio: Decimal
    ) -> None:
        self.baseline_config = baseline_config
        self.reward_risk_ratio = Decimal(str(reward_risk_ratio))
        self._baseline = TrendMomentumBaselineStrategy(baseline_config)
        self.journal: list[HypothesisJournalRecord] = []

    def evaluate(self, context: StrategyContext) -> StrategyDecision:
        decision = self._baseline.evaluate(context)
        if decision.action is StrategyAction.ENTER_LONG:
            return replace(
                decision,
                reward_risk_ratio=self.reward_risk_ratio,
                metadata={**decision.metadata, "candidate_reward_risk_ratio": self.reward_risk_ratio},
            )
        return decision


class PullbackConfirmationStrategy(BaseStrategy):
    """Delay a valid baseline crossover until a causal pullback/reclaim closes."""

    def __init__(
        self, baseline_config: TrendMomentumConfig, confirmation_window_bars: int
    ) -> None:
        self.baseline_config = baseline_config
        self.confirmation_window_bars = confirmation_window_bars
        self._baseline = TrendMomentumBaselineStrategy(baseline_config)
        self._armed = False
        self._bars_observed = 0
        self.journal: list[HypothesisJournalRecord] = []

    @property
    def is_armed(self) -> bool:
        return self._armed

    def evaluate(self, context: StrategyContext) -> StrategyDecision:
        baseline = self._baseline.evaluate(context)
        current = context.indicators

        if context.has_position:
            self._armed = False
            self._bars_observed = 0
            return baseline

        if self._armed:
            if not current.is_ready:
                return self._cancel(context, JournalEvent.INVALIDATED, "Indicators became unavailable")
            assert current.ema_fast is not None
            assert current.ema_slow is not None
            assert current.rsi is not None
            assert current.volume_ratio is not None
            if current.ema_fast <= current.ema_slow:
                return self._cancel(context, JournalEvent.INVALIDATED, "EMA20 is no longer above EMA50")
            if current.close <= current.ema_slow:
                return self._cancel(context, JournalEvent.INVALIDATED, "Close is no longer above EMA50")

            self._bars_observed += 1
            confirmed = (
                context.current_candle.low <= current.ema_fast
                and current.close >= current.ema_fast
                and self.baseline_config.rsi_min <= current.rsi <= self.baseline_config.rsi_max
                and current.volume_ratio >= self.baseline_config.minimum_volume_ratio
            )
            if confirmed:
                decision = self._confirmed_entry(context)
                self._record(
                    context,
                    JournalEvent.CONFIRMED,
                    "EMA20 pullback/reclaim confirmed",
                )
                self._armed = False
                self._bars_observed = 0
                return decision
            if self._bars_observed >= self.confirmation_window_bars:
                return self._cancel(
                    context,
                    JournalEvent.EXPIRED,
                    f"No confirmation within {self.confirmation_window_bars} closed bars",
                )
            return _hold("H3 remains armed awaiting a closed-candle pullback/reclaim")

        if baseline.action is StrategyAction.ENTER_LONG:
            self._armed = True
            self._bars_observed = 0
            self._record(context, JournalEvent.CROSSOVER_DETECTED, baseline.reason)
            self._record(
                context,
                JournalEvent.ARMED,
                f"Armed for {self.confirmation_window_bars} subsequently closed bars",
            )
            return _hold("H3 armed; immediate baseline crossover entry intentionally delayed")
        return baseline

    def _confirmed_entry(self, context: StrategyContext) -> StrategyDecision:
        """Reuse the baseline's exact risk model after H3 timing confirmation."""

        current = context.indicators
        assert current.atr is not None
        assert current.ema_fast is not None
        assert current.ema_slow is not None
        assert current.rsi is not None
        assert current.volume_ratio is not None
        stop_distance = current.atr * self.baseline_config.atr_stop_multiplier
        risk_budget = (
            context.equity
            * self.baseline_config.risk_per_trade_percent
            / Decimal("100")
        )
        affordable = context.cash_usdc / (Decimal("1") + context.entry_fee_rate)
        max_quote = min(affordable, self.baseline_config.maximum_position_notional_usdc)
        if risk_budget <= 0 or stop_distance <= 0 or max_quote <= 0:
            return StrategyDecision(
                StrategyAction.HOLD,
                DecisionReason.INSUFFICIENT_CAPITAL,
                "No positive risk budget or affordable Spot notional",
            )
        return StrategyDecision(
            StrategyAction.ENTER_LONG,
            DecisionReason.EMA_CROSSOVER_ENTRY,
            "H3 causal EMA20 pullback/reclaim confirmed; execute at next bar open",
            risk_budget=risk_budget,
            stop_distance=stop_distance,
            reward_risk_ratio=self.baseline_config.reward_risk_ratio,
            max_quote_amount=max_quote,
            metadata={
                "ema_fast": current.ema_fast,
                "ema_slow": current.ema_slow,
                "rsi": current.rsi,
                "atr": current.atr,
                "volume_ratio": current.volume_ratio,
                "confirmation_bars": self._bars_observed,
            },
        )

    def _cancel(
        self, context: StrategyContext, event: JournalEvent, reason: str
    ) -> StrategyDecision:
        self._record(context, event, reason)
        self._armed = False
        self._bars_observed = 0
        return _hold(f"H3 setup {event.value.lower()}: {reason}")

    def _record(
        self, context: StrategyContext, event: JournalEvent, reason: str
    ) -> None:
        self.journal.append(
            HypothesisJournalRecord(
                timestamp=context.timestamp,
                hypothesis_id=HypothesisId.H3,
                event=event,
                reason=reason,
                values={"bars_observed": self._bars_observed},
            )
        )


def _atr_percent(context: StrategyContext) -> Decimal | None:
    atr = context.indicators.atr
    close = context.indicators.close
    return atr / close * Decimal("100") if atr is not None and close > 0 else None


def _ema_spread_percent(context: StrategyContext) -> Decimal | None:
    fast = context.indicators.ema_fast
    slow = context.indicators.ema_slow
    if fast is None or slow is None or slow == 0:
        return None
    return (fast - slow) / slow * Decimal("100")


def build_candidate(
    hypothesis_id: HypothesisId,
    baseline_config: TrendMomentumConfig,
    suite_config: HypothesisSuiteConfig | None = None,
) -> BaseStrategy:
    """Build one independent candidate; combinations are intentionally impossible."""

    candidate = HypothesisId(hypothesis_id)
    fixed = suite_config or HypothesisSuiteConfig()
    if candidate is HypothesisId.H0:
        return TrendMomentumBaselineStrategy(baseline_config)
    if candidate is HypothesisId.H1:
        return EntryGuardStrategy(
            hypothesis_id=candidate,
            baseline_config=baseline_config,
            lookback=fixed.h1_atr_percentile_lookback,
            percentile_rank=fixed.h1_max_percentile,
            value_name="atr_percent",
            extractor=_atr_percent,
        )
    if candidate is HypothesisId.H2:
        return EntryGuardStrategy(
            hypothesis_id=candidate,
            baseline_config=baseline_config,
            lookback=fixed.h2_ema_spread_lookback,
            percentile_rank=fixed.h2_max_percentile,
            value_name="ema_spread_percent",
            extractor=_ema_spread_percent,
        )
    if candidate is HypothesisId.H3:
        return PullbackConfirmationStrategy(
            baseline_config, fixed.h3_confirmation_window_bars
        )
    if candidate is HypothesisId.H4:
        return OneRTargetStrategy(baseline_config, fixed.h4_reward_risk_ratio)
    raise ValueError(f"Unsupported hypothesis: {candidate}")
