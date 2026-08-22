"""Typed interface for offline, non-mutating research strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from src.strategy.context import StrategyContext
from src.strategy.models import StrategyDecision


class BaseStrategy(ABC):
    """A strategy can only evaluate context and return a decision."""

    @property
    def required_history_bars(self) -> int:
        """Minimum recent-history length required for causal evaluation."""

        return 0

    @property
    def requires_full_history(self) -> bool:
        """Whether evaluation needs every candle available at signal time."""

        return False

    @abstractmethod
    def evaluate(self, context: StrategyContext) -> StrategyDecision:
        """Evaluate one closed candle without mutating the account."""
