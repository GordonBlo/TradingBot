"""Typed interface for offline, non-mutating research strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from src.strategy.context import StrategyContext
from src.strategy.models import StrategyDecision


class BaseStrategy(ABC):
    """A strategy can only evaluate context and return a decision."""

    @abstractmethod
    def evaluate(self, context: StrategyContext) -> StrategyDecision:
        """Evaluate one closed candle without mutating the account."""
