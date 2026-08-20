"""Reusable chronological rolling-percentile state with no current/future leakage."""

from __future__ import annotations

from collections import deque
from decimal import Decimal

class CausalRollingPercentile:
    """Estimate a threshold from the previous N valid observations only."""

    def __init__(self, *, lookback: int, percentile_rank: Decimal) -> None:
        rank = Decimal(str(percentile_rank))
        if lookback < 1:
            raise ValueError("Rolling-percentile lookback must be positive.")
        if not Decimal("0") <= rank <= Decimal("100"):
            raise ValueError("Percentile rank must be between 0 and 100.")
        self.lookback = lookback
        self.percentile_rank = rank
        self._values: deque[Decimal] = deque(maxlen=lookback)

    @property
    def observation_count(self) -> int:
        return len(self._values)

    def threshold(self) -> Decimal | None:
        """Return Qp only after a full prior-observation window exists."""

        if len(self._values) < self.lookback:
            return None
        ordered = sorted(self._values)
        fraction = self.percentile_rank / Decimal("100")
        position = fraction * Decimal(len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - Decimal(lower)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * weight

    def observe(self, value: Decimal | None) -> None:
        """Append the current observation after callers read its threshold."""

        if value is None:
            return
        parsed = Decimal(str(value))
        if not parsed.is_finite():
            raise ValueError("Rolling-percentile observations must be finite.")
        self._values.append(parsed)

    def evaluate_then_observe(self, value: Decimal | None) -> Decimal | None:
        """Convenience operation preserving the strict previous-values policy."""

        result = self.threshold()
        self.observe(value)
        return result
