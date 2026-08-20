"""Reusable risk calculations with no order execution responsibilities."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from src.config.settings import Settings


def _decimal(name: str, value: Decimal) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{name} must be a valid decimal number.") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be finite.")
    return parsed


@dataclass(frozen=True, slots=True)
class RiskManager:
    """Validate capital limits and calculate risk budgets for future use."""

    max_capital_usdc: Decimal
    max_risk_per_trade_percent: Decimal
    max_daily_loss_percent: Decimal

    def __post_init__(self) -> None:
        for name in (
            "max_capital_usdc",
            "max_risk_per_trade_percent",
            "max_daily_loss_percent",
        ):
            object.__setattr__(self, name, _decimal(name, getattr(self, name)))

        if self.max_capital_usdc <= 0:
            raise ValueError("max_capital_usdc must be greater than zero.")
        if not Decimal("0") < self.max_risk_per_trade_percent <= Decimal("100"):
            raise ValueError("max_risk_per_trade_percent must be in (0, 100].")
        if not Decimal("0") < self.max_daily_loss_percent <= Decimal("100"):
            raise ValueError("max_daily_loss_percent must be in (0, 100].")

    @classmethod
    def from_settings(cls, settings: Settings) -> RiskManager:
        """Build a risk manager from the centralized application settings."""

        return cls(
            max_capital_usdc=settings.max_capital_usdc,
            max_risk_per_trade_percent=settings.max_risk_per_trade_percent,
            max_daily_loss_percent=settings.max_daily_loss_percent,
        )

    def validate_capital(self, capital_usdc: Decimal) -> Decimal:
        """Return validated capital or raise when it is outside the configured limit."""

        capital = _decimal("capital_usdc", capital_usdc)
        if capital <= 0:
            raise ValueError("capital_usdc must be greater than zero.")
        if capital > self.max_capital_usdc:
            raise ValueError("capital_usdc exceeds the configured maximum bot capital.")
        return capital

    def calculate_max_risk_amount(self, capital_usdc: Decimal) -> Decimal:
        """Calculate the largest permitted loss for one future trade."""

        capital = self.validate_capital(capital_usdc)
        return capital * self.max_risk_per_trade_percent / Decimal("100")

    def calculate_max_daily_loss(self, capital_usdc: Decimal) -> Decimal:
        """Calculate the maximum permitted daily loss for the given capital."""

        capital = self.validate_capital(capital_usdc)
        return capital * self.max_daily_loss_percent / Decimal("100")

    def remaining_daily_loss_allowance(
        self, capital_usdc: Decimal, realized_daily_pnl_usdc: Decimal
    ) -> Decimal:
        """Calculate loss capacity remaining after realized daily PnL."""

        daily_limit = self.calculate_max_daily_loss(capital_usdc)
        realized_pnl = _decimal("realized_daily_pnl_usdc", realized_daily_pnl_usdc)
        realized_loss = max(-realized_pnl, Decimal("0"))
        return max(daily_limit - realized_loss, Decimal("0"))

    def is_potential_loss_allowed(
        self,
        potential_loss_usdc: Decimal,
        capital_usdc: Decimal,
        realized_daily_pnl_usdc: Decimal = Decimal("0"),
    ) -> bool:
        """Check a hypothetical loss against per-trade and daily limits."""

        potential_loss = _decimal("potential_loss_usdc", potential_loss_usdc)
        if potential_loss < 0:
            raise ValueError("potential_loss_usdc must not be negative.")
        return (
            potential_loss <= self.calculate_max_risk_amount(capital_usdc)
            and potential_loss
            <= self.remaining_daily_loss_allowance(
                capital_usdc, realized_daily_pnl_usdc
            )
        )

