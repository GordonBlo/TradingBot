"""Causal entry-time market/volatility and UTC grouping diagnostics."""

from __future__ import annotations

from src.diagnostics.distributions import grouped_diagnostics
from src.diagnostics.models import (
    GroupDiagnostics,
    MarketRegime,
    TradeDiagnostic,
    VolatilityRegime,
)


def market_regime_diagnostics(
    trades: tuple[TradeDiagnostic, ...], minimum: int
) -> tuple[GroupDiagnostics, ...]:
    return grouped_diagnostics(
        trades,
        dimension="entry_market_regime",
        classifier=lambda trade: trade.market_regime.value,
        groups=tuple(regime.value for regime in MarketRegime),
        minimum_trades_warning=minimum,
    )


def volatility_regime_diagnostics(
    trades: tuple[TradeDiagnostic, ...], minimum: int
) -> tuple[GroupDiagnostics, ...]:
    return grouped_diagnostics(
        trades,
        dimension="entry_volatility_regime_causal_100_bar",
        classifier=lambda trade: trade.volatility_regime.value,
        groups=tuple(regime.value for regime in VolatilityRegime),
        minimum_trades_warning=minimum,
    )


def utc_hour_diagnostics(
    trades: tuple[TradeDiagnostic, ...], minimum: int
) -> tuple[GroupDiagnostics, ...]:
    groups = tuple(f"{hour:02d}" for hour in range(24))
    return grouped_diagnostics(
        trades,
        dimension="entry_utc_hour",
        classifier=lambda trade: f"{trade.entry_utc_hour:02d}",
        groups=groups,
        minimum_trades_warning=minimum,
    )


def utc_day_diagnostics(
    trades: tuple[TradeDiagnostic, ...], minimum: int
) -> tuple[GroupDiagnostics, ...]:
    groups = (
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    )
    return grouped_diagnostics(
        trades,
        dimension="entry_utc_day_of_week",
        classifier=lambda trade: trade.entry_day_of_week,
        groups=groups,
        minimum_trades_warning=minimum,
    )

