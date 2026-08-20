"""Entry-time feature comparisons and descriptive non-overlapping buckets."""

from __future__ import annotations

from decimal import Decimal
from typing import Callable

from src.diagnostics.distributions import (
    grouped_diagnostics,
    optional_mean,
    percentile,
)
from src.diagnostics.models import (
    EntryFeatureComparison,
    GroupDiagnostics,
    TradeDiagnostic,
)


def entry_feature_comparisons(
    trades: tuple[TradeDiagnostic, ...],
) -> tuple[EntryFeatureComparison, ...]:
    features = (
        ("RSI", "entry_rsi"),
        ("Volume Ratio", "entry_volume_ratio"),
        ("ATR %", "entry_atr_percent"),
        ("EMA spread %", "entry_ema_spread_percent"),
        ("Close vs EMA slow %", "close_vs_slow_ema_percent"),
    )
    return tuple(
        EntryFeatureComparison(
            feature=label,
            winner_average=optional_mean(
                getattr(trade, attribute)
                for trade in trades
                if trade.net_pnl > 0
            ),
            loser_average=optional_mean(
                getattr(trade, attribute)
                for trade in trades
                if trade.net_pnl < 0
            ),
            all_average=optional_mean(
                getattr(trade, attribute) for trade in trades
            ),
        )
        for label, attribute in features
    )


def _fixed_buckets(
    trades: tuple[TradeDiagnostic, ...],
    *,
    dimension: str,
    classifier: Callable[[TradeDiagnostic], str],
    groups: tuple[str, ...],
    minimum: int,
) -> tuple[GroupDiagnostics, ...]:
    return grouped_diagnostics(
        trades,
        dimension=dimension,
        classifier=classifier,
        groups=groups,
        minimum_trades_warning=minimum,
    )


def _quantile_buckets(
    trades: tuple[TradeDiagnostic, ...],
    *,
    dimension: str,
    attribute: str,
    minimum: int,
) -> tuple[GroupDiagnostics, ...]:
    values = tuple(
        value
        for trade in trades
        if (value := getattr(trade, attribute)) is not None
    )
    if not values:
        boundaries = (Decimal("0"), Decimal("0"), Decimal("0"))
    else:
        boundaries = (
            percentile(values, Decimal("0.25")) or Decimal("0"),
            percentile(values, Decimal("0.50")) or Decimal("0"),
            percentile(values, Decimal("0.75")) or Decimal("0"),
        )
    labels = (
        f"Q1 < {boundaries[0]}",
        f"Q2 < {boundaries[1]}",
        f"Q3 < {boundaries[2]}",
        f"Q4 >= {boundaries[2]}",
        "UNAVAILABLE",
    )

    def classify(trade: TradeDiagnostic) -> str:
        value = getattr(trade, attribute)
        if value is None:
            return "UNAVAILABLE"
        if value < boundaries[0]:
            return labels[0]
        if value < boundaries[1]:
            return labels[1]
        if value < boundaries[2]:
            return labels[2]
        return labels[3]

    return _fixed_buckets(
        trades,
        dimension=f"{dimension} (DESCRIPTIVE QUARTILES)",
        classifier=classify,
        groups=labels,
        minimum=minimum,
    )


def feature_bucket_diagnostics(
    trades: tuple[TradeDiagnostic, ...], minimum: int
) -> tuple[GroupDiagnostics, ...]:
    def rsi_bucket(trade: TradeDiagnostic) -> str:
        value = trade.entry_rsi
        if value is None:
            return "UNAVAILABLE"
        if value < Decimal("55"):
            return "52-<55"
        if value < Decimal("60"):
            return "55-<60"
        if value < Decimal("64"):
            return "60-<64"
        return "64-68"

    def volume_bucket(trade: TradeDiagnostic) -> str:
        value = trade.entry_volume_ratio
        if value is None:
            return "UNAVAILABLE"
        if value < Decimal("1"):
            return "0.80-<1.00"
        if value < Decimal("1.25"):
            return "1.00-<1.25"
        if value < Decimal("1.50"):
            return "1.25-<1.50"
        return "1.50+"

    rsi = _fixed_buckets(
        trades,
        dimension="entry_rsi",
        classifier=rsi_bucket,
        groups=("52-<55", "55-<60", "60-<64", "64-68", "UNAVAILABLE"),
        minimum=minimum,
    )
    volume = _fixed_buckets(
        trades,
        dimension="entry_volume_ratio",
        classifier=volume_bucket,
        groups=(
            "0.80-<1.00",
            "1.00-<1.25",
            "1.25-<1.50",
            "1.50+",
            "UNAVAILABLE",
        ),
        minimum=minimum,
    )
    atr = _quantile_buckets(
        trades,
        dimension="entry_atr_percent",
        attribute="entry_atr_percent",
        minimum=minimum,
    )
    ema = _quantile_buckets(
        trades,
        dimension="entry_ema_spread_percent",
        attribute="entry_ema_spread_percent",
        minimum=minimum,
    )
    return rsi + volume + atr + ema

