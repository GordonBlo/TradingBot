"""Deterministic Decimal distributions, grouping, holding, and streak metrics."""

from __future__ import annotations

from decimal import Decimal
from typing import Callable, Iterable

from src.diagnostics.models import (
    EvidenceStrength,
    GroupDiagnostics,
    HoldingDiagnostics,
    OutcomeDiagnostics,
    TradeDiagnostic,
)


def decimal_mean(values: Iterable[Decimal]) -> Decimal:
    items = tuple(values)
    return sum(items, start=Decimal("0")) / Decimal(len(items)) if items else Decimal("0")


def percentile(values: Iterable[Decimal], fraction: Decimal) -> Decimal | None:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * Decimal(len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - Decimal(lower)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def median(values: Iterable[Decimal]) -> Decimal:
    result = percentile(values, Decimal("0.5"))
    return result if result is not None else Decimal("0")


def optional_mean(values: Iterable[Decimal | None]) -> Decimal | None:
    present = tuple(value for value in values if value is not None)
    return decimal_mean(present) if present else None


def optional_median(values: Iterable[Decimal | None]) -> Decimal | None:
    present = tuple(value for value in values if value is not None)
    return percentile(present, Decimal("0.5"))


def profit_factor(trades: Iterable[TradeDiagnostic]) -> Decimal | None:
    items = tuple(trades)
    gross_profit = sum(
        (trade.net_pnl for trade in items if trade.net_pnl > 0),
        start=Decimal("0"),
    )
    gross_loss = sum(
        (trade.net_pnl for trade in items if trade.net_pnl < 0),
        start=Decimal("0"),
    )
    return gross_profit / abs(gross_loss) if gross_loss != 0 else None


def evidence_strength(count: int, minimum: int) -> EvidenceStrength:
    if count == 0:
        return EvidenceStrength.DESCRIPTIVE_ONLY
    if count < minimum:
        return EvidenceStrength.LOW_SAMPLE_SIZE
    return EvidenceStrength.MODERATE_SAMPLE


def summarize_group(
    dimension: str,
    group: str,
    trades: Iterable[TradeDiagnostic],
    *,
    total_trades: int,
    minimum_trades_warning: int,
) -> GroupDiagnostics:
    items = tuple(trades)
    count = len(items)
    winners = sum(trade.net_pnl > 0 for trade in items)
    net_values = tuple(trade.net_pnl for trade in items)
    return GroupDiagnostics(
        dimension=dimension,
        group=group,
        trade_count=count,
        percentage_of_trades=(
            Decimal(count) / Decimal(total_trades) * Decimal("100")
            if total_trades
            else Decimal("0")
        ),
        win_rate_percent=(
            Decimal(winners) / Decimal(count) * Decimal("100")
            if count
            else Decimal("0")
        ),
        gross_pnl=sum((trade.gross_pnl for trade in items), start=Decimal("0")),
        net_pnl=sum(net_values, start=Decimal("0")),
        expectancy=decimal_mean(net_values),
        profit_factor=profit_factor(items),
        average_r=optional_mean(trade.r_multiple for trade in items),
        maximum_trade_loss=min(
            (trade.net_pnl for trade in items if trade.net_pnl < 0),
            default=None,
        ),
        average_bars_held=decimal_mean(
            Decimal(trade.bars_held) for trade in items
        ),
        average_mfe_r=optional_mean(trade.mfe_r for trade in items),
        average_mae_r=optional_mean(trade.mae_r for trade in items),
        evidence_strength=evidence_strength(count, minimum_trades_warning),
    )


def outcome_diagnostics(trades: tuple[TradeDiagnostic, ...]) -> OutcomeDiagnostics:
    winners = tuple(trade for trade in trades if trade.net_pnl > 0)
    losers = tuple(trade for trade in trades if trade.net_pnl < 0)
    breakeven = tuple(trade for trade in trades if trade.net_pnl == 0)
    winner_values = tuple(trade.net_pnl for trade in winners)
    loser_values = tuple(trade.net_pnl for trade in losers)
    all_values = tuple(trade.net_pnl for trade in trades)
    win_rate = Decimal(len(winners)) / Decimal(len(trades)) if trades else Decimal("0")
    loss_rate = Decimal(len(losers)) / Decimal(len(trades)) if trades else Decimal("0")
    average_win = decimal_mean(winner_values)
    average_loss = decimal_mean(loser_values)
    expectancy = win_rate * average_win + loss_rate * average_loss
    average_trade = decimal_mean(all_values)
    if abs(expectancy - average_trade) > Decimal("1e-24"):
        raise ValueError("Outcome expectancy does not match average trade PnL.")
    return OutcomeDiagnostics(
        winning_trades=len(winners),
        losing_trades=len(losers),
        breakeven_trades=len(breakeven),
        average_win=average_win,
        median_win=median(winner_values),
        largest_win=max(winner_values, default=Decimal("0")),
        average_loss=average_loss,
        median_loss=median(loser_values),
        largest_loss=min(loser_values, default=Decimal("0")),
        average_winner_r=optional_mean(trade.r_multiple for trade in winners),
        median_winner_r=optional_median(trade.r_multiple for trade in winners),
        average_loser_r=optional_mean(trade.r_multiple for trade in losers),
        median_loser_r=optional_median(trade.r_multiple for trade in losers),
        pnl_percentile_25=percentile(all_values, Decimal("0.25")),
        pnl_percentile_50=percentile(all_values, Decimal("0.50")),
        pnl_percentile_75=percentile(all_values, Decimal("0.75")),
        expectancy_from_outcomes=expectancy,
        average_trade_pnl=average_trade,
    )


def holding_diagnostics(trades: tuple[TradeDiagnostic, ...]) -> HoldingDiagnostics:
    all_bars = tuple(Decimal(trade.bars_held) for trade in trades)
    winner_bars = tuple(
        Decimal(trade.bars_held) for trade in trades if trade.net_pnl > 0
    )
    loser_bars = tuple(
        Decimal(trade.bars_held) for trade in trades if trade.net_pnl < 0
    )
    return HoldingDiagnostics(
        average_bars=decimal_mean(all_bars),
        median_bars=median(all_bars),
        minimum_bars=min((trade.bars_held for trade in trades), default=0),
        maximum_bars=max((trade.bars_held for trade in trades), default=0),
        average_winner_bars=decimal_mean(winner_bars),
        median_winner_bars=median(winner_bars),
        average_loser_bars=decimal_mean(loser_bars),
        median_loser_bars=median(loser_bars),
    )


def grouped_diagnostics(
    trades: tuple[TradeDiagnostic, ...],
    *,
    dimension: str,
    classifier: Callable[[TradeDiagnostic], str],
    groups: tuple[str, ...],
    minimum_trades_warning: int,
) -> tuple[GroupDiagnostics, ...]:
    return tuple(
        summarize_group(
            dimension,
            group,
            (trade for trade in trades if classifier(trade) == group),
            total_trades=len(trades),
            minimum_trades_warning=minimum_trades_warning,
        )
        for group in groups
    )


def holding_buckets(
    trades: tuple[TradeDiagnostic, ...], minimum: int
) -> tuple[GroupDiagnostics, ...]:
    def classify(trade: TradeDiagnostic) -> str:
        bars = trade.bars_held
        if bars <= 4:
            return "1-4"
        if bars <= 16:
            return "5-16"
        if bars <= 32:
            return "17-32"
        if bars <= 64:
            return "33-64"
        if bars <= 96:
            return "65-96"
        return "97+"

    return grouped_diagnostics(
        trades,
        dimension="holding_bars",
        classifier=classify,
        groups=("1-4", "5-16", "17-32", "33-64", "65-96", "97+"),
        minimum_trades_warning=minimum,
    )


def loss_streaks(trades: tuple[TradeDiagnostic, ...]) -> tuple[int, tuple[tuple[str, int], ...]]:
    lengths: list[int] = []
    current = 0
    for trade in trades:
        if trade.net_pnl < 0:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    buckets = {
        "1": sum(length == 1 for length in lengths),
        "2": sum(length == 2 for length in lengths),
        "3": sum(length == 3 for length in lengths),
        "4+": sum(length >= 4 for length in lengths),
    }
    return max(lengths, default=0), tuple(buckets.items())
