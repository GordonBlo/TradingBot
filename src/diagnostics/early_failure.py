"""Post-trade early-failure diagnostics for frozen H5_Q25."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.backtest.models import ExitReason
from src.diagnostics.h5_exit_diagnostics import H5ExitTradeRecord


EARLY_FAILURE_THRESHOLD_R = Decimal("0.5")
EARLY_FAILURE_BARS = 4


@dataclass(frozen=True, slots=True)
class EarlyFailureSummary:
    trades: int
    percentage_of_all_trades: Decimal

    frictionless_expectancy_r: Decimal
    net_expectancy_r: Decimal

    final_positive_percent: Decimal

    stop_loss_trades: int
    take_profit_trades: int
    trend_exit_trades: int
    time_exit_trades: int

    take_profit_percent: Decimal

    later_positive_half_r_percent: Decimal
    later_positive_one_r_percent: Decimal
    later_positive_two_r_percent: Decimal

    same_bar_positive_half_r_ambiguous: int
    same_bar_positive_one_r_ambiguous: int
    same_bar_positive_two_r_ambiguous: int


def _percent(count: int, total: int) -> Decimal:
    if total == 0:
        return Decimal("0")

    return (
        Decimal(count)
        / Decimal(total)
        * Decimal("100")
    )


def _mean(values) -> Decimal:
    selected = tuple(values)

    if not selected:
        return Decimal("0")

    return (
        sum(selected, start=Decimal("0"))
        / Decimal(len(selected))
    )


def _strictly_after(
    *,
    adverse_bar: int | None,
    favorable_bar: int | None,
) -> bool:
    return (
        adverse_bar is not None
        and favorable_bar is not None
        and favorable_bar > adverse_bar
    )


def _same_bar(
    *,
    adverse_bar: int | None,
    favorable_bar: int | None,
) -> bool:
    return (
        adverse_bar is not None
        and favorable_bar is not None
        and favorable_bar == adverse_bar
    )


def summarize_early_failure(
    records: tuple[H5ExitTradeRecord, ...],
) -> EarlyFailureSummary:
    """Analyze trades that hit -0.5R within their first four held bars."""

    selected = tuple(
        record
        for record in records
        if (
            record.bars_to_negative_half_r is not None
            and record.bars_to_negative_half_r
            <= EARLY_FAILURE_BARS
        )
    )

    count = len(selected)

    later_half = sum(
        _strictly_after(
            adverse_bar=record.bars_to_negative_half_r,
            favorable_bar=record.bars_to_positive_half_r,
        )
        for record in selected
    )

    later_one = sum(
        _strictly_after(
            adverse_bar=record.bars_to_negative_half_r,
            favorable_bar=record.bars_to_positive_one_r,
        )
        for record in selected
    )

    later_two = sum(
        _strictly_after(
            adverse_bar=record.bars_to_negative_half_r,
            favorable_bar=record.bars_to_positive_two_r,
        )
        for record in selected
    )

    same_half = sum(
        _same_bar(
            adverse_bar=record.bars_to_negative_half_r,
            favorable_bar=record.bars_to_positive_half_r,
        )
        for record in selected
    )

    same_one = sum(
        _same_bar(
            adverse_bar=record.bars_to_negative_half_r,
            favorable_bar=record.bars_to_positive_one_r,
        )
        for record in selected
    )

    same_two = sum(
        _same_bar(
            adverse_bar=record.bars_to_negative_half_r,
            favorable_bar=record.bars_to_positive_two_r,
        )
        for record in selected
    )

    return EarlyFailureSummary(
        trades=count,
        percentage_of_all_trades=_percent(
            count,
            len(records),
        ),

        frictionless_expectancy_r=_mean(
            record.frictionless_r
            for record in selected
        ),

        net_expectancy_r=_mean(
            record.net_r
            for record in selected
        ),

        final_positive_percent=_percent(
            sum(record.net_r > 0 for record in selected),
            count,
        ),

        stop_loss_trades=sum(
            record.exit_reason is ExitReason.STOP_LOSS
            for record in selected
        ),

        take_profit_trades=sum(
            record.exit_reason is ExitReason.TAKE_PROFIT
            for record in selected
        ),

        trend_exit_trades=sum(
            record.exit_reason is ExitReason.TREND_EXIT
            for record in selected
        ),

        time_exit_trades=sum(
            record.exit_reason is ExitReason.TIME_EXIT
            for record in selected
        ),

        take_profit_percent=_percent(
            sum(
                record.exit_reason is ExitReason.TAKE_PROFIT
                for record in selected
            ),
            count,
        ),

        later_positive_half_r_percent=_percent(
            later_half,
            count,
        ),

        later_positive_one_r_percent=_percent(
            later_one,
            count,
        ),

        later_positive_two_r_percent=_percent(
            later_two,
            count,
        ),

        same_bar_positive_half_r_ambiguous=same_half,
        same_bar_positive_one_r_ambiguous=same_one,
        same_bar_positive_two_r_ambiguous=same_two,
    )