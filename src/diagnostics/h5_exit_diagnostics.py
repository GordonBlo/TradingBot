"""V3.3.1 post-trade H5 R, exit, entry, and excursion summaries."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.backtest.models import ExitReason, Trade
from src.diagnostics.distributions import median
from src.diagnostics.excursions import calculate_excursion_thresholds
from src.diagnostics.models import TradeDiagnostic
from src.diagnostics.r_normalized import RNormalizedTrade
from src.historical.dataset import HistoricalDataset


@dataclass(frozen=True, slots=True)
class H5ExitTradeRecord:
    trade_id: str
    exit_reason: ExitReason
    frictionless_r: Decimal
    fee_r: Decimal
    slippage_r: Decimal
    total_friction_r: Decimal
    net_r: Decimal
    mfe_r: Decimal
    mae_r: Decimal
    first_four_bar_mae_r: Decimal
    holding_bars: int

    entry_atr_percent: Decimal | None
    entry_rsi: Decimal | None
    entry_volume_ratio: Decimal | None
    entry_ema_spread_percent: Decimal | None

    reached_positive_half_r: bool
    reached_positive_one_r: bool
    reached_positive_one_and_half_r: bool
    reached_positive_two_r: bool
    reached_negative_half_r: bool
    reached_negative_one_r: bool

    bars_to_positive_half_r: int | None
    bars_to_positive_one_r: int | None

    # Extended ordering diagnostics.
    # Defaults preserve compatibility with existing tests/helpers.
    bars_to_positive_one_and_half_r: int | None = None
    bars_to_positive_two_r: int | None = None
    bars_to_negative_half_r: int | None = None
    bars_to_negative_one_r: int | None = None


@dataclass(frozen=True, slots=True)
class ExitGroupSummary:
    exit_reason: ExitReason
    trades: int
    percentage_of_all_trades: Decimal
    total_frictionless_r: Decimal
    total_net_r: Decimal
    frictionless_expectancy_r: Decimal
    net_expectancy_r: Decimal
    average_fee_r: Decimal
    average_slippage_r: Decimal
    average_total_friction_r: Decimal
    win_rate_percent: Decimal
    average_winner_r: Decimal
    average_loser_r: Decimal
    payoff_ratio_r: Decimal | None
    profit_factor_r: Decimal | None
    median_mfe_r: Decimal
    median_mae_r: Decimal
    average_holding_bars: Decimal
    median_holding_bars: Decimal
    average_entry_atr_percent: Decimal | None
    average_entry_rsi: Decimal | None
    average_entry_volume_ratio: Decimal | None
    average_entry_ema_spread_percent: Decimal | None


@dataclass(frozen=True, slots=True)
class ExcursionGroupSummary:
    group: str
    trades: int
    reached_positive_half_r_percent: Decimal
    reached_positive_one_r_percent: Decimal
    reached_positive_one_and_half_r_percent: Decimal
    reached_positive_two_r_percent: Decimal
    reached_negative_half_r_percent: Decimal
    reached_negative_one_r_percent: Decimal
    median_bars_to_positive_half_r: Decimal | None
    median_bars_to_positive_one_r: Decimal | None
    first_four_bar_negative_half_r_percent: Decimal


def _mean(values) -> Decimal:
    selected = tuple(values)

    return (
        sum(selected, start=Decimal("0"))
        / Decimal(len(selected))
        if selected
        else Decimal("0")
    )


def _optional_mean(values) -> Decimal | None:
    selected = tuple(
        value
        for value in values
        if value is not None
    )

    return _mean(selected) if selected else None


def _percent(count: int, total: int) -> Decimal:
    return (
        Decimal(count)
        / Decimal(total)
        * Decimal("100")
        if total
        else Decimal("0")
    )


def build_h5_exit_trade_records(
    *,
    trades: tuple[Trade, ...],
    r_records: tuple[RNormalizedTrade, ...],
    diagnostics: tuple[TradeDiagnostic, ...],
    dataset: HistoricalDataset,
) -> tuple[H5ExitTradeRecord, ...]:
    """Join established diagnostics without changing strategy or execution."""

    if not (
        len(trades)
        == len(r_records)
        == len(diagnostics)
    ):
        raise ValueError(
            "H5 diagnostic trade inputs are not aligned."
        )

    timestamp_index = {
        candle.timestamp: index
        for index, candle in enumerate(dataset.candles)
    }

    output: list[H5ExitTradeRecord] = []

    for trade, r_record, diagnostic in zip(
        trades,
        r_records,
        diagnostics,
        strict=True,
    ):
        if not (
            trade.trade_id
            == r_record.trade_id
            == diagnostic.trade_id
            and diagnostic.initial_risk_per_unit is not None
            and diagnostic.initial_risk_per_unit > 0
            and diagnostic.mfe_r is not None
            and diagnostic.mae_r is not None
            and diagnostic.first_four_bar_mae_r is not None
        ):
            raise ValueError(
                f"Incomplete H5 diagnostics for {trade.trade_id}."
            )

        thresholds = calculate_excursion_thresholds(
            trade,
            dataset,
            timestamp_index,
            initial_risk_per_unit=(
                diagnostic.initial_risk_per_unit
            ),
        )

        output.append(
            H5ExitTradeRecord(
                trade_id=trade.trade_id,
                exit_reason=trade.exit_reason,
                frictionless_r=r_record.frictionless_r,
                fee_r=r_record.fee_cost_r,
                slippage_r=r_record.slippage_cost_r,
                total_friction_r=r_record.total_friction_r,
                net_r=r_record.net_r,
                mfe_r=diagnostic.mfe_r,
                mae_r=diagnostic.mae_r,
                first_four_bar_mae_r=(
                    diagnostic.first_four_bar_mae_r
                ),
                holding_bars=diagnostic.bars_held,
                entry_atr_percent=(
                    diagnostic.entry_atr_percent
                ),
                entry_rsi=diagnostic.entry_rsi,
                entry_volume_ratio=(
                    diagnostic.entry_volume_ratio
                ),
                entry_ema_spread_percent=(
                    diagnostic.entry_ema_spread_percent
                ),
                reached_positive_half_r=(
                    thresholds.reached_positive_half_r
                ),
                reached_positive_one_r=(
                    thresholds.reached_positive_one_r
                ),
                reached_positive_one_and_half_r=(
                    thresholds.reached_positive_one_and_half_r
                ),
                reached_positive_two_r=(
                    thresholds.reached_positive_two_r
                ),
                reached_negative_half_r=(
                    thresholds.reached_negative_half_r
                ),
                reached_negative_one_r=(
                    thresholds.reached_negative_one_r
                ),
                bars_to_positive_half_r=(
                    thresholds.bars_to_positive_half_r
                ),
                bars_to_positive_one_r=(
                    thresholds.bars_to_positive_one_r
                ),
                bars_to_positive_one_and_half_r=(
                    thresholds.bars_to_positive_one_and_half_r
                ),
                bars_to_positive_two_r=(
                    thresholds.bars_to_positive_two_r
                ),
                bars_to_negative_half_r=(
                    thresholds.bars_to_negative_half_r
                ),
                bars_to_negative_one_r=(
                    thresholds.bars_to_negative_one_r
                ),
            )
        )

    return tuple(output)


def summarize_exit_groups(
    records: tuple[H5ExitTradeRecord, ...],
) -> tuple[ExitGroupSummary, ...]:
    rows: list[ExitGroupSummary] = []

    for reason in ExitReason:
        selected = tuple(
            record
            for record in records
            if record.exit_reason is reason
        )

        if not selected:
            continue

        winners = tuple(
            record.net_r
            for record in selected
            if record.net_r > 0
        )

        losers = tuple(
            record.net_r
            for record in selected
            if record.net_r < 0
        )

        average_winner = _mean(winners)
        average_loser = _mean(losers)

        gross_winners = sum(
            winners,
            start=Decimal("0"),
        )

        gross_losers = abs(
            sum(
                losers,
                start=Decimal("0"),
            )
        )

        rows.append(
            ExitGroupSummary(
                exit_reason=reason,
                trades=len(selected),
                percentage_of_all_trades=_percent(
                    len(selected),
                    len(records),
                ),
                total_frictionless_r=sum(
                    (
                        record.frictionless_r
                        for record in selected
                    ),
                    Decimal("0"),
                ),
                total_net_r=sum(
                    (
                        record.net_r
                        for record in selected
                    ),
                    Decimal("0"),
                ),
                frictionless_expectancy_r=_mean(
                    record.frictionless_r
                    for record in selected
                ),
                net_expectancy_r=_mean(
                    record.net_r
                    for record in selected
                ),
                average_fee_r=_mean(
                    record.fee_r
                    for record in selected
                ),
                average_slippage_r=_mean(
                    record.slippage_r
                    for record in selected
                ),
                average_total_friction_r=_mean(
                    record.total_friction_r
                    for record in selected
                ),
                win_rate_percent=_percent(
                    len(winners),
                    len(selected),
                ),
                average_winner_r=average_winner,
                average_loser_r=average_loser,
                payoff_ratio_r=(
                    average_winner
                    / abs(average_loser)
                    if winners and losers
                    else None
                ),
                profit_factor_r=(
                    gross_winners
                    / gross_losers
                    if gross_losers > 0
                    else None
                ),
                median_mfe_r=median(
                    record.mfe_r
                    for record in selected
                ),
                median_mae_r=median(
                    record.mae_r
                    for record in selected
                ),
                average_holding_bars=_mean(
                    Decimal(record.holding_bars)
                    for record in selected
                ),
                median_holding_bars=median(
                    Decimal(record.holding_bars)
                    for record in selected
                ),
                average_entry_atr_percent=_optional_mean(
                    record.entry_atr_percent
                    for record in selected
                ),
                average_entry_rsi=_optional_mean(
                    record.entry_rsi
                    for record in selected
                ),
                average_entry_volume_ratio=_optional_mean(
                    record.entry_volume_ratio
                    for record in selected
                ),
                average_entry_ema_spread_percent=_optional_mean(
                    record.entry_ema_spread_percent
                    for record in selected
                ),
            )
        )

    return tuple(rows)


def summarize_excursion_groups(
    records: tuple[H5ExitTradeRecord, ...],
) -> tuple[ExcursionGroupSummary, ...]:
    groups = [
        (
            "ALL",
            records,
        ),
        (
            "LOSING_TRADES",
            tuple(
                record
                for record in records
                if record.net_r < 0
            ),
        ),
        *[
            (
                reason.value,
                tuple(
                    record
                    for record in records
                    if record.exit_reason is reason
                ),
            )
            for reason in ExitReason
            if any(
                record.exit_reason is reason
                for record in records
            )
        ],
    ]

    rows: list[ExcursionGroupSummary] = []

    for label, selected in groups:
        count = len(selected)

        rows.append(
            ExcursionGroupSummary(
                group=label,
                trades=count,
                reached_positive_half_r_percent=_percent(
                    sum(
                        record.reached_positive_half_r
                        for record in selected
                    ),
                    count,
                ),
                reached_positive_one_r_percent=_percent(
                    sum(
                        record.reached_positive_one_r
                        for record in selected
                    ),
                    count,
                ),
                reached_positive_one_and_half_r_percent=_percent(
                    sum(
                        record.reached_positive_one_and_half_r
                        for record in selected
                    ),
                    count,
                ),
                reached_positive_two_r_percent=_percent(
                    sum(
                        record.reached_positive_two_r
                        for record in selected
                    ),
                    count,
                ),
                reached_negative_half_r_percent=_percent(
                    sum(
                        record.reached_negative_half_r
                        for record in selected
                    ),
                    count,
                ),
                reached_negative_one_r_percent=_percent(
                    sum(
                        record.reached_negative_one_r
                        for record in selected
                    ),
                    count,
                ),
                median_bars_to_positive_half_r=(
                    median(
                        Decimal(
                            record.bars_to_positive_half_r
                        )
                        for record in selected
                        if (
                            record.bars_to_positive_half_r
                            is not None
                        )
                    )
                    if any(
                        record.bars_to_positive_half_r
                        is not None
                        for record in selected
                    )
                    else None
                ),
                median_bars_to_positive_one_r=(
                    median(
                        Decimal(
                            record.bars_to_positive_one_r
                        )
                        for record in selected
                        if (
                            record.bars_to_positive_one_r
                            is not None
                        )
                    )
                    if any(
                        record.bars_to_positive_one_r
                        is not None
                        for record in selected
                    )
                    else None
                ),
                first_four_bar_negative_half_r_percent=_percent(
                    sum(
                        record.first_four_bar_mae_r
                        >= Decimal("0.5")
                        for record in selected
                    ),
                    count,
                ),
            )
        )

    return tuple(rows)