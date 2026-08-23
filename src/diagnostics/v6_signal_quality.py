"""Read-only signal-quality diagnostics for the frozen V6-H0 replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from statistics import median

from src.backtest.models import ExitReason, Trade
from src.diagnostics.excursions import calculate_excursions
from src.diagnostics.r_normalized import RNormalizedTrade
from src.historical.dataset import HistoricalDataset
from src.strategy.v6_mtf_continuation import (
    SOURCE_BAR_DURATION,
    V6PreparedContext,
)


POSITIVE_THRESHOLDS = (
    Decimal("0.25"),
    Decimal("0.50"),
    Decimal("1.00"),
    Decimal("1.50"),
    Decimal("2.00"),
)
NEGATIVE_THRESHOLDS = (
    Decimal("0.25"),
    Decimal("0.50"),
    Decimal("0.75"),
    Decimal("1.00"),
)

FEATURE_FIELDS = (
    "ema_spread_4h",
    "ema_spread_percent_4h",
    "close_distance_above_ema20_4h",
    "atr14_4h",
    "atr_percent_4h",
    "previous_close_distance_from_ema20_15m",
    "current_close_distance_above_ema20_15m",
    "continuation_above_previous_high",
    "signal_body_size",
    "signal_range",
    "signal_body_range_ratio",
    "lower_wick_size",
    "upper_wick_size",
    "initial_stop_distance_bps",
    "base_friction_r",
)


@dataclass(frozen=True, slots=True)
class ThresholdPath:
    positive_bars: dict[str, int | None]
    negative_bars: dict[str, int | None]
    negative_075_before_positive_050: str
    same_bar_order_ambiguity: bool


@dataclass(frozen=True, slots=True)
class V6TradeLedger:
    window_id: str
    trade_id: str
    entry_signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    exit_reason: ExitReason
    bars_held: int
    frictionless_r: Decimal
    net_r: Decimal
    fee_r: Decimal
    slippage_r: Decimal
    total_friction_r: Decimal
    entry_fill: Decimal
    initial_stop: Decimal
    take_profit: Decimal
    initial_stop_distance: Decimal
    initial_stop_distance_bps: Decimal
    stop_source: str
    mfe_r: Decimal
    mae_r: Decimal
    first_four_bar_mae_r: Decimal
    positive_025_bar: int | None
    positive_050_bar: int | None
    positive_100_bar: int | None
    positive_150_bar: int | None
    positive_200_bar: int | None
    negative_025_bar: int | None
    negative_050_bar: int | None
    negative_075_bar: int | None
    negative_100_bar: int | None
    negative_075_before_positive_050: str
    same_bar_order_ambiguity: bool
    ema20_4h: Decimal
    ema50_4h: Decimal
    ema_spread_4h: Decimal
    ema_spread_percent_4h: Decimal
    completed_close_4h: Decimal
    close_distance_above_ema20_4h: Decimal
    atr14_4h: Decimal
    atr_percent_4h: Decimal
    previous_close_distance_from_ema20_15m: Decimal
    current_close_distance_above_ema20_15m: Decimal
    continuation_above_previous_high: Decimal
    signal_body_size: Decimal
    signal_range: Decimal
    signal_body_range_ratio: Decimal | None
    lower_wick_size: Decimal
    upper_wick_size: Decimal
    base_friction_r: Decimal


def _mean(values) -> Decimal:
    selected = tuple(values)
    return (
        sum(selected, start=Decimal("0")) / Decimal(len(selected))
        if selected
        else Decimal("0")
    )


def _percent(count: int, total: int) -> Decimal:
    return Decimal(count) / Decimal(total) * Decimal("100") if total else Decimal("0")


def _first_touch(held, *, price: Decimal, favorable: bool) -> int | None:
    return next(
        (
            index
            for index, candle in enumerate(held, start=1)
            if (candle.high >= price if favorable else candle.low <= price)
        ),
        None,
    )


def calculate_threshold_path(
    *, trade: Trade, dataset: HistoricalDataset, initial_risk_per_unit: Decimal
) -> ThresholdPath:
    """Calculate bar-level threshold touches without inventing intrabar order."""

    if initial_risk_per_unit <= 0:
        raise ValueError("Initial risk per unit must be positive.")
    timestamp_index = {
        candle.timestamp: index for index, candle in enumerate(dataset.candles)
    }
    try:
        entry_index = timestamp_index[trade.entry_time]
    except KeyError as exc:
        raise ValueError(f"Entry candle missing for {trade.trade_id}.") from exc
    held = dataset.candles[entry_index : entry_index + trade.bars_held]
    if len(held) != trade.bars_held:
        raise ValueError(f"Held candle range incomplete for {trade.trade_id}.")

    positive = {
        str(threshold): _first_touch(
            held,
            price=trade.entry_price + initial_risk_per_unit * threshold,
            favorable=True,
        )
        for threshold in POSITIVE_THRESHOLDS
    }
    negative = {
        str(threshold): _first_touch(
            held,
            price=trade.entry_price - initial_risk_per_unit * threshold,
            favorable=False,
        )
        for threshold in NEGATIVE_THRESHOLDS
    }
    adverse = negative["0.75"]
    favorable = positive["0.50"]
    if adverse is None:
        ordering = "FALSE"
    elif favorable is None or adverse < favorable:
        ordering = "TRUE"
    elif adverse > favorable:
        ordering = "FALSE"
    else:
        ordering = "AMBIGUOUS"
    ambiguity = any(
        positive_bar is not None
        and negative_bar is not None
        and positive_bar == negative_bar
        for positive_bar in positive.values()
        for negative_bar in negative.values()
    )
    return ThresholdPath(positive, negative, ordering, ambiguity)


def build_trade_ledger(
    *,
    window_id: str,
    trades: tuple[Trade, ...],
    r_records: tuple[RNormalizedTrade, ...],
    stop_observations: tuple,
    replay_dataset: HistoricalDataset,
    evaluation_dataset: HistoricalDataset,
    prepared_context: V6PreparedContext,
) -> tuple[V6TradeLedger, ...]:
    """Join frozen execution/accounting with causal signal-time features."""

    if not (len(trades) == len(r_records) == len(stop_observations)):
        raise ValueError("V6 diagnostic trade inputs are not aligned.")
    candles = {candle.timestamp: candle for candle in replay_dataset.candles}
    evaluation_index = {
        candle.timestamp: index for index, candle in enumerate(evaluation_dataset.candles)
    }
    output: list[V6TradeLedger] = []
    for trade, r_record, observation in zip(
        trades, r_records, stop_observations, strict=True
    ):
        if not (
            trade.trade_id == r_record.trade_id == observation.trade_id
            and observation.initial_stop_distance > 0
        ):
            raise ValueError(f"V6 diagnostic alignment failed for {trade.trade_id}.")
        signal_timestamp = trade.entry_signal_time - SOURCE_BAR_DURATION
        previous_timestamp = signal_timestamp - SOURCE_BAR_DURATION
        try:
            current = candles[signal_timestamp]
            previous = candles[previous_timestamp]
            current_indicators = prepared_context.indicators_15m_by_timestamp[
                signal_timestamp
            ]
            previous_indicators = prepared_context.indicators_15m_by_timestamp[
                previous_timestamp
            ]
        except KeyError as exc:
            raise ValueError(f"V6 causal entry context missing for {trade.trade_id}.") from exc
        state = prepared_context.state_at(signal_timestamp)
        if (
            state is None
            or current_indicators.ema_fast is None
            or previous_indicators.ema_fast is None
        ):
            raise ValueError(f"V6 causal indicators missing for {trade.trade_id}.")
        excursions = calculate_excursions(
            trade,
            evaluation_dataset,
            evaluation_index,
            initial_risk_per_unit=observation.initial_stop_distance,
        )
        path = calculate_threshold_path(
            trade=trade,
            dataset=evaluation_dataset,
            initial_risk_per_unit=observation.initial_stop_distance,
        )
        assert excursions.mfe_r is not None
        assert excursions.mae_r is not None
        assert excursions.first_four_bar_mae_r is not None
        candle_range = current.high - current.low
        body = abs(current.close - current.open)
        spread = state.ema_20 - state.ema_50
        output.append(
            V6TradeLedger(
                window_id=window_id,
                trade_id=trade.trade_id,
                entry_signal_time=trade.entry_signal_time,
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
                exit_reason=trade.exit_reason,
                bars_held=trade.bars_held,
                frictionless_r=r_record.frictionless_r,
                net_r=r_record.net_r,
                fee_r=r_record.fee_cost_r,
                slippage_r=r_record.slippage_cost_r,
                total_friction_r=r_record.total_friction_r,
                entry_fill=trade.entry_price,
                initial_stop=trade.entry_price - observation.initial_stop_distance,
                take_profit=(
                    trade.entry_price + Decimal("2") * observation.initial_stop_distance
                ),
                initial_stop_distance=observation.initial_stop_distance,
                initial_stop_distance_bps=observation.initial_stop_distance_bps,
                stop_source=(
                    "ATR4H"
                    if observation.stop_source == "ATR14_4H"
                    else "COST_FLOOR"
                ),
                mfe_r=excursions.mfe_r,
                mae_r=excursions.mae_r,
                first_four_bar_mae_r=excursions.first_four_bar_mae_r,
                positive_025_bar=path.positive_bars["0.25"],
                positive_050_bar=path.positive_bars["0.50"],
                positive_100_bar=path.positive_bars["1.00"],
                positive_150_bar=path.positive_bars["1.50"],
                positive_200_bar=path.positive_bars["2.00"],
                negative_025_bar=path.negative_bars["0.25"],
                negative_050_bar=path.negative_bars["0.50"],
                negative_075_bar=path.negative_bars["0.75"],
                negative_100_bar=path.negative_bars["1.00"],
                negative_075_before_positive_050=(
                    path.negative_075_before_positive_050
                ),
                same_bar_order_ambiguity=path.same_bar_order_ambiguity,
                ema20_4h=state.ema_20,
                ema50_4h=state.ema_50,
                ema_spread_4h=spread,
                ema_spread_percent_4h=(
                    spread / state.latest_candle.close * Decimal("100")
                ),
                completed_close_4h=state.latest_candle.close,
                close_distance_above_ema20_4h=(
                    state.latest_candle.close - state.ema_20
                ),
                atr14_4h=state.atr_14,
                atr_percent_4h=(
                    state.atr_14 / state.latest_candle.close * Decimal("100")
                ),
                previous_close_distance_from_ema20_15m=(
                    previous.close - previous_indicators.ema_fast
                ),
                current_close_distance_above_ema20_15m=(
                    current.close - current_indicators.ema_fast
                ),
                continuation_above_previous_high=(current.close - previous.high),
                signal_body_size=body,
                signal_range=candle_range,
                signal_body_range_ratio=(
                    body / candle_range if candle_range > 0 else None
                ),
                lower_wick_size=(min(current.open, current.close) - current.low),
                upper_wick_size=(current.high - max(current.open, current.close)),
                base_friction_r=r_record.total_friction_r,
            )
        )
    return tuple(output)


def summarize_records(records: tuple[V6TradeLedger, ...]) -> dict:
    count = len(records)
    winners = tuple(record.net_r for record in records if record.net_r > 0)
    losers = tuple(record.net_r for record in records if record.net_r < 0)
    gross_winners = sum(winners, start=Decimal("0"))
    gross_losers = abs(sum(losers, start=Decimal("0")))
    return {
        "trades": count,
        "frictionless_expectancy_r": _mean(
            record.frictionless_r for record in records
        ),
        "net_expectancy_r": _mean(record.net_r for record in records),
        "profit_factor_r": (
            gross_winners / gross_losers if gross_losers > 0 else None
        ),
        "win_rate_percent": _percent(len(winners), count),
        "average_net_r": _mean(record.net_r for record in records),
        "median_net_r": median(tuple(record.net_r for record in records)) if records else Decimal("0"),
        "average_mfe_r": _mean(record.mfe_r for record in records),
        "median_mfe_r": median(tuple(record.mfe_r for record in records)) if records else Decimal("0"),
        "average_mae_r": _mean(record.mae_r for record in records),
        "median_mae_r": median(tuple(record.mae_r for record in records)) if records else Decimal("0"),
        "average_bars_held": _mean(Decimal(record.bars_held) for record in records),
        "total_net_r": sum((record.net_r for record in records), start=Decimal("0")),
    }


def summarize_exit_groups(
    records: tuple[V6TradeLedger, ...]
) -> tuple[dict, ...]:
    output = []
    for reason in ExitReason:
        selected = tuple(record for record in records if record.exit_reason is reason)
        if not selected:
            continue
        output.append(
            {
                "exit_reason": reason.value,
                "percentage_of_total": _percent(len(selected), len(records)),
                **summarize_records(selected),
            }
        )
    return tuple(output)


def summarize_feature_comparison(
    records: tuple[V6TradeLedger, ...]
) -> tuple[dict, ...]:
    winners = tuple(record for record in records if record.net_r > 0)
    losers = tuple(record for record in records if record.net_r < 0)
    rows = []
    for field in FEATURE_FIELDS:
        winner_values = tuple(
            value for record in winners if (value := getattr(record, field)) is not None
        )
        loser_values = tuple(
            value for record in losers if (value := getattr(record, field)) is not None
        )
        winner_mean = _mean(winner_values)
        loser_mean = _mean(loser_values)
        effect = None
        if len(winner_values) > 1 and len(loser_values) > 1:
            winner_variance = sum(
                ((value - winner_mean) ** 2 for value in winner_values),
                start=Decimal("0"),
            ) / Decimal(len(winner_values) - 1)
            loser_variance = sum(
                ((value - loser_mean) ** 2 for value in loser_values),
                start=Decimal("0"),
            ) / Decimal(len(loser_values) - 1)
            pooled = (
                (
                    Decimal(len(winner_values) - 1) * winner_variance
                    + Decimal(len(loser_values) - 1) * loser_variance
                )
                / Decimal(len(winner_values) + len(loser_values) - 2)
            ).sqrt()
            effect = (winner_mean - loser_mean) / pooled if pooled > 0 else None
        rows.append(
            {
                "feature": field,
                "winner_count": len(winner_values),
                "loser_count": len(loser_values),
                "winner_mean": winner_mean,
                "loser_mean": loser_mean,
                "winner_median": median(winner_values) if winner_values else None,
                "loser_median": median(loser_values) if loser_values else None,
                "mean_difference": winner_mean - loser_mean,
                "standardized_effect_size": effect,
            }
        )
    return tuple(rows)


def summarize_loser_paths(records: tuple[V6TradeLedger, ...]) -> dict:
    losers = tuple(record for record in records if record.net_r < 0)
    ordered = tuple(
        record
        for record in losers
        if record.negative_075_before_positive_050 != "AMBIGUOUS"
    )
    return {
        "trades": len(losers),
        "negative_050_within_4_bars_percent": _percent(
            sum(
                record.negative_050_bar is not None and record.negative_050_bar <= 4
                for record in losers
            ),
            len(losers),
        ),
        "negative_050_within_8_bars_percent": _percent(
            sum(
                record.negative_050_bar is not None and record.negative_050_bar <= 8
                for record in losers
            ),
            len(losers),
        ),
        "negative_075_before_positive_050_unambiguous_percent": _percent(
            sum(
                record.negative_075_before_positive_050 == "TRUE"
                for record in ordered
            ),
            len(ordered),
        ),
        "ordering_eligible_trades": len(ordered),
        "reached_positive_050_percent": _percent(
            sum(record.positive_050_bar is not None for record in losers), len(losers)
        ),
        "reached_positive_100_percent": _percent(
            sum(record.positive_100_bar is not None for record in losers), len(losers)
        ),
        "reached_positive_150_percent": _percent(
            sum(record.positive_150_bar is not None for record in losers), len(losers)
        ),
        "median_mfe_r": median(tuple(record.mfe_r for record in losers)),
        "median_mae_r": median(tuple(record.mae_r for record in losers)),
        "median_holding_bars": median(
            tuple(Decimal(record.bars_held) for record in losers)
        ),
    }


def _median_touch(records, field: str) -> Decimal | None:
    values = tuple(
        Decimal(value)
        for record in records
        if (value := getattr(record, field)) is not None
    )
    return median(values) if values else None


def summarize_winner_paths(records: tuple[V6TradeLedger, ...]) -> dict:
    winners = tuple(record for record in records if record.net_r > 0)
    return {
        "trades": len(winners),
        "average_mfe_r": _mean(record.mfe_r for record in winners),
        "median_mfe_r": median(tuple(record.mfe_r for record in winners)),
        "average_mae_r": _mean(record.mae_r for record in winners),
        "median_mae_r": median(tuple(record.mae_r for record in winners)),
        "reached_negative_025_percent": _percent(
            sum(record.negative_025_bar is not None for record in winners), len(winners)
        ),
        "reached_negative_050_percent": _percent(
            sum(record.negative_050_bar is not None for record in winners), len(winners)
        ),
        "median_bars_to_positive_050": _median_touch(winners, "positive_050_bar"),
        "median_bars_to_positive_100": _median_touch(winners, "positive_100_bar"),
        "median_bars_to_positive_200": _median_touch(winners, "positive_200_bar"),
        "average_bars_held": _mean(Decimal(record.bars_held) for record in winners),
    }


def window_diagnostics(records: tuple[V6TradeLedger, ...]) -> tuple[dict, ...]:
    output = []
    for window_id in sorted({record.window_id for record in records}):
        selected = tuple(record for record in records if record.window_id == window_id)
        row = {"window_id": window_id, **summarize_records(selected)}
        for feature in FEATURE_FIELDS:
            winners = tuple(
                getattr(record, feature)
                for record in selected
                if record.net_r > 0 and getattr(record, feature) is not None
            )
            losers = tuple(
                getattr(record, feature)
                for record in selected
                if record.net_r < 0 and getattr(record, feature) is not None
            )
            row[f"{feature}_winner_loser_mean_difference"] = (
                _mean(winners) - _mean(losers) if winners and losers else None
            )
        output.append(row)
    return tuple(output)


def cost_diagnostics(records: tuple[V6TradeLedger, ...]) -> dict:
    gross = _mean(record.frictionless_r for record in records)
    friction = _mean(record.total_friction_r for record in records)
    net = _mean(record.net_r for record in records)
    winners = tuple(record for record in records if record.net_r > 0)
    losers = tuple(record for record in records if record.net_r < 0)
    average_net_winner = _mean(record.net_r for record in winners)
    average_net_loser = _mean(record.net_r for record in losers)
    average_gross_winner = _mean(record.frictionless_r for record in winners)
    average_gross_loser = _mean(record.frictionless_r for record in losers)

    def required_win_rate(target: Decimal) -> Decimal | None:
        denominator = average_gross_winner - average_gross_loser
        return (
            (target + friction - average_gross_loser)
            / denominator
            * Decimal("100")
            if denominator > 0
            else None
        )

    return {
        "gross_edge_r_per_trade": gross,
        "average_friction_r_per_trade": friction,
        "net_edge_r_per_trade": net,
        "friction_to_absolute_gross_edge_ratio": (
            friction / abs(gross) if gross != 0 else None
        ),
        "minimum_gross_expectancy_to_break_even_r": friction,
        "average_net_winner_r": average_net_winner,
        "average_net_loser_r": average_net_loser,
        "average_gross_winner_r": average_gross_winner,
        "average_gross_loser_r": average_gross_loser,
        "break_even_win_rate_percent": (
            abs(average_net_loser)
            / (average_net_winner + abs(average_net_loser))
            * Decimal("100")
        ),
        "required_win_rate_for_0r_percent": required_win_rate(Decimal("0")),
        "required_win_rate_for_0_10r_percent": required_win_rate(Decimal("0.10")),
        "required_win_rate_for_0_25r_percent": required_win_rate(Decimal("0.25")),
    }


def ledger_rows(records: tuple[V6TradeLedger, ...]) -> list[dict]:
    return [asdict(record) for record in records]
