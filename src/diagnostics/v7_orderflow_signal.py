"""Fixed causal order-flow features for the frozen V6-H0 trade sample."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from statistics import median

from src.models.candle import Candle
from src.orderflow.aggregation import OrderFlowBucket


NUMERIC_FEATURES = (
    "qimb_0",
    "buy_quote_ratio_0",
    "signed_quote_0",
    "total_quote_volume_0",
    "average_aggtrade_quote_size_0",
    "qimb_1",
    "qimb_delta_1",
    "qimb_mean_4",
    "qimb_slope_4",
    "cumulative_quote_imbalance_4",
    "cumulative_signed_quote_4",
    "quote_volume_ratio_vs_prev3",
    "aggtrade_size_ratio_vs_prev3",
    "underlying_trade_count_ratio_vs_prev3",
    "signal_return",
    "four_bar_return",
    "price_flow_interaction_1",
    "price_flow_interaction_4",
)


def _mean(values: tuple[Decimal, ...]) -> Decimal:
    return sum(values, start=Decimal("0")) / Decimal(len(values))


def _ratio(value: Decimal, denominator: Decimal) -> Decimal | None:
    return value / denominator if denominator != 0 else None


def qimb_slope_4(values_oldest_to_newest: tuple[Decimal, ...]) -> Decimal:
    """Least-squares slope for y=[qimb_3,qimb_2,qimb_1,qimb_0], x=0..3."""

    if len(values_oldest_to_newest) != 4:
        raise ValueError("qimb_slope_4 requires exactly four values.")
    # sum((x-1.5)*y) / sum((x-1.5)^2), whose denominator is exactly 5.
    weights = (Decimal("-1.5"), Decimal("-0.5"), Decimal("0.5"), Decimal("1.5"))
    return sum(
        (weight * value for weight, value in zip(weights, values_oldest_to_newest, strict=True)),
        start=Decimal("0"),
    ) / Decimal("5")


def alignment_state(price_return: Decimal, flow: Decimal) -> str:
    if price_return == 0 or flow == 0:
        return "NEUTRAL"
    return f"PRICE_{'UP' if price_return > 0 else 'DOWN'}_FLOW_{'BUY' if flow > 0 else 'SELL'}"


def build_causal_features(
    *,
    buckets: tuple[OrderFlowBucket, ...],
    signal_candle: Candle,
    t_minus_4_candle: Candle,
) -> dict[str, Decimal | str | bool | None]:
    """Build the fixed feature set from t-3..t and completed candle data only."""

    if len(buckets) != 4:
        raise ValueError("Exactly four completed order-flow buckets are required.")
    step = timedelta(minutes=15)
    if any(
        later.bucket_open_time - earlier.bucket_open_time != step
        for earlier, later in zip(buckets, buckets[1:], strict=False)
    ):
        raise ValueError("Order-flow buckets must be four contiguous 15m buckets.")
    current = buckets[-1]
    if (
        current.bucket_open_time != signal_candle.timestamp
        or current.bucket_close_time != signal_candle.timestamp + step
        or t_minus_4_candle.timestamp != signal_candle.timestamp - 4 * step
    ):
        raise ValueError("Causal candle/order-flow timestamp join failed.")
    if signal_candle.open == 0 or t_minus_4_candle.close == 0:
        raise ValueError("Price-return denominators must be non-zero.")

    qimb = tuple(bucket.quote_volume_imbalance for bucket in buckets)
    qimb_3, qimb_2, qimb_1, qimb_0 = qimb
    buy_quote = sum(
        (bucket.taker_buy_quote_volume for bucket in buckets), Decimal("0")
    )
    sell_quote = sum(
        (bucket.taker_sell_quote_volume for bucket in buckets), Decimal("0")
    )
    total_quote = sum((bucket.total_quote_volume for bucket in buckets), Decimal("0"))
    cumulative_imbalance = _ratio(buy_quote - sell_quote, total_quote)
    previous_quote_mean = _mean(tuple(bucket.total_quote_volume for bucket in buckets[:3]))
    previous_size_mean = _mean(
        tuple(bucket.average_aggtrade_quote_size for bucket in buckets[:3])
    )
    previous_count_mean = _mean(
        tuple(Decimal(bucket.underlying_trade_count) for bucket in buckets[:3])
    )
    signal_return = (signal_candle.close - signal_candle.open) / signal_candle.open
    four_bar_return = (
        signal_candle.close - t_minus_4_candle.close
    ) / t_minus_4_candle.close
    if cumulative_imbalance is None:
        raise ValueError("Four-bucket cumulative quote volume is unavailable.")

    return {
        "qimb_0": qimb_0,
        "buy_quote_ratio_0": current.taker_buy_quote_ratio,
        "signed_quote_0": current.signed_quote_volume,
        "total_quote_volume_0": current.total_quote_volume,
        "average_aggtrade_quote_size_0": current.average_aggtrade_quote_size,
        "qimb_1": qimb_1,
        "qimb_delta_1": qimb_0 - qimb_1,
        "qimb_mean_4": _mean((qimb_0, qimb_1, qimb_2, qimb_3)),
        "qimb_slope_4": qimb_slope_4((qimb_3, qimb_2, qimb_1, qimb_0)),
        "cumulative_quote_imbalance_4": cumulative_imbalance,
        "cumulative_signed_quote_4": buy_quote - sell_quote,
        "quote_volume_ratio_vs_prev3": _ratio(
            current.total_quote_volume, previous_quote_mean
        ),
        "aggtrade_size_ratio_vs_prev3": _ratio(
            current.average_aggtrade_quote_size, previous_size_mean
        ),
        "underlying_trade_count_ratio_vs_prev3": _ratio(
            Decimal(current.underlying_trade_count), previous_count_mean
        ),
        "signal_return": signal_return,
        "four_bar_return": four_bar_return,
        "price_flow_interaction_1": signal_return * qimb_0,
        "price_flow_interaction_4": four_bar_return * cumulative_imbalance,
        "current_price_flow_alignment": alignment_state(signal_return, qimb_0),
        "four_bar_price_flow_alignment": alignment_state(
            four_bar_return, cumulative_imbalance
        ),
        "sell_absorption_proxy": four_bar_return >= 0 and cumulative_imbalance < 0,
        "buy_absorption_proxy": four_bar_return <= 0 and cumulative_imbalance > 0,
    }


def cliffs_delta(winners: tuple[Decimal, ...], losers: tuple[Decimal, ...]) -> Decimal | None:
    """Deterministic pairwise Cliff's delta: (greater-less)/(n_w*n_l)."""

    if not winners or not losers:
        return None
    greater = sum(winner > loser for winner in winners for loser in losers)
    less = sum(winner < loser for winner in winners for loser in losers)
    return Decimal(greater - less) / Decimal(len(winners) * len(losers))


def compare_feature(rows: tuple[dict, ...], feature: str) -> dict:
    winners = tuple(
        row[feature] for row in rows if row["net_r"] > 0 and row[feature] is not None
    )
    losers = tuple(
        row[feature] for row in rows if row["net_r"] <= 0 and row[feature] is not None
    )
    winner_mean = _mean(winners) if winners else None
    loser_mean = _mean(losers) if losers else None
    effect = None
    if len(winners) > 1 and len(losers) > 1:
        winner_variance = sum(
            ((value - winner_mean) ** 2 for value in winners), Decimal("0")
        ) / Decimal(len(winners) - 1)
        loser_variance = sum(
            ((value - loser_mean) ** 2 for value in losers), Decimal("0")
        ) / Decimal(len(losers) - 1)
        pooled_variance = (
            Decimal(len(winners) - 1) * winner_variance
            + Decimal(len(losers) - 1) * loser_variance
        ) / Decimal(len(winners) + len(losers) - 2)
        effect = (
            (winner_mean - loser_mean) / pooled_variance.sqrt()
            if pooled_variance > 0
            else None
        )
    return {
        "feature": feature,
        "valid_winner_count": len(winners),
        "valid_loser_count": len(losers),
        "winner_mean": winner_mean,
        "loser_mean": loser_mean,
        "winner_median": median(winners) if winners else None,
        "loser_median": median(losers) if losers else None,
        "mean_difference": (
            winner_mean - loser_mean if winner_mean is not None and loser_mean is not None else None
        ),
        "median_difference": (
            median(winners) - median(losers) if winners and losers else None
        ),
        "standardized_mean_difference": effect,
        "cliffs_delta": cliffs_delta(winners, losers),
    }


def summarize_feature_comparisons(rows: tuple[dict, ...]) -> tuple[dict, ...]:
    return tuple(compare_feature(rows, feature) for feature in NUMERIC_FEATURES)


def feature_window_consistency(
    rows: tuple[dict, ...], comparisons: tuple[dict, ...]
) -> tuple[dict, ...]:
    windows = sorted({row["window_id"] for row in rows})
    output = []
    for comparison in comparisons:
        feature = comparison["feature"]
        differences = []
        for window_id in windows:
            selected = tuple(row for row in rows if row["window_id"] == window_id)
            window = compare_feature(selected, feature)
            if window["median_difference"] is not None:
                differences.append(window["median_difference"])
        aggregate = comparison["median_difference"]
        same_direction = sum(
            (aggregate > 0 and difference > 0)
            or (aggregate < 0 and difference < 0)
            or (aggregate == 0 and difference == 0)
            for difference in differences
        ) if aggregate is not None else 0
        eligible = len(differences)
        ratio = Decimal(same_direction) / Decimal(eligible) if eligible else None
        output.append(
            {
                "feature": feature,
                "aggregate_median_difference": aggregate,
                "eligible_comparison_windows": eligible,
                "winner_median_greater_windows": sum(value > 0 for value in differences),
                "winner_median_lower_windows": sum(value < 0 for value in differences),
                "winner_median_equal_windows": sum(value == 0 for value in differences),
                "same_direction_windows": same_direction,
                "consistency_ratio": ratio,
                "directionally_unstable": (
                    ratio is not None and ratio < Decimal("0.60")
                ),
            }
        )
    return tuple(output)
