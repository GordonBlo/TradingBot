"""Post-trade OHLC excursion calculations isolated from strategy inputs."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.backtest.models import Trade
from src.historical.dataset import HistoricalDataset


@dataclass(frozen=True, slots=True)
class TradeExcursions:
    highest_observed_price: Decimal
    lowest_observed_price: Decimal
    mfe: Decimal
    mae: Decimal
    mfe_percent: Decimal
    mae_percent: Decimal
    mfe_r: Decimal | None
    mae_r: Decimal | None
    bars_to_mfe: int
    bars_to_mae: int
    first_four_bar_mae_r: Decimal | None


def calculate_excursions(
    trade: Trade,
    dataset: HistoricalDataset,
    timestamp_index: dict[object, int],
    *,
    initial_risk_per_unit: Decimal | None,
) -> TradeExcursions:
    """Inspect only the completed trade's held OHLC bars, after the backtest."""

    try:
        entry_index = timestamp_index[trade.entry_time]
    except KeyError as exc:
        raise ValueError(f"Entry candle missing for {trade.trade_id}.") from exc
    held = dataset.candles[entry_index : entry_index + trade.bars_held]
    if len(held) != trade.bars_held:
        raise ValueError(f"Held candle range is incomplete for {trade.trade_id}.")
    highest = max(candle.high for candle in held)
    lowest = min(candle.low for candle in held)
    mfe = max(Decimal("0"), highest - trade.entry_price)
    mae = max(Decimal("0"), trade.entry_price - lowest)
    highest_index = next(
        index for index, candle in enumerate(held, start=1) if candle.high == highest
    )
    lowest_index = next(
        index for index, candle in enumerate(held, start=1) if candle.low == lowest
    )
    first_four_low = min(candle.low for candle in held[:4])
    first_four_mae = max(Decimal("0"), trade.entry_price - first_four_low)
    valid_risk = initial_risk_per_unit is not None and initial_risk_per_unit > 0
    return TradeExcursions(
        highest_observed_price=highest,
        lowest_observed_price=lowest,
        mfe=mfe,
        mae=mae,
        mfe_percent=mfe / trade.entry_price * Decimal("100"),
        mae_percent=mae / trade.entry_price * Decimal("100"),
        mfe_r=mfe / initial_risk_per_unit if valid_risk else None,
        mae_r=mae / initial_risk_per_unit if valid_risk else None,
        bars_to_mfe=highest_index,
        bars_to_mae=lowest_index,
        first_four_bar_mae_r=(
            first_four_mae / initial_risk_per_unit if valid_risk else None
        ),
    )

