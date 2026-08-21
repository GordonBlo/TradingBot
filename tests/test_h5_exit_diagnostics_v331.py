from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from src.analysis.indicators import IndicatorEngine
from src.backtest.models import ExitReason, Trade
from src.diagnostics.excursions import calculate_excursion_thresholds
from src.diagnostics.h5_exit_diagnostics import (
    H5ExitTradeRecord,
    summarize_excursion_groups,
    summarize_exit_groups,
)
from src.historical.dataset import HistoricalDataset
from src.models.candle import Candle
from src.models.market_data_source import MarketDataSource
from src.research.multiregime.models import (
    PartitionKind,
    PartitionStatus,
    ResearchPartitionMetadata,
    ResearchRegion,
    ResearchWindow,
)
from src.diagnostics.window_warmup_parity import compare_indicator_boundary
from src.strategy.models import TrendMomentumConfig


UTC = timezone.utc
BASE = datetime(2024, 1, 1, tzinfo=UTC)


def _candle(index: int, close: Decimal, volume: Decimal | None = None) -> Candle:
    return Candle(
        timestamp=BASE + timedelta(minutes=15 * index),
        symbol="BTCUSDC",
        interval="15m",
        open=close,
        high=close + Decimal("3"),
        low=close - Decimal("2"),
        close=close,
        volume=volume if volume is not None else Decimal("10"),
        is_closed=True,
    )


def _trade() -> Trade:
    return Trade(
        trade_id="T1",
        entry_signal_time=BASE,
        entry_time=BASE,
        entry_price=Decimal("100"),
        exit_signal_time=BASE + timedelta(minutes=30),
        exit_time=BASE + timedelta(minutes=45),
        exit_price=Decimal("101"),
        quantity=Decimal("1"),
        entry_notional=Decimal("100"),
        exit_notional=Decimal("101"),
        entry_fee=Decimal("0"),
        exit_fee=Decimal("0"),
        total_fee=Decimal("0"),
        gross_pnl=Decimal("1"),
        net_pnl=Decimal("1"),
        return_percent=Decimal("1"),
        bars_held=3,
        exit_reason=ExitReason.TIME_EXIT,
    )


def _record(
    trade_id: str,
    *,
    net_r: str,
    frictionless_r: str,
    exit_reason: ExitReason = ExitReason.TIME_EXIT,
    reached_positive_half_r: bool = False,
    first_four_bar_mae_r: str = "0.2",
) -> H5ExitTradeRecord:
    return H5ExitTradeRecord(
        trade_id=trade_id,
        exit_reason=exit_reason,
        frictionless_r=Decimal(frictionless_r),
        fee_r=Decimal("0.4"),
        slippage_r=Decimal("0.1"),
        total_friction_r=Decimal("0.5"),
        net_r=Decimal(net_r),
        mfe_r=Decimal("1.2"),
        mae_r=Decimal("0.7"),
        first_four_bar_mae_r=Decimal(first_four_bar_mae_r),
        holding_bars=4,
        entry_atr_percent=Decimal("0.5"),
        entry_rsi=Decimal("60"),
        entry_volume_ratio=Decimal("1.1"),
        entry_ema_spread_percent=Decimal("0.2"),
        reached_positive_half_r=reached_positive_half_r,
        reached_positive_one_r=reached_positive_half_r,
        reached_positive_one_and_half_r=False,
        reached_positive_two_r=False,
        reached_negative_half_r=True,
        reached_negative_one_r=False,
        bars_to_positive_half_r=2 if reached_positive_half_r else None,
        bars_to_positive_one_r=3 if reached_positive_half_r else None,
    )


def test_exit_group_r_metrics_preserve_cost_identity() -> None:
    records = (
        _record("WIN", net_r="1.5", frictionless_r="2.0"),
        _record("LOSS", net_r="-1.5", frictionless_r="-1.0"),
    )
    row = summarize_exit_groups(records)[0]
    assert row.exit_reason is ExitReason.TIME_EXIT
    assert row.trades == 2
    assert row.frictionless_expectancy_r == Decimal("0.5")
    assert row.net_expectancy_r == Decimal("0")
    assert row.average_total_friction_r == Decimal("0.5")
    assert row.payoff_ratio_r == Decimal("1")
    assert row.profit_factor_r == Decimal("1")
    assert all(
        record.frictionless_r
        - record.slippage_r
        - record.fee_r
        == record.net_r
        for record in records
    )


def test_excursion_thresholds_use_first_held_bar_touch() -> None:
    candles = (
        _candle(0, Decimal("100")),
        _candle(1, Decimal("102")),
        _candle(2, Decimal("98")),
    )
    dataset = HistoricalDataset(
        "BTCUSDC", "15m", MarketDataSource.BINANCE_PUBLIC, candles
    )
    thresholds = calculate_excursion_thresholds(
        _trade(),
        dataset,
        {candle.timestamp: index for index, candle in enumerate(candles)},
        initial_risk_per_unit=Decimal("4"),
    )
    assert thresholds.reached_positive_half_r
    assert thresholds.reached_positive_one_r
    assert not thresholds.reached_positive_one_and_half_r
    assert thresholds.reached_negative_half_r
    assert thresholds.reached_negative_one_r
    assert thresholds.bars_to_positive_half_r == 1
    assert thresholds.bars_to_positive_one_r == 2


def test_losing_excursion_summary_distinguishes_early_adverse_and_reversal() -> None:
    records = (
        _record(
            "EARLY",
            net_r="-1",
            frictionless_r="-0.5",
            first_four_bar_mae_r="0.8",
        ),
        _record(
            "REVERSE",
            net_r="-1",
            frictionless_r="-0.5",
            reached_positive_half_r=True,
        ),
    )
    losing = next(
        row for row in summarize_excursion_groups(records) if row.group == "LOSING_TRADES"
    )
    assert losing.reached_positive_half_r_percent == Decimal("50")
    assert losing.first_four_bar_negative_half_r_percent == Decimal("50")
    assert losing.median_bars_to_positive_half_r == Decimal("2")


def test_finite_warmup_parity_is_measured_without_changing_volume_ratio_semantics() -> None:
    candles = tuple(
        _candle(
            index,
            Decimal("100")
            + Decimal(index) / Decimal("10")
            + Decimal((index * 7) % 11),
            Decimal(index % 20 + 1),
        )
        for index in range(520)
    )
    dataset = HistoricalDataset(
        "BTCUSDC", "15m", MarketDataSource.BINANCE_PUBLIC, candles
    )
    metadata = ResearchPartitionMetadata(
        kind=PartitionKind.RESEARCH_EXPANSION,
        symbol="BTCUSDC",
        interval="15m",
        start=candles[0].timestamp,
        end=candles[-1].timestamp + timedelta(minutes=15),
        status=PartitionStatus.CONSUMED_RESEARCH_DATA,
        candle_count=len(candles),
        created_at=BASE,
        source="SYNTHETIC_TEST_ONLY",
        dataset_sha256="synthetic",
    )
    replay = HistoricalDataset(
        "BTCUSDC",
        "15m",
        MarketDataSource.BINANCE_PUBLIC,
        candles[250:511],
    )
    window = ResearchWindow(
        window_id="PARITY",
        start=candles[500].timestamp,
        end=candles[511].timestamp,
        duration_days=Decimal("1"),
        partition_kind=PartitionKind.RESEARCH_EXPANSION,
        data_status=PartitionStatus.CONSUMED_RESEARCH_DATA,
        partial_window=True,
        duration_eligible=False,
        replay_dataset=replay,
        evaluation_start_index=250,
    )
    rows = {
        row.field: row
        for row in compare_indicator_boundary(
            window, ResearchRegion(metadata, dataset), TrendMomentumConfig()
        )
    }
    assert rows["ema_slow"].absolute_difference > 0
    assert rows["volume_sma"].absolute_difference == 0
    assert rows["volume_ratio"].absolute_difference == 0

    series = IndicatorEngine().calculate_research_series(candles[:20])
    assert series[-1].volume_sma == Decimal("10.5")
    assert series[-1].volume_ratio == Decimal("20") / Decimal("10.5")
