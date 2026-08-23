from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.backtest.models import ExitReason, Trade
from src.cli.run_v6_h0 import validate_v6_h0_implementation
from src.cli.run_v6_signal_quality import (
    verify_frozen_reproduction,
    write_reports,
)
from src.diagnostics.r_normalized import RNormalizedTrade
from src.diagnostics.v6_signal_quality import (
    V6TradeLedger,
    build_trade_ledger,
    calculate_threshold_path,
    summarize_feature_comparison,
)
from src.historical.dataset import HistoricalDataset
from src.models.candle import Candle
from src.models.market_data_source import MarketDataSource
from src.research.v6_mtf_continuation_preregistration import build_manifest
from src.strategy.v6_mtf_continuation import prepare_v6_context


BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def candle(index: int, close: str, *, high: str | None = None, low: str | None = None):
    value = Decimal(close)
    return Candle(
        timestamp=BASE + timedelta(minutes=15 * index),
        symbol="BTCUSDC",
        interval="15m",
        open=value,
        high=Decimal(high) if high is not None else value,
        low=Decimal(low) if low is not None else value,
        close=value,
        volume=Decimal("1"),
        is_closed=True,
    )


def trade(
    *, bars: int = 1, entry_index: int = 2, entry_price: str = "100"
) -> Trade:
    price = Decimal(entry_price)
    return Trade(
        trade_id="T000001",
        entry_signal_time=BASE + timedelta(minutes=15 * entry_index),
        entry_time=BASE + timedelta(minutes=15 * entry_index),
        entry_price=price,
        exit_signal_time=BASE + timedelta(minutes=15 * (entry_index + bars - 1)),
        exit_time=BASE + timedelta(minutes=15 * (entry_index + bars - 1)),
        exit_price=Decimal("102"),
        quantity=Decimal("1"),
        entry_notional=price,
        exit_notional=Decimal("102"),
        entry_fee=Decimal("0.1"),
        exit_fee=Decimal("0.1"),
        total_fee=Decimal("0.2"),
        gross_pnl=Decimal("2"),
        net_pnl=Decimal("1.8"),
        return_percent=Decimal("1.8"),
        bars_held=bars,
        exit_reason=ExitReason.TAKE_PROFIT,
    )


def dataset(held: tuple[Candle, ...]) -> HistoricalDataset:
    return HistoricalDataset(
        symbol="BTCUSDC",
        interval="15m",
        source=MarketDataSource.BINANCE_PUBLIC,
        candles=(candle(0, "100"), candle(1, "100"), *held),
    )


def ledger(net: str, feature: str) -> V6TradeLedger:
    values = {
        "window_id": "W001",
        "trade_id": f"T{net}{feature}",
        "entry_signal_time": BASE,
        "entry_time": BASE,
        "exit_time": BASE,
        "exit_reason": ExitReason.TAKE_PROFIT,
        "bars_held": 2,
        "frictionless_r": Decimal(net) + Decimal("0.1"),
        "net_r": Decimal(net),
        "fee_r": Decimal("0.08"),
        "slippage_r": Decimal("0.02"),
        "total_friction_r": Decimal("0.1"),
        "entry_fill": Decimal("100"),
        "initial_stop": Decimal("98"),
        "take_profit": Decimal("104"),
        "initial_stop_distance": Decimal("2"),
        "initial_stop_distance_bps": Decimal("200"),
        "stop_source": "ATR4H",
        "mfe_r": Decimal("2"),
        "mae_r": Decimal("0.5"),
        "first_four_bar_mae_r": Decimal("0.5"),
        "positive_025_bar": 1,
        "positive_050_bar": 1,
        "positive_100_bar": 2,
        "positive_150_bar": 2,
        "positive_200_bar": 2,
        "negative_025_bar": 1,
        "negative_050_bar": 1,
        "negative_075_bar": None,
        "negative_100_bar": None,
        "negative_075_before_positive_050": "FALSE",
        "same_bar_order_ambiguity": True,
        "ema20_4h": Decimal("110"),
        "ema50_4h": Decimal("100"),
        "ema_spread_4h": Decimal(feature),
        "ema_spread_percent_4h": Decimal(feature),
        "completed_close_4h": Decimal("120"),
        "close_distance_above_ema20_4h": Decimal("10"),
        "atr14_4h": Decimal("2"),
        "atr_percent_4h": Decimal("1"),
        "previous_close_distance_from_ema20_15m": Decimal("-1"),
        "current_close_distance_above_ema20_15m": Decimal("1"),
        "continuation_above_previous_high": Decimal("1"),
        "signal_body_size": Decimal("1"),
        "signal_range": Decimal("2"),
        "signal_body_range_ratio": Decimal("0.5"),
        "lower_wick_size": Decimal("0.5"),
        "upper_wick_size": Decimal("0.5"),
        "base_friction_r": Decimal("0.1"),
    }
    return V6TradeLedger(**values)


def test_mfe_mae_thresholds_and_same_bar_ambiguity() -> None:
    data = dataset((candle(2, "100", high="104", low="98"),))
    path = calculate_threshold_path(
        trade=trade(), dataset=data, initial_risk_per_unit=Decimal("2")
    )

    assert path.positive_bars["2.00"] == 1
    assert path.negative_bars["1.00"] == 1
    assert path.negative_075_before_positive_050 == "AMBIGUOUS"
    assert path.same_bar_order_ambiguity is True


def test_threshold_ordering_requires_strictly_different_bars() -> None:
    data = dataset(
        (
            candle(2, "100", high="100.2", low="98.4"),
            candle(3, "100", high="101.2", low="99"),
        )
    )
    path = calculate_threshold_path(
        trade=trade(bars=2), dataset=data, initial_risk_per_unit=Decimal("2")
    )
    assert path.negative_075_before_positive_050 == "TRUE"


def test_winner_loser_feature_summary_and_effect_size() -> None:
    rows = summarize_feature_comparison(
        (ledger("1", "4"), ledger("2", "6"), ledger("-1", "1"), ledger("-2", "2"))
    )
    spread = next(row for row in rows if row["feature"] == "ema_spread_4h")
    assert spread["winner_mean"] == Decimal("5")
    assert spread["loser_mean"] == Decimal("1.5")
    assert spread["mean_difference"] == Decimal("3.5")
    assert spread["standardized_effect_size"] is not None


def test_entry_features_ignore_future_candle_changes() -> None:
    history = tuple(
        candle(index, str(Decimal("100") + group), high=str(Decimal("101") + group), low=str(Decimal("99") + group))
        for group in range(50)
        for index in range(group * 16, group * 16 + 16)
    )
    previous = candle(800, "140", high="141", low="139")
    signal = candle(801, "160", high="160", low="159")
    entry = candle(802, "200", high="204", low="198")
    future_a = candle(803, "200", high="201", low="199")
    future_b = candle(803, "900", high="999", low="1")

    def build(future):
        replay = HistoricalDataset(
            "BTCUSDC",
            "15m",
            MarketDataSource.BINANCE_PUBLIC,
            (*history, previous, signal, entry, future),
        )
        prepared = prepare_v6_context(replay.candles)
        r_record = RNormalizedTrade(
            "T000001",
            Decimal("2"),
            Decimal("1"),
            Decimal("0.95"),
            Decimal("0.04"),
            Decimal("0.01"),
            Decimal("0.05"),
            Decimal("0.95"),
        )
        observation = SimpleNamespace(
            trade_id="T000001",
            initial_stop_distance=Decimal("2"),
            initial_stop_distance_bps=Decimal("100"),
            stop_source="ATR14_4H",
        )
        return build_trade_ledger(
            window_id="W001",
            trades=(trade(entry_index=802, entry_price="200"),),
            r_records=(r_record,),
            stop_observations=(observation,),
            replay_dataset=replay,
            evaluation_dataset=replay,
            prepared_context=prepared,
        )[0]

    first = build(future_a)
    second = build(future_b)
    for field in (
        "ema20_4h",
        "ema50_4h",
        "atr14_4h",
        "current_close_distance_above_ema20_15m",
        "continuation_above_previous_high",
        "signal_body_size",
    ):
        assert getattr(first, field) == getattr(second, field)


def test_frozen_reproduction_guard_and_holdout_integrity() -> None:
    summary = SimpleNamespace(
        trade_count=443,
        frictionless_expectancy_r=Decimal("0.04722009245553381286395292467"),
        net_expectancy_r=Decimal("-0.1366917232474779417819851766"),
        profit_factor_r=Decimal("0.7695960517596068371318908235"),
    )
    verify_frozen_reproduction(summary=summary, positive_windows=1)
    mismatch = SimpleNamespace(**vars(summary))
    mismatch.trade_count = 442
    with pytest.raises(ValueError, match="trade count"):
        verify_frozen_reproduction(summary=mismatch, positive_windows=1)

    manifest = deepcopy(build_manifest())
    manifest["dataset_policy"]["blind_holdout"]["loaded"] = True
    with pytest.raises(ValueError, match="holdout integrity"):
        validate_v6_h0_implementation(manifest)


def test_report_serialization(tmp_path) -> None:
    output = tmp_path / "e1eef7bdd0c37ad4"
    paths = write_reports(
        output=output,
        summary_payload={"source_run_id": "e1eef7bdd0c37ad4"},
        ledger=[{"trade_id": "T1"}],
        exit_rows=[{"exit_reason": "TAKE_PROFIT"}],
        feature_rows=[{"feature": "ema_spread_4h"}],
        window_rows=[{"window_id": "W001"}],
    )
    assert all(path.is_file() for path in paths)
    with pytest.raises(ValueError, match="exists"):
        write_reports(
            output=output,
            summary_payload={},
            ledger=[{"trade_id": "T1"}],
            exit_rows=[{"exit_reason": "TAKE_PROFIT"}],
            feature_rows=[{"feature": "x"}],
            window_rows=[{"window_id": "W001"}],
        )
