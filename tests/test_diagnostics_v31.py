from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import inspect
from pathlib import Path
from tempfile import TemporaryDirectory
from collections.abc import Iterator

import pytest

from src.backtest.models import BacktestConfig, ExitReason, Trade
from src.diagnostics.analyzer import StrategyDiagnosticsAnalyzer
from src.diagnostics.cost_analysis import decompose_trade_costs
from src.diagnostics.distributions import (
    holding_buckets,
    outcome_diagnostics,
)
from src.diagnostics.entry_analysis import feature_bucket_diagnostics
from src.diagnostics.excursions import calculate_excursions
from src.diagnostics.exit_analysis import exit_reason_diagnostics
from src.diagnostics.models import (
    DiagnosticPeriodInput,
    DiagnosticRunInput,
    EvidenceStrength,
    MarketRegime,
    TradeDiagnostic,
    VolatilityRegime,
)
from src.diagnostics.regime_analysis import (
    utc_day_diagnostics,
    utc_hour_diagnostics,
)
from src.diagnostics.report import DiagnosticsReportWriter
from src.historical.dataset import HistoricalDataset
from src.models.candle import Candle
from src.models.market_data_source import MarketDataSource
from src.research.dataset_split import ResearchPeriod
from src.research.evaluation import SignalRecord
from src.strategy.context import StrategyContext
from src.strategy.baseline_trend import TrendMomentumBaselineStrategy
from src.strategy.models import (
    DecisionReason,
    StrategyAction,
    StrategyDecision,
    TrendMomentumConfig,
)


BASE = datetime(2026, 3, 2, tzinfo=timezone.utc)  # Monday


@pytest.fixture
def diagnostics_tmp_path() -> Iterator[Path]:
    root = Path(__file__).resolve().parents[1] / ".test_artifacts" / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="diagnostics-", dir=root) as directory:
        yield Path(directory)


def candle(index: int, *, high: str, low: str, close: str = "100") -> Candle:
    close_value = Decimal(close)
    return Candle(
        timestamp=BASE + timedelta(minutes=15 * index),
        symbol="BTCUSDC",
        interval="15m",
        open=close_value,
        high=Decimal(high),
        low=Decimal(low),
        close=close_value,
        volume=Decimal("10"),
        is_closed=True,
    )


def dataset(*candles: Candle) -> HistoricalDataset:
    return HistoricalDataset(
        "BTCUSDC", "15m", MarketDataSource.BINANCE_PUBLIC, tuple(candles)
    )


def recorded_trade(
    *,
    entry: str = "100",
    exit: str = "110",
    fee: str = "0",
    bars: int = 3,
    reason: ExitReason = ExitReason.TAKE_PROFIT,
    entry_time: datetime = BASE,
) -> Trade:
    entry_price = Decimal(entry)
    exit_price = Decimal(exit)
    quantity = Decimal("1")
    total_fee = Decimal(fee)
    gross = exit_price - entry_price
    net = gross - total_fee
    return Trade(
        trade_id="T000001",
        entry_signal_time=entry_time,
        entry_time=entry_time,
        entry_price=entry_price,
        exit_signal_time=entry_time + timedelta(minutes=15 * bars),
        exit_time=entry_time + timedelta(minutes=15 * bars),
        exit_price=exit_price,
        quantity=quantity,
        entry_notional=entry_price,
        exit_notional=exit_price,
        entry_fee=total_fee / 2,
        exit_fee=total_fee / 2,
        total_fee=total_fee,
        gross_pnl=gross,
        net_pnl=net,
        return_percent=net / entry_price * Decimal("100"),
        bars_held=bars,
        exit_reason=reason,
    )


def diagnostic(
    *,
    net: str = "1",
    reason: ExitReason = ExitReason.TAKE_PROFIT,
    bars: int = 4,
    rsi: str = "58",
    hour: int = 0,
    trade_id: str = "T1",
) -> TradeDiagnostic:
    net_value = Decimal(net)
    return TradeDiagnostic(
        trade_id=trade_id,
        entry_signal_time=BASE.replace(hour=hour),
        entry_time=BASE.replace(hour=hour),
        exit_time=BASE.replace(hour=hour) + timedelta(minutes=bars * 15),
        entry_price=Decimal("100"),
        exit_price=Decimal("100") + net_value,
        reference_entry_price=Decimal("100"),
        reference_exit_price=Decimal("100") + net_value,
        quantity=Decimal("1"),
        exit_reason=reason,
        frictionless_pnl=net_value,
        gross_pnl=net_value,
        slippage_drag=Decimal("0"),
        fee_cost=Decimal("0"),
        net_pnl=net_value,
        return_percent=net_value,
        initial_stop_price=Decimal("95"),
        initial_risk_per_unit=Decimal("5"),
        initial_risk_amount=Decimal("5"),
        r_multiple=net_value / Decimal("5"),
        bars_held=bars,
        holding_minutes=bars * 15,
        highest_observed_price=Decimal("105"),
        lowest_observed_price=Decimal("97"),
        mfe=Decimal("5"),
        mae=Decimal("3"),
        mfe_percent=Decimal("5"),
        mae_percent=Decimal("3"),
        mfe_r=Decimal("1"),
        mae_r=Decimal("0.6"),
        bars_to_mfe=2,
        bars_to_mae=1,
        first_four_bar_mae_r=Decimal("0.6"),
        capture_ratio=Decimal("0.2") if net_value > 0 else None,
        loss_efficiency=Decimal("0.3") if net_value < 0 else None,
        entry_close=Decimal("100"),
        entry_rsi=Decimal(rsi),
        entry_atr=Decimal("2"),
        entry_atr_percent=Decimal("2"),
        entry_volume_ratio=Decimal("1.1"),
        entry_ema_fast=Decimal("101"),
        entry_ema_slow=Decimal("100"),
        entry_ema_spread_percent=Decimal("1"),
        close_vs_slow_ema_percent=Decimal("0"),
        market_regime=MarketRegime.TREND_UP,
        volatility_regime=VolatilityRegime.MEDIUM,
        entry_utc_hour=hour,
        entry_day_of_week="Monday",
        diagnostic_status="COMPLETE",
    )


def test_mfe_mae_and_r_normalization_are_exact() -> None:
    data = dataset(
        candle(0, high="102", low="99"),
        candle(1, high="105", low="97"),
        candle(2, high="103", low="98"),
    )
    trade = recorded_trade()
    result = calculate_excursions(
        trade,
        data,
        {item.timestamp: index for index, item in enumerate(data.candles)},
        initial_risk_per_unit=Decimal("5"),
    )

    assert result.mfe == Decimal("5")
    assert result.mae == Decimal("3")
    assert result.mfe_r == Decimal("1")
    assert result.mae_r == Decimal("0.6")
    assert (result.bars_to_mfe, result.bars_to_mae) == (2, 2)


def test_r_normalization_known_two_r_and_half_r() -> None:
    data = dataset(candle(0, high="110", low="97.5"))
    result = calculate_excursions(
        recorded_trade(bars=1),
        data,
        {BASE: 0},
        initial_risk_per_unit=Decimal("5"),
    )
    assert result.mfe_r == Decimal("2")
    assert result.mae_r == Decimal("0.5")


def test_post_trade_excursions_do_not_exist_in_strategy_context() -> None:
    post_trade = {
        "mfe",
        "mae",
        "bars_to_mfe",
        "bars_to_mae",
    }
    assert post_trade.isdisjoint(StrategyContext.__dataclass_fields__)
    assert post_trade.issubset(TradeDiagnostic.__dataclass_fields__)
    assert {"future_highs", "future_lows"}.isdisjoint(
        StrategyContext.__dataclass_fields__
    )
    assert post_trade.isdisjoint(StrategyDecision.__dataclass_fields__)
    strategy_source = inspect.getsource(TrendMomentumBaselineStrategy).lower()
    assert "mfe" not in strategy_source
    assert "mae" not in strategy_source


def test_cost_decomposition_recovers_reference_prices_and_drags() -> None:
    trade = recorded_trade(entry="100.10", exit="109.89", fee="0.21", bars=1)
    costs = decompose_trade_costs(trade, Decimal("10"))

    assert costs.reference_entry_price == Decimal("100")
    assert costs.reference_exit_price == Decimal("110")
    assert costs.frictionless_pnl == Decimal("10")
    assert costs.slippage_adjusted_gross_pnl == Decimal("9.79")
    assert costs.slippage_drag == Decimal("-0.21")
    assert costs.fee_drag == Decimal("0.21")
    assert costs.net_pnl == Decimal("9.58")


def test_exit_reason_grouping_counts_percentages_and_pnl() -> None:
    trades = (
        diagnostic(net="-5", reason=ExitReason.STOP_LOSS, trade_id="T1"),
        diagnostic(net="10", reason=ExitReason.TAKE_PROFIT, trade_id="T2"),
        diagnostic(net="2", reason=ExitReason.TREND_EXIT, trade_id="T3"),
        diagnostic(net="-1", reason=ExitReason.TIME_EXIT, trade_id="T4"),
    )
    rows = {row.exit_reason: row for row in exit_reason_diagnostics(trades, 30)}

    assert set(rows) == {
        ExitReason.STOP_LOSS,
        ExitReason.TAKE_PROFIT,
        ExitReason.TREND_EXIT,
        ExitReason.TIME_EXIT,
    }
    assert rows[ExitReason.STOP_LOSS].trade_count == 1
    assert rows[ExitReason.STOP_LOSS].percentage_of_trades == Decimal("25")
    assert rows[ExitReason.TAKE_PROFIT].net_pnl == Decimal("10")


@pytest.mark.parametrize(
    ("bars", "bucket"),
    ((1, "1-4"), (4, "1-4"), (5, "5-16"), (17, "17-32"), (33, "33-64"), (65, "65-96"), (97, "97+")),
)
def test_holding_bucket_boundaries(bars: int, bucket: str) -> None:
    rows = {row.group: row for row in holding_buckets((diagnostic(bars=bars),), 30)}
    assert rows[bucket].trade_count == 1


def test_entry_feature_bin_boundary_is_nonoverlapping_and_warned() -> None:
    rows = feature_bucket_diagnostics((diagnostic(rsi="55"),), 30)
    rsi_rows = {row.group: row for row in rows if row.dimension == "entry_rsi"}

    assert rsi_rows["52-<55"].trade_count == 0
    assert rsi_rows["55-<60"].trade_count == 1
    assert rsi_rows["55-<60"].evidence_strength is EvidenceStrength.LOW_SAMPLE_SIZE


def test_utc_hour_and_day_grouping_use_aware_utc_timestamp() -> None:
    trade = diagnostic(hour=23)
    hours = {row.group: row for row in utc_hour_diagnostics((trade,), 30)}
    days = {row.group: row for row in utc_day_diagnostics((trade,), 30)}

    assert hours["23"].trade_count == 1
    assert days["Monday"].trade_count == 1


def test_expectancy_matches_outcome_formula() -> None:
    trades = tuple(
        diagnostic(net=value, trade_id=f"T{index}")
        for index, value in enumerate(("10", "20", "-5", "-5"))
    )
    result = outcome_diagnostics(trades)
    assert result.expectancy_from_outcomes == Decimal("5")
    assert result.average_trade_pnl == Decimal("5")


def _period_input(
    period: ResearchPeriod, start: datetime, exit_price: str
) -> DiagnosticPeriodInput:
    first = Candle(
        start,
        "BTCUSDC",
        "15m",
        Decimal("99"),
        Decimal("101"),
        Decimal("98"),
        Decimal("100"),
        Decimal("10"),
        True,
    )
    second = Candle(
        start + timedelta(minutes=15),
        "BTCUSDC",
        "15m",
        Decimal("100"),
        max(Decimal(exit_price), Decimal("100")) + 1,
        min(Decimal(exit_price), Decimal("100")) - 1,
        Decimal(exit_price),
        Decimal("10"),
        True,
    )
    trade = recorded_trade(
        exit=exit_price,
        bars=1,
        reason=ExitReason.END_OF_BACKTEST,
        entry_time=second.timestamp,
    )
    signal = SignalRecord(
        timestamp=second.timestamp,
        action=StrategyAction.ENTER_LONG,
        close=Decimal("100"),
        ema_fast=Decimal("101"),
        ema_slow=Decimal("100"),
        rsi=Decimal("57"),
        atr=Decimal("2.5"),
        volume_ratio=Decimal("1.2"),
        position_state="FLAT",
        reason_code=DecisionReason.EMA_CROSSOVER_ENTRY,
        reason="fixture entry",
    )
    return DiagnosticPeriodInput(period, dataset(first, second), (trade,), (signal,))


def diagnostic_run_input() -> DiagnosticRunInput:
    periods = (
        _period_input(ResearchPeriod.DEVELOPMENT, BASE, "110"),
        _period_input(ResearchPeriod.VALIDATION, BASE + timedelta(days=1), "95"),
        _period_input(ResearchPeriod.OUT_OF_SAMPLE, BASE + timedelta(days=2), "102"),
    )
    return DiagnosticRunInput(
        research_run_id="fixture-research",
        symbol="BTCUSDC",
        interval="15m",
        backtest_config=BacktestConfig(fee_bps=Decimal("0"), slippage_bps=Decimal("0")),
        strategy_config=TrendMomentumConfig(),
        minimum_trades_warning=30,
        periods=periods,
    )


def test_split_diagnostics_remain_isolated_and_use_signal_time_features() -> None:
    report = StrategyDiagnosticsAnalyzer().analyze(diagnostic_run_input())

    assert [period.costs.net_pnl for period in report.periods] == [
        Decimal("10"),
        Decimal("-5"),
        Decimal("2"),
    ]
    assert all(period.total_trades == 1 for period in report.periods)
    assert report.periods[0].trades[0].entry_rsi == Decimal("57")
    assert report.periods[0].trades[0].entry_atr == Decimal("2.5")
    assert report.combined.total_trades == 3


def test_diagnostic_reports_are_deterministic(
    diagnostics_tmp_path: Path,
) -> None:
    report = StrategyDiagnosticsAnalyzer().analyze(diagnostic_run_input())
    writer = DiagnosticsReportWriter(diagnostics_tmp_path)

    first = writer.write(report)
    first_contents = {
        path.relative_to(first.directory): path.read_bytes()
        for path in first.directory.rglob("*")
        if path.is_file()
    }
    second = writer.write(report)

    assert first.directory == second.directory
    assert first_contents == {
        path.relative_to(second.directory): path.read_bytes()
        for path in second.directory.rglob("*")
        if path.is_file()
    }
    assert (first.directory / "out_of_sample" / "trade_diagnostics.csv").is_file()
