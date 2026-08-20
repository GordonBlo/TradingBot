from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from collections.abc import Iterator

import pytest

from src.backtest.benchmark import BuyAndHoldBenchmark
from src.backtest.account import InsufficientBalanceError, InvalidOrderIntentError
from src.backtest.engine import BacktestEngine
from src.backtest.execution import SimulatedExecutionModel, basis_points_rate
from src.backtest.metrics import calculate_metrics, maximum_drawdown_percent
from src.backtest.models import (
    AmbiguousBarPolicy,
    BacktestConfig,
    BacktestContext,
    EquityPoint,
    ExitReason,
    OrderAction,
    OrderIntent,
    Trade,
)
from src.backtest.report import BacktestReportWriter
from src.historical.dataset import HistoricalDataset
from src.models.candle import Candle
from src.models.market_data_source import MarketDataSource


BASE = datetime(2024, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def report_tmp_path() -> Iterator[Path]:
    root = Path(__file__).resolve().parents[1] / ".test_artifacts" / "tmp"
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="report-", dir=root) as directory:
        yield Path(directory)


def bar(
    index: int,
    *,
    open: str,
    close: str,
    high: str | None = None,
    low: str | None = None,
) -> Candle:
    open_price = Decimal(open)
    close_price = Decimal(close)
    return Candle(
        timestamp=BASE + timedelta(minutes=15 * index),
        symbol="BTCUSDC",
        interval="15m",
        open=open_price,
        high=Decimal(high) if high is not None else max(open_price, close_price) + 1,
        low=Decimal(low) if low is not None else min(open_price, close_price) - 1,
        close=close_price,
        volume=Decimal("10"),
        is_closed=True,
    )


def dataset(*candles: Candle) -> HistoricalDataset:
    return HistoricalDataset(
        symbol="BTCUSDC",
        interval="15m",
        source=MarketDataSource.BINANCE_PUBLIC,
        candles=tuple(candles),
    )


def scripted(mapping: dict[int, OrderIntent]):
    def decide(context: BacktestContext) -> OrderIntent | None:
        return mapping.get(context.index)

    return decide


def test_basis_points_and_slippage_are_exact_and_adverse() -> None:
    execution = SimulatedExecutionModel(
        fee_bps=Decimal("10"),
        slippage_bps=Decimal("2"),
        ambiguous_bar_policy=AmbiguousBarPolicy.STOP_FIRST,
    )

    assert basis_points_rate(Decimal("100")) == Decimal("0.01")
    assert basis_points_rate(Decimal("10")) == Decimal("0.001")
    assert basis_points_rate(Decimal("1")) == Decimal("0.0001")
    assert execution.buy_fill_price(Decimal("100")) == Decimal("100.0200")
    assert execution.sell_fill_price(Decimal("100")) == Decimal("99.9800")
    assert execution.fee(Decimal("100")) == Decimal("0.100")


def test_signal_sees_no_future_and_executes_at_next_bar_open() -> None:
    data = dataset(
        bar(0, open="100", close="110"),
        bar(1, open="120", close="125"),
    )
    history_lengths: list[int] = []

    def decide(context: BacktestContext) -> OrderIntent | None:
        history_lengths.append(len(context.history))
        assert context.history[-1] == context.candle
        if context.index == 0:
            with pytest.raises(IndexError):
                _ = context.history[1]
            return OrderIntent(OrderAction.BUY, Decimal("100"), reason="test entry")
        return None

    result = BacktestEngine(
        BacktestConfig(fee_bps=Decimal("0"), slippage_bps=Decimal("0"))
    ).run(data, decide)

    assert history_lengths == [1, 2]
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.entry_signal_time == data.candles[1].timestamp
    assert trade.entry_time == data.candles[1].timestamp
    assert trade.entry_price == Decimal("120")
    assert trade.entry_price != data.candles[0].close
    assert trade.exit_reason is ExitReason.END_OF_BACKTEST


def test_final_bar_signal_has_no_fabricated_execution() -> None:
    data = dataset(bar(0, open="100", close="105"))
    result = BacktestEngine(BacktestConfig()).run(
        data,
        scripted(
            {
                0: OrderIntent(
                    OrderAction.BUY, Decimal("100"), reason="last-bar signal"
                )
            }
        ),
    )

    assert result.trades == ()
    assert result.metrics.final_equity == Decimal("1000")


def test_spot_model_rejects_shorting_and_pyramiding() -> None:
    data = dataset(
        bar(0, open="100", close="100"),
        bar(1, open="100", close="100"),
        bar(2, open="100", close="100"),
    )
    engine = BacktestEngine(BacktestConfig())

    with pytest.raises(InvalidOrderIntentError, match="without a Spot position"):
        engine.run(data, scripted({0: OrderIntent(OrderAction.SELL, reason="short")}))
    with pytest.raises(InvalidOrderIntentError, match="while a Spot position is open"):
        engine.run(
            data,
            scripted(
                {
                    0: OrderIntent(OrderAction.BUY, Decimal("100"), reason="entry"),
                    1: OrderIntent(OrderAction.BUY, Decimal("100"), reason="pyramid"),
                }
            ),
        )


def test_spot_model_rejects_borrowed_entry_capital() -> None:
    data = dataset(
        bar(0, open="100", close="100"),
        bar(1, open="100", close="100"),
    )

    with pytest.raises(InsufficientBalanceError):
        BacktestEngine(BacktestConfig()).run(
            data,
            scripted(
                {
                    0: OrderIntent(
                        OrderAction.BUY,
                        Decimal("1000"),
                        reason="fee would require borrowing",
                    )
                }
            ),
        )


def test_accounting_applies_entry_and_exit_fees_exactly() -> None:
    data = dataset(
        bar(0, open="100", close="100"),
        bar(1, open="100", close="100"),
        bar(2, open="110", close="110"),
    )
    decisions = scripted(
        {
            0: OrderIntent(OrderAction.BUY, Decimal("100"), reason="entry"),
            1: OrderIntent(OrderAction.SELL, reason="exit"),
        }
    )
    result = BacktestEngine(
        BacktestConfig(fee_bps=Decimal("10"), slippage_bps=Decimal("0"))
    ).run(data, decisions)
    trade = result.trades[0]

    assert result.equity_curve[1].cash == Decimal("899.900")
    assert result.equity_curve[1].position_value == Decimal("100")
    assert trade.quantity == Decimal("1")
    assert trade.entry_fee == Decimal("0.100")
    assert trade.exit_fee == Decimal("0.110")
    assert trade.total_fee == Decimal("0.210")
    assert trade.gross_pnl == Decimal("10")
    assert trade.net_pnl == Decimal("9.790")
    assert trade.return_percent == Decimal("9.7900")
    assert result.metrics.final_equity == Decimal("1009.790")
    assert result.metrics.total_fees_paid == Decimal("0.210")


def test_known_losing_trade_accounting() -> None:
    data = dataset(
        bar(0, open="100", close="100"),
        bar(1, open="100", close="100"),
        bar(2, open="90", close="90"),
    )
    result = BacktestEngine(
        BacktestConfig(fee_bps=Decimal("10"), slippage_bps=Decimal("0"))
    ).run(
        data,
        scripted(
            {
                0: OrderIntent(OrderAction.BUY, Decimal("100"), reason="entry"),
                1: OrderIntent(OrderAction.SELL, reason="exit"),
            }
        ),
    )
    trade = result.trades[0]

    assert trade.gross_pnl == Decimal("-10")
    assert trade.total_fee == Decimal("0.190")
    assert trade.net_pnl == Decimal("-10.190")
    assert trade.return_percent == Decimal("-10.1900")
    assert result.metrics.final_equity == Decimal("989.810")


@pytest.mark.parametrize(
    ("entry_bar", "expected_reason", "expected_exit"),
    (
        (
            bar(1, open="100", high="101", low="94", close="96"),
            ExitReason.STOP_LOSS,
            Decimal("95"),
        ),
        (
            bar(1, open="100", high="106", low="99", close="104"),
            ExitReason.TAKE_PROFIT,
            Decimal("105"),
        ),
        (
            bar(1, open="100", high="106", low="94", close="101"),
            ExitReason.STOP_LOSS,
            Decimal("95"),
        ),
    ),
)
def test_protective_exits_and_stop_first_ambiguity(
    entry_bar: Candle, expected_reason: ExitReason, expected_exit: Decimal
) -> None:
    result = BacktestEngine(
        BacktestConfig(fee_bps=Decimal("0"), slippage_bps=Decimal("0"))
    ).run(
        dataset(bar(0, open="100", close="100"), entry_bar),
        scripted(
            {
                0: OrderIntent(
                    OrderAction.BUY,
                    Decimal("100"),
                    stop_loss=Decimal("95"),
                    take_profit=Decimal("105"),
                    reason="bracket test",
                )
            }
        ),
    )

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason is expected_reason
    assert result.trades[0].exit_price == expected_exit


def test_gap_through_stop_uses_worse_open_price() -> None:
    result = BacktestEngine(
        BacktestConfig(fee_bps=Decimal("0"), slippage_bps=Decimal("0"))
    ).run(
        dataset(
            bar(0, open="100", close="100"),
            bar(1, open="100", high="102", low="96", close="100"),
            bar(2, open="90", high="92", low="85", close="88"),
        ),
        scripted(
            {
                0: OrderIntent(
                    OrderAction.BUY,
                    Decimal("100"),
                    stop_loss=Decimal("95"),
                    reason="gap stop",
                )
            }
        ),
    )

    trade = result.trades[0]
    assert trade.exit_reason is ExitReason.STOP_LOSS
    assert trade.exit_price == Decimal("90")
    assert trade.exit_price != Decimal("95")


def test_equity_curve_and_end_of_backtest_liquidation_are_exact() -> None:
    result = BacktestEngine(
        BacktestConfig(fee_bps=Decimal("0"), slippage_bps=Decimal("0"))
    ).run(
        dataset(
            bar(0, open="100", close="100"),
            bar(1, open="100", close="110"),
            bar(2, open="115", close="120"),
        ),
        scripted(
            {0: OrderIntent(OrderAction.BUY, Decimal("100"), reason="entry")}
        ),
    )

    assert [point.total_equity for point in result.equity_curve] == [
        Decimal("1000"),
        Decimal("1010"),
        Decimal("1020"),
    ]
    assert result.trades[0].exit_price == Decimal("120")
    assert result.trades[0].exit_reason is ExitReason.END_OF_BACKTEST
    assert result.equity_curve[-1].position_value == 0


def trade(index: int, net_pnl: str, fee: str = "1") -> Trade:
    pnl = Decimal(net_pnl)
    entry_notional = Decimal("100")
    entry_fee = Decimal(fee) / 2
    exit_fee = Decimal(fee) / 2
    return Trade(
        trade_id=f"T{index:06d}",
        entry_signal_time=BASE + timedelta(minutes=index * 30),
        entry_time=BASE + timedelta(minutes=index * 30 + 15),
        entry_price=Decimal("100"),
        exit_signal_time=BASE + timedelta(minutes=index * 30 + 15),
        exit_time=BASE + timedelta(minutes=index * 30 + 30),
        exit_price=Decimal("100") + pnl,
        quantity=Decimal("1"),
        entry_notional=entry_notional,
        exit_notional=entry_notional + pnl + entry_fee + exit_fee,
        entry_fee=entry_fee,
        exit_fee=exit_fee,
        total_fee=entry_fee + exit_fee,
        gross_pnl=pnl + entry_fee + exit_fee,
        net_pnl=pnl,
        return_percent=pnl,
        bars_held=1,
        exit_reason=ExitReason.SIGNAL,
    )


def test_maximum_drawdown_and_performance_metrics_known_values() -> None:
    trades = tuple(
        trade(index, pnl) for index, pnl in enumerate(("20", "10", "-10", "-5", "0"))
    )
    equity = tuple(
        EquityPoint(BASE + timedelta(minutes=index), value, Decimal("0"), value)
        for index, value in enumerate(
            map(Decimal, ("100", "120", "90", "110"))
        )
    )

    metrics = calculate_metrics(
        initial_capital=Decimal("100"),
        trades=trades,
        equity_curve=equity,
        exposed_bars=2,
        total_bars=4,
    )

    assert maximum_drawdown_percent(
        list(map(Decimal, ("100", "120", "90", "110")))
    ) == Decimal("25.00")
    assert metrics.final_equity == Decimal("110")
    assert metrics.net_profit == Decimal("10")
    assert metrics.total_return_percent == Decimal("10.0")
    assert metrics.total_trades == 5
    assert (metrics.winning_trades, metrics.losing_trades, metrics.breakeven_trades) == (
        2,
        2,
        1,
    )
    assert metrics.win_rate_percent == Decimal("40.0")
    assert metrics.gross_profit == Decimal("30")
    assert metrics.gross_loss == Decimal("-15")
    assert metrics.average_winning_trade == Decimal("15")
    assert metrics.average_losing_trade == Decimal("-7.5")
    assert metrics.largest_winning_trade == Decimal("20")
    assert metrics.largest_losing_trade == Decimal("-10")
    assert metrics.payoff_ratio == Decimal("2")
    assert metrics.profit_factor == Decimal("2")
    assert metrics.expectancy_per_trade == Decimal("3")
    assert metrics.maximum_drawdown_percent == Decimal("25.00")
    assert metrics.maximum_consecutive_wins == 2
    assert metrics.maximum_consecutive_losses == 2
    assert metrics.total_fees_paid == Decimal("5")
    assert metrics.market_exposure_percent == Decimal("50.0")


def test_zero_trade_metrics_and_no_loss_profit_factor_are_explicit() -> None:
    point = EquityPoint(BASE, Decimal("100"), Decimal("0"), Decimal("100"))
    no_trades = calculate_metrics(
        initial_capital=Decimal("100"),
        trades=(),
        equity_curve=(point,),
        exposed_bars=0,
        total_bars=1,
    )
    winner_only = calculate_metrics(
        initial_capital=Decimal("100"),
        trades=(trade(1, "10"),),
        equity_curve=(
            point,
            EquityPoint(BASE + timedelta(minutes=1), Decimal("110"), Decimal("0"), Decimal("110")),
        ),
        exposed_bars=1,
        total_bars=2,
    )

    assert no_trades.profit_factor is None
    assert no_trades.payoff_ratio is None
    assert no_trades.expectancy_per_trade == 0
    assert winner_only.profit_factor is None
    assert winner_only.payoff_ratio is None


def test_warmup_delays_decisions_without_hiding_past() -> None:
    data = dataset(
        *(bar(index, open="100", close="100") for index in range(4))
    )
    seen: list[tuple[int, int]] = []

    def decide(context: BacktestContext) -> None:
        seen.append((context.index, len(context.history)))
        return None

    BacktestEngine(BacktestConfig(warmup_candles=2)).run(data, decide)

    assert seen == [(2, 3), (3, 4)]


def test_repeatability_and_report_contents_are_identical(
    report_tmp_path: Path,
) -> None:
    data = dataset(
        bar(0, open="100", close="100"),
        bar(1, open="100", close="105"),
        bar(2, open="110", close="110"),
        bar(3, open="110", close="115"),
    )
    config = BacktestConfig(fee_bps=Decimal("10"), slippage_bps=Decimal("2"))
    mapping = {
        0: OrderIntent(OrderAction.BUY, Decimal("100"), reason="entry"),
        2: OrderIntent(OrderAction.SELL, reason="exit"),
    }

    first = BacktestEngine(config).run(data, scripted(mapping))
    second = BacktestEngine(config).run(data, scripted(mapping))
    writer = BacktestReportWriter(report_tmp_path)
    first_paths = writer.write(first)
    first_contents = {
        path.name: path.read_bytes()
        for path in (
            first_paths.trades,
            first_paths.equity,
            first_paths.summary,
            first_paths.config,
        )
    }
    second_paths = writer.write(second)

    assert first == second
    assert first_paths.directory == second_paths.directory
    assert first_contents == {
        path.name: path.read_bytes()
        for path in (
            second_paths.trades,
            second_paths.equity,
            second_paths.summary,
            second_paths.config,
        )
    }
    assert all(
        path.is_file()
        for path in (
            first_paths.trades,
            first_paths.equity,
            first_paths.summary,
            first_paths.config,
        )
    )


def test_buy_and_hold_benchmark_is_accounting_only_and_affordable() -> None:
    config = BacktestConfig()
    data = dataset(
        bar(0, open="100", close="100"),
        bar(1, open="100", close="105"),
        bar(2, open="105", close="110"),
    )

    result = BacktestEngine(config).run(
        data,
        BuyAndHoldBenchmark(config.initial_capital_usdc, config.fee_bps),
    )

    assert len(result.trades) == 1
    assert result.trades[0].exit_reason is ExitReason.END_OF_BACKTEST
