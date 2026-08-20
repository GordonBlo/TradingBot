from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.backtest.models import (
    BacktestConfig,
    EquityPoint,
    ExitReason,
    Trade,
)
from src.diagnostics.risk_capital_audit import (
    build_risk_capital_audit,
    summarize_risk_capital_audit,
)
from src.research.evaluation import SignalRecord
from src.strategy.models import (
    DecisionReason,
    StrategyAction,
    TrendMomentumConfig,
)


UTC = timezone.utc


def dt(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 1, 1, hour, minute, tzinfo=UTC)


def make_signal(*, atr: str) -> SignalRecord:
    return SignalRecord(
        timestamp=dt(0, 15),
        action=StrategyAction.ENTER_LONG,
        close=Decimal("100"),
        ema_fast=Decimal("101"),
        ema_slow=Decimal("100"),
        rsi=Decimal("60"),
        atr=Decimal(atr),
        volume_ratio=Decimal("1.20"),
        position_state="FLAT",
        reason_code=DecisionReason.EMA_CROSSOVER_ENTRY,
        reason="test entry",
    )


def make_trade(
    *,
    quantity: str,
    entry_price: str = "100",
    exit_price: str = "110",
    entry_fee: str = "0",
    exit_fee: str = "0",
) -> Trade:
    qty = Decimal(quantity)
    entry = Decimal(entry_price)
    exit_ = Decimal(exit_price)
    entry_fee_d = Decimal(entry_fee)
    exit_fee_d = Decimal(exit_fee)

    entry_notional = qty * entry
    exit_notional = qty * exit_
    gross_pnl = (exit_ - entry) * qty
    total_fee = entry_fee_d + exit_fee_d
    net_pnl = gross_pnl - total_fee

    return Trade(
        trade_id="T1",
        entry_signal_time=dt(0, 15),
        entry_time=dt(0, 15),
        entry_price=entry,
        exit_signal_time=dt(1, 0),
        exit_time=dt(1, 15),
        exit_price=exit_,
        quantity=qty,
        entry_notional=entry_notional,
        exit_notional=exit_notional,
        entry_fee=entry_fee_d,
        exit_fee=exit_fee_d,
        total_fee=total_fee,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        return_percent=net_pnl / entry_notional * Decimal("100"),
        bars_held=4,
        exit_reason=ExitReason.TAKE_PROFIT,
    )


def make_equity(*, equity: str, cash: str | None = None) -> EquityPoint:
    equity_d = Decimal(equity)

    return EquityPoint(
        timestamp=dt(0, 15),
        cash=Decimal(cash) if cash is not None else equity_d,
        position_value=Decimal("0"),
        total_equity=equity_d,
    )


def test_position_cap_limits_actual_risk() -> None:
    """
    1000 USDC equity, configured 0.5% risk = 5 USDC risk budget.

    ATR=4 and 2x ATR stop => stop distance=8.
    At entry price=100, risk-sized notional would be 62.5 USDC.

    The configured 50 USDC position cap therefore limits the trade.
    """

    records = build_risk_capital_audit(
        trades=(make_trade(quantity="0.5"),),
        signals=(make_signal(atr="4"),),
        equity_curve=(make_equity(equity="1000"),),
        backtest_config=BacktestConfig(
            initial_capital_usdc=Decimal("1000"),
            fee_bps=Decimal("0"),
            slippage_bps=Decimal("0"),
        ),
        strategy_config=TrendMomentumConfig(
            maximum_position_notional_usdc=Decimal("50"),
        ),
    )

    record = records[0]

    assert record.requested_risk_budget == Decimal("5")
    assert record.stop_distance == Decimal("8")
    assert record.risk_sized_notional == Decimal("62.5")

    assert record.actual_entry_notional == Decimal("50")
    assert record.actual_stop_risk == Decimal("4")

    assert record.actual_risk_percent_of_equity == Decimal("0.4")
    assert record.risk_budget_utilization_percent == Decimal("80")
    assert record.capital_utilization_percent == Decimal("5")

    assert record.limited_by_position_cap is True
    assert record.limited_by_risk_budget is False
    assert record.limited_by_available_cash is False


def test_risk_budget_can_be_the_active_limit() -> None:
    """
    Risk budget = 5 USDC.
    Stop distance = 10.
    Entry price = 100.

    Required notional = 50 USDC, while position cap is 500.
    Therefore risk sizing itself is the limiting factor.
    """

    records = build_risk_capital_audit(
        trades=(make_trade(quantity="0.5"),),
        signals=(make_signal(atr="5"),),
        equity_curve=(make_equity(equity="1000"),),
        backtest_config=BacktestConfig(
            fee_bps=Decimal("0"),
            slippage_bps=Decimal("0"),
        ),
        strategy_config=TrendMomentumConfig(
            maximum_position_notional_usdc=Decimal("500"),
        ),
    )

    record = records[0]

    assert record.requested_risk_budget == Decimal("5")
    assert record.stop_distance == Decimal("10")
    assert record.risk_sized_notional == Decimal("50")

    assert record.actual_stop_risk == Decimal("5")
    assert record.actual_risk_percent_of_equity == Decimal("0.5")
    assert record.risk_budget_utilization_percent == Decimal("100")

    assert record.limited_by_risk_budget is True
    assert record.limited_by_position_cap is False
    assert record.limited_by_available_cash is False


def test_available_cash_can_be_the_active_limit() -> None:
    """
    Artificial small-account fixture used to verify the third limiter.

    Equity/cash = 10 USDC.
    Requested position is much larger than available cash.
    """

    records = build_risk_capital_audit(
        trades=(make_trade(quantity="0.1"),),
        signals=(make_signal(atr="0.5"),),
        equity_curve=(make_equity(equity="10"),),
        backtest_config=BacktestConfig(
            initial_capital_usdc=Decimal("10"),
            fee_bps=Decimal("0"),
            slippage_bps=Decimal("0"),
        ),
        strategy_config=TrendMomentumConfig(
            risk_per_trade_percent=Decimal("50"),
            maximum_position_notional_usdc=Decimal("1000"),
        ),
    )

    record = records[0]

    assert record.requested_risk_budget == Decimal("5")
    assert record.stop_distance == Decimal("1")
    assert record.risk_sized_notional == Decimal("500")

    assert record.actual_entry_notional == Decimal("10")
    assert record.actual_stop_risk == Decimal("0.1")

    assert record.limited_by_available_cash is True
    assert record.limited_by_position_cap is False
    assert record.limited_by_risk_budget is False


def test_summary_exposes_position_cap_frequency_and_actual_risk() -> None:
    records = build_risk_capital_audit(
        trades=(make_trade(quantity="0.5"),),
        signals=(make_signal(atr="4"),),
        equity_curve=(make_equity(equity="1000"),),
        backtest_config=BacktestConfig(
            fee_bps=Decimal("0"),
            slippage_bps=Decimal("0"),
        ),
        strategy_config=TrendMomentumConfig(
            maximum_position_notional_usdc=Decimal("50"),
        ),
    )

    summary = summarize_risk_capital_audit(records)

    assert summary.trade_count == 1
    assert summary.position_cap_limited_count == 1
    assert summary.position_cap_limited_percent == Decimal("100")

    assert summary.average_requested_risk_percent == Decimal("0.50")
    assert summary.average_actual_risk_percent == Decimal("0.4")
    assert summary.maximum_actual_risk_percent == Decimal("0.4")

    assert summary.average_risk_budget_utilization_percent == Decimal("80")
    assert summary.average_capital_utilization_percent == Decimal("5")


def test_missing_entry_signal_is_rejected() -> None:
    with pytest.raises(ValueError, match="Entry signal not found"):
        build_risk_capital_audit(
            trades=(make_trade(quantity="0.5"),),
            signals=(),
            equity_curve=(make_equity(equity="1000"),),
            backtest_config=BacktestConfig(
                fee_bps=Decimal("0"),
                slippage_bps=Decimal("0"),
            ),
            strategy_config=TrendMomentumConfig(),
        )