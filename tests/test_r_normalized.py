from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.backtest.models import ExitReason, Trade
from src.diagnostics.r_normalized import (
    build_r_normalized_trades,
    summarize_r_normalized_trades,
)


UTC = timezone.utc


def make_trade(
    trade_id: str,
    *,
    entry: str,
    exit_: str,
    fee: str,
) -> Trade:
    entry_price = Decimal(entry)
    exit_price = Decimal(exit_)
    quantity = Decimal("1")
    total_fee = Decimal(fee)

    gross = (
        exit_price - entry_price
    ) * quantity

    net = gross - total_fee

    return Trade(
        trade_id=trade_id,
        entry_signal_time=datetime(
            2026, 1, 1, 0, 0, tzinfo=UTC
        ),
        entry_time=datetime(
            2026, 1, 1, 0, 15, tzinfo=UTC
        ),
        entry_price=entry_price,
        exit_signal_time=datetime(
            2026, 1, 1, 1, 0, tzinfo=UTC
        ),
        exit_time=datetime(
            2026, 1, 1, 1, 15, tzinfo=UTC
        ),
        exit_price=exit_price,
        quantity=quantity,
        entry_notional=entry_price,
        exit_notional=exit_price,
        entry_fee=total_fee / Decimal("2"),
        exit_fee=total_fee / Decimal("2"),
        total_fee=total_fee,
        gross_pnl=gross,
        net_pnl=net,
        return_percent=(
            net / entry_price * Decimal("100")
        ),
        bars_held=4,
        exit_reason=ExitReason.TAKE_PROFIT,
    )


def test_r_normalization_without_slippage() -> None:
    trade = make_trade(
        "T1",
        entry="100",
        exit_="110",
        fee="2",
    )

    records = build_r_normalized_trades(
        trades=(trade,),
        actual_stop_risk_by_trade_id={
            "T1": Decimal("5")
        },
        slippage_bps=Decimal("0"),
    )

    record = records[0]

    assert record.frictionless_r == Decimal("2")
    assert record.gross_after_slippage_r == Decimal("2")
    assert record.fee_cost_r == Decimal("0.4")
    assert record.slippage_cost_r == Decimal("0")
    assert record.total_friction_r == Decimal("0.4")
    assert record.net_r == Decimal("1.6")


def test_summary_expectancy_and_break_even_rate() -> None:
    winner = make_trade(
        "WIN",
        entry="100",
        exit_="110",
        fee="2",
    )

    loser = make_trade(
        "LOSS",
        entry="100",
        exit_="95",
        fee="1",
    )

    records = build_r_normalized_trades(
        trades=(winner, loser),
        actual_stop_risk_by_trade_id={
            "WIN": Decimal("5"),
            "LOSS": Decimal("5"),
        },
        slippage_bps=Decimal("0"),
    )

    summary = summarize_r_normalized_trades(records)

    assert summary.trade_count == 2
    assert summary.winning_trades == 1
    assert summary.losing_trades == 1

    assert summary.win_rate_percent == Decimal("50")

    assert summary.average_winner_r == Decimal("1.6")
    assert summary.average_loser_r == Decimal("-1.2")

    assert summary.net_expectancy_r == Decimal("0.2")

    assert summary.payoff_ratio_r == (
        Decimal("1.6") / Decimal("1.2")
    )

    expected_break_even = (
        Decimal("1.2")
        / Decimal("2.8")
        * Decimal("100")
    )

    assert (
        summary.break_even_win_rate_percent
        == expected_break_even
    )


def test_slippage_is_adverse_cost_in_r() -> None:
    trade = make_trade(
        "T1",
        entry="100",
        exit_="110",
        fee="0",
    )

    records = build_r_normalized_trades(
        trades=(trade,),
        actual_stop_risk_by_trade_id={
            "T1": Decimal("5")
        },
        slippage_bps=Decimal("10"),
    )

    record = records[0]

    assert record.slippage_cost_r > 0
    assert record.frictionless_r > record.gross_after_slippage_r
    assert record.total_friction_r == record.slippage_cost_r


def test_missing_risk_is_rejected() -> None:
    trade = make_trade(
        "T1",
        entry="100",
        exit_="110",
        fee="0",
    )

    with pytest.raises(
        ValueError,
        match="Missing initial stop-risk",
    ):
        build_r_normalized_trades(
            trades=(trade,),
            actual_stop_risk_by_trade_id={},
            slippage_bps=Decimal("0"),
        )


def test_zero_risk_is_rejected() -> None:
    trade = make_trade(
        "T1",
        entry="100",
        exit_="110",
        fee="0",
    )

    with pytest.raises(
        ValueError,
        match="must be positive",
    ):
        build_r_normalized_trades(
            trades=(trade,),
            actual_stop_risk_by_trade_id={
                "T1": Decimal("0")
            },
            slippage_bps=Decimal("0"),
        )