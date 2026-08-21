from decimal import Decimal

from src.backtest.models import ExitReason
from src.diagnostics.early_failure import summarize_early_failure
from src.diagnostics.h5_exit_diagnostics import H5ExitTradeRecord


def record(
    trade_id: str,
    *,
    negative_half_bar: int | None,
    positive_two_bar: int | None,
    exit_reason: ExitReason,
    net_r: str,
) -> H5ExitTradeRecord:
    return H5ExitTradeRecord(
        trade_id=trade_id,
        exit_reason=exit_reason,
        frictionless_r=Decimal(net_r),
        fee_r=Decimal("0"),
        slippage_r=Decimal("0"),
        total_friction_r=Decimal("0"),
        net_r=Decimal(net_r),
        mfe_r=Decimal("2"),
        mae_r=Decimal("1"),
        first_four_bar_mae_r=Decimal("0.5"),
        holding_bars=10,
        entry_atr_percent=None,
        entry_rsi=None,
        entry_volume_ratio=None,
        entry_ema_spread_percent=None,
        reached_positive_half_r=positive_two_bar is not None,
        reached_positive_one_r=positive_two_bar is not None,
        reached_positive_one_and_half_r=positive_two_bar is not None,
        reached_positive_two_r=positive_two_bar is not None,
        reached_negative_half_r=negative_half_bar is not None,
        reached_negative_one_r=False,
        bars_to_positive_half_r=positive_two_bar,
        bars_to_positive_one_r=positive_two_bar,
        bars_to_positive_one_and_half_r=positive_two_bar,
        bars_to_positive_two_r=positive_two_bar,
        bars_to_negative_half_r=negative_half_bar,
        bars_to_negative_one_r=None,
    )


def test_early_failure_requires_negative_half_r_within_four_bars() -> None:
    records = (
        record(
            "A",
            negative_half_bar=4,
            positive_two_bar=None,
            exit_reason=ExitReason.STOP_LOSS,
            net_r="-1",
        ),
        record(
            "B",
            negative_half_bar=5,
            positive_two_bar=None,
            exit_reason=ExitReason.STOP_LOSS,
            net_r="-1",
        ),
    )

    summary = summarize_early_failure(records)

    assert summary.trades == 1


def test_later_recovery_requires_strictly_later_bar() -> None:
    records = (
        record(
            "A",
            negative_half_bar=2,
            positive_two_bar=4,
            exit_reason=ExitReason.TAKE_PROFIT,
            net_r="2",
        ),
    )

    summary = summarize_early_failure(records)

    assert summary.later_positive_two_r_percent == Decimal("100")
    assert summary.same_bar_positive_two_r_ambiguous == 0


def test_same_bar_touch_is_not_called_recovery() -> None:
    records = (
        record(
            "A",
            negative_half_bar=2,
            positive_two_bar=2,
            exit_reason=ExitReason.TAKE_PROFIT,
            net_r="2",
        ),
    )

    summary = summarize_early_failure(records)

    assert summary.later_positive_two_r_percent == Decimal("0")
    assert summary.same_bar_positive_two_r_ambiguous == 1