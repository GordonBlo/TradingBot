from copy import deepcopy
from dataclasses import replace
from decimal import Decimal

import pytest

from src.backtest.models import BacktestConfig
from src.cli.run_v5_h0 import (
    validate_v5_h0_implementation,
    validate_v5_h0_manifest,
)
from src.cli.run_v5_h0_replay import (
    WindowResult,
    build_summary_payload,
    main,
    summarize_combined_records,
    validate_eligible_windows,
    validate_replay_configs,
    write_reports,
)
from src.diagnostics.r_normalized import RNormalizedTrade
from src.research.v5_mean_reversion_preregistration import build_manifest
from src.strategy.models import TrendMomentumConfig


def configs():
    base = BacktestConfig(
        fee_bps=Decimal("10"),
        slippage_bps=Decimal("2"),
    )
    return base, replace(
        base,
        fee_bps=Decimal("20"),
        slippage_bps=Decimal("4"),
    )


def record(trade_id: str, net: str) -> RNormalizedTrade:
    value = Decimal(net)
    return RNormalizedTrade(
        trade_id=trade_id,
        initial_risk=Decimal("1"),
        frictionless_r=value,
        gross_after_slippage_r=value,
        fee_cost_r=Decimal("0"),
        slippage_cost_r=Decimal("0"),
        total_friction_r=Decimal("0"),
        net_r=value,
    )


def rows(*, positive: int) -> tuple[WindowResult, ...]:
    return tuple(
        WindowResult(
            window_id=f"W{index + 1:03d}",
            trades=1,
            frictionless_expectancy_r=(
                Decimal("1") if index < positive else Decimal("-0.5")
            ),
            net_expectancy_r=(
                Decimal("1") if index < positive else Decimal("-0.5")
            ),
            profit_factor_r=None,
            win_rate_percent=(
                Decimal("100") if index < positive else Decimal("0")
            ),
            maximum_drawdown_percent=Decimal("1"),
            positive_net=index < positive,
            evaluation_days=Decimal("90"),
            trades_per_day=Decimal("1") / Decimal("90"),
            net_r_per_day=(
                Decimal("1") if index < positive else Decimal("-0.5")
            )
            / Decimal("90"),
        )
        for index in range(11)
    )


def summary_payload(*, positive: int):
    records = tuple(
        record(f"T{index}", "1" if index < positive else "-0.5")
        for index in range(11)
    )
    summary = summarize_combined_records((records,))
    return build_summary_payload(
        manifest=build_manifest(),
        base_summary=summary,
        stress_summary=summary,
        window_results=rows(positive=positive),
        maximum_drawdown_percent=Decimal("1"),
        evaluation_days=Decimal("990"),
    )


def test_execute_flag_is_required_before_any_replay() -> None:
    assert main([]) == 1


def test_manifest_mismatch_is_rejected() -> None:
    manifest = deepcopy(build_manifest())
    manifest["run_id"] = "wrong"

    with pytest.raises(ValueError, match="run ID"):
        validate_v5_h0_manifest(manifest)


def test_holdout_contamination_is_rejected() -> None:
    manifest = deepcopy(build_manifest())
    manifest["dataset_policy"]["blind_holdout"]["loaded"] = True

    with pytest.raises(ValueError, match="holdout integrity"):
        validate_v5_h0_implementation(manifest)


def test_wrong_eligible_window_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly 11"):
        validate_eligible_windows((object(),) * 10)


def test_frozen_strategy_config_mismatch_is_rejected() -> None:
    base, stress = configs()

    with pytest.raises(ValueError, match="atr_period"):
        validate_replay_configs(
            manifest=build_manifest(),
            strategy_config=replace(TrendMomentumConfig(), atr_period=13),
            base_config=base,
            stress_config=stress,
        )


def test_aggregation_uses_combined_trade_records() -> None:
    first_window = tuple(record(f"W1T{index}", "1") for index in range(9))
    second_window = (record("W2T1", "-0.5"),)

    combined = summarize_combined_records((first_window, second_window))

    assert combined.net_expectancy_r == Decimal("0.85")
    assert combined.net_expectancy_r != Decimal("0.25")


def test_seven_of_eleven_integrates_with_frozen_gate() -> None:
    payload = summary_payload(positive=7)

    assert payload["gate"]["positive_net_windows"] == 7
    assert payload["progression_eligible"] is True


def test_six_of_eleven_fails_frozen_gate() -> None:
    payload = summary_payload(positive=6)

    assert payload["gate"]["positive_net_windows"] == 6
    assert payload["progression_eligible"] is False


def test_report_serialization_and_existing_summary_refuses_rerun(tmp_path) -> None:
    payload = summary_payload(positive=7)
    window_results = rows(positive=7)
    output = tmp_path / "b645b6c1d251f374"

    summary_path, csv_path = write_reports(
        output=output,
        summary_payload=payload,
        window_results=window_results,
    )

    assert summary_path.is_file()
    assert csv_path.is_file()
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 12
    assert '"run_id": "b645b6c1d251f374"' in summary_path.read_text(
        encoding="utf-8"
    )
    with pytest.raises(ValueError, match="already exists"):
        write_reports(
            output=output,
            summary_payload=payload,
            window_results=window_results,
        )
