from copy import deepcopy
from dataclasses import replace
from decimal import Decimal

import pytest

from src.backtest.models import BacktestConfig
from src.cli.run_v4_h0 import (
    validate_v4_h0_implementation,
    validate_v4_h0_manifest,
)
from src.cli.run_v4_h0_replay import (
    WindowResult,
    build_summary_payload,
    main,
    validate_eligible_windows,
    validate_replay_configs,
    write_reports,
)
from src.diagnostics.r_normalized import (
    RNormalizedTrade,
    summarize_r_normalized_trades,
)
from src.research.v4_breakout_preregistration import build_manifest
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


def test_execute_flag_is_required_before_any_replay() -> None:
    assert main([]) == 1


def test_manifest_mismatch_is_rejected() -> None:
    manifest = deepcopy(build_manifest())
    manifest["run_id"] = "wrong"

    with pytest.raises(ValueError, match="run ID"):
        validate_v4_h0_manifest(manifest)


def test_holdout_contamination_is_rejected() -> None:
    manifest = deepcopy(build_manifest())
    manifest["dataset_policy"]["blind_holdout"]["loaded"] = True

    with pytest.raises(ValueError, match="holdout integrity"):
        validate_v4_h0_implementation(manifest)


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


def synthetic_records():
    net_values = (
        *(Decimal("1") for _ in range(7)),
        *(Decimal("-0.5") for _ in range(4)),
    )
    records = tuple(
        RNormalizedTrade(
            trade_id=f"T{index}",
            initial_risk=Decimal("1"),
            frictionless_r=net,
            gross_after_slippage_r=net,
            fee_cost_r=Decimal("0"),
            slippage_cost_r=Decimal("0"),
            total_friction_r=Decimal("0"),
            net_r=net,
        )
        for index, net in enumerate(net_values)
    )
    return records, net_values


def test_output_serialization_and_frozen_gate_integration(tmp_path) -> None:
    records, net_values = synthetic_records()
    summary = summarize_r_normalized_trades(records)
    rows = tuple(
        WindowResult(
            window_id=f"W{index + 1:03d}",
            trades=1,
            frictionless_expectancy_r=net,
            net_expectancy_r=net,
            profit_factor_r=None,
            maximum_drawdown_percent=Decimal("1"),
            positive_net=net > 0,
        )
        for index, net in enumerate(net_values)
    )
    payload = build_summary_payload(
        manifest=build_manifest(),
        base_summary=summary,
        stress_summary=summary,
        window_results=rows,
        maximum_drawdown_percent=Decimal("1"),
        evaluation_days=Decimal("990"),
    )

    summary_path, csv_path = write_reports(
        output=tmp_path / "ba79975004b106a5",
        summary_payload=payload,
        window_results=rows,
    )

    assert payload["gate"]["positive_net_windows"] == 7
    assert payload["progression_eligible"] is True
    assert summary_path.is_file()
    assert csv_path.is_file()
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 12
    with pytest.raises(ValueError, match="already exists"):
        write_reports(
            output=summary_path.parent,
            summary_payload=payload,
            window_results=rows,
        )
