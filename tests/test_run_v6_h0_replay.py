from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.backtest.models import BacktestConfig
from src.cli.run_v6_h0 import (
    validate_v6_h0_implementation,
    validate_v6_h0_manifest,
)
from src.cli.run_v6_h0_replay import (
    ATR_STOP_SOURCE,
    COST_FLOOR_STOP_SOURCE,
    WindowResult,
    build_cost_diagnostics,
    build_summary_payload,
    classify_stop_source,
    evaluation_days_total,
    main,
    _window_result,
    validate_eligible_windows,
    validate_replay_configs,
    write_reports,
)
from src.cli.run_v5_h0_replay import summarize_combined_records
from src.diagnostics.r_normalized import RNormalizedTrade
from src.research.v6_mtf_continuation_preregistration import build_manifest
from src.strategy.models import TrendMomentumConfig
from src.strategy.v6_mtf_continuation import V6MTFContinuationStrategy


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


def configs() -> tuple[BacktestConfig, BacktestConfig]:
    base = BacktestConfig(fee_bps=Decimal("10"), slippage_bps=Decimal("2"))
    return base, replace(
        base, fee_bps=Decimal("20"), slippage_bps=Decimal("4")
    )


def record(trade_id: str, net: str, friction: str = "0") -> RNormalizedTrade:
    value = Decimal(net)
    friction_value = Decimal(friction)
    return RNormalizedTrade(
        trade_id=trade_id,
        initial_risk=Decimal("1"),
        frictionless_r=value + friction_value,
        gross_after_slippage_r=value,
        fee_cost_r=friction_value,
        slippage_cost_r=Decimal("0"),
        total_friction_r=friction_value,
        net_r=value,
    )


def rows(*, positive: int) -> tuple[WindowResult, ...]:
    return tuple(
        WindowResult(
            window_id=f"W{index + 1:03d}",
            trades=1,
            evaluation_days=Decimal("90"),
            trades_per_day=Decimal("1") / Decimal("90"),
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
            average_winner_r=Decimal("1"),
            average_loser_r=Decimal("-0.5"),
            payoff_ratio=Decimal("2"),
            maximum_drawdown_percent=Decimal("1"),
            net_r_per_day=(
                Decimal("1") if index < positive else Decimal("-0.5")
            )
            / Decimal("90"),
            positive_net=index < positive,
        )
        for index in range(11)
    )


def payload(*, positive: int) -> dict:
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
        cost_diagnostics={},
    )


def test_execute_flag_is_required_before_any_replay() -> None:
    assert main([]) == 1


def test_manifest_mismatch_is_rejected() -> None:
    manifest = deepcopy(build_manifest())
    manifest["run_id"] = "wrong"
    with pytest.raises(ValueError, match="run ID"):
        validate_v6_h0_manifest(manifest)


def test_holdout_contamination_is_rejected() -> None:
    manifest = deepcopy(build_manifest())
    manifest["dataset_policy"]["blind_holdout"]["loaded"] = True
    with pytest.raises(ValueError, match="holdout integrity"):
        validate_v6_h0_implementation(manifest)


def test_wrong_eligible_window_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly 11"):
        validate_eligible_windows((object(),) * 10)


def test_frozen_strategy_config_mismatch_is_rejected() -> None:
    base, stress = configs()
    with pytest.raises(ValueError, match="fast_ema_period"):
        validate_replay_configs(
            manifest=build_manifest(),
            strategy_config=replace(TrendMomentumConfig(), fast_ema_period=19),
            base_config=base,
            stress_config=stress,
        )


def test_96_bps_floor_mismatch_is_rejected(monkeypatch) -> None:
    base, stress = configs()
    monkeypatch.setattr(
        V6MTFContinuationStrategy,
        "MIN_STOP_DISTANCE_FRACTION",
        Decimal("0.0192"),
    )
    with pytest.raises(ValueError, match="96-bps"):
        validate_replay_configs(
            manifest=build_manifest(),
            strategy_config=TrendMomentumConfig(),
            base_config=base,
            stress_config=stress,
        )


def test_aggregation_uses_combined_trade_records() -> None:
    first = tuple(record(f"A{index}", "1") for index in range(9))
    second = (record("B1", "-0.5"),)
    combined = summarize_combined_records((first, second))
    assert combined.net_expectancy_r == Decimal("0.85")
    assert combined.net_expectancy_r != Decimal("0.25")


def test_evaluation_days_exclude_warmup() -> None:
    windows = (
        SimpleNamespace(duration_days=Decimal("90"), evaluation_start_index=816),
        SimpleNamespace(duration_days=Decimal("60"), evaluation_start_index=1600),
    )
    assert evaluation_days_total(windows) == Decimal("150")


def test_stop_source_and_friction_diagnostics() -> None:
    observations = (
        classify_stop_source(
            trade_id="ATR",
            entry_signal_time=NOW,
            entry_price=Decimal("100"),
            atr_distance=Decimal("1"),
            minimum_stop_distance_fraction=Decimal("0.0096"),
        ),
        classify_stop_source(
            trade_id="FLOOR",
            entry_signal_time=NOW,
            entry_price=Decimal("200"),
            atr_distance=Decimal("1"),
            minimum_stop_distance_fraction=Decimal("0.0096"),
        ),
    )
    diagnostics = build_cost_diagnostics(
        observations=observations,
        records=(record("ATR", "1", "0.2"), record("FLOOR", "1", "0.4")),
        final_buy_signals=2,
    )
    assert observations[0].stop_source == ATR_STOP_SOURCE
    assert observations[1].stop_source == COST_FLOOR_STOP_SOURCE
    assert diagnostics["atr_determined_trades"] == 1
    assert diagnostics["cost_floor_determined_trades"] == 1
    assert diagnostics["average_modeled_base_friction_r_per_trade"] == Decimal(
        "0.3"
    )
    assert diagnostics["average_friction_initial_risk_ratio"] == Decimal("0.3")

    duplicate_window_ids = build_cost_diagnostics(
        observations=(observations[0], replace(observations[1], trade_id="ATR")),
        records=(record("ATR", "1", "0.2"), record("ATR", "1", "0.4")),
        final_buy_signals=2,
    )
    assert duplicate_window_ids["average_modeled_base_friction_r_per_trade"] == (
        Decimal("0.3")
    )


def test_stress_costs_keep_frozen_96_bps_stop_floor() -> None:
    base, stress = configs()
    validate_replay_configs(
        manifest=build_manifest(),
        strategy_config=TrendMomentumConfig(),
        base_config=base,
        stress_config=stress,
    )
    observation = classify_stop_source(
        trade_id="T1",
        entry_signal_time=NOW,
        entry_price=Decimal("100"),
        atr_distance=Decimal("0.5"),
        minimum_stop_distance_fraction=(
            V6MTFContinuationStrategy.MIN_STOP_DISTANCE_FRACTION
        ),
    )
    assert observation.cost_floor_distance == Decimal("0.9600")
    assert observation.initial_stop_distance_bps == Decimal("96.0000")


def test_seven_of_eleven_integrates_with_frozen_gate() -> None:
    result = payload(positive=7)
    assert result["gate"]["positive_net_windows"] == 7
    assert result["progression_eligible"] is True


def test_six_of_eleven_fails_frozen_gate() -> None:
    result = payload(positive=6)
    assert result["gate"]["positive_net_windows"] == 6
    assert result["progression_eligible"] is False


def test_positive_net_uses_exact_window_net_expectancy() -> None:
    summary = summarize_combined_records(((record("T1", "0.0001"),),))
    row = _window_result(
        window=SimpleNamespace(window_id="W001", duration_days=Decimal("1")),
        evaluation=SimpleNamespace(
            backtest=SimpleNamespace(equity_curve=())
        ),
        summary=summary,
    )
    assert row.net_expectancy_r == Decimal("0.0001")
    assert row.positive_net is True
    assert row.positive_net == (row.net_expectancy_r > 0)


def test_report_serialization_and_existing_summary_blocks_rerun(tmp_path) -> None:
    result = payload(positive=7)
    window_results = rows(positive=7)
    output = tmp_path / "e1eef7bdd0c37ad4"
    summary_path, csv_path = write_reports(
        output=output,
        summary_payload=result,
        window_results=window_results,
    )
    assert summary_path.is_file()
    assert csv_path.is_file()
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 12
    assert '"run_id": "e1eef7bdd0c37ad4"' in summary_path.read_text(
        encoding="utf-8"
    )
    with pytest.raises(ValueError, match="already exists"):
        write_reports(
            output=output,
            summary_payload=result,
            window_results=window_results,
        )
