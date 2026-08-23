from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.cli.run_v7_h0_replay as runner
from src.backtest.models import BacktestConfig
from src.cli.run_v7_h0 import (
    EXPECTED_WINDOW_IDS,
    validate_v7_dataset_manifest,
    validate_v7_h0_manifest,
)
from src.cli.run_v7_h0_replay import (
    EntryComparison,
    V7WindowResult,
    build_summary_payload,
    build_window_result,
    main,
    summarize_combined_records,
    validate_partition_paths,
    validate_replay_configs,
    validate_subset_matching,
    verify_frozen_reproduction,
    write_reports,
)
from src.diagnostics.r_normalized import (
    RNormalizedSummary,
    RNormalizedTrade,
    summarize_r_normalized_trades,
)
from src.research.v7_orderflow_preregistration import build_manifest
from src.strategy.models import TrendMomentumConfig
from src.strategy.v6_mtf_continuation import V6MTFContinuationStrategy


NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
DATASET_MANIFEST = Path(
    "data/orderflow/aggregated/15m/BTCUSDC/dataset_manifest.json"
)


def dataset_manifest() -> dict:
    return runner._load_json(DATASET_MANIFEST)


def record(
    trade_id: str,
    net: str,
    *,
    friction: str = "0",
) -> RNormalizedTrade:
    net_r = Decimal(net)
    friction_r = Decimal(friction)
    return RNormalizedTrade(
        trade_id=trade_id,
        initial_risk=Decimal("1"),
        frictionless_r=net_r + friction_r,
        gross_after_slippage_r=net_r,
        fee_cost_r=friction_r,
        slippage_cost_r=Decimal("0"),
        total_friction_r=friction_r,
        net_r=net_r,
    )


def configs() -> tuple[BacktestConfig, BacktestConfig]:
    base = BacktestConfig(fee_bps=Decimal("10"), slippage_bps=Decimal("2"))
    return base, replace(base, fee_bps=Decimal("20"), slippage_bps=Decimal("4"))


def test_execute_flag_is_required_before_any_replay(monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "_load_json",
        lambda path: pytest.fail("preflight must not run without --execute"),
    )

    assert main([]) == 1


def test_wrong_v7_run_id_is_rejected() -> None:
    manifest = deepcopy(build_manifest())
    manifest["run_id"] = "wrong"

    with pytest.raises(ValueError, match="run ID"):
        validate_v7_h0_manifest(manifest)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("dataset_id", "dataset ID"),
        ("dataset_definition_sha256", "dataset SHA-256"),
    ),
)
def test_wrong_dataset_identity_is_rejected(field: str, message: str) -> None:
    payload = dataset_manifest()
    payload[field] = "wrong"

    with pytest.raises(ValueError, match=message):
        validate_v7_dataset_manifest(payload)


def test_corrupted_bucket_coverage_is_rejected() -> None:
    payload = dataset_manifest()
    payload["coverage"]["missing"] = 1

    with pytest.raises(ValueError, match="coverage reconciliation"):
        validate_v7_dataset_manifest(payload)


def test_holdout_partition_overlap_is_rejected_before_loading() -> None:
    payload = dataset_manifest()
    payload["partition_paths"].append(
        "data/orderflow/aggregated/15m/BTCUSDC/2025/08/orderflow.csv"
    )

    with pytest.raises(ValueError, match="overlaps blind holdout"):
        validate_partition_paths(payload)


def test_frozen_v6_reference_mismatch_stops_replay() -> None:
    summary = summarize_r_normalized_trades((record("T1", "1"),))

    with pytest.raises(ValueError, match="reproduction failed"):
        verify_frozen_reproduction(summary=summary, positive_windows=1)


def test_subset_matching_and_retention_ratio_are_exact() -> None:
    v6 = tuple(NOW + timedelta(minutes=15 * index) for index in range(4))
    result = validate_subset_matching(v6_entry_timestamps=v6, v7_entry_timestamps=v6[:3])

    assert result.v6_reference_entries == 4
    assert result.v7_retained_entries == 3
    assert result.filtered_v6_entries == 1
    assert result.retention_ratio == Decimal("0.75")
    assert result.non_v6_entries == 0


def test_non_v6_entry_fails_immediately() -> None:
    with pytest.raises(ValueError, match="absent from the V6"):
        validate_subset_matching(
            v6_entry_timestamps=(NOW,),
            v7_entry_timestamps=(NOW + timedelta(minutes=15),),
        )


def test_zero_trade_window_is_never_reported_as_improvement() -> None:
    empty = summarize_r_normalized_trades(())
    v6_summary = summarize_r_normalized_trades((record("V6", "-1"),))
    evaluation = SimpleNamespace(
        signals=(), backtest=SimpleNamespace(equity_curve=())
    )
    row = build_window_result(
        window=SimpleNamespace(window_id="W002", duration_days=Decimal("1")),
        v6_evaluation=evaluation,
        v6_summary=v6_summary,
        v7_evaluation=evaluation,
        v7_summary=empty,
    )

    assert row.status == "ZERO_TRADES"
    assert row.positive_net is False
    assert row.net_better_than_v6 is False
    assert row.gross_better_than_v6 is False
    assert row.net_delta_vs_v6 is None


def test_combined_metrics_use_combined_trade_records() -> None:
    first = tuple(record(f"A{index}", "1") for index in range(9))
    second = (record("B", "-0.5"),)

    combined = summarize_combined_records((first, second))

    assert combined.net_expectancy_r == Decimal("0.85")
    assert combined.net_expectancy_r != Decimal("0.25")


def synthetic_window_rows(
    records: tuple[RNormalizedTrade, ...], *, better: int
) -> tuple[V7WindowResult, ...]:
    rows = []
    for index, (window_id, item) in enumerate(zip(EXPECTED_WINDOW_IDS, records, strict=True)):
        summary = summarize_r_normalized_trades((item,))
        v6_net = Decimal("-0.2") if index < better else Decimal("2")
        rows.append(
            V7WindowResult(
                window_id=window_id,
                status="TRADES",
                evaluation_days=Decimal("1"),
                v6_trades=1,
                v7_trades=1,
                retention_ratio=Decimal("1"),
                v6_frictionless_expectancy_r=v6_net,
                v6_net_expectancy_r=v6_net,
                v6_profit_factor_r=None,
                v7_frictionless_expectancy_r=summary.frictionless_expectancy_r,
                v7_net_expectancy_r=summary.net_expectancy_r,
                v7_profit_factor_r=summary.profit_factor_r,
                v7_win_rate_percent=summary.win_rate_percent,
                v7_average_winner_r=summary.average_winner_r,
                v7_average_loser_r=summary.average_loser_r,
                v7_payoff_ratio=summary.payoff_ratio_r,
                v7_average_friction_r=summary.average_total_friction_r,
                v7_maximum_drawdown_percent=Decimal("1"),
                v7_trades_per_day=Decimal("1"),
                v7_net_r_per_day=summary.net_expectancy_r,
                positive_net=summary.net_expectancy_r > 0,
                gross_delta_vs_v6=summary.frictionless_expectancy_r - v6_net,
                net_delta_vs_v6=summary.net_expectancy_r - v6_net,
                profit_factor_delta_vs_v6=None,
                gross_better_than_v6=index < better,
                net_better_than_v6=index < better,
                v6_reference_entries=1,
                v7_retained_entries=1,
                filtered_v6_entries=0,
                non_v6_entries=0,
            )
        )
    return tuple(rows)


def frozen_v6_summary() -> RNormalizedSummary:
    base = summarize_r_normalized_trades(
        tuple(record(f"V6-{index}", "-0.1") for index in range(11))
    )
    return replace(
        base,
        frictionless_expectancy_r=Decimal("0.04722009245553381286395292467"),
        net_expectancy_r=Decimal("-0.1366917232474779417819851766"),
        profit_factor_r=Decimal("0.7695960517596068371318908235"),
    )


def summary_payload(
    *, records: tuple[RNormalizedTrade, ...], better: int
) -> dict:
    v7_summary = summarize_r_normalized_trades(records)
    rows = synthetic_window_rows(records, better=better)
    return build_summary_payload(
        manifest=build_manifest(),
        dataset_manifest=dataset_manifest(),
        v6_summary=frozen_v6_summary(),
        v7_summary=v7_summary,
        stress_summary=v7_summary,
        window_results=rows,
        entry_comparison=(
            EntryComparison("W002", NOW.isoformat(), "RETAINED", Decimal("0.1"), Decimal("0.55")),
        ),
        maximum_drawdown_percent=Decimal("1"),
        evaluation_days=Decimal("11"),
    )


def test_seven_net_better_windows_pass_support_consistency() -> None:
    records = tuple(
        record(f"T{index}", "1" if index < 7 else "-0.5")
        for index in range(11)
    )
    result = summary_payload(records=records, better=7)

    assert result["support_gate"]["window_improvement_passed"] is True
    assert result["final_classification"] == "PROGRESSION_ELIGIBLE"


def test_six_net_better_windows_fail_support() -> None:
    records = tuple(
        record(f"T{index}", "1" if index < 7 else "-0.5")
        for index in range(11)
    )
    result = summary_payload(records=records, better=6)

    assert result["support_gate"]["all_conditions_met"] is False
    assert result["final_classification"] == "NOT_SUPPORTED"


def test_support_can_pass_while_progression_fails() -> None:
    records = tuple(
        record(
            f"T{index}",
            "0.1" if index < 7 else "-0.21",
            friction="0.1",
        )
        for index in range(11)
    )
    result = summary_payload(records=records, better=7)

    assert result["support_gate"]["all_conditions_met"] is True
    assert result["progression_gate"]["all_conditions_met"] is False
    assert result["final_classification"] == "SUPPORTED_NOT_PROFITABLE"


def test_stress_costs_keep_frozen_96_bps_floor(monkeypatch) -> None:
    base, stress = configs()
    validate_replay_configs(
        manifest=build_manifest(),
        strategy_config=TrendMomentumConfig(),
        base_config=base,
        stress_config=stress,
    )
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


def test_report_serialization_is_deterministic_and_refuses_rerun(tmp_path) -> None:
    records = tuple(
        record(f"T{index}", "1" if index < 7 else "-0.5")
        for index in range(11)
    )
    payload = summary_payload(records=records, better=7)
    rows = synthetic_window_rows(records, better=7)
    entries = (
        EntryComparison("W002", NOW.isoformat(), "RETAINED", Decimal("0.1"), Decimal("0.55")),
    )
    first = write_reports(
        output=tmp_path / "first",
        summary_payload=payload,
        window_results=rows,
        entry_comparison=entries,
    )
    second = write_reports(
        output=tmp_path / "second",
        summary_payload=payload,
        window_results=rows,
        entry_comparison=entries,
    )

    assert tuple(path.read_bytes() for path in first) == tuple(
        path.read_bytes() for path in second
    )
    with pytest.raises(ValueError, match="already exists"):
        write_reports(
            output=tmp_path / "first",
            summary_payload=payload,
            window_results=rows,
            entry_comparison=entries,
        )
