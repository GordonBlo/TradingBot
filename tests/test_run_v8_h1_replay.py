from __future__ import annotations

import inspect
import json
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

import src.cli.run_v8_h1_replay as runner
from src.backtest.models import BacktestConfig
from src.cli.run_v8_h0_replay import EXPECTED_WINDOW_IDS, FrozenV6Candidate, V8WindowResult
from src.cli.run_v8_h1_replay import (
    CandidateDecision,
    OpenInterestObservation,
    build_summary_payload,
    filter_frozen_candidates,
    load_open_interest_context,
    main,
    validate_replay_configs,
    validate_v8_h1_manifest,
)
from src.diagnostics.r_normalized import summarize_r_normalized_trades
from src.research.v8_open_interest_preregistration import build_manifest
from src.strategy.models import TrendMomentumConfig


T = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)


def _candidate(index: int) -> FrozenV6Candidate:
    return FrozenV6Candidate(
        window_id="W002",
        signal_close=T + timedelta(hours=index * 2),
        atr14_4h=Decimal("100"),
    )


def _observation(
    bucket: datetime,
    value: str | None,
    timestamp: datetime | None = None,
) -> OpenInterestObservation:
    return OpenInterestObservation(
        bucket_close_time=bucket,
        sum_open_interest=Decimal(value) if value is not None else None,
        source_timestamp=timestamp if timestamp is not None else bucket,
    )


def test_execute_flag_is_required_before_preflight(monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "_load_json",
        lambda path: pytest.fail("preflight must not run without --execute"),
    )
    assert main([]) == 1


def test_manifest_mismatch_is_rejected() -> None:
    manifest = deepcopy(build_manifest())
    manifest["run_id"] = "wrong"
    with pytest.raises(ValueError, match="run ID"):
        validate_v8_h1_manifest(manifest)


def test_filter_applies_only_strict_causal_60m_quantity_rule() -> None:
    rows = tuple(_candidate(index) for index in range(7))
    context: dict[datetime, OpenInterestObservation] = {}
    for row in rows:
        context[row.signal_close] = _observation(row.signal_close, "101")
        context[row.signal_close - timedelta(minutes=60)] = _observation(
            row.signal_close - timedelta(minutes=60), "100"
        )

    context[rows[1].signal_close] = _observation(rows[1].signal_close, "100")
    context[rows[2].signal_close] = _observation(rows[2].signal_close, "99")
    del context[rows[3].signal_close]
    del context[rows[4].signal_close - timedelta(minutes=60)]
    context[rows[5].signal_close] = _observation(
        rows[5].signal_close,
        "101",
        rows[5].signal_close + timedelta(microseconds=1),
    )
    historical_bucket = rows[6].signal_close - timedelta(minutes=60)
    context[historical_bucket] = _observation(
        historical_bucket,
        "100",
        historical_bucket + timedelta(microseconds=1),
    )

    retained, decisions = filter_frozen_candidates(
        candidates=rows,
        open_interest_by_bucket_close=context,
    )

    assert retained == (rows[0],)
    assert tuple(row.status for row in decisions) == (
        "RETAINED_OI_EXPANDING_60M",
        "FILTERED_OI_NOT_EXPANDING_60M",
        "FILTERED_OI_NOT_EXPANDING_60M",
        "FILTERED_MISSING_CURRENT_OI",
        "FILTERED_MISSING_60M_OI",
        "FILTERED_FUTURE_CURRENT_OI",
        "FILTERED_FUTURE_60M_OI",
    )


def test_filter_requires_the_exact_t_minus_60m_bucket() -> None:
    row = _candidate(0)
    context = {
        row.signal_close: _observation(row.signal_close, "101"),
        row.signal_close - timedelta(minutes=45): _observation(
            row.signal_close - timedelta(minutes=45), "100"
        ),
    }
    retained, decisions = filter_frozen_candidates(
        candidates=(row,), open_interest_by_bucket_close=context
    )
    assert retained == ()
    assert decisions[0].status == "FILTERED_MISSING_60M_OI"


def test_context_loader_uses_open_interest_quantity_not_value(tmp_path) -> None:
    partition = tmp_path / "context.csv"
    partition.write_text(
        "bucket_close_time,open_interest,open_interest_value,open_interest_timestamp\n"
        "2024-01-01T12:00:00+00:00,123.5,999999,2024-01-01T12:00:00+00:00\n",
        encoding="utf-8",
    )
    loaded = load_open_interest_context(
        {"partition_paths": [str(partition)], "context_buckets": 1}
    )
    assert loaded[T].sum_open_interest == Decimal("123.5")
    assert "open_interest_value" not in inspect.getsource(load_open_interest_context)


def test_replay_reuses_schedule_executor_without_v6_signal_reevaluation() -> None:
    source = inspect.getsource(runner)
    assert "V6MTFContinuationStrategy(" not in source
    assert "evaluate_v6_window" not in source
    assert "require_v8_h1_subset_of_frozen_v6" in source


def test_frozen_execution_and_costs_validate(monkeypatch) -> None:
    base = BacktestConfig()
    stress = replace(base, fee_bps=Decimal("20"), slippage_bps=Decimal("4"))
    validate_replay_configs(
        manifest=build_manifest(),
        strategy_config=TrendMomentumConfig(),
        base_config=base,
        stress_config=stress,
    )
    monkeypatch.setattr(runner.V8FundingScheduleStrategy, "COOLDOWN_BARS", 5)
    with pytest.raises(ValueError, match="execution/risk"):
        validate_replay_configs(
            manifest=build_manifest(),
            strategy_config=TrendMomentumConfig(),
            base_config=base,
            stress_config=stress,
        )


def test_summary_serialization_integrates_frozen_gates(tmp_path) -> None:
    empty = summarize_r_normalized_trades(())
    v6 = replace(
        empty,
        trade_count=11,
        frictionless_expectancy_r=Decimal("0.04"),
        net_expectancy_r=Decimal("-0.14"),
        profit_factor_r=Decimal("0.7"),
    )
    v8 = replace(
        empty,
        trade_count=11,
        frictionless_expectancy_r=Decimal("0.10"),
        net_expectancy_r=Decimal("0.10"),
        profit_factor_r=Decimal("1.2"),
    )
    stress = replace(v8, net_expectancy_r=Decimal("0.05"))
    windows = tuple(
        V8WindowResult(
            window_id=window_id,
            status="TRADES",
            evaluation_days=Decimal("1"),
            v6_candidates=1,
            v8_retained=1,
            filtered_candidates=0,
            non_v6_entries=0,
            v6_gross_expectancy_r=Decimal("0.04"),
            v6_net_expectancy_r=Decimal("-0.14"),
            v6_profit_factor_r=Decimal("0.7"),
            v6_net_result_r=Decimal("-0.14"),
            v8_gross_expectancy_r=Decimal("0.10"),
            v8_net_expectancy_r=Decimal("0.10"),
            v8_profit_factor_r=Decimal("1.2"),
            v8_net_result_r=Decimal("0.10"),
            v8_win_rate_percent=Decimal("50"),
            v8_maximum_drawdown_percent=Decimal("1"),
            positive_net=True,
            net_result_better_than_v6=True,
        )
        for window_id in EXPECTED_WINDOW_IDS
    )
    decisions = tuple(
        CandidateDecision(
            window_id=window_id,
            signal_close_utc=T.isoformat(),
            status="RETAINED_OI_EXPANDING_60M",
            oi_now=Decimal("101"),
            oi_now_timestamp_utc=T.isoformat(),
            oi_60m=Decimal("100"),
            oi_60m_timestamp_utc=(T - timedelta(minutes=60)).isoformat(),
        )
        for window_id in EXPECTED_WINDOW_IDS
    )
    payload = build_summary_payload(
        manifest=build_manifest(),
        dataset_manifest={
            "dataset_id": "eec764735d270f9d",
            "dataset_definition_sha256": "sha",
            "classification": "READY_WITH_SOURCE_LIMITATIONS",
            "context_buckets": 95040,
            "future_data_violations": 0,
        },
        v6_summary=v6,
        v8_summary=v8,
        stress_summary=stress,
        window_results=windows,
        candidate_decisions=decisions,
        maximum_drawdown_percent=Decimal("1"),
        evaluation_days=Decimal("11"),
    )
    assert payload["support_gate"]["all_conditions_met"] is True
    assert payload["progression_gate"]["all_conditions_met"] is True
    assert payload["final_classification"] == "PROGRESSION_ELIGIBLE"

    paths = runner.write_reports(
        output=tmp_path / "result",
        summary_payload=payload,
        window_results=windows,
        candidate_decisions=decisions,
    )
    assert len(paths) == 3
    assert json.loads(paths[0].read_text(encoding="utf-8"))["run_id"] == runner.EXPECTED_RUN_ID
