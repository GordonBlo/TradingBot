from __future__ import annotations

import inspect
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

import src.cli.run_v8_h0_replay as runner
from src.backtest.models import BacktestConfig
from src.cli.run_v8_h0_replay import (
    FrozenV6Candidate,
    FundingObservation,
    doubled_cost_stress_records,
    filter_frozen_candidates,
    load_frozen_v6_reference,
    main,
    validate_derivatives_manifest,
    validate_replay_configs,
    validate_v8_manifest,
)
from src.diagnostics.r_normalized import RNormalizedTrade
from src.models.candle import Candle
from src.models.research_indicator_snapshot import ResearchIndicatorSnapshot
from src.research.v8_funding_context_preregistration import build_manifest
from src.strategy.context import StrategyContext
from src.strategy.models import StrategyAction, TrendMomentumConfig
from src.strategy.v8_funding_schedule import V8FundingScheduleStrategy


SIGNAL_CLOSE = datetime(2024, 1, 1, 0, 15, tzinfo=timezone.utc)
DATASET_MANIFEST = Path(
    "data/derivatives/processed/15m/BTCUSDT/eec764735d270f9d/manifest.json"
)


def test_execute_flag_is_required_before_preflight(monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "_load_json",
        lambda path: pytest.fail("preflight must not run without --execute"),
    )
    assert main([]) == 1


def test_manifest_run_id_mismatch_is_rejected() -> None:
    manifest = deepcopy(build_manifest())
    manifest["run_id"] = "wrong"
    with pytest.raises(ValueError, match="run ID"):
        validate_v8_manifest(manifest)


def test_derivatives_identity_future_data_and_holdout_contamination_are_rejected() -> None:
    payload = runner._load_json(DATASET_MANIFEST)
    validate_derivatives_manifest(payload)

    wrong = deepcopy(payload)
    wrong["dataset_id"] = "wrong"
    with pytest.raises(ValueError, match="identity or integrity"):
        validate_derivatives_manifest(wrong)

    future = deepcopy(payload)
    future["future_data_violations"] = 1
    with pytest.raises(ValueError, match="identity or integrity"):
        validate_derivatives_manifest(future)

    contaminated = deepcopy(payload)
    contaminated["blind_holdout"]["loaded"] = True
    with pytest.raises(ValueError, match="holdout contamination"):
        validate_derivatives_manifest(contaminated)


def candidates() -> tuple[FrozenV6Candidate, ...]:
    return tuple(
        FrozenV6Candidate(
            window_id="W002",
            signal_close=SIGNAL_CLOSE + timedelta(minutes=15 * index),
            atr14_4h=Decimal("100"),
        )
        for index in range(4)
    )


def test_filter_uses_only_non_positive_causal_funding_and_rejects_missing_future() -> None:
    rows = candidates()
    funding = {
        rows[0].signal_close: FundingObservation(
            rows[0].signal_close, Decimal("0"), rows[0].signal_close
        ),
        rows[1].signal_close: FundingObservation(
            rows[1].signal_close, Decimal("0.0001"), rows[1].signal_close
        ),
        rows[2].signal_close: FundingObservation(
            rows[2].signal_close,
            Decimal("-0.0001"),
            rows[2].signal_close + timedelta(microseconds=1),
        ),
    }
    retained, decisions = filter_frozen_candidates(
        candidates=rows, funding_by_signal_close=funding
    )

    assert retained == (rows[0],)
    assert tuple(row.status for row in decisions) == (
        "RETAINED_NON_POSITIVE_FUNDING",
        "FILTERED_POSITIVE_FUNDING",
        "FILTERED_FUTURE_FUNDING",
        "FILTERED_MISSING_FUNDING",
    )


def test_frozen_v6_schedule_is_loaded_directly_and_matches_frozen_result() -> None:
    summary, windows, schedule = load_frozen_v6_reference(
        summary_path="reports/v6_mtf_continuation/e1eef7bdd0c37ad4/summary.json",
        windows_path=(
            "reports/v6_mtf_continuation/e1eef7bdd0c37ad4/window_results.csv"
        ),
        schedule_path=(
            "reports/diagnostics/v6_signal_quality/e1eef7bdd0c37ad4/trade_ledger.csv"
        ),
        manifest=build_manifest(),
    )

    assert summary.trade_count == len(schedule) == 443
    assert tuple(windows) == runner.EXPECTED_WINDOW_IDS
    assert sum(window.trades for window in windows.values()) == 443


def context(
    *,
    timestamp: datetime = SIGNAL_CLOSE,
    has_position: bool = False,
    bars_in_position: int = 0,
    bars_since_exit: int | None = None,
) -> StrategyContext:
    candle_time = timestamp - timedelta(minutes=15)
    candle = Candle(
        timestamp=candle_time,
        symbol="BTCUSDC",
        interval="15m",
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100"),
        volume=Decimal("1"),
        is_closed=True,
    )
    indicators = ResearchIndicatorSnapshot(
        timestamp=candle_time,
        symbol="BTCUSDC",
        interval="15m",
        close=Decimal("100"),
        volume=Decimal("1"),
        ema_fast=None,
        ema_slow=None,
        rsi=None,
        atr=None,
        volume_sma=None,
        volume_ratio=None,
    )
    return StrategyContext(
        timestamp=timestamp,
        current_candle=candle,
        recent_history=(candle,),
        indicators=indicators,
        previous_indicators=None,
        has_position=has_position,
        bars_in_position=bars_in_position,
        equity=Decimal("10000"),
        cash_usdc=Decimal("10000"),
        completed_trade_count=0,
        bars_since_exit=bars_since_exit,
        entry_fee_rate=Decimal("0.001"),
    )


def test_schedule_strategy_emits_only_frozen_retained_timestamp_and_v6_risk() -> None:
    strategy = V8FundingScheduleStrategy(
        TrendMomentumConfig(),
        retained_atr_by_signal_close={SIGNAL_CLOSE: Decimal("123")},
    )

    decision = strategy.evaluate(context())
    assert decision.action is StrategyAction.ENTER_LONG
    assert decision.stop_distance == Decimal("123")
    assert decision.reward_risk_ratio == Decimal("2")
    assert decision.minimum_stop_distance_fraction == Decimal("0.0096")
    assert strategy.evaluate(
        context(timestamp=SIGNAL_CLOSE + timedelta(minutes=15))
    ).action is StrategyAction.HOLD
    assert strategy.emitted_candidate_timestamps == frozenset((SIGNAL_CLOSE,))


def test_schedule_conflict_is_recorded_without_reevaluating_v6() -> None:
    strategy = V8FundingScheduleStrategy(
        TrendMomentumConfig(),
        retained_atr_by_signal_close={SIGNAL_CLOSE: Decimal("123")},
    )

    decision = strategy.evaluate(context(has_position=True, bars_in_position=1))
    assert decision.action is StrategyAction.HOLD
    assert strategy.scheduled_candidate_conflict_timestamps == frozenset(
        (SIGNAL_CLOSE,)
    )
    runner_source = inspect.getsource(runner)
    assert "evaluate_v6_window" not in runner_source
    assert "V6MTFContinuationStrategy(" not in runner_source


def test_frozen_base_stress_execution_and_costs_validate() -> None:
    base = BacktestConfig()
    stress = replace(
        base, fee_bps=Decimal("20"), slippage_bps=Decimal("4")
    )
    validate_replay_configs(
        manifest=build_manifest(),
        strategy_config=TrendMomentumConfig(),
        base_config=base,
        stress_config=stress,
    )

    with pytest.raises(ValueError, match="stress_2x"):
        validate_replay_configs(
            manifest=build_manifest(),
            strategy_config=TrendMomentumConfig(),
            base_config=base,
            stress_config=replace(stress, fee_bps=Decimal("21")),
        )


def test_doubled_cost_stress_freezes_trade_path_and_doubles_only_friction() -> None:
    base = RNormalizedTrade(
        trade_id="T1",
        initial_risk=Decimal("10"),
        frictionless_r=Decimal("1"),
        gross_after_slippage_r=Decimal("0.98"),
        fee_cost_r=Decimal("0.1"),
        slippage_cost_r=Decimal("0.02"),
        total_friction_r=Decimal("0.12"),
        net_r=Decimal("0.88"),
    )

    stressed = doubled_cost_stress_records((base,))[0]
    assert stressed.trade_id == base.trade_id
    assert stressed.initial_risk == base.initial_risk
    assert stressed.frictionless_r == base.frictionless_r
    assert stressed.fee_cost_r == Decimal("0.2")
    assert stressed.slippage_cost_r == Decimal("0.04")
    assert stressed.net_r == Decimal("0.76")
