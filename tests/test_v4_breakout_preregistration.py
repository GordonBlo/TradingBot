import json

from src.research.v4_breakout_preregistration import (
    build_manifest,
    write_manifest,
)


def test_v4_h0_exact_market_and_entry() -> None:
    manifest = build_manifest()

    assert manifest["baseline_id"] == "V4_H0"
    assert manifest["market"]["symbol"] == "BTCUSDC"
    assert manifest["market"]["interval"] == "15m"
    assert manifest["market"]["direction"] == "LONG_ONLY"

    entry = manifest["entry"]

    assert entry["lookback_bars"] == 20
    assert (
        entry["current_candle_excluded_from_reference"]
        is True
    )
    assert entry["execution"] == "NEXT_BAR_OPEN"


def test_v4_h0_has_no_old_strategy_filters() -> None:
    filters = build_manifest()["filters"]

    assert filters == {
        "ema_filter": False,
        "rsi_filter": False,
        "volume_filter": False,
        "atr_percentile_filter": False,
        "trend_regime_filter": False,
    }


def test_v4_h0_exit_semantics_are_frozen() -> None:
    risk = build_manifest()["risk_and_exit"]

    assert risk["atr_period"] == 14
    assert risk["stop_atr_multiple"] == "2"
    assert risk["reward_risk_ratio"] == "2"
    assert risk["maximum_hold_bars"] == 96
    assert risk["cooldown_bars"] == 4
    assert risk["ambiguous_bar_policy"] == "STOP_FIRST"


def test_v4_h0_progression_requires_positive_edge() -> None:
    gate = build_manifest()["progression_gate"]

    assert gate["required_eligible_windows"] == 11
    assert gate["combined_net_expectancy_r_gt"] == "0"
    assert gate["profit_factor_r_gt"] == "1"
    assert (
        gate["positive_net_window_ratio_gte"]
        == "0.60"
    )
    assert gate["all_conditions_required"] is True


def test_v4_h0_keeps_holdout_locked() -> None:
    holdout = (
        build_manifest()
        ["dataset_policy"]
        ["blind_holdout"]
    )

    assert holdout["status"] == "LOCKED_BLIND_HOLDOUT"
    assert holdout["loaded"] is False
    assert holdout["revealed"] is False
    assert holdout["consumed"] is False
    assert holdout["evaluated"] is False


def test_v4_h0_manifest_is_deterministic() -> None:
    first = build_manifest()
    second = build_manifest()

    assert first == second
    assert len(first["run_id"]) == 16
    assert len(first["definition_sha256"]) == 64


def test_v4_h0_manifest_writes_deterministically(
    tmp_path,
) -> None:
    first_path = write_manifest(tmp_path)
    second_path = write_manifest(tmp_path)

    assert first_path == second_path

    stored = json.loads(
        first_path.read_text(
            encoding="utf-8"
        )
    )

    assert stored == build_manifest()