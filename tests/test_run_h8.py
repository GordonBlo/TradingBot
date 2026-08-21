from copy import deepcopy

import pytest

from src.cli.run_h8 import (
    validate_h8_manifest,
    validate_v331_summary,
)


def valid_h8_manifest() -> dict:
    return {
        "run_id": "369586c25375c0ef",
        "source_v33_run_id": "b104ea78b4c88bcf",
        "source_v331_diagnostic_run_id": "ef593be9a0bb2697",
        "dataset_status": "CONSUMED_RESEARCH_DATA",
        "reference_candidate": "H5_Q25",
        "reference_trade_count": 197,
        "blind_holdout_status": "LOCKED_BLIND_HOLDOUT",
        "blind_holdout_revealed": False,
        "blind_holdout_consumed": False,
        "blind_holdout_evaluated": False,
        "definition": {
            "hypothesis_id": "H8",
            "trigger_r": "1.0",
            "activation_timing": "NEXT_BAR_AFTER_CLOSED_TRIGGER",
            "protective_stop": "ENTRY_PRICE",
            "protective_stop_semantics": (
                "PRICE_BREAK_EVEN_NOT_NET_BREAK_EVEN"
            ),
            "take_profit_r": "2.0",
            "preserve_h5_q25_entries": True,
            "preserve_original_atr_stop_until_activation": True,
            "preserve_take_profit": True,
            "partial_exit": False,
            "ambiguous_bar_policy": "STOP_FIRST",
        },
        "research_integrity": {
            "parameter_search": False,
            "alternative_trigger_R_values": False,
            "partial_exit_research": False,
            "trailing_stop_research": False,
            "entry_logic_changes": False,
            "target_changes": False,
            "ATR_stop_changes_before_activation": False,
            "blind_holdout_access": False,
        },
    }


def valid_v331_summary() -> dict:
    return {
        "run_id": "ef593be9a0bb2697",
        "source_v33_run_id": "b104ea78b4c88bcf",
        "candidate": "H5_Q25",
        "dataset_status": "CONSUMED_RESEARCH_DATA",
        "eligible_windows": 11,
        "trade_count": 197,
        "frozen_reproduction_verified": True,
        "r_accounting": {
            "verified": True,
        },
        "warmup_parity": {
            "continuous_state_trade_count": 197,
            "continuous_state_entry_signal_differences": 0,
            "changes_frozen_results": False,
        },
        "blind_holdout": {
            "status": "LOCKED_BLIND_HOLDOUT",
            "loaded": False,
            "revealed": False,
            "consumed": False,
            "evaluated": False,
        },
    }


def test_h8_manifest_accepts_frozen_definition() -> None:
    validate_h8_manifest(valid_h8_manifest())


def test_h8_manifest_rejects_trigger_change() -> None:
    payload = deepcopy(valid_h8_manifest())
    payload["definition"]["trigger_r"] = "0.5"

    with pytest.raises(ValueError, match="trigger"):
        validate_h8_manifest(payload)


def test_v331_reference_must_remain_197_trades() -> None:
    payload = deepcopy(valid_v331_summary())
    payload["trade_count"] = 196

    with pytest.raises(ValueError, match="197"):
        validate_v331_summary(payload)