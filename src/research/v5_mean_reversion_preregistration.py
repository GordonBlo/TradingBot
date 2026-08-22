"""Preregister V5-H0: 15m exhaustion-reclaim mean reversion.

This module defines research intent only. It does not load market data or run
a backtest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


VERSION = "5.0"
BASELINE_ID = "V5_H0"
STRATEGY_FAMILY = "EXHAUSTION_RECLAIM_MEAN_REVERSION"

SYMBOL = "BTCUSDC"
INTERVAL = "15m"

REFERENCE_LOOKBACK_BARS = 20
DEVIATION_MULTIPLE = "2"

ATR_PERIOD = 14
STOP_ATR_MULTIPLE = "2"
REWARD_RISK_RATIO = "2"
MAX_HOLD_BARS = 96
COOLDOWN_BARS = 4

FEE_BPS_PER_SIDE = "10"
SLIPPAGE_BPS_PER_SIDE = "2"

ELIGIBLE_WINDOWS = 11

HOLDOUT_START = "2025-08-01T00:00:00+00:00"
HOLDOUT_END = "2026-02-01T00:00:00+00:00"


def _definition() -> dict:
    return {
        "version": VERSION,
        "baseline_id": BASELINE_ID,
        "strategy_family": STRATEGY_FAMILY,
        "market": {
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "market_type": "BINANCE_PUBLIC_SPOT",
            "direction": "LONG_ONLY",
        },
        "dataset_policy": {
            "research_data_status": "CONSUMED_RESEARCH_DATA",
            "eligible_windows": ELIGIBLE_WINDOWS,
            "blind_holdout": {
                "status": "LOCKED_BLIND_HOLDOUT",
                "start": HOLDOUT_START,
                "end": HOLDOUT_END,
                "loaded": False,
                "revealed": False,
                "consumed": False,
                "evaluated": False,
            },
        },
        "reference_distribution": {
            "lookback_bars": REFERENCE_LOOKBACK_BARS,
            "candles": "PREVIOUS_EXACTLY_20_FULLY_CLOSED_CANDLES",
            "current_candle_excluded": True,
            "price_field": "CLOSE",
            "mean": "ARITHMETIC_MEAN_OF_PREVIOUS_20_CLOSES",
            "standard_deviation": "POPULATION_STANDARD_DEVIATION",
            "standard_deviation_ddof": 0,
            "deviation_multiple": DEVIATION_MULTIPLE,
            "lower_band": "REFERENCE_MEAN_MINUS_2_TIMES_REFERENCE_STD",
            "zero_standard_deviation_rule": "NO_SIGNAL",
        },
        "entry": {
            "mechanism": "DOWNSIDE_EXHAUSTION_LOWER_BAND_RECLAIM",
            "signal_candle": "CURRENT_FULLY_CLOSED_CANDLE",
            "all_conditions_required": True,
            "conditions": [
                "CURRENT_LOW_STRICTLY_LESS_THAN_LOWER_BAND",
                "CURRENT_CLOSE_STRICTLY_GREATER_THAN_LOWER_BAND",
                "CURRENT_CLOSE_STRICTLY_LESS_THAN_REFERENCE_MEAN",
            ],
            "strict_inequalities": True,
            "equality_qualifies": False,
            "execution": "NEXT_BAR_OPEN",
        },
        "filters": {
            "ema_filter": False,
            "rsi_filter": False,
            "volume_filter": False,
            "trend_regime_filter": False,
            "atr_percentile_filter": False,
            "breakout_filter": False,
        },
        "risk_and_exit": {
            "atr_period": ATR_PERIOD,
            "initial_stop": "ENTRY_MINUS_2_TIMES_SIGNAL_CANDLE_ATR14",
            "stop_atr_multiple": STOP_ATR_MULTIPLE,
            "reward_risk_ratio": REWARD_RISK_RATIO,
            "take_profit": "PLUS_2R",
            "maximum_hold_bars": MAX_HOLD_BARS,
            "cooldown_bars": COOLDOWN_BARS,
            "ambiguous_bar_policy": "STOP_FIRST",
            "trailing_stop": False,
            "break_even_stop": False,
            "early_failure_exit": False,
        },
        "costs": {
            "base": {
                "fee_bps_per_side": FEE_BPS_PER_SIDE,
                "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
            },
            "stress_2x": {
                "fee_bps_per_side": "20",
                "slippage_bps_per_side": "4",
            },
        },
        "evaluation": {
            "primary_metric": "NET_EXPECTANCY_R",
            "report": [
                "TOTAL_TRADES",
                "FRICTIONLESS_EXPECTANCY_R_PER_TRADE",
                "NET_EXPECTANCY_R_PER_TRADE",
                "PROFIT_FACTOR_R",
                "WIN_RATE",
                "MAXIMUM_DRAWDOWN_PERCENT",
                "POSITIVE_NET_WINDOWS_OUT_OF_11",
                "STRESS_2X_NET_EXPECTANCY_R",
                "TRADES_PER_DAY",
                "NET_R_PER_DAY",
            ],
        },
        "progression_gate": {
            "required_eligible_windows": 11,
            "combined_net_expectancy_r_gt": "0",
            "profit_factor_r_gt": "1",
            "positive_net_window_ratio_gte": "0.60",
            "minimum_positive_net_windows": 7,
            "all_conditions_required": True,
        },
        "forbidden_during_v5_h0": [
            "PARAMETER_SEARCH",
            "ALTERNATE_REFERENCE_LOOKBACKS",
            "ALTERNATE_DEVIATION_THRESHOLDS",
            "EMA_FILTER",
            "RSI_FILTER",
            "VOLUME_FILTER",
            "TREND_REGIME_FILTER",
            "ATR_PERCENTILE_FILTER",
            "BREAKOUT_FILTER",
            "TRAILING_STOP",
            "BREAK_EVEN_STOP",
            "EARLY_FAILURE_EXIT",
            "BLIND_HOLDOUT_ACCESS",
        ],
        "research_rule": (
            "V5_H0_IS_A_NEW_MEAN_REVERSION_MECHANISM_FAMILY_"
            "AND_NOT_A_V4_BREAKOUT_TUNING_ATTEMPT"
        ),
    }


def _canonical_json(payload: dict) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def build_manifest() -> dict:
    definition = _definition()
    digest = hashlib.sha256(
        _canonical_json(definition).encode("utf-8")
    ).hexdigest()
    return {
        "run_id": digest[:16],
        "definition_sha256": digest,
        **definition,
    }


def write_manifest(
    root: str | Path = "research/v5_mean_reversion",
) -> Path:
    manifest = build_manifest()
    output_dir = Path(root) / manifest["run_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    serialized = json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    ) + "\n"

    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(
                "Existing V5-H0 manifest differs from frozen definition."
            )
        return path

    path.write_text(serialized, encoding="utf-8")
    return path


def main() -> int:
    manifest = build_manifest()
    path = write_manifest()

    print()
    print("V5-H0 EXHAUSTION RECLAIM PREREGISTRATION")
    print(f"Run ID: {manifest['run_id']}")
    print("Market: BTCUSDC Binance Public Spot | 15m | LONG ONLY")
    print("Reference: previous 20 closed CLOSE prices | population std")
    print("Entry: downside exhaustion + strict lower-band reclaim")
    print("Execution: NEXT BAR OPEN | STOP_FIRST")
    print("Dataset: CONSUMED_RESEARCH_DATA | eligible windows: 11")
    print(
        "Blind holdout: LOCKED | NOT LOADED | "
        "NOT REVEALED | NOT CONSUMED | NOT EVALUATED"
    )
    print(f"Manifest: {path}")
    print()
    print("PREREGISTRATION COMPLETE — NO BACKTEST EXECUTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
