"""Preregister V4-H0: 15m price-breakout baseline.

This module defines research intent only.
It does not load market data and does not run a backtest.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


VERSION = "4.0"
BASELINE_ID = "V4_H0"
STRATEGY_FAMILY = "PRICE_BREAKOUT"

SYMBOL = "BTCUSDC"
INTERVAL = "15m"

BREAKOUT_LOOKBACK_BARS = 20
ATR_PERIOD = 14
STOP_ATR_MULTIPLE = "2"
REWARD_RISK_RATIO = "2"

MAX_HOLD_BARS = 96
COOLDOWN_BARS = 4

FEE_BPS_PER_SIDE = "10"
SLIPPAGE_BPS_PER_SIDE = "2"

RISK_PER_TRADE_PERCENT = "0.50"
MAX_CAPITAL_USDC = "50"

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
            "research_data_status": (
                "CONSUMED_RESEARCH_DATA"
            ),
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

        "entry": {
            "mechanism": "PRICE_BREAKOUT",
            "lookback_bars": BREAKOUT_LOOKBACK_BARS,
            "reference": (
                "HIGHEST_HIGH_OF_PREVIOUS_20_"
                "FULLY_CLOSED_CANDLES"
            ),
            "current_candle_excluded_from_reference": True,
            "trigger": (
                "CURRENT_CLOSED_CANDLE_CLOSE_"
                "GREATER_THAN_REFERENCE_HIGH"
            ),
            "execution": "NEXT_BAR_OPEN",
        },

        "filters": {
            "ema_filter": False,
            "rsi_filter": False,
            "volume_filter": False,
            "atr_percentile_filter": False,
            "trend_regime_filter": False,
        },

        "risk_and_exit": {
            "atr_period": ATR_PERIOD,
            "initial_stop": (
                "ENTRY_MINUS_2_TIMES_ATR14"
            ),
            "stop_atr_multiple": STOP_ATR_MULTIPLE,
            "reward_risk_ratio": REWARD_RISK_RATIO,
            "take_profit": "PLUS_2R",
            "maximum_hold_bars": MAX_HOLD_BARS,
            "cooldown_bars": COOLDOWN_BARS,
            "ambiguous_bar_policy": "STOP_FIRST",
        },

        "capital": {
            "risk_per_trade_percent": (
                RISK_PER_TRADE_PERCENT
            ),
            "max_capital_usdc": MAX_CAPITAL_USDC,
            "position_sizing": (
                "EXISTING_BACKTEST_POSITION_SIZING"
            ),
        },

        "costs": {
            "base": {
                "fee_bps_per_side": FEE_BPS_PER_SIDE,
                "slippage_bps_per_side": (
                    SLIPPAGE_BPS_PER_SIDE
                ),
            },
            "stress_2x": {
                "fee_bps_per_side": "20",
                "slippage_bps_per_side": "4",
            },
        },

        "evaluation": {
            "primary_metric": "NET_EXPECTANCY_R",
            "report": [
                "TRADES",
                "FRICTIONLESS_EXPECTANCY_R",
                "NET_EXPECTANCY_R",
                "PROFIT_FACTOR_R",
                "POSITIVE_NET_WINDOWS",
                "MAXIMUM_DRAWDOWN_PERCENT",
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
            "all_conditions_required": True,
        },

        "forbidden_during_v4_h0": [
            "BREAKOUT_LOOKBACK_PARAMETER_SEARCH",
            "ATR_PERIOD_PARAMETER_SEARCH",
            "STOP_MULTIPLE_PARAMETER_SEARCH",
            "TAKE_PROFIT_PARAMETER_SEARCH",
            "EMA_FILTER",
            "RSI_FILTER",
            "VOLUME_FILTER",
            "ATR_PERCENTILE_FILTER",
            "TRAILING_STOP",
            "BREAK_EVEN_STOP",
            "EARLY_FAILURE_EXIT",
            "BLIND_HOLDOUT_ACCESS",
        ],

        "research_rule": (
            "V4_H0_IS_EVALUATED_ONCE_WITH_THE_FROZEN_"
            "DEFINITION_BEFORE_ANY_NEW_MECHANISM_IS_ADDED"
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
    root: str | Path = "research/v4_breakout",
) -> Path:
    manifest = build_manifest()

    output_dir = (
        Path(root)
        / manifest["run_id"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = output_dir / "manifest.json"

    serialized = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    if path.exists():
        existing = path.read_text(
            encoding="utf-8"
        )

        if existing != serialized:
            raise RuntimeError(
                "Existing V4-H0 manifest differs "
                "from frozen definition."
            )

        return path

    path.write_text(
        serialized,
        encoding="utf-8",
    )

    return path


def main() -> int:
    manifest = build_manifest()
    path = write_manifest()

    print()
    print("V4-H0 BREAKOUT PREREGISTRATION")
    print(f"Run ID: {manifest['run_id']}")
    print("Symbol: BTCUSDC")
    print("Interval: 15m")
    print("Breakout lookback: previous 20 closed bars")
    print("Entry: NEXT BAR OPEN")
    print("Stop: 2 x ATR14")
    print("Target: +2R")
    print("Policy: STOP_FIRST")
    print("Dataset: CONSUMED_RESEARCH_DATA")
    print(
        "Blind holdout: LOCKED | NOT LOADED | "
        "NOT REVEALED | NOT CONSUMED | NOT EVALUATED"
    )
    print(f"Manifest: {path}")
    print()
    print(
        "PREREGISTRATION COMPLETE — "
        "NO BACKTEST EXECUTED"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())