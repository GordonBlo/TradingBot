"""Dry-run-only integrity guard for the frozen V4-H0 replay."""

from __future__ import annotations

import argparse
import inspect
import json
from decimal import Decimal
from pathlib import Path

from src.backtest.models import (
    AmbiguousBarPolicy,
    BacktestConfig,
    ExecutionTiming,
)
from src.cli.run_v33_h5 import _read_reference_windows
from src.research.v4_breakout_preregistration import build_manifest
from src.research.v4_breakout_stability import REQUIRED_ELIGIBLE_WINDOWS
from src.strategy.models import TrendMomentumConfig
from src.strategy.v4_breakout import V4BreakoutStrategy


EXPECTED_RUN_ID = "ba79975004b106a5"


def _load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required V4-H0 manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_v4_h0_manifest(payload: dict) -> None:
    if payload.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("Unexpected V4-H0 preregistration run ID.")
    if payload != build_manifest():
        raise ValueError("V4-H0 manifest differs from the frozen definition.")


def validate_v4_h0_implementation(manifest: dict) -> None:
    entry = manifest["entry"]
    risk = manifest["risk_and_exit"]
    filters = manifest["filters"]
    policy = manifest["dataset_policy"]

    if manifest["market"] != {
        "symbol": "BTCUSDC",
        "interval": "15m",
        "market_type": "BINANCE_PUBLIC_SPOT",
        "direction": "LONG_ONLY",
    }:
        raise ValueError("V4-H0 market definition changed.")
    if entry["lookback_bars"] != V4BreakoutStrategy.BREAKOUT_LOOKBACK_BARS:
        raise ValueError("V4-H0 breakout lookback changed.")
    if entry["current_candle_excluded_from_reference"] is not True:
        raise ValueError("V4-H0 current candle must be excluded.")
    if entry["trigger"] != "CURRENT_CLOSED_CANDLE_CLOSE_GREATER_THAN_REFERENCE_HIGH":
        raise ValueError("V4-H0 breakout trigger changed.")

    config = TrendMomentumConfig()
    if config.atr_period != risk["atr_period"]:
        raise ValueError("V4-H0 ATR period changed.")
    if config.atr_stop_multiplier != Decimal(risk["stop_atr_multiple"]):
        raise ValueError("V4-H0 stop multiple changed.")
    if config.reward_risk_ratio != Decimal(risk["reward_risk_ratio"]):
        raise ValueError("V4-H0 reward/risk ratio changed.")
    if config.maximum_bars_in_position != risk["maximum_hold_bars"]:
        raise ValueError("V4-H0 maximum hold changed.")
    if config.cooldown_bars != risk["cooldown_bars"]:
        raise ValueError("V4-H0 cooldown changed.")

    execution = BacktestConfig()
    if entry["execution"] != ExecutionTiming.NEXT_BAR_OPEN.value:
        raise ValueError("V4-H0 execution timing changed.")
    if execution.execution_timing is not ExecutionTiming.NEXT_BAR_OPEN:
        raise ValueError("Backtest next-bar-open execution changed.")
    if risk["ambiguous_bar_policy"] != AmbiguousBarPolicy.STOP_FIRST.value:
        raise ValueError("V4-H0 ambiguity policy changed.")
    if execution.ambiguous_bar_policy is not AmbiguousBarPolicy.STOP_FIRST:
        raise ValueError("Backtest STOP_FIRST behavior changed.")

    if any(filters.get(name) is not False for name in filters):
        raise ValueError("V4-H0 old entry filters must remain absent.")
    if policy["research_data_status"] != "CONSUMED_RESEARCH_DATA":
        raise ValueError("V4-H0 dataset policy changed.")
    if policy["eligible_windows"] != REQUIRED_ELIGIBLE_WINDOWS:
        raise ValueError("V4-H0 eligible window count changed.")
    holdout = policy["blind_holdout"]
    if holdout["status"] != "LOCKED_BLIND_HOLDOUT" or any(
        holdout[name] is not False
        for name in ("loaded", "revealed", "consumed", "evaluated")
    ):
        raise ValueError("V4-H0 blind holdout integrity failed.")

    source = inspect.getsource(V4BreakoutStrategy)
    if "[-required_history:-1]" not in source:
        raise ValueError("V4-H0 reference includes the current candle.")
    if "if close <= reference_high:" not in source:
        raise ValueError("V4-H0 breakout is not strictly greater-than.")
    if any(
        term in source
        for term in ("ema_", ".rsi", "volume_ratio", "atr_percent")
    ):
        raise ValueError("V4-H0 implementation contains an old entry filter.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run-only V4-H0 integrity guard."
    )
    parser.add_argument(
        "--manifest",
        default="research/v4_breakout/ba79975004b106a5/manifest.json",
    )
    parser.add_argument(
        "--mechanism-report",
        default="reports/mechanisms/bc2496aed05555b5",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate V4-H0 only; replay execution remains disabled.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dry_run:
        raise SystemExit(
            "V4-H0 REPLAY DISABLED. Run with --dry-run only."
        )

    manifest = _load_json(args.manifest)
    validate_v4_h0_manifest(manifest)
    validate_v4_h0_implementation(manifest)
    reference_windows = _read_reference_windows(Path(args.mechanism_report))
    if len(reference_windows) != REQUIRED_ELIGIBLE_WINDOWS:
        raise ValueError("V4-H0 requires exactly 11 eligible research windows.")

    print()
    print("V4-H0 DRY-RUN INTEGRITY CHECK")
    print(f"Preregistration: {EXPECTED_RUN_ID} VERIFIED")
    print("Market: BTCUSDC Spot | 15m | LONG only")
    print("Entry: previous 20 closed highs | current excluded | strict >")
    print("Risk: ATR14 | 2x stop | +2R | 96-bar hold | 4-bar cooldown")
    print("Execution: NEXT_BAR_OPEN | STOP_FIRST")
    print("Filters: EMA/RSI/volume/ATR-percentile absent")
    print("Dataset: CONSUMED_RESEARCH_DATA | eligible windows: 11")
    print("Blind holdout: LOCKED | NOT LOADED | NOT REVEALED | NOT CONSUMED | NOT EVALUATED")
    print()
    print("DRY RUN PASSED — NO V4-H0 REPLAY EXECUTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
