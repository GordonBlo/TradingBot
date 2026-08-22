"""Dry-run-only integrity guard for the frozen V5-H0 replay."""

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
from src.research.v5_mean_reversion_preregistration import build_manifest
from src.research.v5_mean_reversion_stability import REQUIRED_ELIGIBLE_WINDOWS
from src.strategy.models import TrendMomentumConfig
from src.strategy.v5_mean_reversion import V5MeanReversionStrategy


EXPECTED_RUN_ID = "b645b6c1d251f374"


def _load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required V5-H0 manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_v5_h0_manifest(payload: dict) -> None:
    if payload.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("Unexpected V5-H0 preregistration run ID.")
    if payload != build_manifest():
        raise ValueError("V5-H0 manifest differs from the frozen definition.")


def validate_v5_h0_implementation(manifest: dict) -> None:
    reference = manifest["reference_distribution"]
    entry = manifest["entry"]
    risk = manifest["risk_and_exit"]
    filters = manifest["filters"]
    costs = manifest["costs"]
    policy = manifest["dataset_policy"]

    if manifest["market"] != {
        "symbol": "BTCUSDC",
        "interval": "15m",
        "market_type": "BINANCE_PUBLIC_SPOT",
        "direction": "LONG_ONLY",
    }:
        raise ValueError("V5-H0 market definition changed.")
    if reference != {
        "lookback_bars": 20,
        "candles": "PREVIOUS_EXACTLY_20_FULLY_CLOSED_CANDLES",
        "current_candle_excluded": True,
        "price_field": "CLOSE",
        "mean": "ARITHMETIC_MEAN_OF_PREVIOUS_20_CLOSES",
        "standard_deviation": "POPULATION_STANDARD_DEVIATION",
        "standard_deviation_ddof": 0,
        "deviation_multiple": "2",
        "lower_band": "REFERENCE_MEAN_MINUS_2_TIMES_REFERENCE_STD",
        "zero_standard_deviation_rule": "NO_SIGNAL",
    }:
        raise ValueError("V5-H0 reference distribution changed.")
    if entry != {
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
    }:
        raise ValueError("V5-H0 entry definition changed.")

    constants = {
        "REFERENCE_LOOKBACK_BARS": reference["lookback_bars"],
        "STANDARD_DEVIATION_DDOF": reference["standard_deviation_ddof"],
        "LOWER_BAND_STANDARD_DEVIATIONS": Decimal(reference["deviation_multiple"]),
        "ATR_PERIOD": risk["atr_period"],
        "ATR_STOP_MULTIPLIER": Decimal(risk["stop_atr_multiple"]),
        "REWARD_RISK_RATIO": Decimal(risk["reward_risk_ratio"]),
        "MAXIMUM_HOLD_BARS": risk["maximum_hold_bars"],
        "COOLDOWN_BARS": risk["cooldown_bars"],
    }
    if any(
        getattr(V5MeanReversionStrategy, name) != expected
        for name, expected in constants.items()
    ):
        raise ValueError("V5-H0 strategy constants changed.")

    config = TrendMomentumConfig()
    if (
        config.atr_period != risk["atr_period"]
        or config.atr_stop_multiplier != Decimal(risk["stop_atr_multiple"])
        or config.reward_risk_ratio != Decimal(risk["reward_risk_ratio"])
        or config.maximum_bars_in_position != risk["maximum_hold_bars"]
        or config.cooldown_bars != risk["cooldown_bars"]
    ):
        raise ValueError("V5-H0 frozen risk configuration changed.")
    if risk["initial_stop"] != "ENTRY_MINUS_2_TIMES_SIGNAL_CANDLE_ATR14":
        raise ValueError("V5-H0 initial stop definition changed.")
    if risk["take_profit"] != "PLUS_2R":
        raise ValueError("V5-H0 target definition changed.")
    if any(
        risk[name] is not False
        for name in ("trailing_stop", "break_even_stop", "early_failure_exit")
    ):
        raise ValueError("V5-H0 forbidden exit logic is enabled.")

    execution = BacktestConfig()
    if entry["execution"] != ExecutionTiming.NEXT_BAR_OPEN.value:
        raise ValueError("V5-H0 execution timing changed.")
    if execution.execution_timing is not ExecutionTiming.NEXT_BAR_OPEN:
        raise ValueError("Backtest next-bar-open execution changed.")
    if risk["ambiguous_bar_policy"] != AmbiguousBarPolicy.STOP_FIRST.value:
        raise ValueError("V5-H0 ambiguity policy changed.")
    if execution.ambiguous_bar_policy is not AmbiguousBarPolicy.STOP_FIRST:
        raise ValueError("Backtest STOP_FIRST behavior changed.")

    if any(filters.get(name) is not False for name in filters):
        raise ValueError("V5-H0 entry filters must remain absent.")
    if costs != {
        "base": {
            "fee_bps_per_side": "10",
            "slippage_bps_per_side": "2",
        },
        "stress_2x": {
            "fee_bps_per_side": "20",
            "slippage_bps_per_side": "4",
        },
    }:
        raise ValueError("V5-H0 cost definition changed.")
    if execution.fee_bps != Decimal("10") or execution.slippage_bps != Decimal("2"):
        raise ValueError("Backtest base costs changed.")

    if policy["research_data_status"] != "CONSUMED_RESEARCH_DATA":
        raise ValueError("V5-H0 dataset policy changed.")
    if policy["eligible_windows"] != REQUIRED_ELIGIBLE_WINDOWS:
        raise ValueError("V5-H0 eligible window count changed.")
    holdout = policy["blind_holdout"]
    if holdout["status"] != "LOCKED_BLIND_HOLDOUT" or any(
        holdout[name] is not False
        for name in ("loaded", "revealed", "consumed", "evaluated")
    ):
        raise ValueError("V5-H0 blind holdout integrity failed.")

    source = inspect.getsource(V5MeanReversionStrategy)
    required_source = (
        "[-required_history:-1]",
        "candle.close for candle in reference_candles",
        "current.low < lower_band",
        "current.close > lower_band",
        "current.close < reference_mean",
        ") / Decimal(self.REFERENCE_LOOKBACK_BARS)",
    )
    if any(fragment not in source for fragment in required_source):
        raise ValueError("V5-H0 causal strict signal implementation changed.")
    if any(
        term in source
        for term in (
            "ema_",
            ".rsi",
            "volume_ratio",
            "trend_regime",
            "atr_percent",
            "breakout_filter",
            "trailing_stop",
            "break_even",
            "early_failure",
        )
    ):
        raise ValueError("V5-H0 implementation contains forbidden logic.")


def validate_eligible_windows(reference_windows: dict) -> None:
    if len(reference_windows) != REQUIRED_ELIGIBLE_WINDOWS:
        raise ValueError("V5-H0 requires exactly 11 eligible research windows.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run-only V5-H0 integrity guard."
    )
    parser.add_argument(
        "--manifest",
        default="research/v5_mean_reversion/b645b6c1d251f374/manifest.json",
    )
    parser.add_argument(
        "--mechanism-report",
        default="reports/mechanisms/bc2496aed05555b5",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate V5-H0 only; replay execution remains disabled.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dry_run:
        raise SystemExit("V5-H0 REPLAY DISABLED. Run with --dry-run only.")

    manifest = _load_json(args.manifest)
    validate_v5_h0_manifest(manifest)
    validate_v5_h0_implementation(manifest)
    validate_eligible_windows(
        _read_reference_windows(Path(args.mechanism_report))
    )

    print()
    print("V5-H0 DRY-RUN INTEGRITY CHECK")
    print(f"Preregistration: {EXPECTED_RUN_ID} VERIFIED")
    print("Market: BTCUSDC Binance Public Spot | 15m | LONG only")
    print("Reference: previous 20 closed closes | current excluded | population std")
    print("Entry: strict low pierce + close reclaim below reference mean")
    print("Risk: ATR14 | 2x stop | +2R | 96-bar hold | 4-bar cooldown")
    print("Execution: NEXT_BAR_OPEN | STOP_FIRST")
    print("Costs: base 10/2 bps | stress 20/4 bps per side")
    print("Filters and additional exit logic: absent")
    print("Dataset: CONSUMED_RESEARCH_DATA | eligible windows: 11")
    print("Blind holdout: LOCKED | NOT LOADED | NOT REVEALED | NOT CONSUMED | NOT EVALUATED")
    print()
    print("DRY RUN PASSED — NO V5-H0 REPLAY EXECUTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
