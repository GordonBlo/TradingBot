"""Dry-run-only integrity guard for the frozen V7-H0 replay."""

from __future__ import annotations

import argparse
import inspect
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from src.cli.run_v33_h5 import _read_reference_windows
from src.cli.run_v6_h0 import validate_v6_h0_implementation
from src.orderflow.models import AggregateTrade, AggressorSide
from src.research.v6_mtf_continuation_preregistration import (
    build_manifest as build_v6_manifest,
)
from src.research.v7_orderflow_preregistration import build_manifest
from src.research.v7_orderflow_stability import REQUIRED_ELIGIBLE_WINDOWS
from src.strategy.models import TrendMomentumConfig
from src.strategy.v6_mtf_continuation import V6MTFContinuationStrategy
from src.strategy.v7_orderflow_confirmation import (
    V7OrderFlowConfirmationStrategy,
)


EXPECTED_RUN_ID = "74b2458cb2812de2"
EXPECTED_DATASET_ID = "8fdfcee8d68b6f48"
EXPECTED_DATASET_SHA256 = (
    "dbca5d298f870068cfdc99ad0d550eda3d0af52b452c484c2e6d1e80e3c79567"
)
EXPECTED_BUCKETS = 95_040
EXPECTED_WINDOW_IDS = tuple(
    [f"W{index:03d}" for index in range(2, 11)] + ["W012", "W013"]
)
HOLDOUT_START = datetime(2025, 8, 1, tzinfo=timezone.utc)
HOLDOUT_END = datetime(2026, 2, 1, tzinfo=timezone.utc)


def _load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required V7-H0 manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_v7_h0_manifest(payload: dict) -> None:
    if payload.get("run_id") != EXPECTED_RUN_ID:
        raise ValueError("Unexpected V7-H0 preregistration run ID.")
    dataset = payload.get("orderflow_dataset", {})
    policy = payload.get("dataset_policy", {})
    if (
        dataset.get("dataset_id") != EXPECTED_DATASET_ID
        or policy.get("only_dataset_id") != EXPECTED_DATASET_ID
    ):
        raise ValueError("V7-H0 preregistration dataset ID changed.")
    if (
        dataset.get("definition_sha256") != EXPECTED_DATASET_SHA256
        or policy.get("only_dataset_definition_sha256")
        != EXPECTED_DATASET_SHA256
    ):
        raise ValueError("V7-H0 preregistration dataset SHA-256 changed.")
    if payload != build_manifest():
        raise ValueError("V7-H0 manifest differs from the frozen definition.")


def _utc(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("V7 dataset coverage timestamps must be timezone-aware.")
    return timestamp.astimezone(timezone.utc)


def validate_v7_dataset_manifest(payload: dict) -> None:
    if payload.get("dataset_id") != EXPECTED_DATASET_ID:
        raise ValueError("Unexpected V7 order-flow dataset ID.")
    if payload.get("dataset_definition_sha256") != EXPECTED_DATASET_SHA256:
        raise ValueError("Unexpected V7 order-flow dataset SHA-256.")
    required = {
        "symbol": "BTCUSDC",
        "interval": "15m",
        "source": "BINANCE_PUBLIC_SPOT_AGGTRADES",
        "classification": "DATASET_READY",
        "aggregated_buckets": EXPECTED_BUCKETS,
    }
    if any(payload.get(name) != expected for name, expected in required.items()):
        raise ValueError("V7 order-flow dataset market or coverage changed.")
    if payload.get("coverage") != {"missing": 0, "extra": 0, "duplicates": 0}:
        raise ValueError("V7 order-flow dataset coverage reconciliation changed.")

    reconciliation = payload.get("reconciliation", {})
    if any(
        (
            reconciliation.get("classification") != "VALIDATED",
            reconciliation.get("bucket_count") != EXPECTED_BUCKETS,
            reconciliation.get("kline_count") != EXPECTED_BUCKETS,
            reconciliation.get("conservation_identities_pass") is not True,
            reconciliation.get("reversed_side_fails") is not True,
        )
    ):
        raise ValueError("V7 order-flow 95,040/95,040 reconciliation changed.")

    holdout = payload.get("blind_holdout", {})
    if (
        holdout.get("status") != "LOCKED_BLIND_HOLDOUT"
        or holdout.get("intersection") != "NONE"
        or any(
            holdout.get(name) is not False
            for name in (
                "downloaded",
                "loaded",
                "revealed",
                "consumed",
                "evaluated",
            )
        )
    ):
        raise ValueError("V7 dataset blind holdout integrity failed.")

    covered_buckets = 0
    for raw_range in payload.get("safe_ranges", ()):
        if not isinstance(raw_range, list) or len(raw_range) != 2:
            raise ValueError("V7 dataset safe coverage range is invalid.")
        start, end = map(_utc, raw_range)
        if start >= end:
            raise ValueError("V7 dataset safe coverage range is invalid.")
        if start < HOLDOUT_END and HOLDOUT_START < end:
            raise ValueError("V7 dataset coverage overlaps the blind holdout.")
        seconds = Decimal(str((end - start).total_seconds()))
        buckets, remainder = divmod(seconds, Decimal(15 * 60))
        if remainder != 0:
            raise ValueError("V7 dataset coverage is not aligned to 15m.")
        covered_buckets += int(buckets)
    if covered_buckets != EXPECTED_BUCKETS:
        raise ValueError("V7 dataset safe coverage does not contain 95,040 buckets.")

    sample = {
        "aggregate_trade_id": 1,
        "price": Decimal("1"),
        "quantity": Decimal("1"),
        "first_trade_id": 1,
        "last_trade_id": 1,
        "timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc),
    }
    if (
        AggregateTrade(**sample, buyer_is_maker=False).aggressor_side
        is not AggressorSide.TAKER_BUY
        or AggregateTrade(**sample, buyer_is_maker=True).aggressor_side
        is not AggressorSide.TAKER_SELL
    ):
        raise ValueError("V7 aggTrade aggressor-side mapping changed.")


def _normalized_source(value: object) -> str:
    return " ".join(inspect.getsource(value).split())


def validate_v7_h0_implementation(manifest: dict) -> None:
    confirmation = manifest["orderflow_confirmation"]
    causality = manifest["causality"]
    relationship = manifest["reference_relationship"]
    policy = manifest["dataset_policy"]

    cls = V7OrderFlowConfirmationStrategy
    constants = {
        "V6_H0_RUN_ID": manifest["frozen_v6_reference"]["run_id"],
        "ORDERFLOW_DATASET_ID": manifest["orderflow_dataset"]["dataset_id"],
        "ORDERFLOW_DATASET_DEFINITION_SHA256": manifest["orderflow_dataset"][
            "definition_sha256"
        ],
        "SYMBOL": manifest["market"]["symbol"],
        "INTERVAL": manifest["market"]["execution_interval"],
        "QUOTE_IMBALANCE_THRESHOLD": Decimal(confirmation["threshold"]),
    }
    if any(getattr(cls, name) != expected for name, expected in constants.items()):
        raise ValueError("V7-H0 strategy constants changed.")

    if manifest["feature_scope"]["allowed"] != [
        "CURRENT_SIGNAL_CANDLE_QUOTE_VOLUME_AGGRESSOR_DIRECTION"
    ] or confirmation["new_feature_count"] != 1:
        raise ValueError("V7-H0 quote-volume-only feature scope changed.")
    if confirmation != build_manifest()["orderflow_confirmation"]:
        raise ValueError("V7-H0 strict order-flow threshold/feature changed.")
    if causality != build_manifest()["causality"]:
        raise ValueError("V7-H0 exact timestamp/causality semantics changed.")
    if relationship != build_manifest()["reference_relationship"]:
        raise ValueError("V7-H0 V6 subset invariant changed.")

    holdout = policy["blind_holdout"]
    if (
        policy["research_data_status"] != "CONSUMED_RESEARCH_DATA"
        or policy["eligible_windows"] != REQUIRED_ELIGIBLE_WINDOWS
        or _utc(holdout["start"]) != HOLDOUT_START
        or _utc(holdout["end"]) != HOLDOUT_END
        or holdout["status"] != "LOCKED_BLIND_HOLDOUT"
        or any(
            holdout[name] is not False
            for name in (
                "downloaded",
                "loaded",
                "revealed",
                "consumed",
                "evaluated",
            )
        )
    ):
        raise ValueError("V7-H0 blind holdout policy changed.")

    if not cls.quote_flow_confirms(
        taker_buy_quote_volume=Decimal("2"),
        taker_sell_quote_volume=Decimal("1"),
        total_quote_volume=Decimal("3"),
    ) or cls.quote_flow_confirms(
        taker_buy_quote_volume=Decimal("1"),
        taker_sell_quote_volume=Decimal("1"),
        total_quote_volume=Decimal("2"),
    ) or cls.quote_flow_confirms(
        taker_buy_quote_volume=Decimal("0"),
        taker_sell_quote_volume=Decimal("0"),
        total_quote_volume=Decimal("0"),
    ):
        raise ValueError("V7-H0 strict quote-flow comparison changed.")

    flow_source = _normalized_source(cls.quote_flow_confirms)
    evaluate_source = _normalized_source(cls.evaluate)
    required_flow = (
        "total_quote_volume > 0",
        "taker_buy_quote_volume > taker_sell_quote_volume",
        "/ total_quote_volume",
        "> V7OrderFlowConfirmationStrategy.QUOTE_IMBALANCE_THRESHOLD",
    )
    required_evaluate = (
        "v6_decision = self._v6_strategy.evaluate(context)",
        "if v6_decision.action is not StrategyAction.ENTER_LONG:",
        "signal_open_time = self._utc(context.current_candle.timestamp)",
        "self._buckets_by_open_time.get(signal_open_time)",
        "bucket.bucket_close_time > self._utc(context.timestamp)",
        "self._missing_bucket_timestamps.add(signal_open_time)",
        "return v6_decision",
    )
    if any(fragment not in flow_source for fragment in required_flow):
        raise ValueError("V7-H0 quote-volume feature implementation changed.")
    if any(fragment not in evaluate_source for fragment in required_evaluate):
        raise ValueError("V7-H0 exact-bucket causal implementation changed.")
    if "StrategyDecision(StrategyAction.ENTER_LONG" in evaluate_source:
        raise ValueError("V7-H0 can independently create a non-V6 BUY.")
    if any(
        term in flow_source
        for term in (
            "base_volume",
            "previous_bucket",
            "rolling",
            "trade_count",
        )
    ):
        raise ValueError("V7-H0 contains an unregistered order-flow feature.")

    strategy = cls(
        TrendMomentumConfig(),
        buckets=(),
        dataset_id=EXPECTED_DATASET_ID,
        dataset_definition_sha256=EXPECTED_DATASET_SHA256,
    )
    if not isinstance(strategy._v6_strategy, V6MTFContinuationStrategy):
        raise ValueError("V7-H0 is not structurally composed from frozen V6-H0.")

    validate_v6_h0_implementation(build_v6_manifest())


def validate_eligible_windows(reference_windows: dict[str, int]) -> None:
    if tuple(sorted(reference_windows)) != EXPECTED_WINDOW_IDS:
        raise ValueError(
            "V7-H0 requires exactly the 11 frozen eligible research windows."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run-only V7-H0 integrity guard.")
    parser.add_argument(
        "--manifest",
        default="research/v7_orderflow/74b2458cb2812de2/manifest.json",
    )
    parser.add_argument(
        "--dataset-manifest",
        default="data/orderflow/aggregated/15m/BTCUSDC/dataset_manifest.json",
    )
    parser.add_argument(
        "--mechanism-report",
        default="reports/mechanisms/bc2496aed05555b5",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate V7-H0 only; replay execution remains disabled.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.dry_run:
        raise SystemExit("V7-H0 REPLAY DISABLED. Run with --dry-run only.")

    manifest = _load_json(args.manifest)
    dataset_manifest = _load_json(args.dataset_manifest)
    validate_v7_h0_manifest(manifest)
    validate_v7_dataset_manifest(dataset_manifest)
    validate_v7_h0_implementation(manifest)
    validate_eligible_windows(
        _read_reference_windows(Path(args.mechanism_report))
    )

    print()
    print("V7-H0 DRY-RUN INTEGRITY CHECK")
    print(f"Preregistration: {EXPECTED_RUN_ID} VERIFIED")
    print(f"Order-flow dataset: {EXPECTED_DATASET_ID} | 95,040/95,040 VERIFIED")
    print("Entry: frozen V6 BUY plus exact closed signal-bucket quote imbalance > 0")
    print("Subset: V7 BUY timestamps structurally restricted to V6 BUY timestamps")
    print("Execution/risk: frozen V6 NEXT_BAR_OPEN | fill floor | +2R | STOP_FIRST")
    print("Dataset: 11 frozen CONSUMED_RESEARCH_DATA windows")
    print(
        "Blind holdout: LOCKED | NOT DOWNLOADED | NOT LOADED | "
        "NOT REVEALED | NOT CONSUMED | NOT EVALUATED"
    )
    print()
    print("DRY RUN PASSED — NO V7-H0 REPLAY EXECUTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
