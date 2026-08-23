"""Execute the fixed read-only V7 order-flow signal diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path

from src.cli.run_multiregime import _require_locked_holdout
from src.cli.run_v33_h5 import _build_regions, _read_reference_windows
from src.cli.run_v6_signal_quality import verify_frozen_reproduction
from src.cli.run_v7_h0 import (
    EXPECTED_BUCKETS,
    EXPECTED_DATASET_ID,
    EXPECTED_DATASET_SHA256,
    EXPECTED_WINDOW_IDS,
    _load_json,
    validate_eligible_windows,
    validate_v7_dataset_manifest,
    validate_v7_h0_manifest,
)
from src.cli.run_v7_h0_replay import load_orderflow_buckets
from src.diagnostics.r_normalized import RNormalizedTrade, summarize_r_normalized_trades
from src.diagnostics.v7_orderflow_signal import (
    NUMERIC_FEATURES,
    build_causal_features,
    feature_window_consistency,
    summarize_feature_comparisons,
)
from src.hypotheses.manifest import HoldoutStatus, ResearchManifestStore
from src.research.multiregime.windows import construct_windows
from src.utils.logger import configure_logging, get_logger


V6_RUN_ID = "e1eef7bdd0c37ad4"
EXPECTED_CANDIDATES = 443
EXPECTED_RETAINED = 324
STEP = timedelta(minutes=15)
DIAGNOSTIC_DEFINITION = {
    "version": "v7-orderflow-signal-diagnostic-1",
    "v6_run_id": V6_RUN_ID,
    "dataset_id": EXPECTED_DATASET_ID,
    "dataset_sha256": EXPECTED_DATASET_SHA256,
    "sample": "EXACT_443_FROZEN_V6_ENTRY_SIGNALS",
    "buckets": ["t-3", "t-2", "t-1", "t"],
    "numeric_features": list(NUMERIC_FEATURES),
    "categorical_features": [
        "current_price_flow_alignment",
        "four_bar_price_flow_alignment",
        "sell_absorption_proxy",
        "buy_absorption_proxy",
    ],
    "threshold_search": False,
}
DIAGNOSTIC_ID = hashlib.sha256(
    json.dumps(DIAGNOSTIC_DEFINITION, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()[:16]


def _plain(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _r_record(row: dict) -> RNormalizedTrade:
    frictionless = Decimal(row["frictionless_r"])
    fee = Decimal(row["fee_r"])
    slippage = Decimal(row["slippage_r"])
    total = Decimal(row["total_friction_r"])
    net = Decimal(row["net_r"])
    if abs(frictionless - total - net) > Decimal("1e-24") or fee + slippage != total:
        raise ValueError("Frozen V6 ledger R accounting changed.")
    return RNormalizedTrade(
        trade_id=f"{row['window_id']}:{row['trade_id']}",
        initial_risk=Decimal("1"),
        frictionless_r=frictionless,
        gross_after_slippage_r=frictionless - slippage,
        fee_cost_r=fee,
        slippage_cost_r=slippage,
        total_friction_r=total,
        net_r=net,
    )


def load_frozen_v6_ledger(path: str | Path) -> tuple[dict, ...]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        raw = tuple(csv.DictReader(stream))
    if len(raw) != EXPECTED_CANDIDATES:
        raise ValueError("Frozen V6 candidate ledger must contain exactly 443 trades.")
    rows = []
    keys = set()
    for source in raw:
        key = (source["window_id"], source["trade_id"])
        if key in keys:
            raise ValueError("Frozen V6 candidate ledger contains duplicate trades.")
        keys.add(key)
        row = dict(source)
        row.update(
            {
                "entry_signal_time": datetime.fromisoformat(source["entry_signal_time"]).astimezone(timezone.utc),
                "frictionless_r": Decimal(source["frictionless_r"]),
                "net_r": Decimal(source["net_r"]),
                "fee_r": Decimal(source["fee_r"]),
                "slippage_r": Decimal(source["slippage_r"]),
                "total_friction_r": Decimal(source["total_friction_r"]),
                "mfe_r": Decimal(source["mfe_r"]),
                "mae_r": Decimal(source["mae_r"]),
            }
        )
        rows.append(row)
    records = tuple(_r_record(row) for row in rows)
    summary = summarize_r_normalized_trades(records)
    positive_windows = sum(
        summarize_r_normalized_trades(
            tuple(record for row, record in zip(rows, records, strict=True) if row["window_id"] == window_id)
        ).net_expectancy_r
        > 0
        for window_id in sorted({row["window_id"] for row in rows})
    )
    verify_frozen_reproduction(summary=summary, positive_windows=positive_windows)
    if tuple(sorted({row["window_id"] for row in rows})) != EXPECTED_WINDOW_IDS:
        raise ValueError("Frozen V6 ledger does not contain the exact 11 windows.")
    return tuple(rows)


def _records(rows: tuple[dict, ...]) -> tuple[RNormalizedTrade, ...]:
    return tuple(_r_record(row) for row in rows)


def summarize_group(rows: tuple[dict, ...]) -> dict:
    summary = summarize_r_normalized_trades(_records(rows))
    sample_windows = sorted({row["window_id"] for row in rows})
    positive_windows = sum(
        summarize_r_normalized_trades(
            _records(tuple(row for row in rows if row["window_id"] == window_id))
        ).net_expectancy_r
        > 0
        for window_id in sample_windows
    )
    return {
        "trades": summary.trade_count,
        "frictionless_expectancy_r": summary.frictionless_expectancy_r,
        "net_expectancy_r": summary.net_expectancy_r,
        "profit_factor_r": summary.profit_factor_r,
        "win_rate_percent": summary.win_rate_percent,
        "average_winner_r": summary.average_winner_r,
        "average_loser_r": summary.average_loser_r,
        "sample_windows": len(sample_windows),
        "positive_net_windows": positive_windows,
    }


def categorical_summaries(rows: tuple[dict, ...]) -> tuple[dict, ...]:
    dimensions = (
        (
            "current_price_flow_alignment",
            ("PRICE_UP_FLOW_BUY", "PRICE_UP_FLOW_SELL", "PRICE_DOWN_FLOW_BUY", "PRICE_DOWN_FLOW_SELL", "NEUTRAL"),
        ),
        (
            "four_bar_price_flow_alignment",
            ("PRICE_UP_FLOW_BUY", "PRICE_UP_FLOW_SELL", "PRICE_DOWN_FLOW_BUY", "PRICE_DOWN_FLOW_SELL", "NEUTRAL"),
        ),
        ("sell_absorption_proxy", (True, False)),
        ("buy_absorption_proxy", (True, False)),
    )
    output = []
    for dimension, states in dimensions:
        for state in states:
            selected = tuple(row for row in rows if row[dimension] == state)
            if selected:
                output.append(
                    {
                        "dimension": dimension,
                        "state": str(state).upper() if isinstance(state, bool) else state,
                        "exploratory_only": True,
                        **summarize_group(selected),
                    }
                )
    return tuple(output)


def v7_zero_boundary_groups(rows: tuple[dict, ...]) -> dict:
    retained = tuple(row for row in rows if row["qimb_0"] > 0)
    filtered = tuple(row for row in rows if row["qimb_0"] <= 0)
    if len(retained) != EXPECTED_RETAINED or len(filtered) != EXPECTED_CANDIDATES - EXPECTED_RETAINED:
        raise ValueError("V7-H0 retained/filtered candidate reproduction changed.")
    return {
        "retained_qimb_0_gt_0": summarize_group(retained),
        "filtered_qimb_0_lte_0": summarize_group(filtered),
    }


def classify_evidence(
    comparisons: tuple[dict, ...], consistency: tuple[dict, ...]
) -> dict:
    """Apply a fixed descriptive rubric, never a trading cutoff."""

    consistency_by_feature = {row["feature"]: row for row in consistency}
    current_only = {
        "qimb_0",
        "buy_quote_ratio_0",
        "signed_quote_0",
        "total_quote_volume_0",
        "average_aggtrade_quote_size_0",
        "signal_return",
    }
    candidates = tuple(row for row in comparisons if row["feature"] not in current_only)
    meaningful = tuple(
        row
        for row in candidates
        if row["cliffs_delta"] is not None and abs(row["cliffs_delta"]) >= Decimal("0.147")
    )
    stable = tuple(
        row
        for row in meaningful
        if consistency_by_feature[row["feature"]]["consistency_ratio"] is not None
        and consistency_by_feature[row["feature"]]["consistency_ratio"] >= Decimal("0.60")
    )
    if stable:
        classification = "ORDERFLOW_SIGNAL_EVIDENCE_FOUND"
        best = max(stable, key=lambda row: abs(row["cliffs_delta"]))["feature"]
        if best in {"qimb_delta_1", "qimb_slope_4", "qimb_1"}:
            recommendation = "FLOW_ACCELERATION_CONFIRMATION"
        else:
            recommendation = "MULTIBAR_FLOW_CONFIRMATION"
    elif meaningful:
        classification = "ORDERFLOW_SIGNAL_EVIDENCE_WEAK"
        recommendation = None
    else:
        classification = "AGGTRADES_SIGNAL_EXHAUSTED"
        recommendation = None
    return {
        "classification": classification,
        "rubric": {
            "meaningful_absolute_cliffs_delta": "0.147",
            "clear_majority_consistency_ratio": "0.60",
            "note": "Descriptive evidence rubric only; not a trading threshold.",
        },
        "stable_features": [row["feature"] for row in stable],
        "one_conceptual_next_hypothesis": recommendation,
        "recommendation": (
            "Test only the named concept in a new preregistration; choose no cutoff here."
            if recommendation
            else "Stop aggTrades-only signal research and consider richer order-book/liquidity data."
        ),
    }


def cost_context(rows: tuple[dict, ...]) -> dict:
    summary = summarize_r_normalized_trades(_records(rows))
    return {
        "gross_edge_r_per_trade": summary.frictionless_expectancy_r,
        "average_friction_r_per_trade": summary.average_total_friction_r,
        "net_edge_r_per_trade": summary.net_expectancy_r,
        "friction_to_gross_edge_ratio": (
            summary.average_total_friction_r / abs(summary.frictionless_expectancy_r)
            if summary.frictionless_expectancy_r != 0
            else None
        ),
        "gross_expectancy_improvement_required_r": {
            "target_net_0": -summary.net_expectancy_r,
            "target_net_0_10": Decimal("0.10") - summary.net_expectancy_r,
            "target_net_0_25": Decimal("0.25") - summary.net_expectancy_r,
        },
    }


def build_summary(
    *, rows: tuple[dict, ...], comparisons: tuple[dict, ...], consistency: tuple[dict, ...], categories: tuple[dict, ...]
) -> dict:
    top = sorted(
        comparisons,
        key=lambda row: abs(row["cliffs_delta"]) if row["cliffs_delta"] is not None else Decimal("-1"),
        reverse=True,
    )[:5]
    consistency_by_feature = {row["feature"]: row for row in consistency}
    strongest_categories = sorted(
        categories, key=lambda row: row["net_expectancy_r"], reverse=True
    )[:5]
    return {
        "diagnostic_id": DIAGNOSTIC_ID,
        "definition": DIAGNOSTIC_DEFINITION,
        "source_v6_run_id": V6_RUN_ID,
        "candidate_count": len(rows),
        "frozen_v6_reproduction": {"verified": True, **summarize_group(rows)},
        "top_5_features_by_absolute_cliffs_delta": [
            {**row, "window_consistency": consistency_by_feature[row["feature"]]}
            for row in top
        ],
        "categorical_states": categories,
        "strongest_categorical_observations": strongest_categories,
        "v7_h0_failure_explanation": {
            **v7_zero_boundary_groups(rows),
            "conclusion": "The natural qimb_0 > 0 boundary removed 119 trades but did not isolate a superior frozen V6 outcome distribution.",
        },
        "cost_context": cost_context(rows),
        "evidence": classify_evidence(comparisons, consistency),
        "integrity": {
            "v6_strategy_changed": False,
            "v7_strategy_changed": False,
            "strategy_replay_executed": False,
            "threshold_search_executed": False,
            "machine_learning_executed": False,
            "orderflow_dataset": {
                "dataset_id": EXPECTED_DATASET_ID,
                "definition_sha256": EXPECTED_DATASET_SHA256,
                "buckets": EXPECTED_BUCKETS,
            },
            "blind_holdout": {
                "status": "LOCKED_BLIND_HOLDOUT",
                "loaded": False,
                "revealed": False,
                "consumed": False,
                "evaluated": False,
            },
        },
    }


def _write_csv(path: Path, rows: tuple[dict, ...]) -> None:
    if not rows:
        raise ValueError(f"Diagnostic CSV rows are empty: {path.name}.")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(_plain(rows))
    temporary.replace(path)


def write_reports(
    *, output: Path, summary: dict, ledger: tuple[dict, ...], comparisons: tuple[dict, ...], consistency: tuple[dict, ...], categories: tuple[dict, ...]
) -> tuple[Path, ...]:
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise ValueError("V7 order-flow signal diagnostic exists; refusing overwrite.")
    output.mkdir(parents=True, exist_ok=True)
    files = (
        (output / "candidate_ledger.csv", ledger),
        (output / "feature_comparison.csv", comparisons),
        (output / "feature_window_consistency.csv", consistency),
        (output / "categorical_states.csv", categories),
    )
    for path, rows in files:
        _write_csv(path, rows)
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(_plain(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    return (summary_path, *(path for path, _ in files))


def _validate_holdout(root_manifest) -> None:
    _require_locked_holdout(root_manifest)
    if (
        root_manifest.holdout_status is not HoldoutStatus.LOCKED_BLIND_HOLDOUT
        or root_manifest.blind_holdout.reveal_timestamp is not None
        or root_manifest.blind_holdout.consumed_timestamp is not None
    ):
        raise ValueError("Blind holdout contamination detected.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the fixed read-only V7 signal diagnostic.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--v6-ledger", default="reports/diagnostics/v6_signal_quality/e1eef7bdd0c37ad4/trade_ledger.csv")
    parser.add_argument("--v6-report", default="reports/v6_mtf_continuation/e1eef7bdd0c37ad4/summary.json")
    parser.add_argument("--v7-manifest", default="research/v7_orderflow/74b2458cb2812de2/manifest.json")
    parser.add_argument("--dataset-manifest", default="data/orderflow/aggregated/15m/BTCUSDC/dataset_manifest.json")
    parser.add_argument("--orderflow-root", default="data/orderflow/aggregated/15m")
    parser.add_argument("--root-manifest", default="research/hypothesis_manifest.json")
    parser.add_argument("--mechanism-report", default="reports/mechanisms/bc2496aed05555b5")
    parser.add_argument("--expansion-data-root", default="data/historical/multiregime_expansion")
    parser.add_argument("--consumed-data-root", default="data/historical")
    parser.add_argument("--output-root", default="reports/diagnostics/v7_orderflow_signal")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    logger = get_logger(__name__)
    if not args.execute:
        logger.error("Diagnostic not executed. Explicit --execute is required.")
        return 1
    try:
        v6_report = _load_json(args.v6_report)
        if v6_report.get("run_id") != V6_RUN_ID:
            raise ValueError("Frozen V6 report run ID changed.")
        v7_manifest = _load_json(args.v7_manifest)
        dataset_manifest = _load_json(args.dataset_manifest)
        validate_v7_h0_manifest(v7_manifest)
        validate_v7_dataset_manifest(dataset_manifest)
        root_manifest = ResearchManifestStore(args.root_manifest).load()
        _validate_holdout(root_manifest)
        reference_windows = _read_reference_windows(Path(args.mechanism_report))
        validate_eligible_windows(reference_windows)

        ledger = load_frozen_v6_ledger(args.v6_ledger)
        from src.config.settings import load_settings

        regions = _build_regions(
            root_manifest=root_manifest,
            expansion_data_root=args.expansion_data_root,
            consumed_data_root=args.consumed_data_root,
        )
        windows = construct_windows(regions, load_settings().multiregime_config())
        eligible = tuple(window for window in windows if window.window_id in reference_windows)
        if tuple(sorted(window.window_id for window in eligible)) != EXPECTED_WINDOW_IDS:
            raise ValueError("Exactly 11 consumed research windows are required.")
        candles_by_window = {
            window.window_id: {candle.timestamp: candle for candle in window.replay_dataset.candles}
            for window in eligible
        }
        buckets = load_orderflow_buckets(dataset_manifest=dataset_manifest, root=args.orderflow_root)
        bucket_index = {bucket.bucket_open_time: bucket for bucket in buckets}

        candidates = []
        for frozen in ledger:
            signal_open = frozen["entry_signal_time"] - STEP
            try:
                flow = tuple(bucket_index[signal_open - offset * STEP] for offset in (3, 2, 1, 0))
                candles = candles_by_window[frozen["window_id"]]
                features = build_causal_features(
                    buckets=flow,
                    signal_candle=candles[signal_open],
                    t_minus_4_candle=candles[signal_open - 4 * STEP],
                )
            except KeyError as exc:
                raise ValueError("Exact causal order-flow/candle coverage failed.") from exc
            candidates.append(
                {
                    "window_id": frozen["window_id"],
                    "trade_id": frozen["trade_id"],
                    "entry_signal_time": frozen["entry_signal_time"],
                    "signal_bucket_open_time": signal_open,
                    **features,
                    "outcome_label": "WINNER" if frozen["net_r"] > 0 else "LOSER",
                    "frictionless_r": frozen["frictionless_r"],
                    "net_r": frozen["net_r"],
                    "fee_r": frozen["fee_r"],
                    "slippage_r": frozen["slippage_r"],
                    "total_friction_r": frozen["total_friction_r"],
                    "exit_reason": frozen["exit_reason"],
                    "mfe_r": frozen["mfe_r"],
                    "mae_r": frozen["mae_r"],
                }
            )
        rows = tuple(candidates)
        if len(rows) != EXPECTED_CANDIDATES:
            raise ValueError("Exact 443-candidate join failed.")
        comparisons = summarize_feature_comparisons(rows)
        consistency = feature_window_consistency(rows, comparisons)
        categories = categorical_summaries(rows)
        summary = build_summary(
            rows=rows,
            comparisons=comparisons,
            consistency=consistency,
            categories=categories,
        )
        output = Path(args.output_root) / DIAGNOSTIC_ID
        paths = write_reports(
            output=output,
            summary=summary,
            ledger=rows,
            comparisons=comparisons,
            consistency=consistency,
            categories=categories,
        )
        logger.info("Frozen V6 reproduction: 443 trades VERIFIED")
        logger.info("Classification: %s", summary["evidence"]["classification"])
        for path in paths:
            logger.info("Report: %s", path.resolve())
        logger.warning("Blind holdout: LOCKED | NOT LOADED | NOT REVEALED | NOT CONSUMED | NOT EVALUATED")
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.error("V7 order-flow signal diagnostic failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
