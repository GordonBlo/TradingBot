"""Run read-only signal-quality diagnostics for frozen V6-H0."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from decimal import Decimal
from enum import Enum
from pathlib import Path

from src.cli.run_multiregime import _require_locked_holdout
from src.cli.run_v33_h5 import _build_regions, _read_reference_windows
from src.cli.run_v6_h0 import (
    EXPECTED_RUN_ID,
    _load_json,
    validate_v6_h0_implementation,
    validate_v6_h0_manifest,
)
from src.cli.run_v6_h0_replay import (
    _evaluate_window,
    expand_complete_region_history,
    prepare_v6_region_contexts,
    prepared_context_for_window,
    summarize_combined_records,
    validate_eligible_windows,
)
from src.diagnostics.r_normalized import RNormalizedSummary, RNormalizedTrade
from src.diagnostics.v6_signal_quality import (
    FEATURE_FIELDS,
    build_trade_ledger,
    cost_diagnostics,
    ledger_rows,
    summarize_exit_groups,
    summarize_feature_comparison,
    summarize_loser_paths,
    summarize_winner_paths,
    window_diagnostics,
)
from src.hypotheses.manifest import HoldoutStatus, ResearchManifestStore
from src.research.multiregime.windows import construct_windows
from src.strategy.models import TrendMomentumConfig
from src.backtest.models import BacktestConfig
from src.utils.logger import configure_logging, get_logger


EXPECTED_TRADES = 443
EXPECTED_FRICTIONLESS_EXPECTANCY_R = Decimal(
    "0.04722009245553381286395292467"
)
EXPECTED_NET_EXPECTANCY_R = Decimal("-0.1366917232474779417819851766")
EXPECTED_PROFIT_FACTOR_R = Decimal("0.7695960517596068371318908235")
EXPECTED_POSITIVE_WINDOWS = 1
REPRODUCTION_TOLERANCE = Decimal("1e-24")


def _plain(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def verify_frozen_reproduction(
    *,
    summary: RNormalizedSummary,
    positive_windows: int,
    expected_trades: int = EXPECTED_TRADES,
    expected_frictionless: Decimal = EXPECTED_FRICTIONLESS_EXPECTANCY_R,
    expected_net: Decimal = EXPECTED_NET_EXPECTANCY_R,
    expected_pf: Decimal = EXPECTED_PROFIT_FACTOR_R,
    expected_positive_windows: int = EXPECTED_POSITIVE_WINDOWS,
    tolerance: Decimal = REPRODUCTION_TOLERANCE,
) -> None:
    """Stop diagnostics unless the exact frozen V6 base result reproduces."""

    checks = {
        "trade count": summary.trade_count == expected_trades,
        "frictionless expectancy": (
            abs(summary.frictionless_expectancy_r - expected_frictionless)
            <= tolerance
        ),
        "net expectancy": abs(summary.net_expectancy_r - expected_net) <= tolerance,
        "profit factor": (
            summary.profit_factor_r is not None
            and abs(summary.profit_factor_r - expected_pf) <= tolerance
        ),
        "positive-net windows": positive_windows == expected_positive_windows,
    }
    failed = tuple(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "Frozen V6-H0 diagnostic reproduction failed: " + ", ".join(failed)
        )


def _feature_consistency(window_rows: tuple[dict, ...]) -> tuple[dict, ...]:
    output = []
    for feature in FEATURE_FIELDS:
        field = f"{feature}_winner_loser_mean_difference"
        values = tuple(row[field] for row in window_rows if row[field] is not None)
        output.append(
            {
                "feature": feature,
                "eligible_windows": len(values),
                "positive_difference_windows": sum(value > 0 for value in values),
                "negative_difference_windows": sum(value < 0 for value in values),
                "zero_difference_windows": sum(value == 0 for value in values),
                "directionally_consistent": (
                    bool(values)
                    and (
                        all(value >= 0 for value in values)
                        or all(value <= 0 for value in values)
                    )
                ),
            }
        )
    return tuple(output)


def _classification(
    *, combined: RNormalizedSummary, exits: tuple[dict, ...], windows: tuple[dict, ...]
) -> dict:
    labels: list[str] = []
    evidence: dict[str, str] = {}
    friction = combined.average_total_friction_r
    gross = combined.frictionless_expectancy_r
    if gross > 0 and combined.net_expectancy_r < 0:
        labels.append("COST_BURDEN_HIGH")
        evidence["COST_BURDEN_HIGH"] = (
            f"Costs of {friction}R exceed and reverse gross edge of {gross}R."
        )
    if gross <= friction:
        labels.append("ENTRY_QUALITY_WEAK")
        evidence["ENTRY_QUALITY_WEAK"] = (
            "Frozen entries generate less gross expectancy than modeled friction."
        )
    weak_non_stop = tuple(
        row
        for row in exits
        if row["exit_reason"] not in ("STOP_LOSS", "TAKE_PROFIT")
        and row["frictionless_expectancy_r"] < 0
    )
    if weak_non_stop:
        labels.append("EXIT_MANAGEMENT_WEAK")
        evidence["EXIT_MANAGEMENT_WEAK"] = (
            "At least one non-protective frozen exit group has negative "
            "frictionless expectancy."
        )
    positive_gross_windows = sum(
        row["frictionless_expectancy_r"] > 0 for row in windows
    )
    if positive_gross_windows < (len(windows) + 1) // 2:
        labels.append("REGIME_SELECTION_WEAK")
        evidence["REGIME_SELECTION_WEAK"] = (
            f"Only {positive_gross_windows}/{len(windows)} windows have gross edge."
        )
    if not labels:
        labels.append("INSUFFICIENT_EVIDENCE")
        evidence["INSUFFICIENT_EVIDENCE"] = "No requested explanatory label is supported."
    return {"labels": labels, "evidence": evidence}


def build_summary_payload(
    *,
    combined: RNormalizedSummary,
    window_rows: tuple[dict, ...],
    exit_rows: tuple[dict, ...],
    feature_rows: tuple[dict, ...],
    loser_paths: dict,
    winner_paths: dict,
    costs: dict,
    frozen_report: dict,
) -> dict:
    classification = _classification(
        combined=combined, exits=exit_rows, windows=window_rows
    )
    return {
        "version": "6.0-signal-quality-diagnostic",
        "source_run_id": EXPECTED_RUN_ID,
        "dataset_status": "CONSUMED_RESEARCH_DATA",
        "eligible_windows": len(window_rows),
        "frozen_reproduction": {
            "verified": True,
            "tolerance": REPRODUCTION_TOLERANCE,
            "summary": asdict(combined),
            "positive_net_windows": sum(
                row["net_expectancy_r"] > 0 for row in window_rows
            ),
        },
        "exit_groups": exit_rows,
        "loser_paths": loser_paths,
        "winner_paths": winner_paths,
        "feature_comparison": feature_rows,
        "feature_window_consistency": _feature_consistency(window_rows),
        "cost_diagnostics": costs,
        "existing_stop_diagnostics": frozen_report["cost_aware_diagnostics"],
        "stress_2x_net_expectancy_r": frozen_report["stress_2x"][
            "net_expectancy_r_per_trade"
        ],
        "classification": classification,
        "integrity": {
            "strategy_changed": False,
            "accounting_changed": False,
            "blind_holdout": {
                "status": "LOCKED_BLIND_HOLDOUT",
                "loaded": False,
                "revealed": False,
                "consumed": False,
                "evaluated": False,
            },
        },
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Diagnostic CSV rows are empty: {path.name}.")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(_plain(rows))
    temporary.replace(path)


def write_reports(
    *,
    output: Path,
    summary_payload: dict,
    ledger: list[dict],
    exit_rows: list[dict],
    feature_rows: list[dict],
    window_rows: list[dict],
) -> tuple[Path, ...]:
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise ValueError("V6 signal-quality report exists; refusing overwrite.")
    output.mkdir(parents=True, exist_ok=True)
    paths = (
        output / "trade_ledger.csv",
        output / "exit_groups.csv",
        output / "feature_comparison.csv",
        output / "window_diagnostics.csv",
    )
    for path, rows in zip(
        paths,
        (ledger, exit_rows, feature_rows, window_rows),
        strict=True,
    ):
        _write_csv(path, rows)
    temporary = output / "summary.json.tmp"
    temporary.write_text(
        json.dumps(_plain(summary_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(summary_path)
    return (summary_path, *paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run read-only V6-H0 diagnostics.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--manifest",
        default="research/v6_mtf_continuation/e1eef7bdd0c37ad4/manifest.json",
    )
    parser.add_argument("--root-manifest", default="research/hypothesis_manifest.json")
    parser.add_argument(
        "--mechanism-report", default="reports/mechanisms/bc2496aed05555b5"
    )
    parser.add_argument(
        "--frozen-report",
        default="reports/v6_mtf_continuation/e1eef7bdd0c37ad4/summary.json",
    )
    parser.add_argument(
        "--expansion-data-root", default="data/historical/multiregime_expansion"
    )
    parser.add_argument("--consumed-data-root", default="data/historical")
    parser.add_argument(
        "--output-root", default="reports/diagnostics/v6_signal_quality"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    logger = get_logger(__name__)
    if not args.execute:
        logger.error("V6 diagnostics not executed. Explicit --execute is required.")
        return 1
    try:
        manifest = _load_json(args.manifest)
        validate_v6_h0_manifest(manifest)
        validate_v6_h0_implementation(manifest)
        frozen_report = _load_json(args.frozen_report)
        if frozen_report.get("run_id") != EXPECTED_RUN_ID:
            raise ValueError("Frozen V6 result report run ID changed.")
        output = Path(args.output_root) / EXPECTED_RUN_ID
        if (output / "summary.json").exists():
            raise ValueError("V6 signal-quality report exists; refusing overwrite.")

        root_manifest = ResearchManifestStore(args.root_manifest).load()
        _require_locked_holdout(root_manifest)
        if (
            root_manifest.holdout_status is not HoldoutStatus.LOCKED_BLIND_HOLDOUT
            or root_manifest.blind_holdout.reveal_timestamp is not None
            or root_manifest.blind_holdout.consumed_timestamp is not None
        ):
            raise ValueError("Blind holdout contamination detected.")

        from src.config.settings import load_settings

        regions = _build_regions(
            root_manifest=root_manifest,
            expansion_data_root=args.expansion_data_root,
            consumed_data_root=args.consumed_data_root,
        )
        windows = construct_windows(regions, load_settings().multiregime_config())
        reference_windows = _read_reference_windows(Path(args.mechanism_report))
        eligible = tuple(
            window for window in windows if window.window_id in reference_windows
        )
        eligible_windows = expand_complete_region_history(eligible, regions)
        validate_eligible_windows(eligible_windows)
        contexts = prepare_v6_region_contexts(
            windows=eligible_windows, regions=regions
        )

        strategy_config = TrendMomentumConfig()
        backtest_config = BacktestConfig()
        combined_r: list[RNormalizedTrade] = []
        ledger = []
        window_summaries = []
        for window in eligible_windows:
            context = prepared_context_for_window(
                window=window, regions=regions, region_contexts=contexts
            )
            evaluation, records, summary, observations = _evaluate_window(
                window=window,
                strategy_config=strategy_config,
                backtest_config=backtest_config,
                prepared_context=context,
            )
            combined_r.extend(records)
            ledger.extend(
                build_trade_ledger(
                    window_id=window.window_id,
                    trades=evaluation.backtest.trades,
                    r_records=records,
                    stop_observations=observations,
                    replay_dataset=window.replay_dataset,
                    evaluation_dataset=evaluation.backtest.dataset,
                    prepared_context=context,
                )
            )
            window_summaries.append(
                {
                    "window_id": window.window_id,
                    "trades": summary.trade_count,
                    "frictionless_expectancy_r": summary.frictionless_expectancy_r,
                    "net_expectancy_r": summary.net_expectancy_r,
                    "profit_factor_r": summary.profit_factor_r,
                }
            )
            logger.info("%s diagnostic ledger: %d trades", window.window_id, len(records))

        combined = summarize_combined_records((tuple(combined_r),))
        window_rows = window_diagnostics(tuple(ledger))
        positive_windows = sum(
            row["net_expectancy_r"] > 0 for row in window_summaries
        )
        verify_frozen_reproduction(
            summary=combined, positive_windows=positive_windows
        )
        exit_rows = summarize_exit_groups(tuple(ledger))
        feature_rows = summarize_feature_comparison(tuple(ledger))
        loser_paths = summarize_loser_paths(tuple(ledger))
        winner_paths = summarize_winner_paths(tuple(ledger))
        costs = cost_diagnostics(tuple(ledger))
        payload = build_summary_payload(
            combined=combined,
            window_rows=window_rows,
            exit_rows=exit_rows,
            feature_rows=feature_rows,
            loser_paths=loser_paths,
            winner_paths=winner_paths,
            costs=costs,
            frozen_report=frozen_report,
        )
        paths = write_reports(
            output=output,
            summary_payload=payload,
            ledger=ledger_rows(tuple(ledger)),
            exit_rows=list(exit_rows),
            feature_rows=list(feature_rows),
            window_rows=list(window_rows),
        )
        logger.info("Frozen V6 reproduction: 443 trades VERIFIED")
        logger.info("Classification: %s", payload["classification"]["labels"])
        for path in paths:
            logger.info("Report: %s", path.resolve())
        logger.warning(
            "Blind holdout: LOCKED | NOT LOADED | NOT REVEALED | "
            "NOT CONSUMED | NOT EVALUATED"
        )
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        logger.error("V6 signal-quality diagnostic failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
