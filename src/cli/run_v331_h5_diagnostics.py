"""Run frozen V3.3.1 H5_Q25 post-trade diagnostics on consumed data only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path

from src.cli.run_multiregime import _require_locked_holdout
from src.cli.run_v33_h5 import (
    EXPECTED_REFERENCE_TRADES,
    EXPECTED_V33_RUN_ID,
    _build_regions,
    _evaluate_window,
    _load_v33_manifest,
    _read_reference_windows,
)
from src.config.settings import load_settings
from src.diagnostics.h5_exit_diagnostics import (
    build_h5_exit_trade_records,
    summarize_excursion_groups,
    summarize_exit_groups,
)
from src.diagnostics.loader import ResearchRunLoader
from src.diagnostics.models import DiagnosticPeriodInput
from src.diagnostics.r_normalized import summarize_r_normalized_trades
from src.diagnostics.risk_capital_audit import (
    build_risk_capital_audit,
    summarize_risk_capital_audit,
)
from src.diagnostics.trade_metrics import build_trade_diagnostics
from src.diagnostics.window_warmup_parity import compare_indicator_boundary
from src.historical.dataset import HistoricalDataset
from src.hypotheses.h5_parameter_research import (
    H5ParameterCandidateId,
    build_h5_parameter_candidate,
)
from src.hypotheses.manifest import ResearchManifestStore
from src.research.evaluation import evaluate_strategy_period
from src.research.multiregime.windows import construct_windows
from src.strategy.models import StrategyAction
from src.utils.logger import configure_logging, get_logger


def _plain(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _owner(window, regions):
    return next(
        region
        for region in regions
        if region.metadata.kind is window.partition_kind
        and region.metadata.start <= window.start
        and window.end <= region.metadata.end
    )


def _continuous_replay(window, region, strategy_config, backtest_config):
    candles = tuple(
        candle for candle in region.dataset.candles if candle.timestamp < window.end
    )
    start_index = next(
        index for index, candle in enumerate(candles) if candle.timestamp >= window.start
    )
    dataset = HistoricalDataset(
        region.dataset.symbol,
        region.dataset.interval,
        region.dataset.source,
        candles,
    )
    return evaluate_strategy_period(
        dataset,
        strategy=build_h5_parameter_candidate(
            H5ParameterCandidateId.H5_Q25, strategy_config
        ),
        strategy_config=strategy_config,
        backtest_config=backtest_config,
        evaluation_start_index=start_index,
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(_plain(rows))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen H5_Q25 R/exit diagnostics; no strategy search."
    )
    parser.add_argument("--root-manifest", default="research/hypothesis_manifest.json")
    parser.add_argument(
        "--v33-manifest",
        default="research/v3_3_h5/b104ea78b4c88bcf/manifest.json",
    )
    parser.add_argument("--research-run", default="adeef00722e9704d")
    parser.add_argument("--research-root", default="reports/research")
    parser.add_argument(
        "--mechanism-report", default="reports/mechanisms/bc2496aed05555b5"
    )
    parser.add_argument(
        "--v33-report", default="reports/v3_3_h5/b104ea78b4c88bcf/summary.json"
    )
    parser.add_argument(
        "--expansion-data-root", default="data/historical/multiregime_expansion"
    )
    parser.add_argument("--consumed-data-root", default="data/historical")
    parser.add_argument("--output-root", default="reports/diagnostics/v331_h5")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    logger = get_logger(__name__)
    try:
        preregistration = _load_v33_manifest(Path(args.v33_manifest))
        if preregistration["run_id"] != EXPECTED_V33_RUN_ID:
            raise ValueError("Frozen V3.3 preregistration changed.")
        root_manifest = ResearchManifestStore(args.root_manifest).load()
        _require_locked_holdout(root_manifest)
        holdout = root_manifest.blind_holdout
        if (
            holdout is None
            or holdout.reveal_timestamp is not None
            or holdout.consumed_timestamp is not None
        ):
            raise ValueError("Blind holdout is not untouched and locked.")

        reference_windows = _read_reference_windows(Path(args.mechanism_report))
        previous = ResearchRunLoader(
            research_root=args.research_root,
            data_root=args.consumed_data_root,
        ).load(args.research_run)
        strategy_config = previous.strategy_config
        backtest_config = previous.backtest_config
        if (
            strategy_config.risk_per_trade_percent != Decimal("0.50")
            or strategy_config.maximum_position_notional_usdc != Decimal("50")
        ):
            raise ValueError("Frozen H5 risk/capital configuration changed.")

        regions = _build_regions(
            root_manifest=root_manifest,
            expansion_data_root=args.expansion_data_root,
            consumed_data_root=args.consumed_data_root,
        )
        windows = construct_windows(regions, load_settings().multiregime_config())
        eligible = tuple(
            window for window in windows if window.window_id in reference_windows
        )
        if len(eligible) != 11:
            raise ValueError(f"Expected 11 eligible windows, got {len(eligible)}.")

        combined_r = []
        combined_risk = []
        combined_exit = []
        parity_rows = []
        continuous_trades = 0
        continuous_signal_differences = 0
        for window in eligible:
            evaluation, r_records, _ = _evaluate_window(
                window=window,
                candidate_id=H5ParameterCandidateId.H5_Q25,
                strategy_config=strategy_config,
                backtest_config=backtest_config,
            )
            actual = len(evaluation.backtest.trades)
            expected = reference_windows[window.window_id]
            if actual != expected:
                raise ValueError(
                    f"Frozen H5_Q25 mismatch in {window.window_id}: "
                    f"expected {expected}, got {actual}."
                )
            risk_records = build_risk_capital_audit(
                trades=evaluation.backtest.trades,
                signals=evaluation.signals,
                equity_curve=evaluation.backtest.equity_curve,
                backtest_config=backtest_config,
                strategy_config=strategy_config,
            )
            diagnostics = build_trade_diagnostics(
                DiagnosticPeriodInput(
                    period=window.window_id,
                    dataset=evaluation.backtest.dataset,
                    trades=evaluation.backtest.trades,
                    signals=evaluation.signals,
                ),
                strategy_config=strategy_config,
                slippage_bps=backtest_config.slippage_bps,
            )
            combined_exit.extend(
                build_h5_exit_trade_records(
                    trades=evaluation.backtest.trades,
                    r_records=r_records,
                    diagnostics=diagnostics,
                    dataset=evaluation.backtest.dataset,
                )
            )
            combined_r.extend(r_records)
            combined_risk.extend(risk_records)

            region = _owner(window, regions)
            parity_rows.extend(
                compare_indicator_boundary(window, region, strategy_config)
            )
            continuous = _continuous_replay(
                window, region, strategy_config, backtest_config
            )
            continuous_trades += len(continuous.backtest.trades)
            local_signals = {
                signal.timestamp
                for signal in evaluation.signals
                if signal.action is StrategyAction.ENTER_LONG
            }
            continuous_signals = {
                signal.timestamp
                for signal in continuous.signals
                if signal.action is StrategyAction.ENTER_LONG
            }
            continuous_signal_differences += len(
                local_signals.symmetric_difference(continuous_signals)
            )

        r_records = tuple(combined_r)
        if len(r_records) != EXPECTED_REFERENCE_TRADES:
            raise ValueError("Frozen H5_Q25 did not reproduce exactly 197 trades.")
        r_summary = summarize_r_normalized_trades(r_records)
        reference = json.loads(Path(args.v33_report).read_text(encoding="utf-8"))[
            "candidates"
        ]["H5_Q25"]["base"]
        for field in (
            "frictionless_expectancy_r",
            "net_expectancy_r",
            "profit_factor_r",
        ):
            if abs(getattr(r_summary, field) - Decimal(reference[field])) > Decimal(
                "1e-18"
            ):
                raise ValueError(f"Frozen H5_Q25 {field} changed.")
        residuals = tuple(
            abs(
                record.frictionless_r
                - record.slippage_cost_r
                - record.fee_cost_r
                - record.net_r
            )
            for record in r_records
        )
        maximum_residual = max(residuals, default=Decimal("0"))
        if maximum_residual > Decimal("1e-20"):
            raise ValueError("H5 R accounting identity failed.")

        risk_summary = summarize_risk_capital_audit(tuple(combined_risk))
        exit_groups = summarize_exit_groups(tuple(combined_exit))
        excursion_groups = summarize_excursion_groups(tuple(combined_exit))
        parity_maxima = {}
        for field in sorted({row.field for row in parity_rows}):
            selected = [
                row for row in parity_rows if row.field == field and row.absolute_difference is not None
            ]
            maximum = max(
                selected,
                key=lambda row: row.absolute_difference,
            )
            parity_maxima[field] = asdict(maximum)

        run_material = json.dumps(
            {
                "version": "3.3.1",
                "source_run": EXPECTED_V33_RUN_ID,
                "candidate": "H5_Q25",
                "windows": reference_windows,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        run_id = hashlib.sha256(run_material.encode("utf-8")).hexdigest()[:16]
        directory = Path(args.output_root) / run_id
        directory.mkdir(parents=True, exist_ok=True)
        exit_rows = [asdict(row) for row in exit_groups]
        excursion_rows = [asdict(row) for row in excursion_groups]
        _write_csv(directory / "exit_groups.csv", exit_rows)
        _write_csv(directory / "excursion_summary.csv", excursion_rows)
        payload = {
            "version": "3.3.1",
            "run_id": run_id,
            "source_v33_run_id": EXPECTED_V33_RUN_ID,
            "candidate": "H5_Q25",
            "dataset_status": "CONSUMED_RESEARCH_DATA",
            "eligible_windows": len(eligible),
            "trade_count": len(r_records),
            "frozen_reproduction_verified": True,
            "r_accounting": {
                "identity": "frictionless_R - slippage_R - fee_R == net_R",
                "tolerance": Decimal("1e-20"),
                "maximum_absolute_residual": maximum_residual,
                "verified": True,
                "summary": asdict(r_summary),
            },
            "risk_semantics": {
                "configured_risk_percent": strategy_config.risk_per_trade_percent,
                "configured_position_cap_usdc": (
                    strategy_config.maximum_position_notional_usdc
                ),
                "actual_risk_is_position_cap_limited": True,
                "summary": asdict(risk_summary),
            },
            "warmup_parity": {
                "historical_design": (
                    "Each fixed window initializes recursive indicators from at most "
                    "250 prior candles inside its safe partition."
                ),
                "changes_frozen_results": False,
                "boundary_maxima": parity_maxima,
                "continuous_state_trade_count": continuous_trades,
                "continuous_state_entry_signal_differences": (
                    continuous_signal_differences
                ),
            },
            "volume_ratio_semantics": {
                "current_closed_candle_included_in_sma20": True,
                "strategy_behavior_changed": False,
            },
            "exit_groups": exit_rows,
            "excursion_groups": excursion_rows,
            "blind_holdout": {
                "status": "LOCKED_BLIND_HOLDOUT",
                "revealed": False,
                "consumed": False,
                "evaluated": False,
                "loaded": False,
            },
        }
        (directory / "summary.json").write_text(
            json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        logger.info("V3.3.1 H5_Q25 R / EXIT DIAGNOSTICS")
        logger.info("Frozen reproduction: 197 trades / 11 windows VERIFIED")
        logger.info("R identity max residual: %s", maximum_residual)
        for row in exit_groups:
            logger.info(
                "%s | n=%d | gross=%sR | net=%sR | friction=%sR",
                row.exit_reason.value,
                row.trades,
                row.frictionless_expectancy_r,
                row.net_expectancy_r,
                row.average_total_friction_r,
            )
        logger.info(
            "Continuous-state parity: trades=%d | entry signal differences=%d",
            continuous_trades,
            continuous_signal_differences,
        )
        logger.warning(
            "Blind holdout: LOCKED | NOT LOADED | NOT REVEALED | "
            "NOT CONSUMED | NOT EVALUATED"
        )
        logger.info("Report: %s", directory.resolve())
        return 0
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        logger.error("V3.3.1 diagnostics failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
