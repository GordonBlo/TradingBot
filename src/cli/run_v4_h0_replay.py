"""Execute the single preregistered V4-H0 consumed-data replay."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from pathlib import Path

from src.backtest.models import BacktestConfig
from src.cli.run_multiregime import _require_locked_holdout
from src.cli.run_v33_h5 import (
    _build_regions,
    _max_drawdown_percent,
    _read_reference_windows,
)
from src.cli.run_v4_h0 import (
    EXPECTED_RUN_ID,
    _load_json,
    validate_v4_h0_implementation,
    validate_v4_h0_manifest,
)
from src.diagnostics.r_normalized import (
    RNormalizedSummary,
    build_r_normalized_trades,
    summarize_r_normalized_trades,
)
from src.diagnostics.risk_capital_audit import build_risk_capital_audit
from src.hypotheses.manifest import (
    HoldoutStatus,
    ResearchManifestStore,
)
from src.research.evaluation import evaluate_strategy_period
from src.research.multiregime.models import PartitionStatus
from src.research.multiregime.windows import construct_windows
from src.research.v4_breakout_stability import (
    REQUIRED_ELIGIBLE_WINDOWS,
    V4H0Metrics,
    evaluate_v4_h0_gate,
)
from src.strategy.models import TrendMomentumConfig
from src.strategy.v4_breakout import V4BreakoutStrategy
from src.utils.logger import configure_logging, get_logger


@dataclass(frozen=True, slots=True)
class WindowResult:
    window_id: str
    trades: int
    frictionless_expectancy_r: Decimal
    net_expectancy_r: Decimal
    profit_factor_r: Decimal | None
    maximum_drawdown_percent: Decimal
    positive_net: bool


def _json_safe(value):
    if isinstance(value, Decimal):
        return str(value)
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def validate_replay_configs(
    *,
    manifest: dict,
    strategy_config: TrendMomentumConfig,
    base_config: BacktestConfig,
    stress_config: BacktestConfig,
) -> None:
    risk = manifest["risk_and_exit"]
    capital = manifest["capital"]
    base_costs = manifest["costs"]["base"]
    stress_costs = manifest["costs"]["stress_2x"]

    expected_strategy = {
        "atr_period": risk["atr_period"],
        "atr_stop_multiplier": Decimal(risk["stop_atr_multiple"]),
        "reward_risk_ratio": Decimal(risk["reward_risk_ratio"]),
        "maximum_bars_in_position": risk["maximum_hold_bars"],
        "cooldown_bars": risk["cooldown_bars"],
        "risk_per_trade_percent": Decimal(
            capital["risk_per_trade_percent"]
        ),
        "maximum_position_notional_usdc": Decimal(
            capital["max_capital_usdc"]
        ),
    }
    for name, expected in expected_strategy.items():
        if getattr(strategy_config, name) != expected:
            raise ValueError(f"Frozen V4-H0 strategy config changed: {name}.")

    expected_base = {
        "fee_bps": Decimal(base_costs["fee_bps_per_side"]),
        "slippage_bps": Decimal(base_costs["slippage_bps_per_side"]),
    }
    expected_stress = {
        "fee_bps": Decimal(stress_costs["fee_bps_per_side"]),
        "slippage_bps": Decimal(stress_costs["slippage_bps_per_side"]),
    }
    for name, expected in expected_base.items():
        if getattr(base_config, name) != expected:
            raise ValueError(f"Frozen V4-H0 base cost changed: {name}.")
    for name, expected in expected_stress.items():
        if getattr(stress_config, name) != expected:
            raise ValueError(f"Frozen V4-H0 stress cost changed: {name}.")

    if stress_config.execution_timing is not base_config.execution_timing:
        raise ValueError("V4-H0 stress execution timing changed.")
    if stress_config.ambiguous_bar_policy is not base_config.ambiguous_bar_policy:
        raise ValueError("V4-H0 stress STOP_FIRST policy changed.")


def validate_eligible_windows(windows: tuple) -> None:
    if len(windows) != REQUIRED_ELIGIBLE_WINDOWS:
        raise ValueError(
            "V4-H0 requires exactly 11 eligible windows, got "
            f"{len(windows)}."
        )
    for window in windows:
        if window.data_status is not PartitionStatus.CONSUMED_RESEARCH_DATA:
            raise ValueError("V4-H0 window is not consumed research data.")
        if (
            window.replay_dataset.symbol != "BTCUSDC"
            or window.replay_dataset.interval != "15m"
        ):
            raise ValueError("V4-H0 window market changed.")


def _evaluate_window(
    *,
    window,
    strategy_config: TrendMomentumConfig,
    backtest_config: BacktestConfig,
):
    evaluation = evaluate_strategy_period(
        window.replay_dataset,
        strategy=V4BreakoutStrategy(strategy_config),
        strategy_config=strategy_config,
        backtest_config=backtest_config,
        evaluation_start_index=window.evaluation_start_index,
    )
    risk_records = build_risk_capital_audit(
        trades=evaluation.backtest.trades,
        signals=evaluation.signals,
        equity_curve=evaluation.backtest.equity_curve,
        backtest_config=backtest_config,
        strategy_config=strategy_config,
    )
    risk_by_trade_id = {
        record.trade_id: record.actual_stop_risk
        for record in risk_records
    }
    r_records = build_r_normalized_trades(
        trades=evaluation.backtest.trades,
        actual_stop_risk_by_trade_id=risk_by_trade_id,
        slippage_bps=backtest_config.slippage_bps,
    )
    if len(r_records) != len(evaluation.backtest.trades):
        raise ValueError(f"R-record mismatch in {window.window_id}.")
    return (
        evaluation,
        r_records,
        summarize_r_normalized_trades(r_records),
    )


def _window_result(*, window_id: str, evaluation, summary) -> WindowResult:
    return WindowResult(
        window_id=window_id,
        trades=len(evaluation.backtest.trades),
        frictionless_expectancy_r=summary.frictionless_expectancy_r,
        net_expectancy_r=summary.net_expectancy_r,
        profit_factor_r=summary.profit_factor_r,
        maximum_drawdown_percent=_max_drawdown_percent(
            evaluation.backtest.equity_curve
        ),
        positive_net=summary.net_expectancy_r > 0,
    )


def build_summary_payload(
    *,
    manifest: dict,
    base_summary: RNormalizedSummary,
    stress_summary: RNormalizedSummary,
    window_results: tuple[WindowResult, ...],
    maximum_drawdown_percent: Decimal,
    evaluation_days: Decimal,
) -> dict:
    gate = evaluate_v4_h0_gate(
        metrics=V4H0Metrics(
            net_expectancy_r=base_summary.net_expectancy_r,
            profit_factor_r=base_summary.profit_factor_r,
        ),
        window_net_expectancies_r=tuple(
            row.net_expectancy_r for row in window_results
        ),
    )
    trades_per_day = (
        Decimal(base_summary.trade_count) / evaluation_days
        if evaluation_days > 0
        else Decimal("0")
    )
    net_r_per_day = (
        base_summary.net_expectancy_r
        * Decimal(base_summary.trade_count)
        / evaluation_days
        if evaluation_days > 0
        else Decimal("0")
    )
    gate_payload = {
        **asdict(gate),
        "positive_net_window_ratio": gate.positive_net_window_ratio,
        "all_conditions_met": gate.all_conditions_met,
    }
    return {
        "version": "4.0",
        "run_id": EXPECTED_RUN_ID,
        "baseline_id": "V4_H0",
        "market": manifest["market"],
        "dataset_status": "CONSUMED_RESEARCH_DATA",
        "eligible_windows": REQUIRED_ELIGIBLE_WINDOWS,
        "blind_holdout": manifest["dataset_policy"]["blind_holdout"],
        "costs": manifest["costs"],
        "aggregate": {
            "total_trades": base_summary.trade_count,
            "frictionless_expectancy_r_per_trade": (
                base_summary.frictionless_expectancy_r
            ),
            "net_expectancy_r_per_trade": base_summary.net_expectancy_r,
            "profit_factor_r": base_summary.profit_factor_r,
            "win_rate_percent": base_summary.win_rate_percent,
            "maximum_drawdown_percent": maximum_drawdown_percent,
            "positive_net_windows": gate.positive_net_windows,
            "positive_net_windows_total": gate.eligible_windows,
            "base_cost_metrics": asdict(base_summary),
            "doubled_cost_stress_net_expectancy_r": (
                stress_summary.net_expectancy_r
            ),
            "doubled_cost_stress_metrics": asdict(stress_summary),
            "evaluation_days": evaluation_days,
            "trades_per_day": trades_per_day,
            "net_r_per_day": net_r_per_day,
        },
        "gate": gate_payload,
        "progression_eligible": gate.all_conditions_met,
    }


def write_reports(
    *,
    output: Path,
    summary_payload: dict,
    window_results: tuple[WindowResult, ...],
) -> tuple[Path, Path]:
    summary_path = output / "summary.json"
    if summary_path.exists():
        raise ValueError(
            "V4-H0 result already exists; refusing to rerun or overwrite."
        )

    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "window_results.csv"
    temporary_csv = output / "window_results.csv.tmp"
    with temporary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=tuple(asdict(window_results[0]).keys()),
        )
        writer.writeheader()
        for row in window_results:
            writer.writerow(_json_safe(asdict(row)))
    temporary_csv.replace(csv_path)

    temporary_summary = output / "summary.json.tmp"
    temporary_summary.write_text(
        json.dumps(
            _json_safe(summary_payload),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_summary.replace(summary_path)
    return summary_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute the single preregistered V4-H0 replay."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly authorize the single V4-H0 consumed-data replay.",
    )
    parser.add_argument(
        "--manifest",
        default="research/v4_breakout/ba79975004b106a5/manifest.json",
    )
    parser.add_argument(
        "--root-manifest",
        default="research/hypothesis_manifest.json",
    )
    parser.add_argument(
        "--mechanism-report",
        default="reports/mechanisms/bc2496aed05555b5",
    )
    parser.add_argument(
        "--expansion-data-root",
        default="data/historical/multiregime_expansion",
    )
    parser.add_argument(
        "--consumed-data-root",
        default="data/historical",
    )
    parser.add_argument("--output-root", default="reports/v4_breakout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    logger = get_logger(__name__)

    if not args.execute:
        logger.error(
            "V4-H0 replay not executed. Explicit --execute is required."
        )
        return 1

    try:
        manifest = _load_json(args.manifest)
        validate_v4_h0_manifest(manifest)
        validate_v4_h0_implementation(manifest)

        output = Path(args.output_root) / EXPECTED_RUN_ID
        if (output / "summary.json").exists():
            raise ValueError(
                "V4-H0 result already exists; refusing to rerun or overwrite."
            )

        root_manifest = ResearchManifestStore(args.root_manifest).load()
        _require_locked_holdout(root_manifest)
        if (
            root_manifest.holdout_status
            is not HoldoutStatus.LOCKED_BLIND_HOLDOUT
            or root_manifest.blind_holdout.reveal_timestamp is not None
            or root_manifest.blind_holdout.consumed_timestamp is not None
        ):
            raise ValueError("Blind holdout contamination detected.")

        reference_windows = _read_reference_windows(
            Path(args.mechanism_report)
        )
        strategy_config = TrendMomentumConfig()
        base_config = BacktestConfig()
        stress_config = replace(
            base_config,
            fee_bps=base_config.fee_bps * Decimal("2"),
            slippage_bps=base_config.slippage_bps * Decimal("2"),
        )
        validate_replay_configs(
            manifest=manifest,
            strategy_config=strategy_config,
            base_config=base_config,
            stress_config=stress_config,
        )

        from src.config.settings import load_settings

        regions = _build_regions(
            root_manifest=root_manifest,
            expansion_data_root=args.expansion_data_root,
            consumed_data_root=args.consumed_data_root,
        )
        windows = construct_windows(
            regions,
            load_settings().multiregime_config(),
        )
        eligible_windows = tuple(
            window
            for window in windows
            if window.window_id in reference_windows
        )
        validate_eligible_windows(eligible_windows)

        logger.info("V4-H0 PREREGISTERED REPLAY")
        logger.info("Windows: 11 | CONSUMED_RESEARCH_DATA")
        logger.warning(
            "Blind holdout: LOCKED | NOT LOADED | "
            "NOT REVEALED | NOT CONSUMED | NOT EVALUATED"
        )

        base_records = []
        stress_records = []
        window_results: list[WindowResult] = []
        maximum_drawdown = Decimal("0")

        for window in eligible_windows:
            evaluation, r_records, summary = _evaluate_window(
                window=window,
                strategy_config=strategy_config,
                backtest_config=base_config,
            )
            row = _window_result(
                window_id=window.window_id,
                evaluation=evaluation,
                summary=summary,
            )
            window_results.append(row)
            base_records.extend(r_records)
            maximum_drawdown = max(
                maximum_drawdown,
                row.maximum_drawdown_percent,
            )
            logger.info(
                "%s | trades=%d | gross=%sR | net=%sR | PF=%s | positive=%s",
                row.window_id,
                row.trades,
                row.frictionless_expectancy_r,
                row.net_expectancy_r,
                row.profit_factor_r,
                row.positive_net,
            )

        for window in eligible_windows:
            _, r_records, _ = _evaluate_window(
                window=window,
                strategy_config=strategy_config,
                backtest_config=stress_config,
            )
            stress_records.extend(r_records)

        base_summary = summarize_r_normalized_trades(tuple(base_records))
        stress_summary = summarize_r_normalized_trades(tuple(stress_records))
        evaluation_days = sum(
            (window.duration_days for window in eligible_windows),
            start=Decimal("0"),
        )
        summary_payload = build_summary_payload(
            manifest=manifest,
            base_summary=base_summary,
            stress_summary=stress_summary,
            window_results=tuple(window_results),
            maximum_drawdown_percent=maximum_drawdown,
            evaluation_days=evaluation_days,
        )
        summary_path, csv_path = write_reports(
            output=output,
            summary_payload=summary_payload,
            window_results=tuple(window_results),
        )

        aggregate = summary_payload["aggregate"]
        logger.info(
            "V4-H0 | trades=%s | gross=%sR | net=%sR | PF=%s | stress=%sR",
            aggregate["total_trades"],
            aggregate["frictionless_expectancy_r_per_trade"],
            aggregate["net_expectancy_r_per_trade"],
            aggregate["profit_factor_r"],
            aggregate["doubled_cost_stress_net_expectancy_r"],
        )
        logger.info(
            "Progression eligible: %s",
            summary_payload["progression_eligible"],
        )
        logger.info("Summary: %s", summary_path.resolve())
        logger.info("Windows: %s", csv_path.resolve())
        logger.warning(
            "Blind holdout: LOCKED | NOT LOADED | "
            "NOT REVEALED | NOT CONSUMED | NOT EVALUATED"
        )
        return 0
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as exc:
        logger.error("V4-H0 replay failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
