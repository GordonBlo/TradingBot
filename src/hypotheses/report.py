"""Reproducible V3.2 comparison, journals, and consumed-data reporting."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from src.backtest.report import BacktestReportWriter
from src.hypotheses.manifest import ResearchManifest
from src.hypotheses.models import hypothesis_registry
from src.hypotheses.runner import (
    HypothesisMetrics,
    HypothesisResearchResult,
)


def _plain(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class HypothesisReportPaths:
    directory: Path
    manifest: Path
    comparison_csv: Path
    comparison_json: Path
    summary: Path


class HypothesisReportWriter:
    def __init__(self, root: str | Path = "reports/hypotheses") -> None:
        self.root = Path(root)

    def write(
        self,
        result: HypothesisResearchResult,
        manifest: ResearchManifest,
    ) -> HypothesisReportPaths:
        fingerprint = {
            "source_research_run_id": result.source_research_run_id,
            "dataset_status": result.dataset_status,
            "base_costs": asdict(result.base_backtest_config),
            "stress_costs": (
                asdict(result.stress_backtest_config)
                if result.stress_backtest_config is not None
                else None
            ),
            "suite_config": asdict(result.suite_config),
            "metrics": {
                item.hypothesis_id.value: asdict(item.combined_metrics)
                for item in result.experiments
            },
        }
        canonical = json.dumps(
            _plain(fingerprint), sort_keys=True, separators=(",", ":")
        )
        run_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        directory = self.root / run_id
        directory.mkdir(parents=True, exist_ok=True)
        paths = HypothesisReportPaths(
            directory=directory,
            manifest=directory / "manifest.json",
            comparison_csv=directory / "comparison.csv",
            comparison_json=directory / "comparison.json",
            summary=directory / "RESEARCH_SUMMARY.md",
        )

        registry = {
            item.hypothesis_id: item for item in hypothesis_registry(result.suite_config)
        }
        report_manifest = {
            "project_version": "3.2",
            "source_research_run_id": result.source_research_run_id,
            "baseline_reproduction_verified": result.baseline_reproduction_verified,
            "dataset_status": result.dataset_status,
            "consumed_data_warning": (
                "The former OOS period is now CONSUMED_RESEARCH_DATA. "
                "No V3.2 result on that period is fresh OOS evidence."
            ),
            "hypotheses": [asdict(item) for item in registry.values()],
            "base_backtest_config": asdict(result.base_backtest_config),
            "stress_backtest_config": (
                asdict(result.stress_backtest_config)
                if result.stress_backtest_config is not None
                else None
            ),
            "research_manifest_snapshot": asdict(manifest),
            "integrity": {
                "parameter_search": False,
                "candidate_combinations": False,
                "automatic_holdout_reveal": False,
                "execution_timing": "NEXT_BAR_OPEN",
                "ambiguous_bar_policy": "STOP_FIRST",
            },
        }
        paths.manifest.write_text(
            json.dumps(_plain(report_manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        rows: list[dict[str, Any]] = []
        comparison_json: dict[str, Any] = {}
        names = {
            "H0": "H0_baseline",
            "H1": "H1_volatility_guard",
            "H2": "H2_extension_guard",
            "H3": "H3_pullback_confirmation",
            "H4": "H4_1R_target",
        }
        for experiment in result.experiments:
            candidate_dir = directory / names[experiment.hypothesis_id.value]
            candidate_dir.mkdir(parents=True, exist_ok=True)
            for period in experiment.periods:
                period_dir = candidate_dir / period.source_period
                BacktestReportWriter(period_dir).write(period.backtest, run_id="base")
                self._write_signals(period_dir / "signals.csv", period.signals)
                self._write_journal(period_dir / "hypothesis_journal.csv", period.journals)
                rows.append(
                    self._row(
                        experiment.hypothesis_id.value,
                        experiment.name,
                        period.source_period,
                        period.data_status.value,
                        period.metrics,
                    )
                )
            rows.append(
                self._row(
                    experiment.hypothesis_id.value,
                    experiment.name,
                    "combined_descriptive_only",
                    result.dataset_status.value,
                    experiment.combined_metrics,
                )
            )
            candidate_summary = {
                "hypothesis": asdict(registry[experiment.hypothesis_id]),
                "dataset_status": result.dataset_status,
                "periods": {
                    period.source_period: {
                        "data_status": period.data_status,
                        "metrics": asdict(period.metrics),
                        "journal_records": len(period.journals),
                    }
                    for period in experiment.periods
                },
                "combined_metrics": asdict(experiment.combined_metrics),
                "delta_to_baseline": (
                    asdict(experiment.delta_to_baseline)
                    if experiment.delta_to_baseline is not None
                    else None
                ),
                "signal_alignment": asdict(experiment.alignment),
                "classification": experiment.classification,
                "verification_status": experiment.verification_status,
                "warnings": experiment.warnings,
                "stress_combined_metrics": (
                    asdict(experiment.stress_combined_metrics)
                    if experiment.stress_combined_metrics is not None
                    else None
                ),
            }
            (candidate_dir / "summary.json").write_text(
                json.dumps(_plain(candidate_summary), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            comparison_json[experiment.hypothesis_id.value] = candidate_summary

        self._write_comparison_csv(paths.comparison_csv, rows)
        paths.comparison_json.write_text(
            json.dumps(_plain(comparison_json), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths.summary.write_text(self._markdown(result), encoding="utf-8")
        return paths

    @staticmethod
    def _row(
        hypothesis_id: str,
        name: str,
        period: str,
        status: str,
        metrics: HypothesisMetrics,
    ) -> dict[str, Any]:
        return {
            "hypothesis_id": hypothesis_id,
            "name": name,
            "period": period,
            "data_status": status,
            **asdict(metrics),
        }

    @staticmethod
    def _write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(_plain(rows))

    @staticmethod
    def _write_signals(path: Path, signals: tuple[Any, ...]) -> None:
        fields = (
            "timestamp", "action", "close", "ema_fast", "ema_slow", "rsi",
            "atr", "volume_ratio", "position_state", "reason_code", "reason",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for signal in signals:
                writer.writerow(_plain(asdict(signal)))

    @staticmethod
    def _write_journal(path: Path, journals: tuple[Any, ...]) -> None:
        fields = ("timestamp", "hypothesis_id", "event", "reason", "values")
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for record in journals:
                row = _plain(asdict(record))
                row["values"] = json.dumps(row["values"], sort_keys=True)
                writer.writerow(row)

    @staticmethod
    def _markdown(result: HypothesisResearchResult) -> str:
        lines = [
            "# V3.2 Hypothesis Research\n",
            "**Dataset status: CONSUMED_RESEARCH_DATA**\n",
            "> The former OOS period has been inspected. No result here is fresh OOS evidence.\n",
            "| ID | Trades | Frictionless | Net | PF | Expectancy | Max DD % | Classification |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for item in result.experiments:
            metric = item.combined_metrics
            lines.append(
                f"| {item.hypothesis_id.value} | {metric.trades} | {metric.frictionless_pnl} | "
                f"{metric.net_pnl} | {metric.profit_factor} | {metric.expectancy} | "
                f"{metric.maximum_drawdown_percent} | {item.classification.value} |"
            )
        lines.extend(
            [
                "\nAll labels are descriptive and all candidates remain **NOT YET VERIFIED ON BLIND HOLDOUT**.",
                "Improvement does not mean profitability or robustness. No parameter search or candidate combination was performed.\n",
            ]
        )
        return "\n".join(lines)
