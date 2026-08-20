"""Deterministic CSV/JSON/Markdown outputs for V3.2.1 research."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from src.research.multiregime.manifest import PreparedMultiRegimeManifest
from src.research.multiregime.models import (
    CandidateStability,
    MultiRegimeResearchResult,
    MultiRegimeWindowResult,
    WindowCandidateResult,
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
class MultiRegimeReportPaths:
    directory: Path
    manifest: Path
    window_comparison: Path
    candidate_stability: Path
    regime_comparison: Path
    summary_json: Path
    summary_markdown: Path


class MultiRegimeReportWriter:
    def __init__(
        self,
        root: str | Path = "reports/multiregime",
        *,
        project_version: str = "3.2.1",
        heading: str = "V3.2.1 Multi-Regime Research Expansion",
        eligibility_label: str = "V3.3_ELIGIBLE",
    ) -> None:
        self.root = Path(root)
        self.project_version = project_version
        self.heading = heading
        self.eligibility_label = eligibility_label

    def write(
        self,
        result: MultiRegimeResearchResult,
        preregistration: PreparedMultiRegimeManifest,
    ) -> MultiRegimeReportPaths:
        directory = self.root / result.run_id
        directory.mkdir(parents=True, exist_ok=True)
        paths = MultiRegimeReportPaths(
            directory=directory,
            manifest=directory / "manifest.json",
            window_comparison=directory / "window_comparison.csv",
            candidate_stability=directory / "candidate_stability.csv",
            regime_comparison=directory / "regime_comparison.csv",
            summary_json=directory / "summary.json",
            summary_markdown=directory / "RESEARCH_SUMMARY.md",
        )
        paths.manifest.write_text(
            json.dumps(_plain(preregistration.payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        window_rows = [
            self._window_row(window, candidate)
            for window in result.windows
            for candidate in window.candidates
        ]
        stability_rows = [self._stability_row(item) for item in result.stability]
        regime_rows = [asdict(item) for item in result.regimes]
        self._write_csv(paths.window_comparison, window_rows)
        self._write_csv(paths.candidate_stability, stability_rows)
        self._write_csv(paths.regime_comparison, regime_rows)
        summary = {
            "run_id": result.run_id,
            "project_version": self.project_version,
            "symbol": result.symbol,
            "interval": result.interval,
            "previous_h0_reproduction_verified": result.previous_h0_reproduction_verified,
            "holdout": {
                "status": "LOCKED_BLIND_HOLDOUT",
                "revealed": result.holdout_revealed,
                "consumed": result.holdout_consumed,
                "evaluated": False,
            },
            "partitions": [asdict(item) for item in result.partitions],
            "window_count": len(result.windows),
            "eligible_window_count": sum(item.eligible for item in result.windows),
            "windows": window_rows,
            "candidate_stability": stability_rows,
            "regime_comparison": regime_rows,
            "extreme_windows": self._extreme_windows(result),
            "warnings": [
                "All evaluated ranges are CONSUMED_RESEARCH_DATA after this run.",
                "BEST HISTORICAL WINDOW IS DESCRIPTIVE ONLY.",
                "Multi-regime consistency does not prove future profitability.",
                "The locked blind holdout was not downloaded, evaluated, or revealed.",
            ],
        }
        if result.previous_h1_reproduction_verified is not None:
            summary["previous_h1_reproduction_verified"] = (
                result.previous_h1_reproduction_verified
            )
        paths.summary_json.write_text(
            json.dumps(_plain(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths.summary_markdown.write_text(
            self._markdown(result), encoding="utf-8"
        )
        return paths

    @staticmethod
    def _window_row(
        window: MultiRegimeWindowResult,
        candidate: WindowCandidateResult,
    ) -> dict[str, Any]:
        metric = candidate.metrics
        delta = candidate.delta_to_h0
        stress = candidate.stress_metrics
        return {
            "window_id": window.window.window_id,
            "start": window.window.start,
            "end": window.window.end,
            "duration_days": window.window.duration_days,
            "partition_kind": window.window.partition_kind,
            "data_status": window.window.data_status,
            "partial_window": window.window.partial_window,
            "eligible": window.eligible,
            "btc_return_percent": window.market.btc_return_percent,
            "btc_maximum_drawdown_percent": window.market.btc_maximum_drawdown_percent,
            "realized_volatility_percent": window.market.realized_candle_volatility_percent,
            "average_atr_percent": window.market.average_atr_percent,
            "median_atr_percent": window.market.median_atr_percent,
            "average_volume_ratio": window.market.average_volume_ratio,
            "hypothesis": candidate.hypothesis_id,
            "trades": metric.trades,
            "frictionless_pnl": metric.frictionless_pnl,
            "gross_after_slippage": metric.gross_after_slippage,
            "slippage_drag": metric.slippage_drag,
            "fees": metric.total_fees,
            "net_pnl": metric.net_pnl,
            "frictionless_expectancy": metric.average_frictionless_pnl_per_trade,
            "net_expectancy": metric.expectancy,
            "profit_factor": metric.profit_factor,
            "return_percent": metric.return_percent,
            "win_rate_percent": metric.win_rate_percent,
            "average_winner": metric.average_winner,
            "average_loser": metric.average_loser,
            "payoff_ratio": metric.payoff_ratio,
            "maximum_drawdown_percent": metric.maximum_drawdown_percent,
            "market_exposure_percent": metric.market_exposure_percent,
            "stop_loss_percent": metric.stop_loss_percent,
            "take_profit_percent": metric.take_profit_percent,
            "trend_exit_percent": metric.trend_exit_percent,
            "time_exit_percent": metric.time_exit_percent,
            "median_mfe_r": metric.median_mfe_r,
            "median_mae_r": metric.median_mae_r,
            "reached_one_r_percent": metric.reached_one_r_percent,
            "reached_two_r_percent": metric.reached_two_r_percent,
            "average_holding_bars": metric.average_holding_bars,
            "delta_frictionless_expectancy_vs_h0": (
                delta.frictionless_expectancy if delta else None
            ),
            "delta_net_expectancy_vs_h0": delta.net_expectancy if delta else None,
            "delta_profit_factor_vs_h0": delta.profit_factor if delta else None,
            "delta_return_vs_h0": delta.return_percent if delta else None,
            "delta_maximum_drawdown_vs_h0": (
                delta.maximum_drawdown_percent if delta else None
            ),
            "delta_trades_vs_h0": delta.trades if delta else None,
            "delta_fees_vs_h0": delta.fees if delta else None,
            "delta_win_rate_vs_h0": delta.win_rate_percent if delta else None,
            "delta_payoff_vs_h0": delta.payoff_ratio if delta else None,
            "stress_expectancy": stress.expectancy if stress else None,
            "stress_expectancy_delta": (
                stress.expectancy - metric.expectancy if stress else None
            ),
            "low_sample_size": candidate.low_sample_size,
            "trade_count_collapse": candidate.trade_count_collapse,
        }

    @staticmethod
    def _stability_row(item: CandidateStability) -> dict[str, Any]:
        metric = item.combined_metrics
        stress = item.stress_combined_metrics
        return {
            "hypothesis": item.hypothesis_id,
            "eligible_windows": item.eligible_windows,
            "combined_trades": metric.trades,
            "combined_frictionless_pnl": metric.frictionless_pnl,
            "combined_slippage_drag": metric.slippage_drag,
            "combined_fees": metric.total_fees,
            "combined_net_pnl": metric.net_pnl,
            "combined_frictionless_expectancy": metric.average_frictionless_pnl_per_trade,
            "combined_net_expectancy": metric.expectancy,
            "combined_profit_factor": metric.profit_factor,
            "combined_return_percent": metric.return_percent,
            "combined_maximum_drawdown_percent": metric.maximum_drawdown_percent,
            "combined_win_rate_percent": metric.win_rate_percent,
            "combined_payoff_ratio": metric.payoff_ratio,
            "average_frictionless_pnl_per_trade": metric.average_frictionless_pnl_per_trade,
            "average_fee_per_trade": metric.average_fee_per_trade,
            "average_slippage_drag_per_trade": metric.average_slippage_drag_per_trade,
            "average_total_friction_per_trade": metric.average_total_friction_per_trade,
            "friction_to_absolute_frictionless_percent": metric.friction_to_absolute_frictionless_percent,
            "median_frictionless_expectancy": item.frictionless_expectancy_distribution.median,
            "frictionless_expectancy_p25": item.frictionless_expectancy_distribution.percentile_25,
            "frictionless_expectancy_p75": item.frictionless_expectancy_distribution.percentile_75,
            "median_net_expectancy": item.net_expectancy_distribution.median,
            "net_expectancy_p25": item.net_expectancy_distribution.percentile_25,
            "net_expectancy_p75": item.net_expectancy_distribution.percentile_75,
            "median_profit_factor": item.profit_factor_distribution.median,
            "profit_factor_p25": item.profit_factor_distribution.percentile_25,
            "profit_factor_p75": item.profit_factor_distribution.percentile_75,
            "median_return_percent": item.return_distribution.median,
            "return_p25": item.return_distribution.percentile_25,
            "return_p75": item.return_distribution.percentile_75,
            "median_maximum_drawdown_percent": item.drawdown_distribution.median,
            "maximum_drawdown_p25": item.drawdown_distribution.percentile_25,
            "maximum_drawdown_p75": item.drawdown_distribution.percentile_75,
            "frictionless_better_windows": item.frictionless_better_windows,
            "frictionless_better_percent": item.frictionless_better_percent,
            "net_better_windows": item.net_better_windows,
            "net_better_percent": item.net_better_percent,
            "profit_factor_better_windows": item.profit_factor_better_windows,
            "profit_factor_better_percent": item.profit_factor_better_percent,
            "lower_drawdown_windows": item.lower_drawdown_windows,
            "lower_drawdown_percent": item.lower_drawdown_percent,
            "trade_count_ratio": item.trade_count_ratio,
            "trade_count_collapse": item.trade_count_collapse,
            "stress_expectancy": stress.expectancy if stress else None,
            "gate": asdict(item.gate),
            "classification": item.classification,
            "gross_edge_costs_still_dominate": item.gross_edge_costs_dominate,
            "verification_status": "BLIND HOLDOUT NOT REVEALED",
        }

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            raise ValueError(f"No report rows exist for {path.name}.")
        flat_rows: list[dict[str, Any]] = []
        for row in rows:
            flat = _plain(row)
            for key, value in tuple(flat.items()):
                if isinstance(value, (dict, list)):
                    flat[key] = json.dumps(value, sort_keys=True)
            flat_rows.append(flat)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)

    @staticmethod
    def _extreme_windows(result: MultiRegimeResearchResult) -> dict[str, Any]:
        output: dict[str, Any] = {}
        eligible = tuple(window for window in result.windows if window.eligible)
        for stability in result.stability:
            hypothesis_id = stability.hypothesis_id
            pairs = tuple(
                (
                    window,
                    next(
                        item
                        for item in window.candidates
                        if item.hypothesis_id is hypothesis_id
                    ),
                )
                for window in eligible
            )
            if not pairs:
                continue
            selectors = {
                "best_net_expectancy": max(
                    pairs, key=lambda item: item[1].metrics.expectancy
                ),
                "worst_net_expectancy": min(
                    pairs, key=lambda item: item[1].metrics.expectancy
                ),
                "best_frictionless_expectancy": max(
                    pairs,
                    key=lambda item: item[1].metrics.average_frictionless_pnl_per_trade,
                ),
                "worst_frictionless_expectancy": min(
                    pairs,
                    key=lambda item: item[1].metrics.average_frictionless_pnl_per_trade,
                ),
                "lowest_maximum_drawdown": min(
                    pairs,
                    key=lambda item: item[1].metrics.maximum_drawdown_percent,
                ),
                "worst_maximum_drawdown": max(
                    pairs,
                    key=lambda item: item[1].metrics.maximum_drawdown_percent,
                ),
            }
            output[hypothesis_id.value] = {
                name: {
                    "window_id": pair[0].window.window_id,
                    "start": pair[0].window.start,
                    "end": pair[0].window.end,
                    "trades": pair[1].metrics.trades,
                    "frictionless_expectancy": pair[1].metrics.average_frictionless_pnl_per_trade,
                    "net_expectancy": pair[1].metrics.expectancy,
                    "maximum_drawdown_percent": pair[1].metrics.maximum_drawdown_percent,
                    "btc_return_percent": pair[0].market.btc_return_percent,
                    "realized_volatility_percent": pair[0].market.realized_candle_volatility_percent,
                }
                for name, pair in selectors.items()
            }
        return output

    def _markdown(self, result: MultiRegimeResearchResult) -> str:
        lines = [
            f"# {self.heading}\n",
            "**All evaluated data is CONSUMED_RESEARCH_DATA.**\n",
            "**Blind holdout: LOCKED_BLIND_HOLDOUT — NOT REVEALED — NOT EVALUATED.**\n",
            f"Windows: {len(result.windows)}; eligible: {sum(item.eligible for item in result.windows)}.\n",
            "| ID | Trades | Frictionless Exp. | Net Exp. | PF | Net PnL | Classification |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for item in result.stability:
            metric = item.combined_metrics
            lines.append(
                f"| {item.hypothesis_id.value} | {metric.trades} | "
                f"{metric.average_frictionless_pnl_per_trade} | {metric.expectancy} | "
                f"{metric.profit_factor} | {metric.net_pnl} | {item.classification.value} |"
            )
        lines.extend(
            [
                "\nBEST HISTORICAL WINDOW IS DESCRIPTIVE ONLY.",
                "Multi-regime historical consistency does not prove future profitability.",
                f"A {self.eligibility_label} label means only eligible for further controlled research; it is not validation.\n",
            ]
        )
        return "\n".join(lines)
