"""Deterministic JSON/CSV output for V3.1 diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

from src.diagnostics.models import (
    ExitReasonDiagnostics,
    GroupDiagnostics,
    PeriodDiagnostics,
    StrategyDiagnosticsReport,
    TradeDiagnostic,
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
class DiagnosticsReportPaths:
    directory: Path
    summary: Path
    comparison: Path


def _comparison_row(period: PeriodDiagnostics, label: str) -> dict[str, Any]:
    exits = {row.exit_reason.value: row for row in period.exit_reasons}
    thresholds = {row.group: row for row in period.excursions.mfe_thresholds}
    features = {row.feature: row for row in period.entry_features}
    return {
        "period": label,
        "trades": period.total_trades,
        "frictionless_pnl": period.costs.frictionless_pnl,
        "gross_pnl": period.costs.slippage_adjusted_gross_pnl,
        "slippage_drag": period.costs.slippage_drag,
        "fees": period.costs.fee_drag,
        "net_pnl": period.costs.net_pnl,
        "win_rate_percent": period.win_rate_percent,
        "profit_factor": period.profit_factor,
        "expectancy": period.expectancy,
        "average_winner": period.outcomes.average_win,
        "average_loser": period.outcomes.average_loss,
        "average_winner_r": period.outcomes.average_winner_r,
        "average_loser_r": period.outcomes.average_loser_r,
        "median_mfe_r": period.excursions.median_mfe_r,
        "median_mae_r": period.excursions.median_mae_r,
        "reached_1r_percent": (
            thresholds[">=1R"].percentage_of_trades if ">=1R" in thresholds else None
        ),
        "reached_2r_percent": (
            thresholds[">=2R"].percentage_of_trades if ">=2R" in thresholds else None
        ),
        "average_holding_bars": period.holding.average_bars,
        "stop_loss_percent": (
            exits["STOP_LOSS"].percentage_of_trades
            if "STOP_LOSS" in exits
            else Decimal("0")
        ),
        "take_profit_percent": (
            exits["TAKE_PROFIT"].percentage_of_trades
            if "TAKE_PROFIT" in exits
            else Decimal("0")
        ),
        "trend_exit_percent": (
            exits["TREND_EXIT"].percentage_of_trades
            if "TREND_EXIT" in exits
            else Decimal("0")
        ),
        "time_exit_percent": (
            exits["TIME_EXIT"].percentage_of_trades
            if "TIME_EXIT" in exits
            else Decimal("0")
        ),
        "average_entry_rsi": features["RSI"].all_average,
        "average_entry_atr_percent": features["ATR %"].all_average,
        "average_entry_volume_ratio": features["Volume Ratio"].all_average,
        "average_entry_ema_spread_percent": features["EMA spread %"].all_average,
        "evidence_strength": period.evidence_strength,
    }


class DiagnosticsReportWriter:
    def __init__(self, root: str | Path = "reports/diagnostics") -> None:
        self.root = Path(root)

    def write(
        self, report: StrategyDiagnosticsReport, *, run_id: str | None = None
    ) -> DiagnosticsReportPaths:
        canonical = json.dumps(
            _plain(asdict(report)), sort_keys=True, separators=(",", ":")
        )
        deterministic_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        selected_id = run_id or deterministic_id
        if not selected_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in selected_id
        ):
            raise ValueError("Diagnostic run_id contains invalid characters.")
        directory = self.root / selected_id
        directory.mkdir(parents=True, exist_ok=True)
        paths = DiagnosticsReportPaths(
            directory=directory,
            summary=directory / "diagnostics_summary.json",
            comparison=directory / "period_comparison.csv",
        )
        summary = {
            "research_run_id": report.research_run_id,
            "symbol": report.symbol,
            "interval": report.interval,
            "strategy_name": report.strategy_name,
            "strategy_version": report.strategy_version,
            "minimum_trades_warning": report.minimum_trades_warning,
            "intrabar_limitation": (
                "OHLC reveals observed highs/lows but not their intrabar ordering. "
                "Exit-bar excursions may include a price extreme whose order relative "
                "to the protective trigger is unknowable."
            ),
            "post_trade_boundary": (
                "MFE, MAE, timing, and ex-post outcomes exist only in diagnostics and "
                "are never exposed to StrategyContext or StrategyDecision."
            ),
            "combined_label": report.combined_label,
            "periods": [
                _comparison_row(period, period.period.value)
                for period in report.periods
            ],
            "combined": _comparison_row(report.combined, report.combined_label),
            "cross_period_observations": [
                asdict(item) for item in report.cross_period_observations
            ],
        }
        paths.summary.write_text(
            json.dumps(_plain(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        comparison_rows = [
            _comparison_row(period, period.period.value) for period in report.periods
        ]
        self._write_dicts(paths.comparison, comparison_rows)
        for period in report.periods:
            self._write_period(directory / period.period.value, period)
        self._write_period(directory / "combined_descriptive_only", report.combined)
        return paths

    def _write_period(self, directory: Path, period: PeriodDiagnostics) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._write_dataclasses(
            directory / "trade_diagnostics.csv", period.trades, TradeDiagnostic
        )
        self._write_dataclasses(
            directory / "exit_analysis.csv",
            period.exit_reasons,
            ExitReasonDiagnostics,
        )
        self._write_dataclasses(
            directory / "feature_buckets.csv",
            period.feature_buckets,
            GroupDiagnostics,
        )
        self._write_dataclasses(
            directory / "regime_analysis.csv",
            period.market_regimes + period.volatility_regimes,
            GroupDiagnostics,
        )
        self._write_dataclasses(
            directory / "holding_analysis.csv",
            period.holding_buckets,
            GroupDiagnostics,
        )
        self._write_dataclasses(
            directory / "time_analysis.csv",
            period.utc_hours + period.utc_day_groups,
            GroupDiagnostics,
        )
        payload = asdict(period)
        payload.pop("trades")
        (directory / "summary.json").write_text(
            json.dumps(_plain(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_dataclasses(
        path: Path, rows: tuple[Any, ...], model: type[Any]
    ) -> None:
        field_names = tuple(item.name for item in fields(model))
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=field_names)
            writer.writeheader()
            for row in rows:
                writer.writerow(_plain(asdict(row)))

    @staticmethod
    def _write_dicts(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow(_plain(row))

