"""Deterministic research fingerprint, comparison, and period reports."""

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
from src.research.runner import ResearchResult
from src.strategy.baseline_trend import TrendMomentumBaselineStrategy


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
class ResearchReportPaths:
    directory: Path
    config: Path
    summary: Path
    comparison: Path


class ResearchReportWriter:
    def __init__(self, root: str | Path = "reports/research") -> None:
        self.root = Path(root)

    @staticmethod
    def _configuration(result: ResearchResult) -> dict[str, Any]:
        first = result.periods[0].baseline
        return {
            "strategy_name": TrendMomentumBaselineStrategy.NAME,
            "strategy_version": TrendMomentumBaselineStrategy.VERSION,
            "strategy_parameters": asdict(result.config.strategy),
            "periods": {
                item.period.value: {
                    "start": item.start,
                    "end": item.end,
                    "boundary": "[start,end)",
                }
                for item in (
                    result.config.development,
                    result.config.validation,
                    result.config.out_of_sample,
                )
            },
            "symbol": first.dataset.symbol,
            "interval": first.dataset.interval,
            "data_source": "BINANCE_PUBLIC_SPOT",
            "data_source_id": first.dataset.source.value,
            "initial_capital_usdc": result.config.backtest.initial_capital_usdc,
            "fee_bps": result.config.backtest.fee_bps,
            "slippage_bps": result.config.backtest.slippage_bps,
            "execution_timing_model": result.config.backtest.execution_timing,
            "ambiguous_bar_policy": result.config.backtest.ambiguous_bar_policy,
            "minimum_trades_warning": result.config.minimum_trades_warning,
            "cost_stress_enabled": result.config.cost_stress,
            "cost_stress_multiplier": 2 if result.config.cost_stress else None,
            "research_accounts": "INDEPENDENT_SAME_INITIAL_CAPITAL",
        }

    def write(
        self, result: ResearchResult, *, run_id: str | None = None
    ) -> ResearchReportPaths:
        config = self._configuration(result)
        fingerprint = {
            "config": config,
            "comparison": [asdict(row) for row in result.comparison],
            "stability": asdict(result.stability),
            "periods": [
                {
                    "period": period.period,
                    "signals": [asdict(signal) for signal in period.signals],
                    "trades": [asdict(trade) for trade in period.baseline.trades],
                    "equity": [asdict(point) for point in period.baseline.equity_curve],
                    "benchmark": asdict(period.benchmark.metrics),
                    "stress": (
                        asdict(period.cost_stress.metrics)
                        if period.cost_stress is not None
                        else None
                    ),
                }
                for period in result.periods
            ],
        }
        canonical = json.dumps(
            _plain(fingerprint), sort_keys=True, separators=(",", ":")
        )
        deterministic_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        selected_id = run_id or deterministic_id
        if not selected_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in selected_id
        ):
            raise ValueError("Research run_id contains invalid characters.")

        directory = self.root / selected_id
        directory.mkdir(parents=True, exist_ok=True)
        paths = ResearchReportPaths(
            directory=directory,
            config=directory / "config.json",
            summary=directory / "research_summary.json",
            comparison=directory / "comparison.csv",
        )
        paths.config.write_text(
            json.dumps(_plain(config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths.summary.write_text(
            json.dumps(
                _plain(
                    {
                        "stability": asdict(result.stability),
                        "comparison": [asdict(row) for row in result.comparison],
                    }
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self._write_comparison(paths.comparison, result)

        for period in result.periods:
            period_directory = directory / period.period.value
            period_directory.mkdir(parents=True, exist_ok=True)
            BacktestReportWriter(period_directory).write(
                period.baseline, run_id="baseline"
            )
            BacktestReportWriter(period_directory).write(
                period.benchmark, run_id="buy_and_hold_benchmark"
            )
            if period.cost_stress is not None:
                BacktestReportWriter(period_directory).write(
                    period.cost_stress, run_id="cost_stress_2x"
                )
            self._write_signals(period_directory / "signals.csv", period.signals)
        return paths

    @staticmethod
    def _write_comparison(path: Path, result: ResearchResult) -> None:
        rows = [asdict(row) for row in result.comparison]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow(_plain(row))

    @staticmethod
    def _write_signals(path: Path, signals: tuple[Any, ...]) -> None:
        fields = (
            "timestamp",
            "action",
            "close",
            "ema_fast",
            "ema_slow",
            "rsi",
            "atr",
            "volume_ratio",
            "position_state",
            "reason_code",
            "reason",
        )
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for signal in signals:
                writer.writerow(_plain(asdict(signal)))

