"""Deterministic CSV/JSON backtest report generation."""

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

from src.backtest.engine import BacktestEngine
from src.backtest.models import BacktestResult
from src.historical.dataset import HistoricalDataset
from src.market.intervals import next_open_time


def _plain(value: Any) -> Any:
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


@dataclass(frozen=True, slots=True)
class ReportPaths:
    directory: Path
    trades: Path
    equity: Path
    summary: Path
    config: Path


def dataset_checksum(dataset: HistoricalDataset) -> str:
    """Return the canonical V2 candle checksum used in report fingerprints."""

    rows = (
        "|".join(
            (
                candle.timestamp.isoformat(),
                candle.symbol,
                candle.interval,
                str(candle.open),
                str(candle.high),
                str(candle.low),
                str(candle.close),
                str(candle.volume),
                str(candle.is_closed),
            )
        )
        for candle in dataset.candles
    )
    payload = "\n".join(rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class BacktestReportWriter:
    """Write one reproducible directory identified by deterministic contents."""

    def __init__(self, root: str | Path = "reports/backtests") -> None:
        self.root = Path(root)

    @staticmethod
    def _dataset_checksum(result: BacktestResult) -> str:
        return dataset_checksum(result.dataset)

    @staticmethod
    def _configuration(result: BacktestResult) -> dict[str, Any]:
        dataset = result.dataset
        return {
            "engine_version": BacktestEngine.ENGINE_VERSION,
            "symbol": dataset.symbol,
            "interval": dataset.interval,
            "data_source": "BINANCE_PUBLIC_SPOT",
            "data_source_id": dataset.source.value,
            "start": dataset.candles[0].timestamp,
            "end": next_open_time(dataset.candles[-1].timestamp, dataset.interval),
            "candle_count": len(dataset.candles),
            "dataset_sha256": BacktestReportWriter._dataset_checksum(result),
            "initial_capital_usdc": result.config.initial_capital_usdc,
            "fee_bps": result.config.fee_bps,
            "slippage_bps": result.config.slippage_bps,
            "execution_timing_model": result.config.execution_timing,
            "ambiguous_bar_policy": result.config.ambiguous_bar_policy,
            "end_of_backtest_policy": "FINAL_CLOSED_CANDLE_WITH_SELL_COSTS",
            "warmup_candles": result.config.warmup_candles,
        }

    def write(self, result: BacktestResult, *, run_id: str | None = None) -> ReportPaths:
        config = self._configuration(result)
        fingerprint_input = {
            "config": config,
            "trades": [asdict(trade) for trade in result.trades],
            "equity": [asdict(point) for point in result.equity_curve],
            "summary": asdict(result.metrics),
        }
        canonical = json.dumps(
            _plain(fingerprint_input), sort_keys=True, separators=(",", ":")
        )
        deterministic_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        selected_run_id = run_id or deterministic_id
        if not selected_run_id or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for character in selected_run_id
        ):
            raise ValueError("Backtest run_id contains invalid characters.")

        directory = self.root / selected_run_id
        directory.mkdir(parents=True, exist_ok=True)
        paths = ReportPaths(
            directory=directory,
            trades=directory / "trades.csv",
            equity=directory / "equity.csv",
            summary=directory / "summary.json",
            config=directory / "config.json",
        )
        self._write_trades(paths.trades, result)
        self._write_equity(paths.equity, result)
        paths.summary.write_text(
            json.dumps(_plain(asdict(result.metrics)), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths.config.write_text(
            json.dumps(_plain(config), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return paths

    @staticmethod
    def _write_trades(path: Path, result: BacktestResult) -> None:
        fields = tuple(asdict(result.trades[0])) if result.trades else (
            "trade_id",
            "entry_signal_time",
            "entry_time",
            "entry_price",
            "exit_signal_time",
            "exit_time",
            "exit_price",
            "quantity",
            "entry_notional",
            "exit_notional",
            "entry_fee",
            "exit_fee",
            "total_fee",
            "gross_pnl",
            "net_pnl",
            "return_percent",
            "bars_held",
            "exit_reason",
        )
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for trade in result.trades:
                writer.writerow(_plain(asdict(trade)))

    @staticmethod
    def _write_equity(path: Path, result: BacktestResult) -> None:
        fields = ("timestamp", "cash", "position_value", "total_equity")
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for point in result.equity_curve:
                writer.writerow(_plain(asdict(point)))
