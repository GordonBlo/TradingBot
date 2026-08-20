"""Validated loader for existing V3 research artifacts and local candles."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.backtest.models import (
    AmbiguousBarPolicy,
    BacktestConfig,
    ExitReason,
    Trade,
)
from src.backtest.report import dataset_checksum
from src.diagnostics.models import (
    DiagnosticPeriodInput,
    DiagnosticRunInput,
)
from src.historical.storage import HistoricalDatasetStore
from src.research.dataset_split import (
    ResearchPeriod,
    ResearchPeriodRange,
    split_dataset,
)
from src.research.evaluation import SignalRecord
from src.strategy.baseline_trend import TrendMomentumBaselineStrategy
from src.strategy.models import (
    DecisionReason,
    StrategyAction,
    TrendMomentumConfig,
)


def _decimal_or_none(value: str) -> Decimal | None:
    return Decimal(value) if value.strip() else None


class ResearchRunLoader:
    """Load diagnostics from reports without rerunning or contacting Binance."""

    def __init__(
        self,
        *,
        research_root: str | Path = "reports/research",
        data_root: str | Path = "data/historical",
    ) -> None:
        self.research_root = Path(research_root)
        self.data_root = Path(data_root)

    def resolve(self, path_or_id: str | Path) -> Path:
        supplied = Path(path_or_id)
        directory = supplied if supplied.is_dir() else self.research_root / supplied
        if not directory.is_dir() or not (directory / "config.json").is_file():
            raise ValueError(f"V3 research run not found: {path_or_id}")
        return directory

    def load(self, path_or_id: str | Path) -> DiagnosticRunInput:
        directory = self.resolve(path_or_id)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if config.get("strategy_name") != TrendMomentumBaselineStrategy.NAME:
            raise ValueError("Diagnostics support the frozen V3 baseline report only.")
        if config.get("strategy_version") != TrendMomentumBaselineStrategy.VERSION:
            raise ValueError("Research strategy version does not match this V3 build.")
        if config.get("execution_timing_model") != "NEXT_BAR_OPEN":
            raise ValueError("Research report does not use next-bar-open execution.")

        symbol = str(config["symbol"])
        interval = str(config["interval"])
        dataset = HistoricalDatasetStore(self.data_root).load(symbol, interval)
        if dataset is None:
            raise ValueError("Validated local candle cache is required for diagnostics.")

        period_ranges = tuple(
            ResearchPeriodRange(
                period,
                datetime.fromisoformat(config["periods"][period.value]["start"]),
                datetime.fromisoformat(config["periods"][period.value]["end"]),
            )
            for period in (
                ResearchPeriod.DEVELOPMENT,
                ResearchPeriod.VALIDATION,
                ResearchPeriod.OUT_OF_SAMPLE,
            )
        )
        split = split_dataset(dataset, *period_ranges)
        backtest_config = BacktestConfig(
            initial_capital_usdc=Decimal(config["initial_capital_usdc"]),
            fee_bps=Decimal(config["fee_bps"]),
            slippage_bps=Decimal(config["slippage_bps"]),
            ambiguous_bar_policy=AmbiguousBarPolicy(
                config["ambiguous_bar_policy"]
            ),
        )
        strategy_config = TrendMomentumConfig(**config["strategy_parameters"])

        periods: list[DiagnosticPeriodInput] = []
        for period, period_dataset in split.items():
            period_directory = directory / period.value
            baseline_directory = period_directory / "baseline"
            baseline_config = json.loads(
                (baseline_directory / "config.json").read_text(encoding="utf-8")
            )
            actual_checksum = dataset_checksum(period_dataset)
            if baseline_config.get("dataset_sha256") != actual_checksum:
                raise ValueError(
                    f"Cached candles do not match {period.value} research checksum."
                )
            trades = self._load_trades(baseline_directory / "trades.csv")
            signals = self._load_signals(period_directory / "signals.csv")
            periods.append(
                DiagnosticPeriodInput(period, period_dataset, trades, signals)
            )

        typed_periods = tuple(periods)
        assert len(typed_periods) == 3
        return DiagnosticRunInput(
            research_run_id=directory.name,
            symbol=symbol,
            interval=interval,
            backtest_config=backtest_config,
            strategy_config=strategy_config,
            minimum_trades_warning=int(config["minimum_trades_warning"]),
            periods=typed_periods,  # type: ignore[arg-type]
        )

    @staticmethod
    def _load_trades(path: Path) -> tuple[Trade, ...]:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = tuple(csv.DictReader(stream))
        return tuple(
            Trade(
                trade_id=row["trade_id"],
                entry_signal_time=datetime.fromisoformat(row["entry_signal_time"]),
                entry_time=datetime.fromisoformat(row["entry_time"]),
                entry_price=Decimal(row["entry_price"]),
                exit_signal_time=datetime.fromisoformat(row["exit_signal_time"]),
                exit_time=datetime.fromisoformat(row["exit_time"]),
                exit_price=Decimal(row["exit_price"]),
                quantity=Decimal(row["quantity"]),
                entry_notional=Decimal(row["entry_notional"]),
                exit_notional=Decimal(row["exit_notional"]),
                entry_fee=Decimal(row["entry_fee"]),
                exit_fee=Decimal(row["exit_fee"]),
                total_fee=Decimal(row["total_fee"]),
                gross_pnl=Decimal(row["gross_pnl"]),
                net_pnl=Decimal(row["net_pnl"]),
                return_percent=Decimal(row["return_percent"]),
                bars_held=int(row["bars_held"]),
                exit_reason=ExitReason(row["exit_reason"]),
            )
            for row in rows
        )

    @staticmethod
    def _load_signals(path: Path) -> tuple[SignalRecord, ...]:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows: tuple[dict[str, Any], ...] = tuple(csv.DictReader(stream))
        return tuple(
            SignalRecord(
                timestamp=datetime.fromisoformat(row["timestamp"]),
                action=StrategyAction(row["action"]),
                close=Decimal(row["close"]),
                ema_fast=_decimal_or_none(row["ema_fast"]),
                ema_slow=_decimal_or_none(row["ema_slow"]),
                rsi=_decimal_or_none(row["rsi"]),
                atr=_decimal_or_none(row["atr"]),
                volume_ratio=_decimal_or_none(row["volume_ratio"]),
                position_state=row["position_state"],
                reason_code=DecisionReason(row["reason_code"]),
                reason=row["reason"],
            )
            for row in rows
        )

