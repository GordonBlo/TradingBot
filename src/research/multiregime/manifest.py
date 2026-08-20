"""Immutable V3.2.1 preregistration built before expanded strategy replay."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from src.backtest.models import BacktestConfig
from src.hypotheses.models import HypothesisSuiteConfig, hypothesis_registry
from src.research.multiregime.models import (
    MultiRegimeConfig,
    ResearchPartitionMetadata,
)
from src.strategy.models import TrendMomentumConfig


def plain(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class PreparedMultiRegimeManifest:
    run_id: str
    configuration_sha256: str
    path: Path
    payload: dict[str, Any]


class MultiRegimeManifestStore:
    def __init__(
        self,
        root: str | Path = "research/multiregime",
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.root = Path(root)
        self._clock = clock

    def prepare(
        self,
        *,
        symbol: str,
        interval: str,
        requested_expansion_start: datetime,
        actual_expansion_start: datetime,
        expansion_end: datetime,
        partitions: tuple[ResearchPartitionMetadata, ...],
        multiregime_config: MultiRegimeConfig,
        hypothesis_config: HypothesisSuiteConfig,
        strategy_config: TrendMomentumConfig,
        backtest_config: BacktestConfig,
        cost_stress_enabled: bool,
    ) -> PreparedMultiRegimeManifest:
        configuration = {
            "project_version": "3.2.1",
            "symbol": symbol,
            "interval": interval,
            "historical_expansion": {
                "requested_start": requested_expansion_start,
                "actual_available_start": actual_expansion_start,
                "end": expansion_end,
            },
            "partitions": [asdict(item) for item in partitions],
            "window_definition": asdict(multiregime_config),
            "frozen_hypotheses": [
                asdict(item) for item in hypothesis_registry(hypothesis_config)
            ],
            "frozen_baseline_parameters": asdict(strategy_config),
            "base_cost_assumptions": asdict(backtest_config),
            "cost_stress_enabled": cost_stress_enabled,
            "stress_cost_assumptions": {
                **asdict(backtest_config),
                "fee_bps": backtest_config.fee_bps * Decimal("2"),
                "slippage_bps": backtest_config.slippage_bps * Decimal("2"),
            },
            "regime_definitions": {
                "long_trend": {
                    "LONG_TREND_UP": "close > causal EMA200 and causal EMA50 > EMA200",
                    "LONG_TREND_DOWN": "close < causal EMA200 and causal EMA50 < EMA200",
                    "LONG_TREND_MIXED": "otherwise",
                },
                "volatility": {
                    "measure": "ATR14 / close * 100",
                    "lookback": 200,
                    "current_observation_excluded": True,
                    "low": "<= causal Q33",
                    "medium": "between causal Q33 and Q67",
                    "high": ">= causal Q67",
                },
                "strategy_effect": "DIAGNOSTIC_ONLY",
            },
            "support_criteria": {
                "minimum_eligible_windows": 6,
                "combined_frictionless_expectancy": "candidate > H0",
                "combined_net_expectancy": "candidate > H0",
                "combined_profit_factor": "candidate >= H0",
                "frictionless_window_consistency_percent": "at least 60",
                "net_window_consistency_percent": "at least 60",
                "combined_trade_count_ratio": "at least 0.40 of H0",
                "combined_max_drawdown": "no more than 1.25 times H0",
                "stress_expectancy": "candidate >= H0 under identical doubled costs",
                "v3_3_eligibility_extra": "combined frictionless expectancy > 0",
            },
            "holdout_policy": {
                "required_status": "LOCKED_BLIND_HOLDOUT",
                "revealed": False,
                "evaluated": False,
                "warmup_allowed": False,
            },
        }
        canonical = json.dumps(
            plain(configuration), sort_keys=True, separators=(",", ":")
        )
        configuration_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        run_id = configuration_sha256[:16]
        path = self.root / run_id / "manifest.json"
        payload = {
            "run_id": run_id,
            "configuration_sha256": configuration_sha256,
            "created_at": self._clock().astimezone(timezone.utc),
            **configuration,
        }
        serialized = json.dumps(plain(payload), indent=2, sort_keys=True) + "\n"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("configuration_sha256") != configuration_sha256:
                raise ValueError("Existing multi-regime manifest is not immutable.")
            payload = existing
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            try:
                temporary.write_text(serialized, encoding="utf-8")
                temporary.replace(path)
            finally:
                if temporary.exists():
                    temporary.unlink()
        return PreparedMultiRegimeManifest(
            run_id=run_id,
            configuration_sha256=configuration_sha256,
            path=path,
            payload=payload,
        )
