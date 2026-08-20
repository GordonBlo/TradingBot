"""Immutable V3.2.2 preregistration for H0/H1/H5/H6/H7 research."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from src.backtest.models import BacktestConfig
from src.hypotheses.mechanisms import MechanismSuiteConfig, mechanism_registry
from src.hypotheses.models import HypothesisSuiteConfig
from src.research.multiregime.manifest import PreparedMultiRegimeManifest, plain
from src.research.multiregime.models import MultiRegimeConfig, ResearchPartitionMetadata
from src.strategy.models import TrendMomentumConfig


class MechanismManifestStore:
    def __init__(
        self,
        root: str | Path = "research/mechanisms",
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
        source_multiregime_run_id: str,
        source_multiregime_manifest_sha256: str,
        source_control_metrics: dict[str, Any],
        partitions: tuple[ResearchPartitionMetadata, ...],
        multiregime_config: MultiRegimeConfig,
        v32_config: HypothesisSuiteConfig,
        mechanism_config: MechanismSuiteConfig,
        strategy_config: TrendMomentumConfig,
        backtest_config: BacktestConfig,
        cost_stress_enabled: bool,
    ) -> PreparedMultiRegimeManifest:
        configuration = {
            "project_version": "3.2.2",
            "symbol": symbol,
            "interval": interval,
            "source_multiregime_run_id": source_multiregime_run_id,
            "source_multiregime_manifest_sha256": source_multiregime_manifest_sha256,
            "source_control_metrics": source_control_metrics,
            "partitions": [asdict(item) for item in partitions],
            "fixed_window_definition": asdict(multiregime_config),
            "frozen_v32_control_parameters": asdict(v32_config),
            "frozen_baseline_parameters": asdict(strategy_config),
            "frozen_mechanisms": [asdict(item) for item in mechanism_registry(mechanism_config)],
            "base_cost_assumptions": asdict(backtest_config),
            "cost_stress_enabled": cost_stress_enabled,
            "stress_cost_assumptions": {
                **asdict(backtest_config),
                "fee_bps": backtest_config.fee_bps * 2,
                "slippage_bps": backtest_config.slippage_bps * 2,
            },
            "cost_stress_semantics": (
                "Execution costs double; H7's preregistered opportunity threshold "
                "remains based on frozen base-cost assumptions so stress does not "
                "change strategy signals."
            ),
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
                "next_stage_extra": "combined frictionless expectancy > 0",
            },
            "holdout_policy": {
                "required_status": "LOCKED_BLIND_HOLDOUT",
                "revealed": False,
                "consumed": False,
                "evaluated": False,
                "warmup_allowed": False,
            },
            "research_integrity": {
                "dataset_status": "CONSUMED_RESEARCH_DATA",
                "parameter_search": False,
                "candidate_combinations": False,
                "new_candidates": ["H5", "H6", "H7"],
                "frozen_controls": ["H0", "H1"],
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
                raise ValueError("Existing V3.2.2 mechanism manifest is not immutable.")
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
